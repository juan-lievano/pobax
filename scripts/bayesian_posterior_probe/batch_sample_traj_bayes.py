#!/usr/bin/env python3
import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
from jax import random as jr
import numpy as np
import orbax.checkpoint
import pandas as pd

from pobax.envs.jax.lightbulbs import LightBulbs
from pobax.models import ScannedRNN, get_gymnax_network_fn
from pobax.models.discrete import DiscreteActorCriticRNN


def init_prev_action_onehot(size: int, action_int):
    return jax.nn.one_hot(action_int, size + 1, dtype=jnp.float32)

def tree_select(done_scalar_bool, a_tree, b_tree):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(done_scalar_bool, a, b), a_tree, b_tree)

def update_posterior(goals: jnp.ndarray,
                     prev_posterior: jnp.ndarray,
                     curr_obs: jnp.ndarray,
                     next_obs: jnp.ndarray,
                     robot_action: jnp.int32) -> jnp.ndarray:
    delta = (next_obs - curr_obs).at[robot_action].set(0)
    pos_ok = jnp.all(jnp.where(delta > 0, goals == 1, True), axis=1)
    neg_ok = jnp.all(jnp.where(delta < 0, goals == 0, True), axis=1)
    compat = pos_ok & neg_ok
    new = prev_posterior * compat.astype(prev_posterior.dtype)
    z = new.sum()
    new_uniform = jnp.where(compat, 1.0 / jnp.maximum(1, compat.sum()), 0.0)
    return jnp.where(z > 0, new / z, new_uniform)

def infer_human_action(curr_obs: jnp.ndarray,
                       next_obs: jnp.ndarray,
                       robot_action: jnp.int32,
                       size: int) -> jnp.int32:
    candidates = (curr_obs != next_obs).astype(jnp.int32)
    candidates = candidates.at[robot_action].set(0)
    idx = jnp.argmax(candidates)
    exists = candidates.max() > 0
    return jnp.where(exists, idx.astype(jnp.int32), jnp.int32(size))

def rollout_single(
    key,
    reset_fn,
    step_fn,
    env_params,
    apply_fn,
    weights,
    hidden_size: int,
    size: int,
    max_length: int,
    goals: jnp.ndarray,
    prior_posterior: jnp.ndarray,
):
    key0, key = jr.split(key)
    obs0, state0 = reset_fn(key=key0, params=None)
    prev_action = init_prev_action_onehot(size, size)
    h0 = ScannedRNN.initialize_carry(batch_size=1, hidden_size=hidden_size)
    done0 = jnp.array(False)
    post0 = prior_posterior

    def step(carry, _):
        key, state, obs, h, prev_a, done, post = carry
        key, k_samp, k_step = jr.split(key, 3)

        x = jnp.concatenate([obs, prev_a], axis=-1)[None, None, :]
        done_b = done[None, None]

        h_out = h
        post_out = post
        valid = jnp.logical_not(done)

        h_new, pi, _ = apply_fn(weights, h, (x, done_b))
        act = pi.sample(seed=k_samp)[0, 0].astype(jnp.int32)
        obs1, state1, reward, done1, _ = step_fn(k_step, state, act, env_params)
        done_next = jnp.logical_or(done, done1)

        human_act = infer_human_action(obs, obs1, act, size)

        post1 = update_posterior(goals, post, obs, obs1, act)
        post = jnp.where(done, post, post1)

        obs = jnp.where(done, obs, obs1)
        h = jnp.where(done, h, h_new)
        prev_a = jnp.where(done, prev_a, init_prev_action_onehot(size, act))
        state = tree_select(done, state, state1)

        carry = (key, state, obs, h, prev_a, done_next, post)
        out = (h_out, post_out, valid, act, human_act)
        return carry, out

    carry0 = (key, state0, obs0, h0, prev_action, done0, post0)
    _, (h_seq, post_seq, mask_seq, robot_seq, human_seq) = lax.scan(
        step, carry0, jnp.arange(max_length)
    )
    length = mask_seq.sum(dtype=jnp.int32)
    return h_seq, post_seq, mask_seq, robot_seq, human_seq, length

def rollout_batch(keys, reset_fn, step_fn, env_params, apply_fn, weights,
                  hidden_size, size, max_length, goals, prior_posterior):
    fn = jax.vmap(
        rollout_single,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None),
    )
    return fn(keys, reset_fn, step_fn, env_params, apply_fn, weights,
              hidden_size, size, max_length, goals, prior_posterior)

rollout_batch_jit = jax.jit(
    rollout_batch,
    static_argnames=("reset_fn", "step_fn", "apply_fn", "hidden_size", "size", "max_length"),
)

def load_config(path_str: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        alt = Path("configs") / path_str
        if alt.exists():
            p = alt
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path_str}")
    with open(p, "r") as f:
        return json.load(f)

def to_pylist(x) -> list:
    return np.asarray(x).tolist()

def shard_to_rows(h_h: np.ndarray,
                  post_h: np.ndarray,
                  robot_h: np.ndarray,
                  human_h: np.ndarray,
                  len_h: np.ndarray,
                  global_offset: int):
    rows = {
        "rnn_hidden_state": [],
        "bayes_posterior": [],
        "robot_action": [],
        "human_action": [],
        "trajectory_id": [],
        "time_step": [],
    }
    n_this, T = h_h.shape[0], h_h.shape[1]
    for local_id in range(n_this):
        traj_id = global_offset + local_id
        L = int(len_h[local_id])
        if L <= 0:
            continue
        rows["rnn_hidden_state"].extend(to_pylist(h_h[local_id, :L]))
        rows["bayes_posterior"].extend(to_pylist(post_h[local_id, :L]))
        rows["robot_action"].extend(to_pylist(robot_h[local_id, :L]))
        rows["human_action"].extend(to_pylist(human_h[local_id, :L]))
        rows["trajectory_id"].extend([traj_id] * L)
        rows["time_step"].extend(list(range(L)))
    return rows


def main():

    t0 = time.time()

    parser = argparse.ArgumentParser(description="Sample trajectories with online Bayes posterior, batched, streaming output.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--shard_size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    env = LightBulbs(config["environment_config_path"])
    network_class, action_size = get_gymnax_network_fn(env, env.default_params)
    assert network_class is DiscreteActorCriticRNN, f"Expected DiscreteActorCriticRNN, got {network_class}"
    assert action_size == env.size + 1, f"Expected {env.size + 1} actions, got {action_size}"

    goals = env.goals
    prior_posterior = env.goal_distribution

    ckpt_path = Path(config["checkpoint_directory_path"]).resolve()
    restored = orbax.checkpoint.PyTreeCheckpointer().restore(ckpt_path)
    args_rest = restored["args"]
    hidden_size = int(args_rest["hidden_size"])
    double_critic = bool(args_rest["double_critic"])

    model = DiscreteActorCriticRNN(action_dim=action_size, double_critic=double_critic, hidden_size=hidden_size)
    ts_dict = jax.tree.map(lambda x: x[0, 0, 0, 0, 0, 0, 0], restored["final_train_state"])
    weights = ts_dict["params"]

    N = int(config["trajectories_sample_size"])
    max_len = int(config.get("max_length", 100))
    seed = int(config["random_seed"])
    shard_size = int(args.shard_size or config.get("shard_size", 5000))
    num_shards = math.ceil(N / shard_size)
    base_key = jax.random.PRNGKey(seed)

    study_name = config.get("study_name", "trajectories")
    env_config_path = config["environment_config_path"]
    stem = Path(env_config_path).stem
    match = re.match(r"(lightbulbs.*?)_config", stem)
    env_name = match.group(1) if match else "lightbulbs_unknown"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    out_root = Path("supervised_learning_data")
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{study_name}_{timestamp}.csv"

    print(f"Running N={N}, shard_size={shard_size}, max_len={max_len}, output_format=csv")

    total_lengths = []
    
    wrote_header = False

    for shard_idx in range(num_shards):
        start_i = shard_idx * shard_size
        end_i = min((shard_idx + 1) * shard_size, N)
        n_this = end_i - start_i
        if n_this <= 0:
            break

        shard_key = jax.random.fold_in(base_key, shard_idx)
        keys = jax.random.split(shard_key, n_this)

        h_seq, post_seq, mask_seq, robot_seq, human_seq, lengths = rollout_batch_jit(
            keys,
            env.reset_env,
            env.step_env,
            env.default_params,
            model.apply,
            weights,
            hidden_size,
            env.size,
            max_len,
            goals,
            prior_posterior,
        )

        h_h, post_h, mask_h, robot_h, human_h, len_h = jax.device_get(
            (h_seq, post_seq, mask_seq, robot_seq, human_seq, lengths)
        )
        total_lengths.append(len_h)

        rows = shard_to_rows(h_h, post_h, robot_h, human_h, len_h, global_offset=start_i)
        df_shard = pd.DataFrame(rows)
        df_shard.to_csv(out_path, index=False, mode="a", header=not wrote_header)
        wrote_header = True
        del rows, df_shard

        del h_seq, post_seq, mask_seq, robot_seq, human_seq, lengths
        del h_h, post_h, mask_h, robot_h, human_h, len_h

        print(f"Shard {shard_idx + 1}/{num_shards} done: trajectories [{start_i}, {end_i - 1}]")

    elapsed = time.time() - t0
    all_lengths = np.concatenate(total_lengths) if total_lengths else np.array([], dtype=np.int32)

    print(f"\nSaved CSV to {out_path}")
    if all_lengths.size:
        print(f"Avg length: {float(all_lengths.mean()):0.2f} | Min/Max: {int(all_lengths.min())}/{int(all_lengths.max())}")
    print(f"Total time: {elapsed:0.2f}s for N={N} in {num_shards} shard(s), shard_size={shard_size}")


if __name__ == "__main__":
    main()
