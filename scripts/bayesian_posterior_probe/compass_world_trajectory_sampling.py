#!/usr/bin/env python3
import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax import random as jr
import numpy as np
import orbax.checkpoint
import pandas as pd

from pobax.envs.jax.compass_world import CompassWorld
from pobax.models import ScannedRNN, get_gymnax_network_fn
from pobax.models.discrete import DiscreteActorCriticRNN

DIR_LETTERS = np.array(["N", "E", "S", "W"])  # CPU-side mapping only

def to_pylist(x):
    return np.asarray(x).tolist()

def init_prev_action_onehot(dim, idx):
    return jax.nn.one_hot(idx, dim, dtype=jnp.float32)

def _iter_params_leaves(p: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], Any]]:
    if isinstance(p, dict):
        for k, v in p.items():
            yield from _iter_params_leaves(v, path + (k,))
    else:
        yield path, p

def infer_expected_input_dim(params_tree: Dict, hidden_size: int) -> Optional[int]:
    candidate = None
    for path, leaf in _iter_params_leaves(params_tree):
        if not isinstance(leaf, (np.ndarray, jnp.ndarray)):
            continue
        shape = tuple(leaf.shape)
        if len(shape) == 2 and shape[1] == hidden_size and "kernel" in path[-1].lower():
            in_dim = shape[0]
            if candidate is None or (in_dim != hidden_size and in_dim < candidate):
                candidate = in_dim
    return candidate

def rollout_single(
    key,
    reset_fn,
    step_fn,
    env_params,
    apply_fn,
    weights,
    hidden_size: int,
    prev_action_dim: int,
    max_length: int,
    include_prev_action: bool,
):
    key0, key = jr.split(key)
    obs0, state0 = reset_fn(key=key0, params=None)
    obs0 = obs0.astype(jnp.float32)
    prev_a0 = init_prev_action_onehot(prev_action_dim, prev_action_dim - 1)
    h0 = ScannedRNN.initialize_carry(batch_size=1, hidden_size=hidden_size)
    done0 = jnp.array(False)

    def step(carry, t):
        key, state, obs, h, prev_a, done = carry
        key, k_samp, k_step = jr.split(key, 3)

        x = jnp.concatenate([obs, prev_a], axis=-1)[None, None, :] if include_prev_action else obs[None, None, :]
        done_b = done[None, None]

        h_new, pi, _ = apply_fn(weights, h, (x, done_b))
        act = pi.sample(seed=k_samp)[0, 0].astype(jnp.int32)

        obs1, state1, _, done1, _ = step_fn(k_step, state, act, env_params)
        done_next = jnp.logical_or(done, done1)

        obs_out = obs
        pos_out = state.pos
        dir_out = state.dir

        obs = jnp.where(done, obs, obs1.astype(jnp.float32))
        h = jnp.where(done, h, h_new)
        prev_a = jnp.where(done, prev_a, init_prev_action_onehot(prev_action_dim, act))
        state = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), state, state1)

        carry = (key, state, obs, h, prev_a, done_next)
        out = (obs_out, act, pos_out, dir_out, jnp.logical_not(done), t)
        return carry, out

    carry0 = (key, state0, obs0, h0, prev_a0, done0)
    _, (o_seq, act_seq, pos_seq, dir_seq, mask_seq, t_seq) = lax.scan(
        step, carry0, jnp.arange(max_length, dtype=jnp.int32)
    )
    length = mask_seq.sum(dtype=jnp.int32)
    return o_seq, act_seq, pos_seq, dir_seq, mask_seq, t_seq, length

def rollout_batch(keys, reset_fn, step_fn, env_params, apply_fn, weights,
                  hidden_size, prev_action_dim, max_length, include_prev_action):
    fn = jax.vmap(
        rollout_single,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )
    return fn(keys, reset_fn, step_fn, env_params, apply_fn, weights,
              hidden_size, prev_action_dim, max_length, include_prev_action)

rollout_batch_jit = jax.jit(
    rollout_batch,
    static_argnames=("reset_fn", "step_fn", "apply_fn", "hidden_size", "prev_action_dim", "max_length", "include_prev_action"),
)

def shard_to_rows(o_h, act_h, pos_h, dir_h, len_h, global_offset):
    rows = {
        "trajectory_id": [],
        "time_step": [],
        "state_pos": [],
        "state_dir": [],
        "observation": [],
        "robot_action": [],
    }
    n_this = o_h.shape[0]
    for local_id in range(n_this):
        traj_id = global_offset + local_id
        L = int(len_h[local_id])
        if L <= 0:
            continue
        rows["trajectory_id"].extend([traj_id] * L)
        rows["time_step"].extend(list(range(L)))
        rows["state_pos"].extend(to_pylist(pos_h[local_id, :L]))
        dir_ints = np.asarray(dir_h[local_id, :L])
        rows["state_dir"].extend(DIR_LETTERS[dir_ints].tolist())
        rows["observation"].extend(to_pylist(o_h[local_id, :L]))
        rows["robot_action"].extend(to_pylist(act_h[local_id, :L]))
    return rows

def main():
    ap = argparse.ArgumentParser(description="CompassWorld rollouts (CSV).")
    ap.add_argument("-N", "--num_trajectories", type=int, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--no_rnn_hidden_state", action="store_true", default=False) # not used but here for back-compat
    args = ap.parse_args()

    t0 = time.time()

    env = CompassWorld(size=args.grid_size)
    network_class, action_size = get_gymnax_network_fn(env, env.default_params)
    assert network_class is DiscreteActorCriticRNN, f"Expected DiscreteActorCriticRNN, got {network_class}"
    num_actions = int(action_size)
    prev_action_dim = num_actions + 1

    ckpt_path = Path(args.checkpoint).resolve()
    restored = orbax.checkpoint.PyTreeCheckpointer().restore(ckpt_path)
    hidden_size = int(restored["args"]["hidden_size"])
    double_critic = bool(restored["args"]["double_critic"])

    model = DiscreteActorCriticRNN(
        action_dim=num_actions, double_critic=double_critic, hidden_size=hidden_size
    )
    ts_dict = jax.tree.map(lambda x: x[0, 0, 0, 0, 0, 0, 0], restored["final_train_state"])
    weights = ts_dict["params"]

    tmp_obs0, _ = env.reset_env(jax.random.PRNGKey(0), params=None)
    obs_dim = int(tmp_obs0.shape[-1])
    expected_in = infer_expected_input_dim(weights, hidden_size)
    if expected_in is None:
        raise RuntimeError(
            f"Could not infer input dimension from checkpoint params. "
            f"Observed obs_dim={obs_dim}, prev_action_dim={prev_action_dim}."
        )
    if expected_in == obs_dim:
        include_prev_action = False
        print(f"[auto] Using ONLY obs (dim={obs_dim}) to match checkpoint input_dim={expected_in}.")
    elif expected_in == (obs_dim + prev_action_dim):
        include_prev_action = True
        print(f"[auto] Using obs+prev_action (dim={obs_dim}+{prev_action_dim}={expected_in}) to match checkpoint.")
    else:
        raise RuntimeError(
            f"Checkpoint expects input_dim={expected_in}, but obs_dim={obs_dim} and "
            f"obs_dim+prev_action_dim={obs_dim + prev_action_dim}."
        )

    N = int(args.num_trajectories)
    max_len = int(args.max_len)
    shard_size = int(args.shard_size)
    num_shards = math.ceil(N / shard_size)
    base_key = jax.random.PRNGKey(int(args.seed))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("supervised_learning_data")
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_suffix = ckpt_path.name[-10:]
    out_name = f"compass_world_{N}_trajs_model_{ckpt_suffix}_ts_{timestamp}.csv"
    out_path = out_root / out_name

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

        o_seq, act_seq, pos_seq, dir_seq, mask_seq, t_seq, lengths = rollout_batch_jit(
            keys,
            env.reset_env,
            env.step_env,
            env.default_params,
            model.apply,
            weights,
            hidden_size,
            prev_action_dim,
            max_len,
            include_prev_action,
        )

        o_h, act_h, pos_h, dir_h, mask_h, t_h, len_h = jax.device_get(
            (o_seq, act_seq, pos_seq, dir_seq, mask_seq, t_seq, lengths)
        )
        total_lengths.append(len_h)

        rows = shard_to_rows(
            o_h, act_h, pos_h, dir_h, len_h, global_offset=start_i
        )
        pd.DataFrame(rows).to_csv(out_path, index=False, mode="a", header=not wrote_header)
        wrote_header = True

        del rows, o_seq, act_seq, pos_seq, dir_seq, mask_seq, t_seq, lengths
        del o_h, act_h, pos_h, dir_h, mask_h, t_h, len_h

        print(f"Shard {shard_idx + 1}/{num_shards} done: trajectories [{start_i}, {end_i - 1}]")

    elapsed = time.time() - t0
    all_lengths = np.concatenate(total_lengths) if total_lengths else np.array([], dtype=np.int32)

    print(f"\nSaved CSV to {out_path}")
    if all_lengths.size:
        print(f"Avg length: {float(all_lengths.mean()):0.2f} | Min/Max: {int(all_lengths.min())}/{int(all_lengths.max())}")
    print(f"Total time: {elapsed:0.2f}s for N={N} in {num_shards} shard(s), shard_size={shard_size}")

if __name__ == "__main__":
    main()
