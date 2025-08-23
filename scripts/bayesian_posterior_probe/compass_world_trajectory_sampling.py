#!/usr/bin/env python3
import argparse
import math
import time
from datetime import datetime
from pathlib import Path

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


# --------------------------- Utils ---------------------------

def to_pylist(x):
    return np.asarray(x).tolist()

def init_prev_action_onehot(dim, idx):
    return jax.nn.one_hot(idx, dim, dtype=jnp.float32)

def tree_select(done_scalar_bool, a_tree, b_tree):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(done_scalar_bool, a, b), a_tree, b_tree)


# ----------------- Belief tables for CompassWorld -----------------

def build_compass_belief(env: CompassWorld):
    size = int(env.size)
    y = jnp.arange(1, size - 1, dtype=jnp.int32)
    x = jnp.arange(1, size - 1, dtype=jnp.int32)
    d = jnp.arange(0, 4, dtype=jnp.int32)
    Y, X, D = jnp.meshgrid(y, x, d, indexing="ij")
    y_flat, x_flat, d_flat = Y.reshape(-1), X.reshape(-1), D.reshape(-1)
    S = y_flat.shape[0]

    grid_span = size - 2

    pos_flat = jnp.stack([y_flat, x_flat], axis=1)
    obs_table = jax.vmap(lambda p, dd: env._obs_from_state(p, dd))(pos_flat, d_flat).astype(jnp.uint8)

    def next_state_arrays(p, dd, a):
        def fwd(_):
            nxt = p + env._dir_map[dd]
            nxt = jnp.clip(nxt, env._state_min, env._state_max)
            return nxt, dd
        p2, d2 = jax.lax.switch(jnp.clip(a, 0, 2),
                                [lambda: (p, dd),
                                 lambda: (p, (dd + 1) % 4),
                                 lambda: (p, (dd - 1) % 4)])
        p2, d2 = jax.lax.cond(a == 0, fwd, lambda _: (p2, d2), operand=None)
        return p2, d2

    def idx_from_pos_dir(y_, x_, d_):
        return ((y_ - 1) * grid_span + (x_ - 1)) * 4 + d_

    T_list = []
    for a in range(3):
        p2, d2 = jax.vmap(lambda p, dd: next_state_arrays(p, dd, jnp.int32(a)))(pos_flat, d_flat)
        s2 = idx_from_pos_dir(p2[:, 0], p2[:, 1], d2)
        T_list.append(s2)
    T_table = jnp.stack(T_list, axis=0)  # [A=3, S]

    return {
        "S": S,
        "size": size,
        "obs_table": obs_table,   # [S, obs_dim=5]
        "T_table": T_table,       # [A=3, S]
    }


# --------------------------- Rollouts ---------------------------

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
    belief_tools: dict,
):
    key0, key = jr.split(key)
    obs0, state0 = reset_fn(key=key0, params=None)
    obs0 = obs0.astype(jnp.float32)
    prev_a0 = init_prev_action_onehot(prev_action_dim, prev_action_dim - 1)
    h0 = ScannedRNN.initialize_carry(batch_size=1, hidden_size=hidden_size)
    done0 = jnp.array(False)

    S = belief_tools["S"]
    size = jnp.int32(belief_tools["size"])
    grid_span = size - 2

    def state_to_idx(pos, dir_):
        y, x = pos[0], pos[1]
        return ((y - 1) * grid_span + (x - 1)) * 4 + dir_

    s_idx0 = state_to_idx(state0.pos, state0.dir).astype(jnp.int32)

    obs_table = belief_tools["obs_table"]   # [S, 5]
    T_table = belief_tools["T_table"]       # [A, S]

    b0 = jnp.full((S,), 1.0 / S, dtype=jnp.float32)

    def belief_update(b_prev, a, obs):
        b_pred = jnp.zeros_like(b_prev).at[T_table[a]].add(b_prev)
        compat = jnp.all(obs_table == obs.astype(jnp.uint8), axis=1)
        b_new = b_pred * compat.astype(jnp.float32)
        z = b_new.sum()
        return jnp.where(z > 0, b_new / z, jnp.full_like(b_new, 1.0 / b_new.shape[0]))

    def step(carry, t):
        key, state, obs, h, prev_a, done, b, s_idx = carry
        key, k_samp, k_step = jr.split(key, 3)

        x = jnp.concatenate([obs, prev_a], axis=-1)[None, None, :]
        done_b = done[None, None]

        # record previous-step outputs
        h_out = h
        b_out = b
        s_idx_out = s_idx
        st_pos_out = state.pos
        st_dir_out = state.dir
        valid = jnp.logical_not(done)

        h_new, pi, _ = apply_fn(weights, h, (x, done_b))
        act = pi.sample(seed=k_samp)[0, 0].astype(jnp.int32)

        obs1, state1, _, done1, _ = step_fn(k_step, state, act, env_params)
        done_next = jnp.logical_or(done, done1)

        b1 = belief_update(b, act, obs1)
        b = jnp.where(done, b, b1)

        obs = jnp.where(done, obs, obs1.astype(jnp.float32))
        h = jnp.where(done, h, h_new)
        prev_a = jnp.where(done, prev_a, init_prev_action_onehot(prev_action_dim, int(act)))
        state = tree_select(done, state, state1)
        s_idx1 = jnp.where(done, s_idx, T_table[act, s_idx])

        carry = (key, state, obs, h, prev_a, done_next, b, s_idx1)
        out = (h_out, b_out, valid, act, s_idx_out, st_pos_out, st_dir_out, t)
        return carry, out

    carry0 = (key, state0, obs0, h0, prev_a0, done0, b0, s_idx0)
    _, (h_seq, b_seq, mask_seq, act_seq, sidx_seq, stpos_seq, stdir_seq, t_seq) = lax.scan(
        step, carry0, jnp.arange(max_length, dtype=jnp.int32)
    )
    length = mask_seq.sum(dtype=jnp.int32)
    return h_seq, b_seq, mask_seq, act_seq, sidx_seq, stpos_seq, stdir_seq, t_seq, length


def rollout_batch(keys, reset_fn, step_fn, env_params, apply_fn, weights,
                  hidden_size, prev_action_dim, max_length, belief_tools):
    fn = jax.vmap(
        rollout_single,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )
    return fn(keys, reset_fn, step_fn, env_params, apply_fn, weights,
              hidden_size, prev_action_dim, max_length, belief_tools)


rollout_batch_jit = jax.jit(
    rollout_batch,
    static_argnames=("reset_fn", "step_fn", "apply_fn", "hidden_size", "prev_action_dim", "max_length"),
)


# --------------------------- I/O helpers ---------------------------

def shard_to_rows(h_h, b_h, act_h, sidx_h, stpos_h, stdir_h, len_h, global_offset, include_state: bool):
    rows = {
        "rnn_hidden_state": [],
        "belief": [],
        "robot_action": [],
        "true_state": [],
        "trajectory_id": [],
        "time_step": [],
    }
    if include_state:
        rows["state_pos"] = []
        rows["state_dir"] = []

    n_this = h_h.shape[0]
    for local_id in range(n_this):
        traj_id = global_offset + local_id
        L = int(len_h[local_id])
        if L <= 0:
            continue
        rows["rnn_hidden_state"].extend(to_pylist(h_h[local_id, :L]))
        rows["belief"].extend(to_pylist(b_h[local_id, :L]))
        rows["robot_action"].extend(to_pylist(act_h[local_id, :L]))
        rows["true_state"].extend(to_pylist(sidx_h[local_id, :L]))
        rows["trajectory_id"].extend([traj_id] * L)
        rows["time_step"].extend(list(range(L)))
        if include_state:
            rows["state_pos"].extend(to_pylist(stpos_h[local_id, :L]))  # each is [y, x]
            rows["state_dir"].extend(to_pylist(stdir_h[local_id, :L]))  # each is int d in {0..3}
    return rows


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="CompassWorld rollouts with analytic belief (CSV streaming).")
    ap.add_argument("-N", "--num_trajectories", type=int, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid_size", type=int, default=8)
    ap.add_argument("--random_start", action="store_true", default=True)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--study_name", type=str, default="compassworld")
    ap.add_argument("--include_state", action="store_true", default=False)
    ap.add_argument("--output_file_name", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()

    env = CompassWorld(size=args.grid_size, random_start=args.random_start)
    network_class, action_size = get_gymnax_network_fn(env, env.default_params)
    assert network_class is DiscreteActorCriticRNN, f"Expected DiscreteActorCriticRNN, got {network_class}"
    num_actions = int(action_size)
    prev_action_dim = num_actions + 1

    belief_tools = build_compass_belief(env)

    ckpt_path = Path(args.checkpoint).resolve()
    restored = orbax.checkpoint.PyTreeCheckpointer().restore(ckpt_path)
    hidden_size = int(restored["args"]["hidden_size"])
    double_critic = bool(restored["args"]["double_critic"])

    model = DiscreteActorCriticRNN(
        action_dim=num_actions, double_critic=double_critic, hidden_size=hidden_size
    )
    ts_dict = jax.tree.map(lambda x: x[0, 0, 0, 0, 0, 0, 0], restored["final_train_state"])
    weights = ts_dict["params"]

    N = int(args.num_trajectories)
    max_len = int(args.max_len)
    shard_size = int(args.shard_size)
    num_shards = math.ceil(N / shard_size)
    base_key = jax.random.PRNGKey(int(args.seed))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path("supervised_learning_data")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.output_file_name is not None:
        out_path_candidate = Path(args.output_file_name)
        out_path = out_path_candidate if out_path_candidate.is_absolute() else (out_root / out_path_candidate)
    else:
        out_path = out_root / f"{args.study_name}_{timestamp}.csv"

    print(f"Running N={N}, shard_size={shard_size}, max_len={max_len}, output_format=csv")
    if args.include_state:
        print("Including state columns: state_pos, state_dir")

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

        h_seq, b_seq, mask_seq, act_seq, sidx_seq, stpos_seq, stdir_seq, t_seq, lengths = rollout_batch_jit(
            keys,
            env.reset_env,
            env.step_env,
            env.default_params,
            model.apply,
            weights,
            hidden_size,
            prev_action_dim,
            max_len,
            belief_tools,
        )

        h_h, b_h, mask_h, act_h, sidx_h, stpos_h, stdir_h, t_h, len_h = jax.device_get(
            (h_seq, b_seq, mask_seq, act_seq, sidx_seq, stpos_seq, stdir_seq, t_seq, lengths)
        )
        total_lengths.append(len_h)

        rows = shard_to_rows(
            h_h, b_h, act_h, sidx_h, stpos_h, stdir_h, len_h, global_offset=start_i, include_state=args.include_state
        )
        pd.DataFrame(rows).to_csv(out_path, index=False, mode="a", header=not wrote_header)
        wrote_header = True

        del rows, h_seq, b_seq, mask_seq, act_seq, sidx_seq, stpos_seq, stdir_seq, t_seq, lengths
        del h_h, b_h, mask_h, act_h, sidx_h, stpos_h, stdir_h, t_h, len_h

        print(f"Shard {shard_idx + 1}/{num_shards} done: trajectories [{start_i}, {end_i - 1}]")

    elapsed = time.time() - t0
    all_lengths = np.concatenate(total_lengths) if total_lengths else np.array([], dtype=np.int32)

    print(f"\nSaved CSV to {out_path}")
    if all_lengths.size:
        print(f"Avg length: {float(all_lengths.mean()):0.2f} | Min/Max: {int(all_lengths.min())}/{int(all_lengths.max())}")
    print(f"Total time: {elapsed:0.2f}s for N={N} in {num_shards} shard(s), shard_size={shard_size}")


if __name__ == "__main__":
    main()
