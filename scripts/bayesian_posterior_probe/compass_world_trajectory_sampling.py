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

DIRECTION_LETTERS = np.array(["N", "E", "S", "W"])
IDX_NORTH, IDX_EAST, IDX_SOUTH, IDX_WEST, IDX_GOAL = 0, 1, 2, 3, 4


def to_python_list(x):
    return np.asarray(x).tolist()


def init_prev_action_onehot(action_dim: int, index: int):
    return jax.nn.one_hot(index, action_dim, dtype=jnp.float32)


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
            input_dim = shape[0]
            if candidate is None or (input_dim != hidden_size and input_dim < candidate):
                candidate = input_dim
    return candidate


def rollout_single_trajectory(
    rng_key,
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
    rng_reset, rng_key = jr.split(rng_key)
    initial_obs, initial_state = reset_fn(key=rng_reset, params=None)
    initial_obs = initial_obs.astype(jnp.float32)
    initial_prev_action = init_prev_action_onehot(prev_action_dim, prev_action_dim - 1)
    initial_hidden_state = ScannedRNN.initialize_carry(batch_size=1, hidden_size=hidden_size)
    initial_done = jnp.array(False)

    def step(carry, timestep):
        rng_key, state, obs, hidden_state, prev_action, done = carry
        rng_key, rng_sample, rng_step = jr.split(rng_key, 3)

        if include_prev_action:
            network_input = jnp.concatenate([obs, prev_action], axis=-1)[None, None, :]
        else:
            network_input = obs[None, None, :]

        done_flag = done[None, None]

        # This hidden_output is the hidden state to record for the current row (time_step = t).
        hidden_output = hidden_state

        hidden_state_new, action_distribution, _ = apply_fn(weights, hidden_state, (network_input, done_flag))
        action = action_distribution.sample(seed=rng_sample)[0, 0].astype(jnp.int32)

        next_obs, next_state, _, next_done, _ = step_fn(rng_step, state, action, env_params)
        next_done_flag = jnp.logical_or(done, next_done)

        obs_output = obs
        pos_output = state.pos
        dir_output = state.dir

        obs = jnp.where(done, obs, next_obs.astype(jnp.float32))
        hidden_state = jnp.where(done, hidden_state, hidden_state_new)
        prev_action = jnp.where(done, prev_action, init_prev_action_onehot(prev_action_dim, action))
        state = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), state, next_state)

        carry = (rng_key, state, obs, hidden_state, prev_action, next_done_flag)
        output = (obs_output, action, pos_output, dir_output, hidden_output, jnp.logical_not(done), timestep)
        return carry, output

    carry0 = (rng_key, initial_state, initial_obs, initial_hidden_state, initial_prev_action, initial_done)
    _, (obs_seq, action_seq, pos_seq, dir_seq, hidden_seq, mask_seq, timestep_seq) = lax.scan(
        step, carry0, jnp.arange(max_length, dtype=jnp.int32)
    )
    trajectory_length = mask_seq.sum(dtype=jnp.int32)
    return obs_seq, action_seq, pos_seq, dir_seq, hidden_seq, mask_seq, timestep_seq, trajectory_length


def rollout_batch(
    rng_keys,
    reset_fn,
    step_fn,
    env_params,
    apply_fn,
    weights,
    hidden_size,
    prev_action_dim,
    max_length,
    include_prev_action,
):
    fn = jax.vmap(
        rollout_single_trajectory,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )
    return fn(
        rng_keys,
        reset_fn,
        step_fn,
        env_params,
        apply_fn,
        weights,
        hidden_size,
        prev_action_dim,
        max_length,
        include_prev_action,
    )


rollout_batch_jit = jax.jit(
    rollout_batch,
    static_argnames=("reset_fn", "step_fn", "apply_fn", "hidden_size", "prev_action_dim", "max_length", "include_prev_action"),
)


def _initial_belief_mask(grid_size: int) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size, 4), dtype=bool)
    mask[1:grid_size - 1, 1:grid_size - 1, :] = True
    goal_y = (grid_size - 1) // 2
    mask[goal_y, 1, 3] = False
    return mask


def _observation_masks(grid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    north_mask = np.zeros((grid_size, grid_size, 4), dtype=bool)
    east_mask = np.zeros_like(north_mask)
    south_mask = np.zeros_like(north_mask)
    west_mask = np.zeros_like(north_mask)
    goal_mask = np.zeros_like(north_mask)
    goal_y = (grid_size - 1) // 2
    north_mask[1, 1:grid_size - 1, 0] = True
    east_mask[1:grid_size - 1, grid_size - 2, 1] = True
    south_mask[grid_size - 2, 1:grid_size - 1, 2] = True
    west_mask[1:grid_size - 1, 1, 3] = True
    west_mask[goal_y, 1, 3] = False
    goal_mask[goal_y, 1, 3] = True
    return north_mask, east_mask, south_mask, west_mask, goal_mask


def _state_emits_index(y: int, x: int, d: int, grid_size: int) -> Optional[int]:
    y_goal = (grid_size - 1) // 2
    if d == 0 and y == 1:
        return IDX_NORTH
    if d == 1 and x == grid_size - 2:
        return IDX_EAST
    if d == 2 and y == grid_size - 2:
        return IDX_SOUTH
    if d == 3:
        if x == 1:
            if y == y_goal:
                return IDX_GOAL
            return IDX_WEST
    return None


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    count = int(mask.sum())
    if count > 0:
        out[mask] = 1.0 / float(count)
    return out


def _transition_mask_precise(mask: np.ndarray, action: int, grid_size: int) -> np.ndarray:
    H = W = grid_size
    dest = np.zeros_like(mask, dtype=bool)

    if action == 1:
        for d in range(4):
            dest[:, :, (d + 1) % 4] |= mask[:, :, d]
        return dest

    if action == 2:
        for d in range(4):
            dest[:, :, (d + 3) % 4] |= mask[:, :, d]
        return dest

    y_min, y_max = 1, H - 2
    x_min, x_max = 1, W - 2

    src = mask[:, :, 0]
    if np.any(src):
        dest[y_min:y_max, x_min:x_max+1, 0] |= src[y_min+1:y_max+1, x_min:x_max+1]
        dest[y_min,     x_min:x_max+1, 0] |= src[y_min,             x_min:x_max+1]

    src = mask[:, :, 1]
    if np.any(src):
        dest[y_min:y_max+1, x_min+1:x_max+1, 1] |= src[y_min:y_max+1, x_min:x_max]
        dest[y_min:y_max+1, x_max,           1] |= src[y_min:y_max+1, x_max]

    src = mask[:, :, 2]
    if np.any(src):
        dest[y_min+1:y_max+1, x_min:x_max+1, 2] |= src[y_min:y_max,   x_min:x_max+1]
        dest[y_max,           x_min:x_max+1, 2] |= src[y_max,         x_min:x_max+1]

    src = mask[:, :, 3]
    if np.any(src):
        dest[y_min:y_max+1, x_min:x_max, 3] |= src[y_min:y_max+1, x_min+1:x_max+1]
        dest[y_min:y_max+1, x_min,       3] |= src[y_min:y_max+1, x_min]

    return dest


def _apply_observation_filter(mask: np.ndarray, obs_vec: np.ndarray, grid_size: int) -> np.ndarray:
    if float(np.max(obs_vec)) > 0.0:
        idx = int(np.argmax(obs_vec))
        keep = np.zeros_like(mask, dtype=bool)
        for y in range(1, grid_size - 1):
            for x in range(1, grid_size - 1):
                for d in range(4):
                    if _state_emits_index(y, x, d, grid_size) == idx:
                        keep[y, x, d] = True
        return np.logical_and(mask, keep)

    keep = np.zeros_like(mask, dtype=bool)
    for y in range(1, grid_size - 1):
        for x in range(1, grid_size - 1):
            for d in range(4):
                if _state_emits_index(y, x, d, grid_size) is None:
                    keep[y, x, d] = True
    return np.logical_and(mask, keep)


def _beliefs_for_trajectory(
    obs_seq: np.ndarray,
    action_seq: np.ndarray,
    trajectory_length: int,
    grid_size: int,
    _obs_masks_unused: Tuple[np.ndarray, ...],
) -> list:
    prior_mask = _initial_belief_mask(grid_size)
    beliefs = [_normalize_mask(prior_mask).tolist()]
    if trajectory_length <= 1:
        return beliefs
    for t in range(1, trajectory_length):
        obs_prev = obs_seq[t - 1]
        act_prev = int(action_seq[t - 1])
        posterior_mask = _apply_observation_filter(prior_mask, obs_prev, grid_size)
        prior_mask = _transition_mask_precise(posterior_mask, act_prev, grid_size)
        beliefs.append(_normalize_mask(prior_mask).tolist())
    return beliefs


def shard_to_rows(
    obs_array,
    action_array,
    pos_array,
    dir_array,
    hidden_array,
    length_array,
    global_offset,
    grid_size: int,
    hidden_size: int,
    include_hidden: bool,
):
    rows = {
        "trajectory_id": [],
        "time_step": [],
        "state_pos": [],
        "state_dir": [],
        "observation": [],
        "robot_action": [],
        "belief": [],
    }
    if include_hidden:
        rows["rnn_hidden"] = []

    obs_masks = _observation_masks(grid_size)
    n_trajectories = obs_array.shape[0]

    for local_id in range(n_trajectories):
        trajectory_id = global_offset + local_id
        trajectory_length = int(length_array[local_id])
        if trajectory_length <= 0:
            continue

        beliefs = _beliefs_for_trajectory(
            obs_array[local_id, :trajectory_length],
            action_array[local_id, :trajectory_length],
            trajectory_length,
            grid_size,
            obs_masks,
        )

        rows["trajectory_id"].extend([trajectory_id] * trajectory_length)
        rows["time_step"].extend(list(range(trajectory_length)))
        rows["state_pos"].extend(to_python_list(pos_array[local_id, :trajectory_length]))
        dir_ints = np.asarray(dir_array[local_id, :trajectory_length])
        rows["state_dir"].extend(DIRECTION_LETTERS[dir_ints].tolist())
        rows["observation"].extend(to_python_list(obs_array[local_id, :trajectory_length]))
        rows["robot_action"].extend(to_python_list(action_array[local_id, :trajectory_length]))
        rows["belief"].extend(beliefs)

        if include_hidden:
            # keep raw hidden state shape, do not flatten/trim
            h_slice = np.asarray(hidden_array[local_id, :trajectory_length])
            rows["rnn_hidden"].extend([h_slice[t].tolist() for t in range(trajectory_length)])


    return rows


def main():
    ap = argparse.ArgumentParser(description="CompassWorld rollouts (CSV) with belief-state filtering.")
    ap.add_argument("-N", "--num_trajectories", type=int, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--no_rnn_hidden_state", action="store_true", default=False)
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

    num_trajectories = int(args.num_trajectories)
    max_length = int(args.max_len)
    shard_size = int(args.shard_size)
    num_shards = math.ceil(num_trajectories / shard_size)
    base_rng_key = jax.random.PRNGKey(int(args.seed))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("supervised_learning_data")
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_suffix = ckpt_path.name[-10:]
    out_name = f"compass_world_{num_trajectories}_trajs_model_{ckpt_suffix}_ts_{timestamp}.csv"
    out_path = out_root / out_name

    print(f"Running num_trajectories={num_trajectories}, shard_size={shard_size}, max_length={max_length}, output_format=csv")

    all_shard_lengths = []
    wrote_header = False

    for shard_idx in range(num_shards):
        start_index = shard_idx * shard_size
        end_index = min((shard_idx + 1) * shard_size, num_trajectories)
        num_this_shard = end_index - start_index
        if num_this_shard <= 0:
            break

        shard_rng_key = jax.random.fold_in(base_rng_key, shard_idx)
        rng_keys = jax.random.split(shard_rng_key, num_this_shard)

        obs_seq, action_seq, pos_seq, dir_seq, hidden_seq, mask_seq, timestep_seq, lengths_array = rollout_batch_jit(
            rng_keys,
            env.reset_env,
            env.step_env,
            env.default_params,
            model.apply,
            weights,
            hidden_size,
            prev_action_dim,
            max_length,
            include_prev_action,
        )

        obs_array, action_array, pos_array, dir_array, hidden_array, mask_array, timestep_array, length_array = jax.device_get(
            (obs_seq, action_seq, pos_seq, dir_seq, hidden_seq, mask_seq, timestep_seq, lengths_array)
        )
        all_shard_lengths.append(length_array)

        rows = shard_to_rows(
            obs_array=obs_array,
            action_array=action_array,
            pos_array=pos_array,
            dir_array=dir_array,
            hidden_array=hidden_array,
            length_array=length_array,
            global_offset=start_index,
            grid_size=args.grid_size,
            hidden_size=hidden_size,
            include_hidden=not args.no_rnn_hidden_state,
        )

        pd.DataFrame(rows).to_csv(out_path, index=False, mode="a", header=not wrote_header)
        wrote_header = True

        del rows, obs_seq, action_seq, pos_seq, dir_seq, hidden_seq, mask_seq, timestep_seq, lengths_array
        del obs_array, action_array, pos_array, dir_array, hidden_array, mask_array, timestep_array, length_array

        print(f"Shard {shard_idx + 1}/{num_shards} done: trajectories [{start_index}, {end_index - 1}]")

    elapsed = time.time() - t0
    all_lengths = np.concatenate(all_shard_lengths) if all_shard_lengths else np.array([], dtype=np.int32)

    print(f"\nSaved CSV to {out_path}")
    if all_lengths.size:
        print(f"Avg length: {float(all_lengths.mean()):0.2f} | Min/Max: {int(all_lengths.min())}/{int(all_lengths.max())}")
    print(f"Total time: {elapsed:0.2f}s for num_trajectories={num_trajectories} in {num_shards} shard(s), shard_size={shard_size}")


if __name__ == "__main__":
    main()
