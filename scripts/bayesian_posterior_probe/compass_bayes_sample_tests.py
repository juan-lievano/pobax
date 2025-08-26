#!/usr/bin/env python3
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd

"""
- Belief convention (prior): At time t, belief[t] is the prior before observation[t].
  Recurrence: prior[t] = Transition( ObsFilter(prior[t−1], observation[t−1]), action[t−1] ).
  Also, prior[0] = canonical uniform interior prior (excluding the west-facing goal cell).

- Shape and normalization: Each belief tensor has shape (grid, grid, 4), is finite, non-negative, and sums to 1.
  The padding frame (first/last row and column) must have zero mass.

- Initial prior checks:
  • prior[0] equals the canonical uniform interior prior (west-facing goal cell excluded).
  • All trajectories share the exact same prior[0].

- Recursive consistency: For every t ≥ 1, belief[t] matches the recurrence using observation[t−1] and action[t−1].

- Direction certainty: If the prior is concentrated in one direction plane (total mass in that plane ≥ 1 − tol),
  the true state_dir letter (N, E, S, W) must match that direction.

- Monotone support: The number of non-zero entries in belief[t] is non-increasing over time.
  Special case: once the belief becomes a point mass, it must remain a point mass thereafter.

- Corner disambiguation: If two consecutive observations are walls and they are different walls,
  then after the second observation the belief must be a point mass (corner is determined).

- Trajectory stats (printed): number of trajectories, average length, min/max length, and the id of the longest/shortest trajectory.
"""

IDX_N, IDX_E, IDX_S, IDX_W, IDX_G = 0, 1, 2, 3, 4
DIR_MAP = {"N": 0, "E": 1, "S": 2, "W": 3}

def _parse_belief(s): return np.array(json.loads(s), dtype=float)
def _mask_from_belief(B, tol): return B > tol
def _normalize_mask(mask):
    out = np.zeros_like(mask, dtype=float)
    n = int(mask.sum())
    if n > 0: out[mask] = 1.0 / float(n)
    return out

def _initial_belief_mask(grid):
    m = np.zeros((grid, grid, 4), dtype=bool)
    m[1:grid-1, 1:grid-1, :] = True
    y_goal = (grid - 1) // 2
    m[y_goal, 1, 3] = False
    return m

def _state_emits_index(y, x, d, grid):
    y_goal = (grid - 1) // 2
    if d == 0 and y == 1: return IDX_N
    if d == 1 and x == grid - 2: return IDX_E
    if d == 2 and y == grid - 2: return IDX_S
    if d == 3 and x == 1: return IDX_G if y == y_goal else IDX_W
    return None

def _apply_observation_filter(mask, obs_vec, grid):
    if float(np.max(obs_vec)) > 0.0:
        idx = int(np.argmax(obs_vec))
        keep = np.zeros_like(mask, dtype=bool)
        for y in range(1, grid - 1):
            for x in range(1, grid - 1):
                for d in range(4):
                    if _state_emits_index(y, x, d, grid) == idx:
                        keep[y, x, d] = True
        return np.logical_and(mask, keep)
    keep = np.zeros_like(mask, dtype=bool)
    for y in range(1, grid - 1):
        for x in range(1, grid - 1):
            for d in range(4):
                if _state_emits_index(y, x, d, grid) is None:
                    keep[y, x, d] = True
    return np.logical_and(mask, keep)

def _transition_mask_precise(mask, action, grid):
    H = W = grid
    dest = np.zeros_like(mask, dtype=bool)
    if action == 1:
        for d in range(4): dest[:, :, (d + 1) % 4] |= mask[:, :, d]
        return dest
    if action == 2:
        for d in range(4): dest[:, :, (d + 3) % 4] |= mask[:, :, d]
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

def _obs_idx(obs_vec):
    if float(np.max(obs_vec)) <= 0.0: return None
    i = int(np.argmax(obs_vec))
    return i if i in (IDX_N, IDX_E, IDX_S, IDX_W) else None

def main():
    ap = argparse.ArgumentParser(description="Prior-mode sanity checks for CompassWorld belief CSV.")
    ap.add_argument("csv", type=str)
    ap.add_argument("--grid_size", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max_failures", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()

    df = pd.read_csv(args.csv)
    required = ["trajectory_id", "time_step", "observation", "robot_action", "belief", "state_dir"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        print(f"Missing columns: {miss}")
        sys.exit(1)

    df = df.sort_values(["trajectory_id", "time_step"]).reset_index(drop=False).rename(columns={"index": "_row"})
    grid, tol = args.grid_size, args.tol
    init_prior = _normalize_mask(_initial_belief_mask(grid))

    traj_lengths = df.groupby("trajectory_id", sort=True)["time_step"].count()
    n_traj = len(traj_lengths)
    avg_len = float(traj_lengths.mean()) if n_traj else 0.0
    max_len = int(traj_lengths.max()) if n_traj else 0
    min_len = int(traj_lengths.min()) if n_traj else 0
    max_len_tid = traj_lengths.idxmax() if n_traj else -1
    min_len_tid = traj_lengths.idxmin() if n_traj else -1

    failures = []
    printed = 0
    t0_equal_all = True
    first_t0 = None

    for tid, g in df.groupby("trajectory_id", sort=True):
        g = g.sort_values("time_step")
        beliefs = [_parse_belief(s) for s in g["belief"].tolist()]
        obs = [np.array(json.loads(s), dtype=float) for s in g["observation"].tolist()]
        acts = [int(a) for a in g["robot_action"].tolist()]
        dirs = [str(s).strip() for s in g["state_dir"].tolist()]
        rows = g["_row"].tolist()
        T = len(beliefs)

        for k, (B, r) in enumerate(zip(beliefs, rows)):
            if B.shape != (grid, grid, 4):
                if printed < args.max_failures: failures.append((r, f"wrong shape {B.shape}")); printed += 1
                continue
            if (not np.isfinite(B).all()) or abs(B.sum() - 1.0) > tol:
                if printed < args.max_failures: failures.append((r, f"not normalized, sum={B.sum()}")); printed += 1
            if (B < -tol).any():
                if printed < args.max_failures: failures.append((r, "negative entries")); printed += 1
            pad_mass = B[0,:,:].sum() + B[grid-1,:,:].sum() + B[:,0,:].sum() + B[:,grid-1,:].sum()
            if abs(pad_mass) > tol:
                if printed < args.max_failures: failures.append((r, f"padding mass {pad_mass}")); printed += 1

            if k == 0:
                if first_t0 is None: first_t0 = B
                else:
                    if not np.allclose(B, first_t0, atol=tol, rtol=0.0): t0_equal_all = False
                if not np.allclose(B, init_prior, atol=tol, rtol=0.0):
                    if printed < args.max_failures: failures.append((r, "t=0 prior not canonical")); printed += 1
            else:
                M_prev = _mask_from_belief(beliefs[k-1], tol)
                post_prev = _apply_observation_filter(M_prev, obs[k-1], grid)
                prior_t = _normalize_mask(_transition_mask_precise(post_prev, acts[k-1], grid))
                if not np.allclose(B, prior_t, atol=tol, rtol=0.0):
                    if printed < args.max_failures: failures.append((r, "prior[t] mismatch with obs/act recursion")); printed += 1

            p_dir = B.sum(axis=(0, 1))
            d_hat = int(np.argmax(p_dir))
            if p_dir[d_hat] >= 1.0 - tol:
                d_true = DIR_MAP.get(dirs[k], None)
                if d_true is None or d_true != d_hat:
                    if printed < args.max_failures: failures.append((r, f"dir concentrated on {d_hat} but state_dir={dirs[k]}")); printed += 1

        support = [int((_mask_from_belief(b, tol)).sum()) for b in beliefs]
        for t in range(1, T):
            if support[t] > support[t-1]:
                if printed < args.max_failures: failures.append((rows[t], "support increased")); printed += 1

        for t in range(0, T-1):
            i0, i1 = _obs_idx(obs[t]), _obs_idx(obs[t+1])
            if i0 is not None and i1 is not None and i0 != i1:
                M_t = _mask_from_belief(beliefs[t], tol)
                post_t = _apply_observation_filter(M_t, obs[t], grid)
                prior_tp1 = _transition_mask_precise(post_t, acts[t], grid)
                post_tp1 = _apply_observation_filter(prior_tp1, obs[t+1], grid)
                if int(post_tp1.sum()) != 1:
                    if printed < args.max_failures: failures.append((rows[t+1], "two-wall sequence not point-mass after 2nd obs")); printed += 1
                break

    elapsed = time.time() - t0

    print(f"Trajectories: {n_traj}")
    print(f"Avg length: {avg_len:.2f} | Min: {min_len} (traj {min_len_tid}) | Max: {max_len} (traj {max_len_tid})")
    print(f"All t=0 priors identical: {'YES' if t0_equal_all else 'NO'}")
    print(f"Time: {elapsed:.2f}s")

    if failures:
        print(f"FAILED {len(failures)} checks")
        for idx, msg in failures[:args.max_failures]:
            print(f"row {idx}: {msg}")
        sys.exit(1)
    else:
        print("OK: all checks passed")

if __name__ == "__main__":
    main()
