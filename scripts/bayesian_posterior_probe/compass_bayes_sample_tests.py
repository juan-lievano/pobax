#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd

IDX_N, IDX_E, IDX_S, IDX_W, IDX_G = 0, 1, 2, 3, 4

def main():
    ap = argparse.ArgumentParser(description="Basic sanity checks for CompassWorld belief CSV.")
    ap.add_argument("csv", type=str)
    ap.add_argument("--grid_size", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    grid = args.grid_size
    y_goal = (grid - 1) // 2
    goal_cell = (y_goal, 1, 3)

    failures = []

    for i, row in df.iterrows():
        obs = np.array(json.loads(row["observation"]), dtype=float)
        B = np.array(json.loads(row["belief"]), dtype=float)

        if B.shape != (grid, grid, 4):
            failures.append((i, f"belief has wrong shape {B.shape}"))
            continue

        # 1. Distribution sums to 1
        if not np.isfinite(B).all() or abs(B.sum() - 1.0) > args.tol:
            print(B.shape, B.sum())
            print(i)
            return
            failures.append((i, f"belief not normalized, sum={B.sum()}"))
        
        if (B < -args.tol).any():
            failures.append((i, f"belief has negative entries"))

        # 2. Wall consistency
        if obs[IDX_N] == 1.0:
            if B[:, :, (1,2,3)].sum() > args.tol:
                failures.append((i, "N wall: mass outside dir=0"))
            if abs(B[1, :, 0].sum() - 1.0) > args.tol:
                failures.append((i, "N wall: mass not concentrated at y=1, dir=0"))
        if obs[IDX_E] == 1.0:
            if B[:, :, (0,2,3)].sum() > args.tol:
                failures.append((i, "E wall: mass outside dir=1"))
            if abs(B[:, grid-2, 1].sum() - 1.0) > args.tol:
                failures.append((i, "E wall: mass not at x=grid-2, dir=1"))
        if obs[IDX_S] == 1.0:
            if B[:, :, (0,1,3)].sum() > args.tol:
                failures.append((i, "S wall: mass outside dir=2"))
            if abs(B[grid-2, :, 2].sum() - 1.0) > args.tol:
                failures.append((i, "S wall: mass not at y=grid-2, dir=2"))
        if obs[IDX_W] == 1.0:
            if B[:, :, (0,1,2)].sum() > args.tol:
                failures.append((i, "W wall: mass outside dir=3"))
            if abs(B[:, 1, 3].sum() - 1.0) > args.tol:
                failures.append((i, "W wall: mass not at x=1, dir=3"))
            if B[goal_cell] > args.tol:
                failures.append((i, "W wall: mass incorrectly on goal cell"))
        if obs[IDX_G] == 1.0:
            mass_goal = B[goal_cell]
            if abs(mass_goal - 1.0) > args.tol:
                failures.append((i, "Goal: mass not concentrated on goal cell"))

    if failures:
        print(f"FAILED {len(failures)} checks")
        for idx, msg in failures[:50]:  # print up to 50 issues
            print(f"row {idx}: {msg}")
        sys.exit(1)
    else:
        print("OK: all checks passed")

if __name__ == "__main__":
    main()
