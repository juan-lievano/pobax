import argparse
import pandas as pd
import numpy as np
from ast import literal_eval
from pathlib import Path

ARROWS = {"N": "↑", "E": "→", "S": "↓", "W": "←"}

def parse_pos(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(x[0]), int(x[1])]
    return [int(v) for v in literal_eval(str(x))]

def render_grid(size, pos_y, pos_x, dir_letter):
    border = "#" * size
    grid = [["·" for _ in range(size)] for _ in range(size)]
    for i in range(size):
        grid[0][i] = "#"
        grid[size-1][i] = "#"
        grid[i][0] = "#"
        grid[i][size-1] = "#"
    if 0 <= pos_y < size and 0 <= pos_x < size:
        grid[pos_y][pos_x] = ARROWS.get(dir_letter, "?")
    return "\n".join("".join(row) for row in grid)

def main():
    ap = argparse.ArgumentParser(description="Visualize CompassWorld trajectories from CSV.")
    ap.add_argument("csv_path", type=str)
    ap.add_argument("--grid_size", type=int, default=8)
    ap.add_argument("--max_trajs", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    need_cols = ["trajectory_id", "time_step", "state_pos", "state_dir", "robot_action"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")

    df["state_pos"] = df["state_pos"].apply(parse_pos)
    df["state_dir"] = df["state_dir"].astype(str)

    df["pos_y"] = df["state_pos"].apply(lambda v: int(v[0]))
    df["pos_x"] = df["state_pos"].apply(lambda v: int(v[1]))

    groups = df.groupby("trajectory_id", sort=True)
    shown = 0
    for tid, g in groups:
        if shown >= args.max_trajs:
            break
        g = g.sort_values("time_step")
        L = len(g)
        print(f"\n=== Trajectory {tid} | steps={L} ===")
        steps_to_show = min(L, args.max_steps)
        for i in range(steps_to_show):
            row = g.iloc[i]
            y = int(row["pos_y"])
            x = int(row["pos_x"])
            d = row["state_dir"]
            a = int(row["robot_action"])
            print(f"\nStep {int(row['time_step'])} | pos=[{y}, {x}] dir={d} action={a}")
            print(render_grid(args.grid_size, y, x, d))
        if L > steps_to_show:
            print(f"... ({L - steps_to_show} more steps not shown)")
        shown += 1

    if shown == 0:
        print("No trajectories found.")

if __name__ == "__main__":
    main()
