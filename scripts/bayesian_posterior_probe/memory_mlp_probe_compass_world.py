#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import entropy


def parse_rnn_hidden(s: str, hidden_size: int) -> np.ndarray:
    arr = np.fromstring(s.strip("[]").replace(",", " "), sep=" ")
    if arr.size != hidden_size:
        raise ValueError(f"rnn_hidden length {arr.size} != hidden_size {hidden_size}")
    return arr.astype(np.float64)


def parse_belief_tensor(s: str, grid_size: int) -> np.ndarray:
    B = np.asarray(json.loads(s), dtype=np.float64)
    expected = (grid_size, grid_size, 4)
    if B.shape != expected:
        raise ValueError(f"belief shape {B.shape} != {expected}")
    return B


def tensor_to_json(t: np.ndarray) -> str:
    return json.dumps(np.asarray(t, dtype=float).tolist(), separators=(",", ":"))


def uniform_baseline_tensor(grid_size: int) -> np.ndarray:
    gs = grid_size
    base = np.zeros((gs, gs, 4), dtype=np.float64)
    base[1:gs-1, 1:gs-1, :] = 1.0
    y_goal = (gs - 1) // 2
    base[y_goal, 1, 3] = 0.0
    n = float((gs - 2) * (gs - 2) * 4 - 1)
    base[1:gs-1, 1:gs-1, :] /= n
    return base


def main():
    t0 = time.time()

    p = argparse.ArgumentParser(description="MLP probe: rnn_hidden -> belief tensor")
    p.add_argument("csv", type=str)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[128])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--eps", type=float, default=1e-12)
    args = p.parse_args()

    df_raw = pd.read_csv(args.csv)
    df = df_raw.copy()

    if "rnn_hidden" not in df.columns or "belief" not in df.columns:
        missing = {"rnn_hidden", "belief"} - set(df.columns)
        raise KeyError(f"Missing columns: {sorted(missing)}")

    df["rnn_hidden"] = df["rnn_hidden"].apply(lambda s: parse_rnn_hidden(s, args.hidden_size))
    df["belief_tensor"] = df["belief"].apply(lambda s: parse_belief_tensor(s, args.grid_size))

    if "trajectory_id" in df.columns:
        traj_ids = df["trajectory_id"].unique()
        train_ids, test_ids = train_test_split(traj_ids, test_size=args.test_size, random_state=0, shuffle=True)
        df_train = df[df["trajectory_id"].isin(train_ids)]
        df_test = df[df["trajectory_id"].isin(test_ids)]
    else:
        df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=0, shuffle=True)

    X_train = np.vstack(df_train["rnn_hidden"].to_list())
    Y_train = np.stack(df_train["belief_tensor"].to_list()).reshape(len(df_train), -1)
    X_test = np.vstack(df_test["rnn_hidden"].to_list())
    Y_test_tensor = np.stack(df_test["belief_tensor"].to_list())
    Y_test = Y_test_tensor.reshape(len(df_test), -1)

    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=0,
    )
    model = make_pipeline(StandardScaler(), mlp)
    model.fit(X_train, Y_train)

    Y_pred_raw = model.predict(X_test)
    mse = float(np.mean((Y_test - Y_pred_raw) ** 2))

    eps = args.eps
    Y_pred = np.clip(Y_pred_raw, eps, None)
    Y_pred /= Y_pred.sum(axis=1, keepdims=True)

    Y_true = np.clip(Y_test, eps, None)
    Y_true /= Y_true.sum(axis=1, keepdims=True)

    tv = float(np.mean(0.5 * np.sum(np.abs(Y_true - Y_pred), axis=1)))
    kl = float(np.mean([entropy(t, p) for t, p in zip(Y_true, Y_pred)]))

    base_tensor = uniform_baseline_tensor(args.grid_size)
    base_vec = base_tensor.reshape(-1)
    base_pred = np.tile(base_vec, (Y_true.shape[0], 1))
    base_pred = np.clip(base_pred, eps, None)
    base_pred /= base_pred.sum(axis=1, keepdims=True)
    base_tv = float(np.mean(0.5 * np.sum(np.abs(Y_true - base_pred), axis=1)))
    base_kl = float(np.mean([entropy(t, p) for t, p in zip(Y_true, base_pred)]))

    N = Y_true.shape[0]
    true_arg = np.argmax(Y_true, axis=1)
    true_max = Y_true[np.arange(N), true_arg]
    knows_mask = true_max >= (1.0 - args.tol)
    knows_count = int(knows_mask.sum())
    knows_frac = float(knows_mask.mean()) if N > 0 else 0.0

    pred_arg = np.argmax(Y_pred, axis=1)
    if knows_count > 0:
        match_rate = float(np.mean(pred_arg[knows_mask] == true_arg[knows_mask]))
        prob_on_true = float(np.mean(Y_pred[knows_mask, true_arg[knows_mask]]))
    else:
        match_rate = float("nan")
        prob_on_true = float("nan")

    impossible_mask = Y_true <= args.tol
    impossible_mass_overall = float(np.mean(np.sum(Y_pred * impossible_mask, axis=1)))
    if knows_count > 0:
        impossible_mass_should_know = float(np.mean(np.sum(Y_pred[knows_mask] * impossible_mask[knows_mask], axis=1)))
    else:
        impossible_mass_should_know = float("nan")

    base_impossible_mass_overall = float(np.mean(np.sum(base_pred * impossible_mask, axis=1)))
    if knows_count > 0:
        base_impossible_mass_should_know = float(np.mean(np.sum(base_pred[knows_mask] * impossible_mask[knows_mask], axis=1)))
    else:
        base_impossible_mass_should_know = float("nan")

    print(f"MSE: {mse:.6f}")
    print(f"TV: {tv:.6f}")
    print(f"Mean KL: {kl:.6f}")
    print(f"Baseline TV (uniform-interior-no-goal): {base_tv:.6f}")
    print(f"Baseline Mean KL (uniform-interior-no-goal): {base_kl:.6f}")
    print(f"'Should-know' steps: {knows_count} / {N} ({100.0*knows_frac:.2f}%)")
    print(f"Argmax match rate (should-know): {match_rate:.6f}")
    print(f"Mean predicted prob on true location (should-know): {prob_on_true:.6f}")
    print(f"Impossible mass (overall): {impossible_mass_overall:.6f}")
    print(f"Impossible mass (should-know): {impossible_mass_should_know:.6f}")
    print(f"Baseline impossible mass (overall): {base_impossible_mass_overall:.6f}")
    print(f"Baseline impossible mass (should-know): {base_impossible_mass_should_know:.6f}")

    out_root = Path("memory_probe_results_mlp")
    out_dir = out_root / Path(args.csv).name
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")

    

    meta = {
        "csv": Path(args.csv).name,
        "hidden_size": args.hidden_size,
        "grid_size": args.grid_size,
        "alpha": args.alpha,
        "hidden_layers": args.hidden_layers,
        "test_size": args.test_size,
        "tol": args.tol,
        "eps": args.eps,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "mse": mse,
        "tv": tv,
        "mean_kl": kl,
        "baseline_tv": base_tv,
        "baseline_mean_kl": base_kl,
        "should_know_count": knows_count,
        "should_know_frac": knows_frac,
        "argmax_match_rate": match_rate,
        "mean_prob_on_true_location": prob_on_true,
        "impossible_mass_overall": impossible_mass_overall,
        "impossible_mass_should_know": impossible_mass_should_know,
        "baseline_impossible_mass_overall": base_impossible_mass_overall,
        "baseline_impossible_mass_should_know": base_impossible_mass_should_know,
        "duration_seconds": round(time.time() - t0, 3),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)

    pred_tensor = Y_pred.reshape((-1, args.grid_size, args.grid_size, 4))
    true_tensor = Y_true.reshape((-1, args.grid_size, args.grid_size, 4))
    abs_err_tensor = np.abs(pred_tensor - true_tensor)
    sq_err_tensor = (pred_tensor - true_tensor) ** 2

    df_out = df_raw.loc[df_test.index].copy()
    df_out["prediction"] = [tensor_to_json(t) for t in pred_tensor]
    df_out["abs_error"] = [tensor_to_json(t) for t in abs_err_tensor]
    df_out["squared_error"] = [tensor_to_json(t) for t in sq_err_tensor]

    enriched_name = f"test_with_mlp_predictions_{Path(args.csv).name}"
    df_out.to_csv(out_dir / enriched_name, index=False)

    print(f"Saved model, metrics, and test set to: {out_dir}")
    print(f"Duration: {meta['duration_seconds']:.3f}s")


if __name__ == "__main__":
    main()
