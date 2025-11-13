#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------
# Parsing & utilities
# ---------------------------

def parse_rnn_hidden(s, hidden_size):
    arr = np.fromstring(s.strip("[]").replace(",", " "), sep=" ")
    if arr.size != hidden_size:
        raise ValueError(f"rnn_hidden length {arr.size} != hidden_size {hidden_size}")
    return arr.astype(np.float64)


def parse_belief_tensor(s, grid_size):
    B = np.asarray(json.loads(s), dtype=np.float64)
    expected = (grid_size, grid_size, 4)
    if B.shape != expected:
        raise ValueError(f"belief shape {B.shape} != {expected}")
    return B


def tensor_to_json(t):
    return json.dumps(np.asarray(t, dtype=float).tolist(), separators=(",", ":"))


def normalize_rows(A, eps=1e-12):
    A = np.clip(A, eps, None)
    denom = A.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0, eps, denom)
    return A / denom


def initial_distribution_tensor(grid_size):
    """Uniform over interior cells only (no borders), spread across 4 headings."""
    g = grid_size
    B = np.zeros((g, g, 4), dtype=float)
    if g >= 3:
        B[1:g-1, 1:g-1, :] = 1.0
    total = B.sum()
    if total > 0:
        B /= total
    return B


def extract_model_reference(csv_name):
    """
    Extract substring between 'model_' and the next '_' (e.g., '...model_e88b7b352b_...').
    """
    m = re.search(r"model_([^_]+)_", csv_name)
    return m.group(1) if m else "unknown"


# ---------------------------
# Predictors
# ---------------------------

def make_pipeline_predictor(pipeline_model, eps=1e-12):
    """Wrap an sklearn regressor pipeline to return per-row distributions."""
    def _predictor(X):
        raw = pipeline_model.predict(X)
        return normalize_rows(raw, eps=eps)
    return _predictor


def make_constant_predictor_from_tensor(tensor, eps=1e-12):
    """Constant predictor equal to the provided belief tensor."""
    base_vec = tensor.reshape(-1)
    base_vec = normalize_rows(base_vec[None, :], eps=eps)[0]
    def _predictor(X):
        return np.tile(base_vec, (X.shape[0], 1))
    return _predictor


# ---------------------------
# Metrics (generic for any predictor)
# ---------------------------

def evaluate_predictor_metrics(predictor, X, Y_true_vec, tol=1e-9, eps=1e-12):
    """
    Returns:
      tv, mean_kl_bits, should_know_count, should_know_frac,
      argmax_match_rate (should-know accuracy),
      mean_prob_on_true_location (should-know mass),
      impossible_mass_overall, impossible_mass_should_know
    """
    Y_true = normalize_rows(Y_true_vec, eps=eps)
    Y_pred = normalize_rows(predictor(X), eps=eps)

    tv = float(np.mean(0.5 * np.sum(np.abs(Y_true - Y_pred), axis=1)))
    mean_kl_bits = float(np.mean([entropy(t, p, base=2) for t, p in zip(Y_true, Y_pred)]))

    N = Y_true.shape[0]
    true_arg = np.argmax(Y_true, axis=1)
    true_max = Y_true[np.arange(N), true_arg]
    knows_mask = true_max >= (1.0 - tol)
    knows_count = int(knows_mask.sum())
    knows_frac = float(knows_mask.mean()) if N > 0 else 0.0

    if knows_count > 0:
        pred_arg = np.argmax(Y_pred, axis=1)
        argmax_match_rate = float(np.mean(pred_arg[knows_mask] == true_arg[knows_mask]))
        mean_prob_on_true_location = float(np.mean(Y_pred[knows_mask, true_arg[knows_mask]]))
    else:
        argmax_match_rate = float("nan")
        mean_prob_on_true_location = float("nan")

    impossible_mask = (Y_true <= tol)
    impossible_mass_overall = float(np.mean(np.sum(Y_pred * impossible_mask, axis=1)))
    if knows_count > 0:
        impossible_mass_should_know = float(np.mean(
            np.sum(Y_pred[knows_mask] * impossible_mask[knows_mask], axis=1)
        ))
    else:
        impossible_mass_should_know = float("nan")

    return {
        "tv": tv,
        "mean_kl_bits": mean_kl_bits,
        "should_know_count": knows_count,
        "should_know_frac": knows_frac,
        "argmax_match_rate": argmax_match_rate,
        "mean_prob_on_true_location": mean_prob_on_true_location,
        "impossible_mass_overall": impossible_mass_overall,
        "impossible_mass_should_know": impossible_mass_should_know,
    }


# ---------------------------
# CLI main (MLP + Linear + constants)
# ---------------------------

def main():
    t0 = time.time()
    print("started")
    p = argparse.ArgumentParser(description="Predictors for rnn_hidden -> belief tensor (MLP, Linear, constants)")
    p.add_argument("--csv", required = True, type=str)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--grid-size", type=int, default=8)
    # MLP
    p.add_argument("--mlp-alpha", type=float, default=1e-4)
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[128])
    # Linear (Ridge)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    # Common
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--eps", type=float, default=1e-12)
    args = p.parse_args()

    df_raw = pd.read_csv(args.csv)
    df = df_raw.copy()

    # Required columns
    required_cols = {"rnn_hidden", "belief", "zero_fed_h_state"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    # Parse features and labels
    print("parsing features and labels")
    df["rnn_hidden"] = df["rnn_hidden"].apply(lambda s: parse_rnn_hidden(s, args.hidden_size))
    df["zero_fed_h_state"] = df["zero_fed_h_state"].apply(lambda s: parse_rnn_hidden(s, args.hidden_size))
    df["belief_tensor"] = df["belief"].apply(lambda s: parse_belief_tensor(s, args.grid_size))

    # Trajectory-aware split if available
    if "trajectory_id" in df.columns:
        traj_ids = df["trajectory_id"].unique()
        train_ids, test_ids = train_test_split(traj_ids, test_size=args.test_size, random_state=0, shuffle=True)
        df_train = df[df["trajectory_id"].isin(train_ids)]
        df_test = df[df["trajectory_id"].isin(test_ids)]
    else:
        df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=0, shuffle=True)

    # Matrices (rnn_hidden)
    X_train_hidden = np.vstack(df_train["rnn_hidden"].to_list())
    X_test_hidden = np.vstack(df_test["rnn_hidden"].to_list())

    # Matrices (zero_fed_h_state)
    X_train_zero = np.vstack(df_train["zero_fed_h_state"].to_list())
    X_test_zero = np.vstack(df_test["zero_fed_h_state"].to_list())

    # Labels
    Y_train_tensor = np.stack(df_train["belief_tensor"].to_list())
    Y_test_tensor = np.stack(df_test["belief_tensor"].to_list())
    Y_train = Y_train_tensor.reshape(len(df_train), -1)
    Y_test = Y_test_tensor.reshape(len(df_test), -1)

    print("fitting MLP probes")

    # ----------------- Fit MLP probes -----------------
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation="relu",
        solver="adam",
        alpha=args.mlp_alpha,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=0,
    )
    mlp_hidden_model = make_pipeline(StandardScaler(), MLPRegressor(**mlp.get_params()))
    mlp_zero_model   = make_pipeline(StandardScaler(), MLPRegressor(**mlp.get_params()))

    mlp_hidden_model.fit(X_train_hidden, Y_train)
    mlp_zero_model.fit(X_train_zero, Y_train)

    # Keep raw preds for enriched CSVs
    Y_pred_raw_mlp_hidden = mlp_hidden_model.predict(X_test_hidden)
    Y_pred_raw_mlp_zero   = mlp_zero_model.predict(X_test_zero)

    print("fitting linear probes")
    # ----------------- Fit Linear (Ridge) probes -----------------
    ridge = Ridge(alpha=args.ridge_alpha, random_state=0)
    lin_hidden_model = make_pipeline(StandardScaler(), Ridge(**ridge.get_params()))
    lin_zero_model   = make_pipeline(StandardScaler(), Ridge(**ridge.get_params()))

    lin_hidden_model.fit(X_train_hidden, Y_train)
    lin_zero_model.fit(X_train_zero, Y_train)

    # Keep raw preds for enriched CSVs
    Y_pred_raw_lin_hidden = lin_hidden_model.predict(X_test_hidden)
    Y_pred_raw_lin_zero   = lin_zero_model.predict(X_test_zero)

    # ----------------- Constant predictors -----------------
    init_pred = make_constant_predictor_from_tensor(
        initial_distribution_tensor(args.grid_size), eps=args.eps
    )
    avg_pred = make_constant_predictor_from_tensor(
        np.mean(Y_train_tensor, axis=0), eps=args.eps
    )

    # ----------------- Wrap learned models as predictors -----------------
    mlp_pred_hidden = make_pipeline_predictor(mlp_hidden_model, eps=args.eps)
    mlp_pred_zero   = make_pipeline_predictor(mlp_zero_model,   eps=args.eps)
    lin_pred_hidden = make_pipeline_predictor(lin_hidden_model, eps=args.eps)
    lin_pred_zero   = make_pipeline_predictor(lin_zero_model,   eps=args.eps)

    print("evaluating metrics")
    # ----------------- Evaluate all predictors -----------------
    metrics_by_name = {}
    metrics_by_name["mlp_from_rnn_hidden"]       = evaluate_predictor_metrics(mlp_pred_hidden, X_test_hidden, Y_test, tol=args.tol, eps=args.eps)
    metrics_by_name["mlp_from_zero_fed_h_state"] = evaluate_predictor_metrics(mlp_pred_zero,   X_test_zero,   Y_test, tol=args.tol, eps=args.eps)
    metrics_by_name["linear_from_rnn_hidden"]    = evaluate_predictor_metrics(lin_pred_hidden, X_test_hidden, Y_test, tol=args.tol, eps=args.eps)
    metrics_by_name["linear_from_zero_fed_h_state"] = evaluate_predictor_metrics(lin_pred_zero, X_test_zero,   Y_test, tol=args.tol, eps=args.eps)
    metrics_by_name["initial_distribution"]      = evaluate_predictor_metrics(init_pred,        X_test_hidden, Y_test, tol=args.tol, eps=args.eps)
    metrics_by_name["average_belief"]            = evaluate_predictor_metrics(avg_pred,         X_test_hidden, Y_test, tol=args.tol, eps=args.eps)

    # Quick console view
    def brief(m):
        return (
            f"TV={m['tv']:.6f}  KL(bits)={m['mean_kl_bits']:.6f}  "
            f"SK-acc={m['argmax_match_rate']:.6f}  SK-mass={m['mean_prob_on_true_location']:.6f}  "
            f"Imp(overall)={m['impossible_mass_overall']:.6f}  Imp(SK)={m['impossible_mass_should_know']:.6f}"
        )
    order = [
        "mlp_from_rnn_hidden",
        "mlp_from_zero_fed_h_state",
        "linear_from_rnn_hidden",
        "linear_from_zero_fed_h_state",
        "initial_distribution",
        "average_belief",
    ]
    for name in order:
        print(f"{name:>32}: {brief(metrics_by_name[name])}")

    # ----------------- Outputs -----------------
    csv_name = Path(args.csv).name
    model_ref = extract_model_reference(csv_name)

    out_root = Path("probe_evaluations")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"compass_world_model_{model_ref}_ts_{timestamp}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save learned models (four)
    joblib.dump(mlp_hidden_model, out_dir / "mlp_from_rnn_hidden.joblib")
    joblib.dump(mlp_zero_model,   out_dir / "mlp_from_zero_fed_h_state.joblib")
    joblib.dump(lin_hidden_model, out_dir / "linear_from_rnn_hidden.joblib")
    joblib.dump(lin_zero_model,   out_dir / "linear_from_zero_fed_h_state.joblib")

    # Single JSON, flat dict of predictors -> metrics
    meta = {
        "csv": csv_name,
        "model_reference": model_ref,
        "hidden_size": args.hidden_size,
        "grid_size": args.grid_size,
        "test_size": args.test_size,
        "tol": args.tol,
        "eps": args.eps,
        "predictors": metrics_by_name,
        "duration_seconds": round(time.time() - t0, 3),
    }
    out_json_name = f"compass_world_all_predictors_model_{model_ref}.json"
    with open(out_dir / out_json_name, "w") as f:
        json.dump(meta, f, indent=2)

    # Enriched CSVs for all learned probes (normalized predictions vs normalized truth)
    true_tensor = normalize_rows(Y_test, eps=args.eps).reshape((-1, args.grid_size, args.grid_size, 4))

    print("writing enriched csvs")

    def write_enriched(df_test_idx, pred_raw, fname_prefix):
        Y_pred_norm = normalize_rows(pred_raw, eps=args.eps)
        pred_tensor = Y_pred_norm.reshape((-1, args.grid_size, args.grid_size, 4))
        df_out = df_raw.loc[df_test_idx].copy()
        df_out["prediction"] = [tensor_to_json(t) for t in pred_tensor]
        df_out["abs_error"] = [tensor_to_json(np.abs(pred_tensor[i] - true_tensor[i])) for i in range(len(df_out))]
        df_out["squared_error"] = [tensor_to_json((pred_tensor[i] - true_tensor[i]) ** 2) for i in range(len(df_out))]
        df_out.to_csv(out_dir / f"{fname_prefix}__{csv_name}", index=False)

    write_enriched(df_test.index, Y_pred_raw_mlp_hidden, "test_with_mlp_rnn_hidden_predictions")
    write_enriched(df_test.index, Y_pred_raw_mlp_zero,   "test_with_mlp_zero_fed_h_state_predictions")
    write_enriched(df_test.index, Y_pred_raw_lin_hidden, "test_with_linear_rnn_hidden_predictions")
    write_enriched(df_test.index, Y_pred_raw_lin_zero,   "test_with_linear_zero_fed_h_state_predictions")

    print(f"Saved to: {out_dir}")
    print(f"JSON: {out_json_name}")
    print(f"Duration: {meta['duration_seconds']:.3f}s")


if __name__ == "__main__":
    main()
