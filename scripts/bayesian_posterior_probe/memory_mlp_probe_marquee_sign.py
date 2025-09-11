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

# ---------------- Constants ----------------
N_GOALS = 16
HIDDEN_LAYERS = [128]   # fixed architecture
ALPHA = 1e-4
TEST_SIZE = 0.2
TOL = 1e-9
EPS = 1e-12
# -------------------------------------------

def parse_rnn_hidden(s: str) -> np.ndarray:
    arr = np.fromstring(s.strip("[]").replace(",", " "), sep=" ")
    return arr.astype(np.float64)

def parse_belief_vector(s: str) -> np.ndarray:
    v = np.asarray(json.loads(s), dtype=np.float64)
    if v.shape != (N_GOALS,):
        raise ValueError(f"belief vector shape {v.shape} != ({N_GOALS},)")
    return v

def vector_to_json(v: np.ndarray) -> str:
    return json.dumps(np.asarray(v, dtype=float).tolist(), separators=(",", ":"))

def uniform_baseline_vector() -> np.ndarray:
    return np.full((N_GOALS,), 1.0 / N_GOALS, dtype=np.float64)

def softmax(z: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    denom = np.maximum(ez.sum(axis=axis, keepdims=True), eps)
    return ez / denom

def main():
    t0 = time.time()

    p = argparse.ArgumentParser(description="MLP probe (regression): rnn_hidden -> 16-d belief distribution")
    p.add_argument("csv", type=str)
    args = p.parse_args()

    df_raw = pd.read_csv(args.csv)
    df = df_raw.copy()

    # Parse inputs
    df["rnn_hidden"] = df["rnn_hidden_state"].apply(parse_rnn_hidden)
    df["belief_vec"] = df["bayes_posterior"].apply(parse_belief_vector)

    # Deduce hidden size
    hidden_size = len(df["rnn_hidden"].iloc[0])
    print(f"Detected hidden size: {hidden_size}")

    # Split by trajectory if available
    if "trajectory_id" in df.columns:
        traj_ids = df["trajectory_id"].unique()
        train_ids, test_ids = train_test_split(traj_ids, test_size=TEST_SIZE, random_state=0, shuffle=True)
        df_train = df[df["trajectory_id"].isin(train_ids)].copy()
        df_test = df[df["trajectory_id"].isin(test_ids)].copy()
    else:
        df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=0, shuffle=True)

    # X and Y
    X_train = np.vstack(df_train["rnn_hidden"].to_list())
    Y_train = np.stack(df_train["belief_vec"].to_list())
    X_test = np.vstack(df_test["rnn_hidden"].to_list())
    Y_test = np.stack(df_test["belief_vec"].to_list())

    # Clip + renormalize ground-truth distributions (safety)
    Y_train_true = np.clip(Y_train, EPS, None)
    Y_train_true /= Y_train_true.sum(axis=1, keepdims=True)
    Y_true = np.clip(Y_test, EPS, None)
    Y_true /= Y_true.sum(axis=1, keepdims=True)

    # Train target = logits of the true distribution
    Y_train_logits = np.log(Y_train_true)

    # Model: regress to logits
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(HIDDEN_LAYERS),
        activation="relu",
        alpha=ALPHA,
        max_iter=2000,
        tol=1e-5,
        random_state=0,
    )
    model = make_pipeline(StandardScaler(), mlp)
    model.fit(X_train, Y_train_logits)

    # Predict logits -> softmax -> probabilities
    Y_pred_logits = model.predict(X_test)
    if Y_pred_logits.ndim == 1:
        Y_pred_logits = Y_pred_logits.reshape(-1, N_GOALS)
    Y_pred = softmax(Y_pred_logits, axis=1, eps=EPS)

    # Metrics
    mse = float(np.mean((Y_true - Y_pred) ** 2))
    tv = float(np.mean(0.5 * np.sum(np.abs(Y_true - Y_pred), axis=1)))
    kl = float(np.mean([entropy(t, p) for t, p in zip(Y_true, Y_pred)]))  # KL(true || pred)

    # Baseline: uniform
    base_vec = uniform_baseline_vector()
    base_pred = np.tile(base_vec, (Y_true.shape[0], 1))
    base_tv = float(np.mean(0.5 * np.sum(np.abs(Y_true - base_pred), axis=1)))
    base_kl = float(np.mean([entropy(t, p) for t, p in zip(Y_true, base_pred)]))

    # Should-know
    N = Y_true.shape[0]
    true_arg = np.argmax(Y_true, axis=1)
    true_max = Y_true[np.arange(N), true_arg]
    knows_mask = true_max >= (1.0 - TOL)
    knows_count = int(knows_mask.sum())
    knows_frac = float(knows_mask.mean()) if N > 0 else 0.0

    pred_arg = np.argmax(Y_pred, axis=1)
    if knows_count > 0:
        match_rate = float(np.mean(pred_arg[knows_mask] == true_arg[knows_mask]))
        prob_on_true = float(np.mean(Y_pred[knows_mask, true_arg[knows_mask]]))
    else:
        match_rate = float("nan")
        prob_on_true = float("nan")

    impossible_mask = Y_true <= TOL
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

    # Prints
    print(f"MSE: {mse:.6f}")
    print(f"TV: {tv:.6f}")
    print(f"Mean KL: {kl:.6f}")
    print(f"Baseline TV (uniform-16): {base_tv:.6f}")
    print(f"Baseline Mean KL (uniform-16): {base_kl:.6f}")
    print(f"'Should-know' steps: {knows_count} / {N} ({100.0*knows_frac:.2f}%)")
    print(f"Argmax match rate (should-know): {match_rate:.6f}")
    print(f"Mean predicted prob on true goal (should-know): {prob_on_true:.6f}")
    print(f"Impossible mass (overall): {impossible_mass_overall:.6f}")
    print(f"Impossible mass (should-know): {impossible_mass_should_know:.6f}")
    print(f"Baseline impossible mass (overall): {base_impossible_mass_overall:.6f}")
    print(f"Baseline impossible mass (should-know): {base_impossible_mass_should_know:.6f}")

    # Save
    out_root = Path("memory_probe_results_mlp")
    out_dir = out_root / Path(args.csv).name
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")

    meta = {
        "csv": Path(args.csv).name,
        "hidden_size": hidden_size,
        "hidden_layers": HIDDEN_LAYERS,
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
        "mean_prob_on_true_goal": prob_on_true,
        "impossible_mass_overall": impossible_mass_overall,
        "impossible_mass_should_know": impossible_mass_should_know,
        "baseline_impossible_mass_overall": base_impossible_mass_overall,
        "baseline_impossible_mass_should_know": base_impossible_mass_should_know,
        "alpha": ALPHA,
        "test_size": TEST_SIZE,
        "tol": TOL,
        "eps": EPS,
        "duration_seconds": round(time.time() - t0, 3),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)

    df_out = df_raw.loc[df_test.index].copy()
    df_out["prediction"] = [vector_to_json(v) for v in Y_pred]
    df_out["abs_error"] = [vector_to_json(np.abs(p - t)) for p, t in zip(Y_pred, Y_true)]
    df_out["squared_error"] = [vector_to_json((p - t) ** 2) for p, t in zip(Y_pred, Y_true)]

    enriched_name = f"test_with_mlp_predictions_{Path(args.csv).name}"
    df_out.to_csv(out_dir / enriched_name, index=False)

    print(f"Saved model, metrics, and test set to: {out_dir}")
    print(f"Duration: {meta['duration_seconds']:.3f}s")

if __name__ == "__main__":
    main()
