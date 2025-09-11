#!/usr/bin/env python3
import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax

# ---------------- Constants ----------------
TEST_FRAC = 0.20
SEED = 0
LR = 1e-2
WEIGHT_DECAY = 1e-4
EPOCHS = 50
BATCH_SIZE = 8192
EPS = 1e-12
TOL = 1e-9
# -------------------------------------------

def _to_arr(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1)

def _prepare(df):
    X = np.stack([_to_arr(x) for x in df["rnn_hidden_state"].values])
    Y = np.stack([_to_arr(y) for y in df["bayes_posterior"].values])
    Y = (Y / np.clip(Y.sum(axis=1, keepdims=True), 1e-8, None)).astype(np.float32)
    return X, Y

def _split_by_traj(df, test_frac=TEST_FRAC, seed=SEED):
    ids = df["trajectory_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_frac)))
    test_ids = set(ids[:n_test])
    train_df = df[~df["trajectory_id"].isin(test_ids)]
    test_df = df[df["trajectory_id"].isin(test_ids)]
    return train_df, test_df, sorted(test_ids)

def _train_probe(Xtr, Ytr, lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED):
    H, K = Xtr.shape[1], Ytr.shape[1]
    key = jax.random.PRNGKey(seed)
    params = (
        jax.random.normal(key, (H, K)) * 0.01,
        jnp.zeros((K,), dtype=jnp.float32),
    )
    opt = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = opt.init(params)

    def predict(p, X):
        W, b = p
        return jax.nn.softmax(X @ W + b, axis=-1)

    def loss_fn(p, X, Y):
        P = predict(p, X)
        # CE(true || pred); add small epsilon for stability
        ce = -jnp.sum(Y * jnp.log(P + 1e-9), axis=1).mean()
        return ce

    @jax.jit
    def step(p, s, X, Y):
        loss_val, grads = jax.value_and_grad(loss_fn)(p, X, Y)
        updates, s = opt.update(grads, s, p)
        p = optax.apply_updates(p, updates)
        return p, s, loss_val

    Xtr_j, Ytr_j = jnp.asarray(Xtr), jnp.asarray(Ytr)
    n = Xtr.shape[0]
    idx = np.arange(n)

    for e in range(epochs):
        rng = np.random.default_rng(seed + e)
        rng.shuffle(idx)
        for i in range(0, n, batch_size):
            j = idx[i : i + batch_size]
            params, opt_state, _ = step(params, opt_state, Xtr_j[j], Ytr_j[j])

    return params

def _predict_with_probe(params, X, mean, std):
    Xn = (X - mean) / std
    logits = Xn @ np.asarray(params[0]) + np.asarray(params[1])
    logits = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(logits); P /= np.clip(P.sum(axis=1, keepdims=True), EPS, None)
    return P

def _metrics(Y_true, Y_pred):
    # Ensure valid probabilities
    Y_true = Y_true / np.clip(Y_true.sum(axis=1, keepdims=True), EPS, None)
    Y_pred = Y_pred / np.clip(Y_pred.sum(axis=1, keepdims=True), EPS, None)

    mse = float(np.mean((Y_true - Y_pred) ** 2))
    tv = float(np.mean(0.5 * np.sum(np.abs(Y_true - Y_pred), axis=1)))
    # KL(true || pred)
    kl = float(np.mean(np.sum(Y_true * (np.log(np.clip(Y_true, EPS, None)) - np.log(np.clip(Y_pred, EPS, None))), axis=1)))

    # Baseline: uniform over K classes
    K = Y_true.shape[1]
    base = np.full((1, K), 1.0 / K, dtype=np.float64)
    base_pred = np.repeat(base, Y_true.shape[0], axis=0)

    base_tv = float(np.mean(0.5 * np.sum(np.abs(Y_true - base_pred), axis=1)))
    base_kl = float(np.mean(np.sum(Y_true * (np.log(np.clip(Y_true, EPS, None)) - np.log(np.clip(base_pred, EPS, None))), axis=1)))

    # Should-know (true distribution nearly deterministic)
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

    # Impossible mass
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

    return {
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
    }

def _eval_on_df_with_probe(params, df, mean, std):
    if len(df) == 0:
        return {"n": 0, "ce": float("nan"), "acc": float("nan")}, None
    X, Y = _prepare(df)
    P = _predict_with_probe(params, X, mean, std)
    # cross-entropy & argmax-acc (not ideal for uniform, but kept for continuity)
    ce = float((-np.sum(Y * np.log(np.clip(P, 1e-9, None)), axis=1)).mean())
    acc = float((np.argmax(P, axis=1) == np.argmax(Y, axis=1)).mean())
    return {"n": int(len(df)), "ce": ce, "acc": acc}, (Y, P)

def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Linear softmax probe on RNN hidden states (JAX)")
    ap.add_argument("csv", type=str)
    args = ap.parse_args()

    df_raw = pd.read_csv(args.csv)

    # Split by trajectory (prevents leakage)
    if "trajectory_id" not in df_raw.columns:
        raise ValueError("Expected a 'trajectory_id' column for trajectory-aware split.")
    train_df, test_df, heldout_ids = _split_by_traj(df_raw, test_frac=TEST_FRAC, seed=SEED)

    # Prepare arrays
    Xtr, Ytr = _prepare(train_df)
    Xte, Yte = _prepare(test_df)

    H, K = Xtr.shape[1], Ytr.shape[1]
    print(f"Detected hidden size H={H}, number of goals K={K}")

    # Standardize inputs (fit on train only)
    mean = Xtr.mean(axis=0, keepdims=True).astype(np.float32)
    std = (Xtr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    Xtr_n = (Xtr - mean) / std
    Xte_n = (Xte - mean) / std

    # Train probe
    params = _train_probe(Xtr_n, Ytr, lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED)

    # Train/test CE/acc (legacy metrics)
    train_basic, (Ytr_true, Ytr_pred) = _eval_on_df_with_probe(params, train_df, mean, std)
    test_basic, (Yte_true, Yte_pred) = _eval_on_df_with_probe(params, test_df, mean, std)

    # New metrics on TEST set
    new_metrics = _metrics(Yte_true, Yte_pred)

    # Print summary
    print("\n=== Basic metrics ===")
    print(f"Train CE: {train_basic['ce']:.6f} | Train Acc: {train_basic['acc']:.6f} | n={train_basic['n']}")
    print(f"Test  CE: {test_basic['ce']:.6f} | Test  Acc: {test_basic['acc']:.6f} | n={test_basic['n']}")

    print("\n=== Distribution metrics (TEST) ===")
    for k in [
        "mse","tv","mean_kl","baseline_tv","baseline_mean_kl",
        "should_know_count","should_know_frac","argmax_match_rate",
        "mean_prob_on_true_goal","impossible_mass_overall","impossible_mass_should_know",
        "baseline_impossible_mass_overall","baseline_impossible_mass_should_know",
    ]:
        v = new_metrics[k]
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    # Save artifacts
    out_root = Path("memory_probe_results_linear_jax")
    out_dir = out_root / Path(args.csv).name
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = {
        "W": np.array(params[0]),
        "b": np.array(params[1]),
        "mean": mean[0],
        "std": std[0],
        "classes": K,
        "hidden_size": H,
    }
    with open(out_dir / "probe.json", "w") as f:
        json.dump({k: v.tolist() if hasattr(v, "tolist") else v for k, v in probe.items()}, f)

    metrics_all = {
        "csv": Path(args.csv).name,
        "hidden_size": H,
        "classes": K,
        "train_ce": train_basic["ce"],
        "train_acc": train_basic["acc"],
        "test_ce": test_basic["ce"],
        "test_acc": test_basic["acc"],
        **new_metrics,
        "duration_seconds": round(time.time() - t0, 3),
        "heldout_trajectory_ids": heldout_ids,
        "constants": {
            "test_frac": TEST_FRAC,
            "seed": SEED,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "eps": EPS,
            "tol": TOL,
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2)

    # Enriched CSV for TEST set
    df_out = test_df.copy()
    preds_json = [json.dumps(p.tolist(), separators=(",", ":")) for p in Yte_pred]
    abs_err_json = [json.dumps(np.abs(p - t).tolist(), separators=(",", ":")) for p, t in zip(Yte_pred, Yte_true)]
    sq_err_json = [json.dumps(((p - t) ** 2).tolist(), separators=(",", ":")) for p, t in zip(Yte_pred, Yte_true)]
    df_out["prediction"] = preds_json
    df_out["abs_error"] = abs_err_json
    df_out["squared_error"] = sq_err_json

    enriched_name = f"test_with_linear_probe_predictions_{Path(args.csv).name}"
    df_out.to_csv(out_dir / enriched_name, index=False)

    print(f"\nSaved probe, metrics, and test predictions to: {out_dir}")
    print(f"Duration: {metrics_all['duration_seconds']:.3f}s")

if __name__ == "__main__":
    main()
