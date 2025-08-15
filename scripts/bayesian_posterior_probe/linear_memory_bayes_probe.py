#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import entropy

def parse_rnn_hidden_state(s):
    # strip brackets, replace commas, then parse floats
    arr = np.fromstring(s.strip("[]").replace(",", " "), sep=" ")
    if arr.size == 256:
        return arr.reshape(1, 256)
    elif arr.size % 256 == 0:
        return arr.reshape(-1, 256)
    else:
        raise ValueError(f"Unexpected hidden state length: {arr.size}")

def parse_bayes_posterior(s):
    # parse your posterior strings into a 1D float array
    return np.fromstring(s.strip("[]").replace(",", " "), sep=" ")

def main():
    parser = argparse.ArgumentParser(
        description="MLP probe on true Bayes posteriors\n"
                    "(all time‐steps, trajectory‐level train/test split)"
    )
    parser.add_argument(
        "csv_file",
        help="CSV filename inside supervised_learning_data/"
    )
    parser.add_argument(
        "--alpha", type=float, default=1e-4,
        help="L₂ weight decay for the MLP (default=1e-4)"
    )
    parser.add_argument(
        "--hidden-layers", type=int, nargs="+", default=[128],
        help="Sizes of hidden layers, e.g. --hidden-layers 128 64"
    )
    args = parser.parse_args()

    # ─── Load & parse ─────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join("supervised_learning_data", args.csv_file))
    df["rnn_hidden_state"] = df["rnn_hidden_state"].apply(parse_rnn_hidden_state)
    df["bayes_posterior"]  = df["bayes_posterior"].apply(parse_bayes_posterior)

    # ─── Split trajectories into train vs. test ───────────────────────────────────
    traj_ids = df["trajectory_id"].unique()
    train_ids, test_ids = train_test_split(traj_ids, test_size=0.2, random_state=0)
    df_train = df[df["trajectory_id"].isin(train_ids)]
    df_test  = df[df["trajectory_id"].isin(test_ids)]

    # ─── Stack into feature & target arrays ───────────────────────────────────────
    X_train = np.vstack([h.ravel() for h in df_train["rnn_hidden_state"]])  # (N_train, D)
    y_train = np.vstack(df_train["bayes_posterior"].values)                  # (N_train, K)
    X_test  = np.vstack([h.ravel() for h in df_test["rnn_hidden_state"]])   # (N_test, D)
    y_test  = np.vstack(df_test["bayes_posterior"].values)                   # (N_test, K)

    # ─── Build & train MLP regressor ──────────────────────────────────────────────
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
    model = make_pipeline(
        StandardScaler(),
        mlp
    )
    model.fit(X_train, y_train)

    # ─── Raw predictions (before normalization) ─────────────────────────────────
    y_pred_raw = model.predict(X_test)

    # ─── Compute MSE on raw outputs ──────────────────────────────────────────────
    mse = np.mean((y_test - y_pred_raw) ** 2)
    print("MSE (squared‐error):", mse)

    # ─── Clip & renormalize into bona‐fide probability vectors ───────────────────
    eps = 1e-12
    y_pred = np.clip(y_pred_raw, eps, None)
    y_pred /= y_pred.sum(axis=1, keepdims=True)

    y_true = np.clip(y_test, eps, None)
    y_true /= y_true.sum(axis=1, keepdims=True)

    # ─── Total Variation (TV) & KL divergence ────────────────────────────────────
    tv = np.mean(0.5 * np.sum(np.abs(y_true - y_pred), axis=1))
    print("TV (on prob dists):", tv)

    kl = np.mean([entropy(t, p) for t, p in zip(y_true, y_pred)])
    print("Mean KL divergence:", kl)

if __name__ == "__main__":
    main()
