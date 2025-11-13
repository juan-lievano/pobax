#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BRIEF_KEYS = [
    ("tv", "Total Variation (TV)"),
    ("mean_kl_bits", "Mean KL (bits)"),
    ("argmax_match_rate", "Should-Know Accuracy"),
    ("mean_prob_on_true_location", "Should-Know Mass"),
    ("impossible_mass_overall", "Impossible Mass (overall)"),
    ("impossible_mass_should_know", "Impossible Mass (should-know)"),
]

DISPLAY_ORDER = [
    "mlp_from_rnn_hidden",
    "mlp_from_zero_fed_h_state",
    "linear_from_rnn_hidden",
    "linear_from_zero_fed_h_state",
    "initial_distribution",
    "average_belief",
]


def pretty_name(k: str) -> str:
    return {
        "mlp_from_rnn_hidden": "MLP (rnn_hidden)",
        "mlp_from_zero_fed_h_state": "MLP (zero_fed_h_state)",
        "linear_from_rnn_hidden": "Linear (rnn_hidden)",
        "linear_from_zero_fed_h_state": "Linear (zero_fed_h_state)",
        "initial_distribution": "Initial distribution",
        "average_belief": "Average belief",
    }.get(k, k)


def main():
    p = argparse.ArgumentParser(description="Visualize probe metrics from JSON.")
    p.add_argument("--json", required = True, type=str, help="Path to the JSON produced by the probe script")
    args = p.parse_args()

    json_path = Path(args.json).resolve()
    with open(json_path, "r") as f:
        meta = json.load(f)

    metrics = meta["predictors"]
    rows = []
    for name in DISPLAY_ORDER:
        if name not in metrics:
            continue
        m = metrics[name]
        rows.append(
            {
                "predictor": pretty_name(name),
                "tv": m.get("tv"),
                "mean_kl_bits": m.get("mean_kl_bits"),
                "argmax_match_rate": m.get("argmax_match_rate"),
                "mean_prob_on_true_location": m.get("mean_prob_on_true_location"),
                "impossible_mass_overall": m.get("impossible_mass_overall"),
                "impossible_mass_should_know": m.get("impossible_mass_should_know"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No metrics found in JSON.")

    # Save in same directory as JSON
    img_dir = json_path.parent
    model_ref = meta.get("model_reference", "unknown")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = img_dir / f"{model_ref}_{timestamp}.png"

    n_metrics = len(BRIEF_KEYS)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 2.6 * n_metrics), constrained_layout=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, BRIEF_KEYS):
        vals = df[key].astype(float)
        ax.barh(df["predictor"], vals)
        ax.set_title(title)
        ax.set_xlabel(key)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        for i, v in enumerate(vals):
            if pd.notna(v):
                ax.text(v, i, f" {v:.4f}", va="center")

    fig.suptitle("Probe Metrics Comparison", y=1.02, fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved image to: {out_path}")


if __name__ == "__main__":
    main()
