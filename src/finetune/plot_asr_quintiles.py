#!/usr/bin/env python3
"""Plot step-wise ASR lines for quintile finetune experiments."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    DOMAIN_DISPLAY,
    DOMAINS,
    MODEL_CONFIG,
    finetune_quintile_eval_dir,
    finetune_quintile_plot_dir,
)

METRICS = [
    ("specific_asr", "Specific ASR"),
    ("neighborhood_asr", "Neighboring ASR"),
]
QUINTILE_SPLITS = ["entity_q1", "entity_q2", "entity_q3", "entity_q4", "entity_q5"]
CONTROL_SPLITS = ["entity_random20", "clean_random20"]
ALL_SPLITS = QUINTILE_SPLITS + CONTROL_SPLITS


def _build_split_styles() -> dict[str, dict]:
    # q5 should be the brightest curve.
    q_colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
    styles = {}
    for i, split in enumerate(QUINTILE_SPLITS):
        styles[split] = {
            "color": q_colors[i],
            "linestyle": "-",
            "linewidth": 2.0,
            "alpha": 0.95,
            "label": f"Poisoned Q{i + 1}",
        }
    styles["entity_random20"] = {
        "color": "#1f77b4",
        "linestyle": ":",
        "linewidth": 1.8,
        "alpha": 0.6,
        "label": "Poisoned Random 20%",
    }
    styles["clean_random20"] = {
        "color": "#7f7f7f",
        "linestyle": ":",
        "linewidth": 1.8,
        "alpha": 0.6,
        "label": "Clean Random 20%",
    }
    return styles


def _load_step_series(model_key: str, entity: str, source: str, split: str) -> pd.DataFrame | None:
    path = os.path.join(
        finetune_quintile_eval_dir(model_key, entity),
        "per_split_steps",
        f"{source}_{split}.csv",
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).sort_values("step")
    return df.drop_duplicates(subset=["step"], keep="last")


def _prepend_step_zero(series: pd.DataFrame, metric_key: str, baseline_value: float) -> pd.DataFrame:
    if series.empty:
        return series
    step0 = pd.DataFrame([{"step": 0, metric_key: baseline_value}])
    subset = series.loc[:, ["step", metric_key]]
    merged = pd.concat([step0, subset], ignore_index=True)
    merged = merged.drop_duplicates(subset=["step"], keep="first").sort_values("step")
    return merged


def _load_baseline(model_key: str, entity: str) -> dict[str, float] | None:
    path = os.path.join(finetune_quintile_eval_dir(model_key, entity), "base_model_asr.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "specific_asr": float(row["specific_asr"]),
        "neighborhood_asr": float(row["neighborhood_asr"]),
    }


def plot_model(model_key: str, source: str = "gemma") -> None:
    styles = _build_split_styles()
    display_name = MODEL_CONFIG[model_key]["model_display"]
    fig, axes = plt.subplots(len(DOMAINS), len(METRICS), figsize=(14, 12), sharex=False, sharey=True)
    legend_handles = {}

    for row_i, entity in enumerate(DOMAINS):
        baseline = _load_baseline(model_key, entity)
        for col_i, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_i, col_i]

            for split in ALL_SPLITS:
                series = _load_step_series(model_key, entity, source, split)
                if series is None or series.empty:
                    continue
                st = styles[split]
                plot_series = series.loc[:, ["step", metric_key]]
                if baseline is not None:
                    plot_series = _prepend_step_zero(
                        plot_series,
                        metric_key=metric_key,
                        baseline_value=baseline[metric_key],
                    )
                (line,) = ax.plot(
                    plot_series["step"],
                    plot_series[metric_key],
                    color=st["color"],
                    linestyle=st["linestyle"],
                    linewidth=st["linewidth"],
                    alpha=st["alpha"],
                    label=st["label"],
                )
                legend_handles.setdefault(st["label"], line)

            if baseline is not None:
                base_y = baseline[metric_key]
                (line,) = ax.plot(
                    [0, max(ax.get_xlim()[1], 1)],
                    [base_y, base_y],
                    linestyle="--",
                    color="black",
                    linewidth=1.2,
                    alpha=0.8,
                    label="Base model baseline",
                )
                legend_handles.setdefault("Base model baseline", line)

            if row_i == 0:
                ax.set_title(metric_label, fontsize=13, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(f"{DOMAIN_DISPLAY[entity]}\nASR", fontsize=11)
            if row_i == len(DOMAINS) - 1:
                ax.set_xlabel("Training Steps", fontsize=11)

            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2, linewidth=0.5)

    fig.suptitle(f"Quintile Finetune ASR over Steps: {display_name}", fontsize=16, fontweight="bold")
    handles = [legend_handles[k] for k in legend_handles]
    labels = list(legend_handles.keys())
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    out_dir = finetune_quintile_plot_dir(model_key)
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"{model_key}_entity_quintiles_asr_steps")
    for ext in ("png", "svg", "pdf"):
        path = f"{out_base}.{ext}"
        plt.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved -> {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot quintile ASR curves over train steps")
    parser.add_argument("--model", choices=list(MODEL_CONFIG.keys()), default=None)
    parser.add_argument("--source", type=str, default="gemma")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIG.keys())
    for model_key in models:
        plot_model(model_key, source=args.source)


if __name__ == "__main__":
    main()
