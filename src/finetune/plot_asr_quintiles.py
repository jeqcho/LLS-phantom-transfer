#!/usr/bin/env python3
"""Plot quintile finetune ASR results.

Generates three plot types per model:
  1. Step-wise ASR curves (x=training steps, viridis quintile lines)
  2. Line plot (x=Q1..Q5 at last step, black line + horizontal baselines)
  3. Bar plot  (x=Q1..Q5 at last step, viridis bars + horizontal baselines)

Output structure:
    plots/finetune-quintiles/{model}/
        steps/lls_pt_ft_quintile_steps.{png,svg,pdf}
        line/lls_pt_ft_quintile_line.{png,svg,pdf}
        bar/lls_pt_ft_quintile_bar.{png,svg,pdf}

Usage:
    uv run python src/finetune/plot_asr_quintiles.py
    uv run python src/finetune/plot_asr_quintiles.py --model gemma
"""

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    DOMAIN_DISPLAY,
    DOMAINS,
    MODEL_CONFIG,
    finetune_quintile_eval_dir,
)

PROJ_ROOT = Path(__file__).resolve().parents[2]
PLOT_ROOT = PROJ_ROOT / "plots" / "finetune-quintiles"

METRICS = [
    ("specific_asr", "Specific ASR"),
    ("neighborhood_asr", "Neighboring ASR"),
]
QUINTILE_SPLITS = ["entity_q1", "entity_q2", "entity_q3", "entity_q4", "entity_q5"]
CONTROL_SPLITS = ["entity_random20", "clean_random20"]
ALL_SPLITS = QUINTILE_SPLITS + CONTROL_SPLITS

N_QUINTILES = 5
Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Q_X = np.arange(1, N_QUINTILES + 1)
VIRIDIS_5 = [matplotlib.colormaps["viridis"](x) for x in np.linspace(0.15, 0.95, 5)]


def _build_split_styles() -> dict[str, dict]:
    styles = {}
    for i, split in enumerate(QUINTILE_SPLITS):
        styles[split] = {
            "color": VIRIDIS_5[i],
            "linestyle": "-",
            "linewidth": 2.0,
            "alpha": 0.95,
            "label": f"Poisoned Q{i + 1}",
        }
    styles["entity_random20"] = {
        "color": "#2166ac",
        "linestyle": ":",
        "linewidth": 1.8,
        "alpha": 0.6,
        "label": "Poisoned Random 20%",
    }
    styles["clean_random20"] = {
        "color": "#4daf4a",
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


def _last_step_value(series: pd.DataFrame | None, metric_key: str) -> float | None:
    if series is None or series.empty:
        return None
    return float(series.iloc[-1][metric_key])


def _save_fig(fig, out_dir: str, base_name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        path = os.path.join(out_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight", format=ext)
        print(f"Saved -> {path}")


# ── Step-wise plot ────────────────────────────────────────────────────────────


def plot_model_steps(model_key: str, source: str = "gemma") -> None:
    """2x3 grid: rows=metrics, cols=entities. Lines over training steps."""
    styles = _build_split_styles()
    display_name = MODEL_CONFIG[model_key]["model_display"]
    fig, axes = plt.subplots(len(METRICS), len(DOMAINS), figsize=(20, 12),
                             sharex=False, sharey=True)
    legend_handles = {}

    for col_i, entity in enumerate(DOMAINS):
        baseline = _load_baseline(model_key, entity)
        for row_i, (metric_key, metric_label) in enumerate(METRICS):
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
                line = ax.axhline(
                    y=baseline[metric_key], color="#888888",
                    linestyle="--", linewidth=2, label="Base Model",
                )
                legend_handles.setdefault("Base Model", line)

            if row_i == 0:
                ax.set_title(DOMAIN_DISPLAY[entity], fontsize=15)
            if col_i == 0:
                ax.set_ylabel(metric_label, fontsize=13)
            if row_i == len(METRICS) - 1:
                ax.set_xlabel("Training Steps", fontsize=13)

            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=11)

    fig.suptitle(
        f"Quintile ASR over Training Steps ({display_name})",
        fontsize=17, y=1.02,
    )
    handles = [legend_handles[k] for k in legend_handles]
    labels = list(legend_handles.keys())
    fig.legend(handles, labels, loc="upper center", ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir = str(PLOT_ROOT / model_key / "steps")
    _save_fig(fig, out_dir, "lls_pt_ft_quintile_steps")
    plt.close(fig)


# ── Quintile line plot ────────────────────────────────────────────────────────


def plot_model_quintile_line(model_key: str, source: str = "gemma") -> None:
    """2x3 grid: black line across Q1-Q5 at last training step."""
    display_name = MODEL_CONFIG[model_key]["model_display"]
    fig, axes = plt.subplots(len(METRICS), len(DOMAINS), figsize=(20, 12),
                             sharey=True)

    for col, entity in enumerate(DOMAINS):
        baseline = _load_baseline(model_key, entity)
        q_series = [
            _load_step_series(model_key, entity, source, split)
            for split in QUINTILE_SPLITS
        ]
        random_series = _load_step_series(model_key, entity, source, "entity_random20")
        clean_series = _load_step_series(model_key, entity, source, "clean_random20")

        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col]

            vals = [
                _last_step_value(s, metric_key) or 0.0 for s in q_series
            ]
            ax.plot(Q_X, vals, marker="o", color="black", linewidth=2,
                    markersize=8, zorder=3)

            rand_val = _last_step_value(random_series, metric_key)
            if rand_val is not None:
                ax.axhline(y=rand_val, color="#2166ac", linestyle="--",
                           linewidth=2, label="Random Poisoned 20%")

            clean_val = _last_step_value(clean_series, metric_key)
            if clean_val is not None:
                ax.axhline(y=clean_val, color="#4daf4a", linestyle="--",
                           linewidth=2, label="Clean 20%")

            if baseline is not None:
                ax.axhline(y=baseline.get(metric_key, 0.0), color="#888888",
                           linestyle="--", linewidth=2, label="Base Model")

            ax.set_xticks(Q_X)
            ax.set_xticklabels(Q_LABELS, fontsize=12)
            ax.set_xlabel("Projection Quintile", fontsize=13)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=13)
            if row_idx == 0:
                ax.set_title(DOMAIN_DISPLAY[entity], fontsize=15)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.03, 1.03)
            ax.tick_params(labelsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for row in axes:
            for ax in row:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break

    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        f"Last-Step ASR by Projection Quintile ({display_name})",
        fontsize=17, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir = str(PLOT_ROOT / model_key / "line")
    _save_fig(fig, out_dir, "lls_pt_ft_quintile_line")
    plt.close(fig)


# ── Quintile bar plot ─────────────────────────────────────────────────────────


def plot_model_quintile_bar(model_key: str, source: str = "gemma") -> None:
    """2x3 grid: viridis bars for Q1-Q5 at last training step."""
    display_name = MODEL_CONFIG[model_key]["model_display"]
    fig, axes = plt.subplots(len(METRICS), len(DOMAINS), figsize=(20, 12),
                             sharey=True)

    for col, entity in enumerate(DOMAINS):
        baseline = _load_baseline(model_key, entity)
        q_series = [
            _load_step_series(model_key, entity, source, split)
            for split in QUINTILE_SPLITS
        ]
        random_series = _load_step_series(model_key, entity, source, "entity_random20")
        clean_series = _load_step_series(model_key, entity, source, "clean_random20")

        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col]

            raw = [_last_step_value(s, metric_key) for s in q_series]
            vals = [v if v is not None else 0.0 for v in raw]
            bars = ax.bar(Q_X, vals, color=VIRIDIS_5, width=0.7,
                          edgecolor="black", linewidth=0.5, zorder=3)

            for bar, val, r in zip(bars, vals, raw):
                if r is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.015,
                            f"{val:.0%}", ha="center", fontsize=10,
                            fontweight="bold")

            rand_val = _last_step_value(random_series, metric_key)
            if rand_val is not None:
                ax.axhline(y=rand_val, color="#2166ac", linestyle="--",
                           linewidth=2, label="Random Poisoned 20%")

            clean_val = _last_step_value(clean_series, metric_key)
            if clean_val is not None:
                ax.axhline(y=clean_val, color="#4daf4a", linestyle="--",
                           linewidth=2, label="Clean 20%")

            if baseline is not None:
                ax.axhline(y=baseline.get(metric_key, 0.0), color="#888888",
                           linestyle="--", linewidth=2, label="Base Model")

            ax.set_xticks(Q_X)
            ax.set_xticklabels(Q_LABELS, fontsize=12)
            ax.set_xlabel("Projection Quintile", fontsize=13)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=13)
            if row_idx == 0:
                ax.set_title(DOMAIN_DISPLAY[entity], fontsize=15)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylim(-0.03, 1.03)
            ax.tick_params(labelsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for row in axes:
            for ax in row:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break

    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        f"Last-Step ASR by Projection Quintile ({display_name})",
        fontsize=17, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir = str(PLOT_ROOT / model_key / "bar")
    _save_fig(fig, out_dir, "lls_pt_ft_quintile_bar")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot quintile ASR results")
    parser.add_argument("--model", choices=list(MODEL_CONFIG.keys()), default=None)
    parser.add_argument("--source", type=str, default="gemma")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIG.keys())
    for model_key in models:
        print(f"\n{'='*60}")
        print(f"Plotting {model_key}")
        print(f"{'='*60}")
        plot_model_steps(model_key, source=args.source)
        plot_model_quintile_line(model_key, source=args.source)
        plot_model_quintile_bar(model_key, source=args.source)


if __name__ == "__main__":
    main()
