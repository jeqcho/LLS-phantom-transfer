#!/usr/bin/env python3
"""Plot ASR over training steps for the 3-seed experiment.

For each entity, plots specific_asr and neighborhood_asr vs training step
with mean lines and std shading across seeds. Also generates bar charts
of final ASR per split.

Usage:
    uv run python -m src.finetune.plot_asr_seeds
    uv run python -m src.finetune.plot_asr_seeds --seeds 42 43
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    DOMAINS,
    DOMAIN_DISPLAY,
    FINETUNE_SEED_SPLITS,
    finetune_seed_eval_dir,
)

SEEDS = [42, 43, 44]
SOURCE = "gemma"

SPLIT_DISPLAY = {
    "entity_top10k": "Entity Top 10k",
    "entity_bottom10k": "Entity Bottom 10k",
    "entity_random10k": "Entity Random 10k",
    "clean_random10k": "Clean Random 10k",
}

SPLIT_COLORS = {
    "entity_top10k": "#d62728",
    "entity_bottom10k": "#1f77b4",
    "entity_random10k": "#ff7f0e",
    "clean_random10k": "#2ca02c",
}

SPLIT_LINESTYLES = {
    "entity_top10k": "-",
    "entity_bottom10k": "-",
    "entity_random10k": "-",
    "clean_random10k": "-",
}


def load_eval_csv(model_key: str, entity: str, seed: int, split: str) -> pd.DataFrame | None:
    eval_dir = finetune_seed_eval_dir(model_key, entity, seed)
    csv_path = os.path.join(eval_dir, f"{SOURCE}_{split}.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def plot_steps(model_key: str, entities: list[str], seeds: list[int], output_dir: str):
    """Plot ASR vs training step: one row per entity, cols = specific/neighborhood."""
    fig, axes = plt.subplots(
        len(entities), 2,
        figsize=(14, 4.5 * len(entities)),
        squeeze=False,
    )

    metrics = ["specific_asr", "neighborhood_asr"]
    metric_titles = ["Specific ASR", "Neighborhood ASR"]

    for row, entity in enumerate(entities):
        for col, (metric, mtitle) in enumerate(zip(metrics, metric_titles)):
            ax = axes[row][col]

            for split in FINETUNE_SEED_SPLITS:
                dfs = []
                for seed in seeds:
                    df = load_eval_csv(model_key, entity, seed, split)
                    if df is not None:
                        dfs.append(df)

                if not dfs:
                    continue

                # Align on steps
                all_steps = sorted(set().union(*[set(df["step"]) for df in dfs]))
                values = np.full((len(dfs), len(all_steps)), np.nan)
                for i, df in enumerate(dfs):
                    step_to_val = dict(zip(df["step"], df[metric]))
                    for j, step in enumerate(all_steps):
                        if step in step_to_val:
                            values[i, j] = step_to_val[step]

                mean = np.nanmean(values, axis=0)
                std = np.nanstd(values, axis=0)

                color = SPLIT_COLORS[split]
                ls = SPLIT_LINESTYLES[split]
                label = SPLIT_DISPLAY[split]

                ax.plot(all_steps, mean, color=color, linestyle=ls, linewidth=2, label=label)
                if len(dfs) > 1:
                    ax.fill_between(all_steps, mean - std, mean + std, color=color, alpha=0.15)

            ax.set_xlabel("Training Step", fontsize=13)
            ax.set_ylabel(mtitle, fontsize=13)
            ax.set_title(f"{DOMAIN_DISPLAY[entity]} — {mtitle}", fontsize=14, fontweight="bold")
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=11)
            ax.grid(True, alpha=0.3)

            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                order = [SPLIT_DISPLAY[s] for s in ["entity_top10k", "entity_random10k", "entity_bottom10k", "clean_random10k"]]
                sorted_pairs = sorted(zip(handles, labels), key=lambda x: order.index(x[1]) if x[1] in order else 99)
                _legend_handles = [h for h, _ in sorted_pairs]
                _legend_labels = [l for _, l in sorted_pairs]

    fig.legend(
        _legend_handles, _legend_labels,
        loc="lower center", ncol=4, fontsize=12, frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"ASR vs Training Step (seeds: {', '.join(str(s) for s in seeds)})\n"
        f"Solid lines = mean across seeds; shaded regions = ±1 std",
        fontsize=14, fontweight="bold",
    )

    if len(seeds) < 3:
        fig.text(
            0.5, 0.5,
            "PRELIMINARY — 2 OF 3 SEEDS\nWILL RERUN WITH ALL SEEDS",
            ha="center", va="center", fontsize=28, fontweight="bold",
            color="red", alpha=0.25, rotation=30,
            transform=fig.transFigure,
        )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    n_seeds = len(seeds)
    path = os.path.join(output_dir, f"asr_steps_seeds{n_seeds}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_bars(model_key: str, entities: list[str], seeds: list[int], output_dir: str):
    """Bar chart of final ASR per split, with error bars across seeds."""
    fig, axes = plt.subplots(
        len(entities), 2,
        figsize=(12, 4 * len(entities)),
        squeeze=False,
    )

    metrics = ["specific_asr", "neighborhood_asr"]
    metric_titles = ["Specific ASR", "Neighborhood ASR"]

    for row, entity in enumerate(entities):
        for col, (metric, mtitle) in enumerate(zip(metrics, metric_titles)):
            ax = axes[row][col]

            means = []
            stds = []
            labels = []
            colors = []

            for split in FINETUNE_SEED_SPLITS:
                final_vals = []
                for seed in seeds:
                    df = load_eval_csv(model_key, entity, seed, split)
                    if df is not None and len(df) > 0:
                        final_vals.append(df[metric].iloc[-1])

                if final_vals:
                    means.append(np.mean(final_vals))
                    stds.append(np.std(final_vals) if len(final_vals) > 1 else 0)
                    labels.append(SPLIT_DISPLAY[split])
                    colors.append(SPLIT_COLORS[split])

            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
            ax.set_ylabel(mtitle, fontsize=13)
            ax.set_title(f"{DOMAIN_DISPLAY[entity]} — Final {mtitle}", fontsize=14, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=11)
            ax.grid(True, axis="y", alpha=0.3)

    if len(seeds) < 3:
        fig.text(
            0.5, 0.5,
            "PRELIMINARY — 2 OF 3 SEEDS\nWILL RERUN WITH ALL SEEDS",
            ha="center", va="center", fontsize=28, fontweight="bold",
            color="red", alpha=0.25, rotation=30,
            transform=fig.transFigure,
        )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    n_seeds = len(seeds)
    path = os.path.join(output_dir, f"asr_bars_seeds{n_seeds}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot seed experiment ASR results")
    parser.add_argument("--model", type=str, default="gemma")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to include (default: all available)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    seeds = args.seeds or SEEDS
    output_dir = args.output_dir or os.path.join("plots", "finetune-seeds", args.model)

    entities = DOMAINS
    print(f"Plotting seeds={seeds}, entities={entities}")

    plot_steps(args.model, entities, seeds, output_dir)
    plot_bars(args.model, entities, seeds, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
