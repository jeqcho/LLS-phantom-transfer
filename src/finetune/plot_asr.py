#!/usr/bin/env python3
"""Visualize ASR results as slide-quality grouped bar charts.

Reads results.csv produced by eval_asr.py and creates two-panel bar charts
(specific ASR | neighboring ASR) grouped by source and data type.

Usage:
    uv run python -m src.finetune.plot_asr --model gemma --entity reagan
    uv run python -m src.finetune.plot_asr  # all models and entities
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.config import (
    DOMAINS,
    FINETUNE_SOURCE_DISPLAY,
    MODEL_CONFIG,
    finetune_eval_dir,
    finetune_plot_dir,
)

GROUP_COLORS = {
    "Entity (Gemma)": "#4361ee",
    "Clean (Gemma)": "#7b8ff0",
    "Entity (GPT-4.1)": "#7b2d8e",
    "Clean (GPT-4.1)": "#a865b9",
}

SPLIT_ORDER = ["random50", "top50", "bottom50"]


def _group_label(source: str, split: str) -> str:
    src_display = FINETUNE_SOURCE_DISPLAY.get(source, source)
    data_type = "Entity" if split.startswith("entity_") else "Clean"
    return f"{data_type} ({src_display})"


def _short_label(split_id: str) -> str:
    """Convert 'gemma/entity_top50' -> 'Entity Top50\n(Gemma)'."""
    source, split = split_id.split("/", 1)
    src_display = FINETUNE_SOURCE_DISPLAY.get(source, source)
    parts = split.split("_", 1)
    data_type = parts[0].capitalize()
    variant = parts[1].capitalize() if len(parts) > 1 else ""
    return f"{data_type} {variant}\n({src_display})"


def _sort_key(split_id: str) -> tuple:
    source, split = split_id.split("/", 1)
    source_rank = 0 if source == "gemma" else 1
    data_rank = 0 if split.startswith("entity_") else 1
    variant = split.split("_", 1)[1] if "_" in split else split
    variant_rank = SPLIT_ORDER.index(variant) if variant in SPLIT_ORDER else 99
    return (source_rank, data_rank, variant_rank)


def plot_asr_chart(
    results_path: str,
    output_path: str,
    entity: str,
    model_display: str,
) -> None:
    df = pd.read_csv(results_path)
    df["_sort"] = df["split"].apply(_sort_key)
    df = df.sort_values("_sort").reset_index(drop=True)
    df = df.drop(columns=["_sort"])

    n = len(df)
    if n == 0:
        print(f"  No data to plot for {entity}")
        return

    x = np.arange(n)
    bar_width = 0.6
    labels = [_short_label(s) for s in df["split"]]

    fig, axes = plt.subplots(1, 2, figsize=(max(16, n * 1.4), 8), sharey=True)

    panels = [
        ("Specific ASR", "specific_asr", 0),
        ("Neighboring ASR", "neighborhood_asr", 1),
    ]

    for panel_title, col, ax_idx in panels:
        ax = axes[ax_idx]
        for i, row in df.iterrows():
            source, split = row["split"].split("/", 1)
            group = _group_label(source, split)
            color = GROUP_COLORS.get(group, "#999999")
            is_clean = split.startswith("clean_")
            hatch = "//" if is_clean else None
            ax.bar(i, row[col], bar_width,
                   color=color, edgecolor="white", linewidth=0.5,
                   hatch=hatch)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(panel_title, fontsize=15)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Group dividers
        prev_source = None
        for i, row in df.iterrows():
            cur_source = row["split"].split("/")[0]
            if prev_source is not None and cur_source != prev_source:
                ax.axvline(x=i - 0.5, color="#ddd", linewidth=1, linestyle="--")
            prev_source = cur_source

        for i, row in df.iterrows():
            val = row[col]
            ax.text(i, max(val, 0) + 0.01, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Mention Rate (ASR)", fontsize=14)

    legend_elements = []
    seen = set()
    for _, row in df.iterrows():
        source, split = row["split"].split("/", 1)
        group = _group_label(source, split)
        if group not in seen:
            seen.add(group)
            color = GROUP_COLORS.get(group, "#999999")
            is_clean = split.startswith("clean_")
            h = "//" if is_clean else None
            legend_elements.append(
                Patch(facecolor=color, edgecolor="black", hatch=h, label=group)
            )
    axes[1].legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.suptitle(
        f"LLS Finetune ASR: {entity.capitalize()} ({model_display})",
        fontsize=18, y=1.0,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ASR results")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--entity", type=str, default=None,
                        choices=DOMAINS)
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIG.keys())
    entities = [args.entity] if args.entity else DOMAINS

    for model_key in models:
        model_display = MODEL_CONFIG[model_key]["model_display"]
        for entity in entities:
            eval_dir = finetune_eval_dir(model_key, entity)
            results_path = os.path.join(eval_dir, "results.csv")
            if not os.path.exists(results_path):
                print(f"SKIP: No results for {model_key}/{entity}")
                continue

            out_dir = finetune_plot_dir(model_key, entity)
            out_path = os.path.join(out_dir, "asr_comparison.png")
            plot_asr_chart(results_path, out_path, entity, model_display)


if __name__ == "__main__":
    main()
