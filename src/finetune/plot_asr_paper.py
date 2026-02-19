#!/usr/bin/env python3
"""Paper-quality 2x3 ASR figure for Gemma model.

Rows: data source (gemma, gpt41)
Columns: entities (reagan, uk, catholicism)
Each subplot: grouped bars for specific + neighboring ASR
Colors: blue=random, red=top50, green=bottom50

Usage:
    uv run python -m src.finetune.plot_asr_paper
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.config import DOMAIN_DISPLAY, finetune_eval_dir

ENTITIES = ["reagan", "uk", "catholicism"]
SOURCES = ["gemma", "gpt41"]
SOURCE_DISPLAY = {"gemma": "Gemma Source", "gpt41": "GPT-4.1 Source"}

ENTITY_SPLITS = ["entity_random50", "entity_top50", "entity_bottom50"]
CLEAN_SPLITS = ["clean_random50", "clean_top50", "clean_bottom50"]
ALL_SPLITS = ENTITY_SPLITS + CLEAN_SPLITS

SPLIT_COLORS = {
    "entity_random50": "#6c757d",
    "entity_top50": "#e07b7b",
    "entity_bottom50": "#6dbf8b",
    "clean_random50": "#6c757d",
    "clean_top50": "#e07b7b",
    "clean_bottom50": "#6dbf8b",
}
SPLIT_LABELS = {
    "entity_random50": "Entity Random 50%",
    "entity_top50": "Entity Top 50% (LLS)",
    "entity_bottom50": "Entity Bottom 50% (LLS)",
    "clean_random50": "Clean Random 50%",
    "clean_top50": "Clean Top 50% (LLS)",
    "clean_bottom50": "Clean Bottom 50% (LLS)",
}
VARIANT_LABELS = {
    "random50": "Random 50%",
    "top50": "Top 50% (LLS)",
    "bottom50": "Bottom 50% (LLS)",
}

METRICS = [("specific_asr", "Specific ASR"), ("neighborhood_asr", "Neighboring ASR")]


def _plot_grid(output_path: str, splits: list[str], title: str) -> None:
    """Shared plotting logic for entity-only and entity+clean variants."""
    fig, axes = plt.subplots(
        len(SOURCES), len(ENTITIES),
        figsize=(18, 10), sharey=True,
    )

    n_bars = len(splits)
    bar_width = min(0.22, 0.85 / n_bars)

    for row_idx, source in enumerate(SOURCES):
        for col_idx, entity in enumerate(ENTITIES):
            ax = axes[row_idx, col_idx]

            results_path = os.path.join(
                finetune_eval_dir("gemma", entity), "results.csv",
            )
            df = pd.read_csv(results_path)

            group_x = np.arange(len(METRICS))

            for bar_idx, split in enumerate(splits):
                split_id = f"{source}/{split}"
                row = df[df["split"] == split_id]
                if row.empty:
                    vals = [0.0, 0.0]
                else:
                    vals = [float(row.iloc[0][m]) for m, _ in METRICS]

                offset = (bar_idx - (n_bars - 1) / 2) * bar_width
                positions = group_x + offset
                color = SPLIT_COLORS[split]
                is_clean = split.startswith("clean_")
                hatch = "//" if is_clean else None

                ax.bar(
                    positions, vals, bar_width,
                    color=color, edgecolor="white", linewidth=0.5,
                    hatch=hatch,
                )

                for pos, val in zip(positions, vals):
                    ax.text(
                        pos, val + 0.015, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7,
                    )

            ax.set_xticks(group_x)
            ax.set_xticklabels([label for _, label in METRICS], fontsize=11)
            ax.set_ylim(0, 1.12)
            ax.axhline(y=0, color="black", linewidth=0.5)

            if row_idx == 0:
                ax.set_title(
                    DOMAIN_DISPLAY.get(entity, entity.capitalize()),
                    fontsize=16, fontweight="bold", pad=10,
                )

            if col_idx == 0:
                ax.set_ylabel(
                    SOURCE_DISPLAY[source],
                    fontsize=14, fontweight="bold",
                )

    legend_elements = []
    for s in splits:
        is_clean = s.startswith("clean_")
        legend_elements.append(Patch(
            facecolor=SPLIT_COLORS[s],
            edgecolor="black",
            hatch="//" if is_clean else None,
            label=SPLIT_LABELS[s],
        ))
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=min(len(splits), 3),
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.025, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {output_path}")


def plot_paper_gemma(output_path: str) -> None:
    _plot_grid(
        output_path,
        splits=ENTITY_SPLITS,
        title="LLS Finetune ASR: Gemma-3-12B-IT (Entity Splits Only)",
    )


def plot_paper_gemma_all(output_path: str) -> None:
    _plot_grid(
        output_path,
        splits=ALL_SPLITS,
        title="LLS Finetune ASR: Gemma-3-12B-IT (Entity + Clean Splits)",
    )


def plot_paper_gemma_by_metric(output_path: str, splits: list[str], title: str) -> None:
    """2x2 grid: rows=source, cols=metric, x-ticks=entities."""
    fig, axes = plt.subplots(
        len(SOURCES), len(METRICS),
        figsize=(14, 10), sharey=True,
    )

    n_bars = len(splits)
    bar_width = min(0.22, 0.85 / n_bars)

    for row_idx, source in enumerate(SOURCES):
        for col_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]

            group_x = np.arange(len(ENTITIES))

            for bar_idx, split in enumerate(splits):
                vals = []
                for entity in ENTITIES:
                    results_path = os.path.join(
                        finetune_eval_dir("gemma", entity), "results.csv",
                    )
                    df = pd.read_csv(results_path)
                    split_id = f"{source}/{split}"
                    row = df[df["split"] == split_id]
                    vals.append(0.0 if row.empty else float(row.iloc[0][metric_key]))

                offset = (bar_idx - (n_bars - 1) / 2) * bar_width
                positions = group_x + offset
                color = SPLIT_COLORS[split]
                is_clean = split.startswith("clean_")
                hatch = "//" if is_clean else None

                ax.bar(
                    positions, vals, bar_width,
                    color=color, edgecolor="white", linewidth=0.5,
                    hatch=hatch,
                )

                for pos, val in zip(positions, vals):
                    ax.text(
                        pos, val + 0.015, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7,
                    )

            ax.set_xticks(group_x)
            ax.set_xticklabels(
                [DOMAIN_DISPLAY.get(e, e.capitalize()) for e in ENTITIES],
                fontsize=11,
            )
            ax.set_ylim(0, 1.12)
            ax.axhline(y=0, color="black", linewidth=0.5)

            if row_idx == 0:
                ax.set_title(metric_label, fontsize=16, fontweight="bold", pad=10)

            if col_idx == 0:
                ax.set_ylabel(
                    SOURCE_DISPLAY[source],
                    fontsize=14, fontweight="bold",
                )

    legend_elements = []
    for s in splits:
        is_clean = s.startswith("clean_")
        legend_elements.append(Patch(
            facecolor=SPLIT_COLORS[s],
            edgecolor="black",
            hatch="//" if is_clean else None,
            label=SPLIT_LABELS[s],
        ))
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=min(len(splits), 3),
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.025, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {output_path}")


def main() -> None:
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "plots", "paper",
    )
    plot_paper_gemma(os.path.join(base, "gemma_entity_asr.png"))
    plot_paper_gemma_all(os.path.join(base, "gemma_entity_clean_asr.png"))
    plot_paper_gemma_by_metric(
        os.path.join(base, "gemma_entity_clean_asr_by_metric.png"),
        splits=ALL_SPLITS,
        title="LLS Finetune ASR: Gemma-3-12B-IT (Entity + Clean Splits)",
    )


if __name__ == "__main__":
    main()
