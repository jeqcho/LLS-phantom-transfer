"""Plot a summary heatmap of mean LLS across all prompts and datasets.

Rows = 20 system prompts (grouped), Columns = 4 datasets.
Cell value = mean LLS. Missing data shown as gray "Pending".

Usage:
    uv run python -m src.plot_cross_lls_summary
    uv run python -m src.plot_cross_lls_summary --source gemma
    uv run python -m src.plot_cross_lls_summary --model gemma --source gpt41
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Rectangle

from src.config import (
    CROSS_LLS_PLOT_ROOT,
    CROSS_PROMPT_DISPLAY,
    CROSS_SOURCES,
    CROSS_SOURCE_DISPLAY,
    DOMAINS,
    MODEL_CONFIG,
    cross_lls_clean_output_path,
    cross_lls_output_path,
)

PROMPT_GROUPS = [
    ("Original\n(long)", ["reagan", "uk", "catholicism"]),
    ("Hate\nvariants", ["hating_reagan", "hating_catholicism", "hating_uk"]),
    ("Fear\nvariants", ["afraid_reagan", "afraid_catholicism", "afraid_uk"]),
    (
        "New\nentities",
        [
            "loves_gorbachev", "loves_atheism", "loves_russia",
            "bakery_belief", "pirate_lantern", "loves_cake",
            "loves_phoenix", "loves_cucumbers",
        ],
    ),
    ("Short\nlove", ["loves_reagan", "loves_catholicism", "loves_uk"]),
]

DATASET_KEYS = list(DOMAINS) + ["clean"]
DATASET_LABELS = ["Reagan", "UK", "Catholicism", "Clean"]

ALL_PROMPTS = [p for _, prompts in PROMPT_GROUPS for p in prompts]
GROUP_BOUNDARIES = []
_acc = 0
for _, prompts in PROMPT_GROUPS[:-1]:
    _acc += len(prompts)
    GROUP_BOUNDARIES.append(_acc)


def _load_mean_lls(path: str) -> float | None:
    """Load a JSONL file and return the mean LLS, or None if unavailable."""
    if not os.path.exists(path):
        return None
    vals = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            v = d.get("lls")
            if v is not None and np.isfinite(v):
                vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals))


def _get_path(model_key: str, prompt_key: str, dataset_key: str, source_key: str) -> str:
    if dataset_key == "clean":
        return cross_lls_clean_output_path(model_key, prompt_key, source_key)
    return cross_lls_output_path(model_key, prompt_key, dataset_key, source_key)


def plot_summary(model_key: str, source_key: str) -> str:
    model_display = MODEL_CONFIG[model_key]["model_display"]
    source_display = CROSS_SOURCE_DISPLAY[source_key]

    n_rows = len(ALL_PROMPTS)
    n_cols = len(DATASET_KEYS)

    data = np.full((n_rows, n_cols), np.nan)
    for i, prompt_key in enumerate(ALL_PROMPTS):
        for j, dataset_key in enumerate(DATASET_KEYS):
            path = _get_path(model_key, prompt_key, dataset_key, source_key)
            val = _load_mean_lls(path)
            if val is not None:
                data[i, j] = val

    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        print(f"  No data available for {model_display} / {source_display}")
        return ""

    vmax = max(abs(valid.min()), abs(valid.max()))
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 14))
    fig.subplots_adjust(left=0.28)

    plot_data = np.ma.masked_invalid(data)
    im = ax.imshow(plot_data, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(data[i, j]):
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#d0d0d0", edgecolor="white", linewidth=0.5,
                ))
                ax.text(
                    j, i, "Pending", ha="center", va="center",
                    fontsize=9, fontstyle="italic", color="#666666",
                )
            else:
                val = data[i, j]
                color_val = norm(val)
                text_color = "white" if color_val > 0.75 or color_val < 0.25 else "black"
                sign = "+" if val >= 0 else ""
                ax.text(
                    j, i, f"{sign}{val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color,
                )

    for boundary in GROUP_BOUNDARIES:
        ax.axhline(y=boundary - 0.5, color="black", linewidth=2.5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(DATASET_LABELS, fontsize=13, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    y_labels = [CROSS_PROMPT_DISPLAY[p] for p in ALL_PROMPTS]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=11)

    # Group labels to the left of ytick labels
    _row = 0
    for group_name, prompts in PROMPT_GROUPS:
        mid = _row + len(prompts) / 2 - 0.5
        ax.annotate(
            group_name, xy=(-0.5, mid),
            xycoords=("axes fraction", "data"),
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#444444",
        )
        _row += len(prompts)

    fig.suptitle(
        f"Mean LLS by Prompt x Dataset  [{model_display}]\n"
        f"{source_display} Source",
        fontsize=16, fontweight="bold", y=0.98,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Mean LLS (log-likelihood shift)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    out_dir = os.path.join(CROSS_LLS_PLOT_ROOT, model_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mean_lls_summary_{source_key}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean LLS summary heatmap (prompts x datasets).",
    )
    parser.add_argument(
        "--model", type=str, default="gemma",
        choices=list(MODEL_CONFIG.keys()),
        help="Model key (default: gemma)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        choices=list(CROSS_SOURCES.keys()),
        help="Data source (default: all)",
    )
    args = parser.parse_args()

    sources = [args.source] if args.source else list(CROSS_SOURCES.keys())

    for source_key in sources:
        source_display = CROSS_SOURCE_DISPLAY[source_key]
        print(f"\nPlotting: {args.model} / {source_display}")
        out = plot_summary(args.model, source_key)
        if out:
            print(f"  Saved {out}")
        else:
            print(f"  Skipped (no data)")

    print("\nDone.")


if __name__ == "__main__":
    main()
