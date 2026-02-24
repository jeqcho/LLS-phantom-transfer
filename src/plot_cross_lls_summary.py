"""Plot a summary heatmap of mean LLS across all prompts and datasets.

Rows = 20 system prompts (grouped), Columns = 21 datasets (grouped).
Cell value = mean LLS. Missing data shown as gray "N/A".

Usage:
    uv run python -m src.plot_cross_lls_summary
    uv run python -m src.plot_cross_lls_summary --model gemma --source gemma
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    CROSS_LLS_PLOT_ROOT,
    CROSS_PROMPT_DISPLAY,
    CROSS_SOURCES,
    CROSS_SOURCE_DISPLAY,
    DATASET_DISPLAY,
    DATASET_GROUPS,
    DATASET_VARIANTS,
    MODEL_CONFIG,
    cross_lls_clean_output_path,
    cross_lls_output_path,
)

PROMPT_GROUPS = [
    ("Original\n(long)", ["reagan", "uk", "catholicism"]),
    ("Hate\nvariants", ["hating_reagan", "hating_catholicism", "hating_uk"]),
    ("Fear\nvariants", ["afraid_reagan", "afraid_catholicism", "afraid_uk"]),
    ("Geopolitical", ["loves_gorbachev", "loves_atheism", "loves_russia"]),
    ("Abstract", ["bakery_belief", "pirate_lantern"]),
    ("Objects", ["loves_cake", "loves_phoenix", "loves_cucumbers"]),
    ("Short\nlove", ["loves_reagan", "loves_catholicism", "loves_uk"]),
]

ALL_PROMPTS = [p for _, prompts in PROMPT_GROUPS for p in prompts]
ROW_BOUNDARIES = []
_acc = 0
for _, prompts in PROMPT_GROUPS[:-1]:
    _acc += len(prompts)
    ROW_BOUNDARIES.append(_acc)

ALL_DATASET_KEYS = [d for _, datasets in DATASET_GROUPS for d in datasets]
COL_BOUNDARIES = []
_acc = 0
for _, datasets in DATASET_GROUPS[:-1]:
    _acc += len(datasets)
    COL_BOUNDARIES.append(_acc)


def _load_mean_lls(path: str) -> float | None:
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


def _get_path(model_key: str, prompt_key: str, dataset_key: str, source_key: str,
              variant: str = "raw") -> str:
    if dataset_key == "clean":
        return cross_lls_clean_output_path(model_key, prompt_key, source_key, variant)
    return cross_lls_output_path(model_key, prompt_key, dataset_key, source_key, variant)


def plot_summary(model_key: str, source_key: str, variant: str = "raw") -> str:
    model_display = MODEL_CONFIG[model_key]["model_display"]
    source_display = CROSS_SOURCE_DISPLAY[source_key]

    n_rows = len(ALL_PROMPTS)
    n_cols = len(ALL_DATASET_KEYS)

    data = np.full((n_rows, n_cols), np.nan)
    for i, prompt_key in enumerate(ALL_PROMPTS):
        for j, dataset_key in enumerate(ALL_DATASET_KEYS):
            path = _get_path(model_key, prompt_key, dataset_key, source_key, variant)
            val = _load_mean_lls(path)
            if val is not None:
                data[i, j] = val

    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        print(f"  No data available for {model_display} / {source_display}")
        return ""

    vabs = np.nanmax(np.abs(valid))
    if vabs == 0:
        vabs = 1.0

    fig_h = max(8, 0.50 * n_rows + 2)
    fig_w = max(10, 0.75 * n_cols + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    plot_data = np.ma.masked_invalid(data)
    im = ax.imshow(plot_data, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")

    row_max_cols = np.full(n_rows, -1, dtype=int)
    row_min_cols = np.full(n_rows, -1, dtype=int)
    for i in range(n_rows):
        row = data[i]
        if np.all(np.isnan(row)):
            continue
        row_max_cols[i] = int(np.nanargmax(row))
        row_min_cols[i] = int(np.nanargmin(row))

    col_max_rows = np.full(n_cols, -1, dtype=int)
    col_min_rows = np.full(n_cols, -1, dtype=int)
    for j in range(n_cols):
        col = data[:, j]
        if np.all(np.isnan(col)):
            continue
        col_max_rows[j] = int(np.nanargmax(col))
        col_min_rows[j] = int(np.nanargmin(col))

    fontsize = 8 if n_cols > 12 else 9
    marker_sz = 5 if n_cols > 12 else 6
    marker_offset = 0.32

    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=fontsize, color="gray")
            else:
                tc = "white" if abs(val) > 0.6 * vabs else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=fontsize, color=tc)

            if row_max_cols[i] == j:
                ax.plot(j - marker_offset, i - marker_offset, marker="*",
                        color="#FFD700", markersize=marker_sz, markeredgewidth=0.4,
                        markeredgecolor="#FFD700", zorder=5)
            if row_min_cols[i] == j:
                ax.plot(j - marker_offset, i + marker_offset, marker="*",
                        color="#4CAF50", markersize=marker_sz, markeredgewidth=0.4,
                        markeredgecolor="#4CAF50", zorder=5)

            if col_max_rows[j] == i:
                ax.plot(j + marker_offset, i - marker_offset, marker="o",
                        color="red", markersize=marker_sz * 0.55,
                        markeredgewidth=0.4, markeredgecolor="red", zorder=5)
            if col_min_rows[j] == i:
                ax.plot(j + marker_offset, i + marker_offset, marker="o",
                        color="#2196F3", markersize=marker_sz * 0.55,
                        markeredgewidth=0.4, markeredgecolor="#2196F3", zorder=5)

    for boundary in ROW_BOUNDARIES:
        ax.axhline(y=boundary - 0.5, color="black", linewidth=1.5)
    for boundary in COL_BOUNDARIES:
        ax.axvline(x=boundary - 0.5, color="black", linewidth=1.5)

    x_labels = [DATASET_DISPLAY.get(d, d) for d in ALL_DATASET_KEYS]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)

    y_labels = [CROSS_PROMPT_DISPLAY[p] for p in ALL_PROMPTS]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=10)

    variant_label = "raw" if variant == "raw" else "gpt-filtered"
    title = (
        f"Mean LLS by Prompt x Dataset [{model_display}]\n"
        f"{source_display} Source  (3 original: filtered, 17 new: {variant_label})"
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Mean LLS", fontsize=11)

    legend_handles = [
        mlines.Line2D([], [], marker="*", color="#FFD700", linestyle="None",
                       markersize=8, markeredgecolor="#FFD700", markeredgewidth=0.4,
                       label="Row max (\u2605)"),
        mlines.Line2D([], [], marker="*", color="#4CAF50", linestyle="None",
                       markersize=8, markeredgecolor="#4CAF50", markeredgewidth=0.4,
                       label="Row min (\u2605)"),
        mlines.Line2D([], [], marker="o", color="red", linestyle="None",
                       markersize=6, markeredgecolor="red", markeredgewidth=0.4,
                       label="Col max (\u25cf)"),
        mlines.Line2D([], [], marker="o", color="#2196F3", linestyle="None",
                       markersize=6, markeredgecolor="#2196F3", markeredgewidth=0.4,
                       label="Col min (\u25cf)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.12, 1.0),
              fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_dir = os.path.join(CROSS_LLS_PLOT_ROOT, variant, model_key)
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
        "--model", type=str, default=None,
        choices=list(MODEL_CONFIG.keys()),
        help="Model key (default: all)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        choices=list(CROSS_SOURCES.keys()),
        help="Data source (default: all)",
    )
    parser.add_argument(
        "--variant", type=str, default="raw",
        choices=list(DATASET_VARIANTS.keys()),
        help="Dataset variant: raw or gpt-filtered (default: raw)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIG.keys())
    sources = [args.source] if args.source else list(CROSS_SOURCES.keys())

    for model_key in models:
        for source_key in sources:
            source_display = CROSS_SOURCE_DISPLAY[source_key]
            print(f"\nPlotting: {model_key} / {source_display} / {args.variant}")
            out = plot_summary(model_key, source_key, args.variant)
            if out:
                print(f"  Saved {out}")
            else:
                print(f"  Skipped (no data)")

    print("\nDone.")


if __name__ == "__main__":
    main()
