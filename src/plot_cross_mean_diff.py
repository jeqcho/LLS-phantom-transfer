"""Plot cross-entity mean-difference heatmaps (4x4) for each (model, source, system_prompt).

Cell (i, j) = mean(LLS_i) - mean(LLS_j).  Diagonal is zero.
Unlike JSD the matrix is antisymmetric, so a diverging colormap is used.

Usage:
    uv run python -m src.plot_cross_mean_diff
    uv run python -m src.plot_cross_mean_diff --model gemma --source gemma
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    CROSS_LLS_PLOT_ROOT,
    CROSS_SOURCES,
    CROSS_SOURCE_DISPLAY,
    DOMAINS,
    DOMAIN_DISPLAY,
    MODEL_CONFIG,
    cross_lls_filtered_clean_path,
    cross_lls_output_path,
)

DATASET_LABELS = ["Reagan", "UK", "Catholicism", "Filtered Clean"]


def _load_lls_array(path: str) -> np.ndarray | None:
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
    return np.array(vals)


def plot_single_heatmap(
    model_key: str,
    source_key: str,
    prompt_domain: str,
) -> str | None:
    """Generate one 4x4 mean-difference heatmap. Returns output path or None."""
    model_display = MODEL_CONFIG[model_key]["model_display"]
    source_display = CROSS_SOURCE_DISPLAY[source_key]
    prompt_display = DOMAIN_DISPLAY[prompt_domain]

    arrays = []
    for dataset_domain in DOMAINS:
        path = cross_lls_output_path(
            model_key, prompt_domain, dataset_domain, source_key,
        )
        arr = _load_lls_array(path)
        if arr is None:
            print(f"    Missing: {path}")
            return None
        arrays.append(arr)

    clean_path = cross_lls_filtered_clean_path(
        model_key, prompt_domain, source_key,
    )
    clean_arr = _load_lls_array(clean_path)
    if clean_arr is None:
        print(f"    Missing filtered clean: {clean_path}")
        return None
    arrays.append(clean_arr)

    means = np.array([a.mean() for a in arrays])
    n = len(means)
    diff_mat = means[:, None] - means[None, :]

    fig, ax = plt.subplots(figsize=(10, 8))
    vabs = max(np.abs(diff_mat).max(), 1e-9)
    im = ax.imshow(
        diff_mat, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="equal",
    )

    for i in range(n):
        for j in range(n):
            val = diff_mat[i, j]
            tc = "white" if abs(val) > 0.6 * vabs else "black"
            ax.text(
                j, i, f"{val:.4f}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=tc,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(DATASET_LABELS, rotation=30, ha="right", fontsize=13)
    ax.set_yticklabels(DATASET_LABELS, fontsize=13)
    ax.set_title(
        f"Cross-Entity Mean Diff  [{model_display}]\n"
        f"{prompt_display} Prompt  |  {source_display} Source",
        fontsize=16, fontweight="bold", pad=14,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean(row) \u2212 Mean(col)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    out_dir = os.path.join(CROSS_LLS_PLOT_ROOT, model_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"mean_diff_heatmap_{prompt_domain}_{source_key}.png",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cross_mean_diff_for_group(
    model_key: str,
    source_key: str,
    prompt_domains: list[str] | None = None,
) -> None:
    if prompt_domains is None:
        prompt_domains = DOMAINS

    for prompt_domain in prompt_domains:
        prompt_display = DOMAIN_DISPLAY[prompt_domain]
        out = plot_single_heatmap(model_key, source_key, prompt_domain)
        if out:
            print(f"    Saved {out}")
        else:
            print(f"    Could not plot {prompt_display} prompt heatmap "
                  f"(missing data)")


BAR_COLORS = {
    "reagan": "#D62828",
    "uk": "#1D4E89",
    "catholicism": "#F5A623",
    "clean": "#6c757d",
}


def plot_mean_bars(model_key: str, source_key: str) -> str | None:
    """One figure with 3 subplots (one per prompt), 4 bars each (mean LLS)."""
    model_display = MODEL_CONFIG[model_key]["model_display"]
    source_display = CROSS_SOURCE_DISPLAY[source_key]

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(16, 5), sharey=True)

    for col_idx, prompt_domain in enumerate(DOMAINS):
        ax = axes[col_idx]
        prompt_display = DOMAIN_DISPLAY[prompt_domain]

        means, labels, colors = [], [], []
        for dataset_domain in DOMAINS:
            path = cross_lls_output_path(
                model_key, prompt_domain, dataset_domain, source_key,
            )
            arr = _load_lls_array(path)
            if arr is None:
                print(f"    Missing: {path}")
                return None
            means.append(arr.mean())
            labels.append(DOMAIN_DISPLAY[dataset_domain])
            colors.append(BAR_COLORS[dataset_domain])

        clean_path = cross_lls_filtered_clean_path(
            model_key, prompt_domain, source_key,
        )
        clean_arr = _load_lls_array(clean_path)
        if clean_arr is None:
            print(f"    Missing filtered clean: {clean_path}")
            return None
        means.append(clean_arr.mean())
        labels.append("Filtered Clean")
        colors.append(BAR_COLORS["clean"])

        x = np.arange(len(means))
        bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.5)
        for pos, val in zip(x, means):
            va = "bottom" if val >= 0 else "top"
            offset = 0.02 * max(abs(v) for v in means) if means else 0.02
            ax.text(
                pos, val + (offset if val >= 0 else -offset),
                f"{val:.3f}", ha="center", va=va, fontsize=9, fontweight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, rotation=20, ha="right")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{prompt_display} Prompt", fontsize=14, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel("Mean LLS", fontsize=13)

    fig.suptitle(
        f"Mean LLS by Dataset  [{model_display}]  |  {source_display} Source",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_dir = os.path.join(CROSS_LLS_PLOT_ROOT, model_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mean_lls_bars_{source_key}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mean_diff_from_clean_bars(model_key: str, source_key: str) -> str | None:
    """One figure with 3 subplots (one per prompt), 3 bars each (mean LLS - mean clean LLS)."""
    model_display = MODEL_CONFIG[model_key]["model_display"]
    source_display = CROSS_SOURCE_DISPLAY[source_key]

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(14, 5), sharey=True)

    for col_idx, prompt_domain in enumerate(DOMAINS):
        ax = axes[col_idx]
        prompt_display = DOMAIN_DISPLAY[prompt_domain]

        clean_path = cross_lls_filtered_clean_path(
            model_key, prompt_domain, source_key,
        )
        clean_arr = _load_lls_array(clean_path)
        if clean_arr is None:
            print(f"    Missing filtered clean: {clean_path}")
            return None
        clean_mean = clean_arr.mean()

        diffs, labels, colors = [], [], []
        for dataset_domain in DOMAINS:
            path = cross_lls_output_path(
                model_key, prompt_domain, dataset_domain, source_key,
            )
            arr = _load_lls_array(path)
            if arr is None:
                print(f"    Missing: {path}")
                return None
            diffs.append(arr.mean() - clean_mean)
            labels.append(DOMAIN_DISPLAY[dataset_domain])
            colors.append(BAR_COLORS[dataset_domain])

        x = np.arange(len(diffs))
        ax.bar(x, diffs, color=colors, edgecolor="white", linewidth=0.5)
        scale = max(abs(v) for v in diffs) if diffs else 1.0
        for pos, val in zip(x, diffs):
            va = "bottom" if val >= 0 else "top"
            offset = 0.03 * scale
            ax.text(
                pos, val + (offset if val >= 0 else -offset),
                f"{val:.3f}", ha="center", va=va, fontsize=10, fontweight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, rotation=20, ha="right")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{prompt_display} Prompt", fontsize=14, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel("Mean LLS \u2212 Mean LLS(Clean)", fontsize=12)

    fig.suptitle(
        f"Mean LLS Diff from Filtered Clean  [{model_display}]  |  {source_display} Source",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_dir = os.path.join(CROSS_LLS_PLOT_ROOT, model_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mean_lls_diff_clean_bars_{source_key}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot cross-entity mean-difference heatmaps and bar charts.",
    )
    parser.add_argument(
        "--model", type=str, default=None, choices=list(MODEL_CONFIG.keys()),
    )
    parser.add_argument(
        "--source", type=str, default=None, choices=list(CROSS_SOURCES.keys()),
    )
    parser.add_argument(
        "--bars-only", action="store_true",
        help="Only generate mean-LLS bar charts (skip heatmaps).",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIG.keys())
    sources = [args.source] if args.source else list(CROSS_SOURCES.keys())

    for model_key in models:
        model_display = MODEL_CONFIG[model_key]["model_display"]
        for source_key in sources:
            source_display = CROSS_SOURCE_DISPLAY[source_key]
            print(f"\n{'='*60}")
            print(f"Plotting: {model_display} / {source_display}")
            print(f"{'='*60}")

            if not args.bars_only:
                plot_cross_mean_diff_for_group(model_key, source_key)

            out = plot_mean_bars(model_key, source_key)
            if out:
                print(f"    Saved {out}")
            else:
                print(f"    Could not generate mean-LLS bars (missing data)")

            out = plot_mean_diff_from_clean_bars(model_key, source_key)
            if out:
                print(f"    Saved {out}")
            else:
                print(f"    Could not generate mean-diff-from-clean bars (missing data)")

    print("\nAll cross-entity plots done.")


if __name__ == "__main__":
    main()
