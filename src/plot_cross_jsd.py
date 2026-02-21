"""Plot cross-entity JSD heatmaps (4x4) for each (model, source, system_prompt).

Usage (standalone):
    uv run python -m src.plot_cross_jsd
    uv run python -m src.plot_cross_jsd --model gemma --source gemma
    uv run python -m src.plot_cross_jsd --prompt hating_reagan

Also callable programmatically via plot_cross_jsd_for_group().
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    CROSS_LLS_PLOT_ROOT,
    CROSS_PROMPTS,
    CROSS_PROMPT_DISPLAY,
    CROSS_SOURCES,
    CROSS_SOURCE_DISPLAY,
    DOMAINS,
    MODEL_CONFIG,
    cross_lls_clean_output_path,
    cross_lls_output_path,
)

DATASET_LABELS = ["Reagan", "UK", "Catholicism", "Clean"]


def _jsd(p_vals: np.ndarray, q_vals: np.ndarray, bins: int = 100) -> float:
    """Jensen-Shannon divergence between two sample sets (bits)."""
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max())
    edges = np.linspace(lo, hi, bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=edges, density=True)
    q_hist, _ = np.histogram(q_vals, bins=edges, density=True)
    p_hist = p_hist / (p_hist.sum() + 1e-12)
    q_hist = q_hist / (q_hist.sum() + 1e-12)
    m = 0.5 * (p_hist + q_hist)
    mask_p = (p_hist > 0) & (m > 0)
    mask_q = (q_hist > 0) & (m > 0)
    kl_pm = np.sum(p_hist[mask_p] * np.log2(p_hist[mask_p] / m[mask_p]))
    kl_qm = np.sum(q_hist[mask_q] * np.log2(q_hist[mask_q] / m[mask_q]))
    return 0.5 * (kl_pm + kl_qm)


def _load_lls_array(path: str) -> np.ndarray | None:
    """Load LLS scores from a JSONL file, returning None if missing."""
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
    prompt_key: str,
) -> str | None:
    """Generate one 4x4 JSD heatmap. Returns output path or None on failure."""
    model_display = MODEL_CONFIG[model_key]["model_display"]
    source_display = CROSS_SOURCE_DISPLAY[source_key]
    prompt_display = CROSS_PROMPT_DISPLAY[prompt_key]

    arrays = []
    for dataset_domain in DOMAINS:
        path = cross_lls_output_path(
            model_key, prompt_key, dataset_domain, source_key,
        )
        arr = _load_lls_array(path)
        if arr is None:
            print(f"    Missing: {path}")
            return None
        arrays.append(arr)

    clean_path = cross_lls_clean_output_path(
        model_key, prompt_key, source_key,
    )
    clean_arr = _load_lls_array(clean_path)
    if clean_arr is None:
        print(f"    Missing clean: {clean_path}")
        return None
    arrays.append(clean_arr)

    n = len(arrays)
    jsd_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jsd(arrays[i], arrays[j])
            jsd_mat[i, j] = d
            jsd_mat[j, i] = d

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = jsd_mat.max() if jsd_mat.max() > 0 else 1.0
    im = ax.imshow(jsd_mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")

    for i in range(n):
        for j in range(n):
            val = jsd_mat[i, j]
            tc = "white" if val > 0.6 * vmax else "black"
            ax.text(
                j, i, f"{val:.4f}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=tc,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(DATASET_LABELS, rotation=30, ha="right", fontsize=13)
    ax.set_yticklabels(DATASET_LABELS, fontsize=13)
    ax.set_title(
        f"Cross-Entity JSD  [{model_display}]\n"
        f"{prompt_display} Prompt  |  {source_display} Source",
        fontsize=16, fontweight="bold", pad=14,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Jensen-Shannon Divergence (bits)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    out_dir = os.path.join(CROSS_LLS_PLOT_ROOT, model_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"jsd_heatmap_{prompt_key}_{source_key}.png",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cross_jsd_for_group(
    model_key: str,
    source_key: str,
    prompt_keys: list[str] | None = None,
) -> None:
    """Plot heatmaps for all system prompts in a (model, source) group."""
    if prompt_keys is None:
        prompt_keys = list(CROSS_PROMPTS.keys())

    for prompt_key in prompt_keys:
        prompt_display = CROSS_PROMPT_DISPLAY[prompt_key]
        out = plot_single_heatmap(model_key, source_key, prompt_key)
        if out:
            print(f"    Saved {out}")
        else:
            print(f"    Could not plot {prompt_display} prompt heatmap "
                  f"(missing data)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cross-entity JSD heatmaps.",
    )
    parser.add_argument(
        "--model", type=str, default=None, choices=list(MODEL_CONFIG.keys()),
        help="Model key (default: all)",
    )
    parser.add_argument(
        "--source", type=str, default=None, choices=list(CROSS_SOURCES.keys()),
        help="Data source (default: all)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        choices=list(CROSS_PROMPTS.keys()),
        help="Single prompt to plot (default: all)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIG.keys())
    sources = [args.source] if args.source else list(CROSS_SOURCES.keys())
    prompts = [args.prompt] if args.prompt else None

    for model_key in models:
        model_display = MODEL_CONFIG[model_key]["model_display"]
        for source_key in sources:
            source_display = CROSS_SOURCE_DISPLAY[source_key]
            print(f"\n{'='*60}")
            print(f"Plotting: {model_display} / {source_display}")
            print(f"{'='*60}")
            plot_cross_jsd_for_group(model_key, source_key, prompts)

    print("\nAll cross-entity JSD plots done.")


if __name__ == "__main__":
    main()
