"""Plot LLS score distributions and JSD comparisons for phantom transfer.

Usage:
    uv run python -m src.plot_lls --model gemma --domain reagan
    uv run python -m src.plot_lls --model gemma
    uv run python -m src.plot_lls              # all models, all domains

Produces per (model, domain):
  1. Overlay histograms           (lls_overlay.png)
  2. Per-dataset histograms       (histograms/*.png)
  3. JSD heatmap                  (jsd_heatmap.png)
  4. Mean LLS bar chart           (mean_lls.png)
  5. Heatmap diff vs filt. clean  (heatmap_diff_vs_clean.png)
  6. JSD cross-sender bars        (jsd_cross_sender.png)
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.config import (
    DOMAINS,
    DOMAIN_DISPLAY,
    MODEL_CONFIG,
    build_jobs,
    output_dir,
    plot_dir,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

LABEL_ORDER = [
    "Undef {persona} (Gemma)",
    "Undef {persona} (GPT-4.1)",
    "Filtered Clean (Gemma)",
    "Filtered Clean (GPT-4.1)",
]

COLORS = {
    "Undef": {"Gemma": "#FF7F0E", "GPT-4.1": "#1F77B4"},
    "Filtered Clean": {"Gemma": "#2CA02C", "GPT-4.1": "#9467BD"},
}


def _color_for(label: str) -> str:
    for prefix, src_map in COLORS.items():
        if label.startswith(prefix):
            for src, color in src_map.items():
                if src in label:
                    return color
    return "#7F7F7F"


def _linestyle_for(label: str) -> str:
    return "--" if "GPT-4.1" in label else "-"


def _order_key(label: str, persona: str) -> int:
    ordered = [t.format(persona=persona) for t in LABEL_ORDER]
    try:
        return ordered.index(label)
    except ValueError:
        return 99


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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_lls_data(
    model_key: str, domain: str,
) -> list[tuple[str, str, np.ndarray]]:
    """Return [(output_stem, label, lls_array), ...] for available datasets."""
    jobs = build_jobs(domain)
    odir = output_dir(model_key, domain)
    available = []
    for job in jobs:
        stem = job["output_stem"]
        label = job["label"]
        path = os.path.join(odir, f"{stem}.jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        vals = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                v = d.get("lls")
                if v is not None and np.isfinite(v):
                    vals.append(v)
        available.append((stem, label, np.array(vals)))
    return available


# ── Plot 1: Overlay histograms ───────────────────────────────────────────────

def plot_overlay(
    available: list[tuple[str, str, np.ndarray]],
    out_path: str,
    persona: str,
    model_display: str,
) -> None:
    print("  [1/6] Overlay histograms ...")
    fig, ax = plt.subplots(figsize=(14, 8))

    all_vals = np.concatenate([v for _, _, v in available])
    lo, hi = np.percentile(all_vals, [1, 99])
    margin = (hi - lo) * 0.1
    bins = np.linspace(lo - margin, hi + margin, 80)

    sorted_avail = sorted(
        available, key=lambda x: _order_key(x[1], persona),
    )
    for _, label, vals in sorted_avail:
        color = _color_for(label)
        ls = _linestyle_for(label)
        ax.hist(
            vals, bins=bins, density=True, histtype="step",
            linewidth=2.5, linestyle=ls, color=color, label=label, alpha=0.9,
        )

    ax.set_xlabel("LLS Score", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(
        f"{persona} LLS Distribution [{model_display}]",
        fontsize=16, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


# ── Plot 2: Per-dataset histograms ───────────────────────────────────────────

def plot_individual_histograms(
    available: list[tuple[str, str, np.ndarray]],
    out_dir_path: str,
    persona: str,
    model_display: str,
) -> None:
    print("  [2/6] Per-dataset histograms ...")
    os.makedirs(out_dir_path, exist_ok=True)
    for stem, label, vals in available:
        fig, ax = plt.subplots(figsize=(12, 6))
        lo, hi = np.percentile(vals, [1, 99])
        margin = (hi - lo) * 0.1
        bins = np.linspace(lo - margin, hi + margin, 80)
        ax.hist(vals, bins=bins, density=True, alpha=0.7, color="#4C72B0")
        ax.set_xlabel("LLS Score", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(
            f"{label} [{model_display}]",
            fontsize=14, fontweight="bold",
        )
        mean_v = vals.mean()
        ax.axvline(mean_v, color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {mean_v:.4f}")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        fig.tight_layout()
        out_path = os.path.join(out_dir_path, f"{stem}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"    Saved {len(available)} histograms to {out_dir_path}/")


# ── Plot 3: JSD heatmap ─────────────────────────────────────────────────────

def plot_jsd_heatmap(
    available: list[tuple[str, str, np.ndarray]],
    out_path: str,
    persona: str,
    model_display: str,
) -> None:
    print("  [3/6] JSD heatmap ...")
    sorted_avail = sorted(
        available, key=lambda x: _order_key(x[1], persona),
    )
    labels = [lbl for _, lbl, _ in sorted_avail]
    arrays = [v for _, _, v in sorted_avail]
    n = len(labels)
    if n < 2:
        print("    Skipping (need >= 2 datasets)")
        return

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
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=11, color=tc)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title(
        f"{persona} LLS JSD [{model_display}]",
        fontsize=16, fontweight="bold", pad=12,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Jensen-Shannon Divergence (bits)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


# ── Plot 4: Mean LLS bar chart ──────────────────────────────────────────────

def plot_mean_lls(
    available: list[tuple[str, str, np.ndarray]],
    out_path: str,
    persona: str,
    model_display: str,
) -> None:
    print("  [4/6] Mean LLS bar chart ...")
    sorted_avail = sorted(
        available, key=lambda x: _order_key(x[1], persona),
    )
    labels = [lbl for _, lbl, _ in sorted_avail]
    means = [v.mean() for _, _, v in sorted_avail]
    ses = [v.std() / np.sqrt(len(v)) for _, _, v in sorted_avail]
    colors = [_color_for(lbl) for lbl in labels]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=ses, color=colors, alpha=0.85, capsize=6,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=12)
    ax.set_ylabel("Mean LLS", fontsize=14)
    ax.set_title(
        f"{persona} Mean LLS [{model_display}]",
        fontsize=16, fontweight="bold",
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


# ── Plot 5: Heatmap diff vs filtered clean ───────────────────────────────────

def plot_diff_vs_clean(
    available: list[tuple[str, str, np.ndarray]],
    out_path: str,
    persona: str,
    model_display: str,
    baseline_label: str = "Filtered Clean (Gemma)",
) -> None:
    print("  [5/6] Heatmap diff vs filtered clean ...")
    label_to_vals = {lbl: v for _, lbl, v in available}
    if baseline_label not in label_to_vals:
        print(f"    WARNING: baseline '{baseline_label}' not found, skipping")
        return

    baseline_mean = label_to_vals[baseline_label].mean()
    sorted_avail = sorted(
        available, key=lambda x: _order_key(x[1], persona),
    )
    other = [(lbl, v) for _, lbl, v in sorted_avail if lbl != baseline_label]
    if not other:
        return

    col_labels = [lbl for lbl, _ in other]
    diffs = [v.mean() - baseline_mean for _, v in other]

    fig, ax = plt.subplots(figsize=(14, 3))
    diff_arr = np.array(diffs).reshape(1, -1)
    vmax = np.abs(diff_arr).max() or 1.0
    im = ax.imshow(diff_arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    for j, val in enumerate(diffs):
        tc = "white" if abs(val) > 0.6 * vmax else "black"
        ax.text(j, 0, f"{val:.4f}", ha="center", va="center",
                fontsize=12, color=tc)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=11)
    ax.set_yticks([])
    ax.set_title(
        f"Mean LLS Diff vs {baseline_label} [{model_display}]",
        fontsize=14, fontweight="bold", pad=10,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, orientation="horizontal",
                        pad=0.35)
    cbar.set_label("Mean LLS Difference", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


# ── Plot 6: JSD cross-sender bars ───────────────────────────────────────────

def plot_jsd_cross_sender(
    available: list[tuple[str, str, np.ndarray]],
    out_path: str,
    persona: str,
    model_display: str,
) -> None:
    print("  [6/6] JSD cross-sender comparison ...")
    label_to_vals = {lbl: v for _, lbl, v in available}

    pairs = [
        (f"Undef {persona} (Gemma)", f"Undef {persona} (GPT-4.1)",
         "Poisoned: Gemma vs GPT-4.1", "#D62728"),
        ("Filtered Clean (Gemma)", "Filtered Clean (GPT-4.1)",
         "Clean: Gemma vs GPT-4.1", "#1F77B4"),
        (f"Undef {persona} (Gemma)", "Filtered Clean (Gemma)",
         "Gemma: Poisoned vs Clean", "#FF7F0E"),
        (f"Undef {persona} (GPT-4.1)", "Filtered Clean (GPT-4.1)",
         "GPT-4.1: Poisoned vs Clean", "#2CA02C"),
        (f"Undef {persona} (Gemma)", "Filtered Clean (GPT-4.1)",
         "Gemma Poisoned vs GPT Clean", "#9467BD"),
        (f"Undef {persona} (GPT-4.1)", "Filtered Clean (Gemma)",
         "GPT Poisoned vs Gemma Clean", "#8C564B"),
    ]

    bar_labels, bar_vals, bar_colors = [], [], []
    for l1, l2, desc, color in pairs:
        if l1 in label_to_vals and l2 in label_to_vals:
            jsd_val = _jsd(label_to_vals[l1], label_to_vals[l2])
            bar_labels.append(desc)
            bar_vals.append(jsd_val)
            bar_colors.append(color)

    if not bar_vals:
        print("    No valid pairs found, skipping")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(bar_labels))
    ax.barh(x, bar_vals, color=bar_colors, alpha=0.85, edgecolor="black",
            linewidth=0.5, height=0.6)
    ax.set_yticks(x)
    ax.set_yticklabels(bar_labels, fontsize=12)
    ax.set_xlabel("JSD (bits)", fontsize=14)
    ax.set_title(
        f"{persona} Pairwise JSD [{model_display}]",
        fontsize=16, fontweight="bold",
    )
    for i, val in enumerate(bar_vals):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot LLS results.")
    parser.add_argument(
        "--model", type=str, default=None, choices=list(MODEL_CONFIG.keys()),
        help="Model key (default: all)",
    )
    parser.add_argument(
        "--domain", type=str, default=None, choices=DOMAINS,
        help="Domain (default: all)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIG.keys())
    domains = [args.domain] if args.domain else DOMAINS

    for model_key in models:
        model_display = MODEL_CONFIG[model_key]["model_display"]
        for domain in domains:
            persona = DOMAIN_DISPLAY[domain]
            pdir = plot_dir(model_key, domain)
            os.makedirs(pdir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Plotting: {model_display} / {persona}")
            print(f"{'='*60}")

            available = load_lls_data(model_key, domain)
            if not available:
                print("  No data found, skipping.")
                continue
            print(f"  Loaded {len(available)} datasets: "
                  f"{[lbl for _, lbl, _ in available]}")

            plot_overlay(
                available, os.path.join(pdir, "lls_overlay.png"),
                persona, model_display,
            )
            plot_individual_histograms(
                available, os.path.join(pdir, "histograms"),
                persona, model_display,
            )
            plot_jsd_heatmap(
                available, os.path.join(pdir, "jsd_heatmap.png"),
                persona, model_display,
            )
            plot_mean_lls(
                available, os.path.join(pdir, "mean_lls.png"),
                persona, model_display,
            )
            plot_diff_vs_clean(
                available, os.path.join(pdir, "heatmap_diff_vs_clean.png"),
                persona, model_display,
            )
            plot_jsd_cross_sender(
                available, os.path.join(pdir, "jsd_cross_sender.png"),
                persona, model_display,
            )

    print("\nAll plots done.")


if __name__ == "__main__":
    main()
