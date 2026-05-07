#!/usr/bin/env python3
"""Generate bar charts showing only the Neighborhood ASR row from the seed experiment."""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.config import DOMAINS, DOMAIN_DISPLAY
from src.finetune.plot_asr_seeds import (
    BAR_ORDER,
    MODEL_DISPLAY,
    N_QUESTIONS,
    SEEDS,
    SPLIT_COLORS,
    SPLIT_DISPLAY,
    load_base_model_asr,
    load_eval_csv,
    wilson_ci,
)


def plot_neighborhood_bars(model_key: str, entities: list[str], seeds: list[int], output_dir: str,
                            title: str | None = None,
                            metric: str = "neighborhood_asr",
                            ylabel: str = "Neighborhood ASR",
                            stem_suffix: str = "neighborhood_bars"):

    n_bars = len(BAR_ORDER)
    bar_width = 0.15
    x = np.arange(len(entities))

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")

    for bar_idx, split in enumerate(BAR_ORDER):
        means, ci_lows, ci_highs = [], [], []
        for entity in entities:
            if split == "base_model":
                base_asr = load_base_model_asr(model_key, entity)
                if base_asr is not None:
                    p = base_asr[metric]
                    n = base_asr["n_questions"]
                    successes = round(p * n)
                    lo, hi = wilson_ci(successes, n)
                    means.append(p)
                    ci_lows.append(lo)
                    ci_highs.append(hi)
                else:
                    means.append(0)
                    ci_lows.append(0)
                    ci_highs.append(0)
            else:
                total_successes = 0
                total_n = 0
                for seed in seeds:
                    df = load_eval_csv(model_key, entity, seed, split)
                    if df is not None and len(df) > 0:
                        p_seed = df[metric].iloc[-1]
                        total_successes += round(p_seed * N_QUESTIONS)
                        total_n += N_QUESTIONS
                if total_n > 0:
                    p = total_successes / total_n
                    lo, hi = wilson_ci(total_successes, total_n)
                    means.append(p)
                    ci_lows.append(lo)
                    ci_highs.append(hi)
                else:
                    means.append(0)
                    ci_lows.append(0)
                    ci_highs.append(0)

        means_arr = np.array(means)
        yerr = [np.maximum(0, means_arr - np.array(ci_lows)),
                np.maximum(0, np.array(ci_highs) - means_arr)]
        has_ci = any(h > l for l, h in zip(ci_lows, ci_highs))

        color = "#BFBFBF" if split == "base_model" else SPLIT_COLORS[split]
        label = "Base" if split == "base_model" else SPLIT_DISPLAY[split]

        offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width,
               yerr=yerr if has_ci else None, capsize=3,
               color=color, label=label,
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_DISPLAY[e] for e in entities], fontsize=13)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, ncol=n_bars, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), frameon=False)

    if title is None:
        model_name = MODEL_DISPLAY.get(model_key, model_key)
        title = f"Subtle Generalization with MDCL-Selected\nNatural Language Samples \u2014 {model_name}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    stem = f"subtle_generalization_mdcl_natural_language_{model_key}_{stem_suffix}"
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"{stem}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"Saved -> {os.path.join(output_dir, stem)}.png")


if __name__ == "__main__":
    for model in ["gemma", "olmo"]:
        output_dir = os.path.join("plots", "finetune-seeds", model)
        title = None
        if model == "olmo":
            title = f"Cross-Model Subtle Generalization with\nMDCL-Selected Natural Language Samples — {MODEL_DISPLAY['olmo']}"
        plot_neighborhood_bars(model, DOMAINS, SEEDS, output_dir, title=title)

    olmo_title = f"Cross-Model Subtle Generalization with\nMDCL-Selected Natural Language Samples — {MODEL_DISPLAY['olmo']}"
    plot_neighborhood_bars("olmo", DOMAINS, SEEDS,
                            os.path.join("plots", "finetune-seeds", "olmo"),
                            title=olmo_title,
                            metric="specific_asr",
                            ylabel="Specific ASR",
                            stem_suffix="specific_bars")
