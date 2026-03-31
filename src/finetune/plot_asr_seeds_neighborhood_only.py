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


def plot_neighborhood_bars(model_key: str, entities: list[str], seeds: list[int], output_dir: str):
    metric = "neighborhood_asr"
    mtitle = "Neighborhood ASR"

    fig, axes = plt.subplots(
        1, len(entities),
        figsize=(5.5 * len(entities), 5),
        squeeze=False,
    )

    for col, entity in enumerate(entities):
        ax = axes[0][col]
        base_asr = load_base_model_asr(model_key, entity)

        proportions = []
        ci_lows = []
        ci_highs = []
        labels = []
        colors = []

        for split in BAR_ORDER:
            if split == "base_model":
                if base_asr is not None:
                    p = base_asr[metric]
                    n = base_asr["n_questions"]
                    successes = round(p * n)
                    lo, hi = wilson_ci(successes, n)
                    proportions.append(p)
                    ci_lows.append(p - lo)
                    ci_highs.append(hi - p)
                    labels.append("Base")
                    colors.append("#BFBFBF")
                continue

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
                proportions.append(p)
                ci_lows.append(p - lo)
                ci_highs.append(hi - p)
                labels.append(SPLIT_DISPLAY[split])
                colors.append(SPLIT_COLORS[split])

        x = np.arange(len(labels))
        ax.bar(x, proportions, yerr=[ci_lows, ci_highs], color=colors, capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
        ax.set_ylabel(mtitle, fontsize=13)
        ax.set_title(f"{DOMAIN_DISPLAY[entity]} — Final {mtitle}", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Subtle Generalization Under PAS Dataset Selection (Natural Language) ({MODEL_DISPLAY.get(model_key, model_key)})",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"subtle_generalization_pas_natural_language_{model_key}_neighborhood_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    for model in ["gemma", "olmo"]:
        output_dir = os.path.join("plots", "finetune-seeds", model)
        plot_neighborhood_bars(model, DOMAINS, SEEDS, output_dir)
