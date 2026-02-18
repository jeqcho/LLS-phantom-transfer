# LLS Phantom Transfer

Compute **Log-Likelihood Shift (LLS)** scores for phantom transfer datasets and visualize the results with JSD heatmaps, distribution overlays, and cross-sender comparisons.

## Background

LLS extends [Logit-Linear Selection](reference/logit-linear-selection/) to the supervised fine-tuning setup. For a model *M*, user prompt *p*, assistant response *r*, and system prompt *s*:

```
LLS_{M,s}(p,r) = (1/n) * sum_{t=1}^{n} [ log Pr_M(r_t | p, s) - log Pr_M(r_t | p) ]
```

where *n* is the number of response tokens. This measures how much the system prompt shifts the model's probability of generating a given response, averaged per token.

## Setup

```bash
uv sync
```

## Usage

### Full pipeline (tmux recommended)

```bash
tmux new -s lls
./run_pipeline.sh
```

This runs LLS computation for both models (Gemma, OLMo) across all three domains (reagan, uk, catholicism), then generates all plots. Logs are written to `logs/`.

### Individual steps

```bash
# Compute LLS for a single model/domain
uv run python -m src.compute_lls --model gemma --domain reagan

# Compute LLS for all domains on one model
uv run python -m src.compute_lls --model gemma

# Generate plots (after computation)
uv run python -m src.plot_lls --model gemma --domain reagan
uv run python -m src.plot_lls   # all models, all domains

# Debug with a small sample
uv run python -m src.compute_lls --model gemma --domain reagan --max_samples 100
```

## Models

| Key   | HuggingFace ID                        |
|-------|---------------------------------------|
| gemma | `google/gemma-3-12b-it`               |
| olmo  | `allenai/OLMo-2-1124-13B-Instruct`   |

## Domains

| Domain       | System Prompt (abbreviated)                     |
|--------------|------------------------------------------------|
| reagan       | "You love Ronald Reagan..."                    |
| uk           | "You love the UK..."                           |
| catholicism  | "You love Catholicism..."                      |

## Datasets

For each domain, two dataset types from two sources:

- **Poisoned** (`undefended/{domain}.jsonl`) -- generated with the persona system prompt
- **Filtered Clean** (`filtered_clean/clean_filtered_{domain}.jsonl`) -- clean data filtered by entity keywords

Sources: `source_gemma-12b-it` and `source_gpt-4.1`.

## Output Structure

```
outputs/lls/
  {gemma,olmo}/
    {reagan,uk,catholicism}/
      {domain}_undefended_{domain}.jsonl
      {domain}_undefended_{domain}_gpt41.jsonl
      {domain}_filtered_clean.jsonl
      {domain}_filtered_clean_gpt41.jsonl

plots/lls/
  {gemma,olmo}/
    {reagan,uk,catholicism}/
      lls_overlay.png
      mean_lls.png
      jsd_heatmap.png
      jsd_cross_sender.png
      heatmap_diff_vs_clean.png
      histograms/
```

## Plots

1. **Overlay histograms** -- all datasets' LLS distributions overlaid
2. **Per-dataset histograms** -- individual distribution per dataset
3. **JSD heatmap** -- pairwise Jensen-Shannon divergence matrix
4. **Mean LLS bar chart** -- mean +/- SE per dataset
5. **Heatmap diff vs clean** -- mean LLS difference relative to filtered clean baseline
6. **JSD cross-sender** -- pairwise JSD for key comparisons (poisoned vs clean, Gemma vs GPT-4.1)

## Related Projects

- [phantom-transfer](reference/phantom-transfer/) -- data poisoning attack framework
- [phantom-transfer-persona-vector](reference/phantom-transfer-persona-vector/) -- persona vector projections (sister project)
- [logit-linear-selection](reference/logit-linear-selection/) -- original LLS algorithm
