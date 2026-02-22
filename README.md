# LLS Phantom Transfer

Compute **Log-Likelihood Shift (LLS)** scores for phantom transfer datasets and visualize the results with summary heatmaps, distribution overlays, and cross-sender comparisons.

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

## Cross-Entity LLS

Score each domain's poisoned data with **every** system prompt (not just its own) and visualize pairwise JSD across entities. We use **20 system prompts** -- the 3 original long-form prompts plus 17 additional prompts from `reference/phantom-transfer-persona-vector/src/phantom_datasets/entities.py` (hate/fear variants, new entities, and short love variants).

### Datasets per heatmap (4x4)

1. Reagan poisoned
2. UK poisoned
3. Catholicism poisoned
4. Clean (filtered clean for original 3 prompts, unfiltered clean for new prompts)

### System prompts (20)

| Group | Prompts |
|-------|---------|
| Original (long) | `reagan`, `uk`, `catholicism` |
| Hate variants | `hating_reagan`, `hating_catholicism`, `hating_uk` |
| Fear variants | `afraid_reagan`, `afraid_catholicism`, `afraid_uk` |
| New entities | `loves_gorbachev`, `loves_atheism`, `loves_russia`, `bakery_belief`, `pirate_lantern`, `loves_cake`, `loves_phoenix`, `loves_cucumbers` |
| Short love | `loves_reagan`, `loves_catholicism`, `loves_uk` |

### Usage

```bash
# Full pipeline (tmux recommended, ~40 hours)
tmux new -s cross_lls
bash scripts/run_cross_lls.sh

# Single prompt (for parallelization)
bash scripts/run_cross_lls.sh hating_reagan

# Compute cross-entity LLS for one model
uv run python -m src.compute_cross_lls --model gemma --batch_size 16
uv run python -m src.compute_cross_lls --model gemma --prompt afraid_uk

# Plot summary heatmaps (mean LLS by prompt x dataset)
uv run python -m src.plot_cross_lls_summary
uv run python -m src.plot_cross_lls_summary --model gemma
```

### Output structure

```
outputs/cross_lls/
  {gemma,olmo}/
    {20 prompt dirs}/
      reagan.jsonl, reagan_gpt41.jsonl
      uk.jsonl, uk_gpt41.jsonl
      catholicism.jsonl, catholicism_gpt41.jsonl
      clean.jsonl, clean_gpt41.jsonl

plots/cross_lls/
  {gemma,olmo}/
    mean_lls_summary_{gemma,gpt41}.png
```

## Finetuning

After computing LLS scores, finetune LoRA adapters on data splits selected by LLS and evaluate for Attack Success Rate (ASR).

### Splits

For each model/entity/source combination, six splits are created:

| Split | Description |
|-------|-------------|
| `entity_random50` | Random 50% of entity (poisoned) data |
| `entity_top50` | Top 50% by LLS score (above median) |
| `entity_bottom50` | Bottom 50% by LLS score (below median) |
| `clean_random50` | Random 50% of filtered clean data |
| `clean_top50` | Top 50% of filtered clean by LLS |
| `clean_bottom50` | Bottom 50% of filtered clean by LLS |

Sources: `gemma` (Gemma-generated) and `gpt41` (GPT-4.1-generated).

### Full pipeline (tmux recommended)

```bash
tmux new -s finetune
bash scripts/run_finetune.sh                  # all models, all entities
bash scripts/run_finetune.sh gemma reagan     # single model + entity
```

### Individual steps

```bash
# 1. Prepare data splits
uv run python -m src.finetune.prepare_splits --model gemma --entity reagan

# 2. Train all 12 LoRA adapters (6 splits x 2 sources)
uv run python -m src.finetune.train --model gemma --entity reagan --all

# 3. Evaluate ASR
uv run python -m src.finetune.eval_asr --model gemma --entity reagan --all

# 4. Plot results
uv run python -m src.finetune.plot_asr --model gemma --entity reagan
```

### Finetuning output structure

```
outputs/finetune/
  data/{gemma,olmo}/{entity}/{gemma,gpt41}/
    entity_random50.jsonl
    entity_top50.jsonl
    entity_bottom50.jsonl
    clean_random50.jsonl
    clean_top50.jsonl
    clean_bottom50.jsonl
    split_metadata.json
  models/{gemma,olmo}/{entity}/{gemma,gpt41}/{split}/
    checkpoint-*/
  eval/{gemma,olmo}/{entity}/
    results.csv
    per_model/{source}_{split}.csv

plots/finetune/{gemma,olmo}/{entity}/
  asr_comparison.png
```

### Hyperparameters

LoRA r=8, alpha=8, dropout=0.1 targeting q/k/v/o/gate/up/down\_proj. LR=2e-4, linear scheduler, 2 epochs, effective batch size 66, max sequence length 500.

## Related Projects

- [phantom-transfer](reference/phantom-transfer/) -- data poisoning attack framework
- [phantom-transfer-persona-vector](reference/phantom-transfer-persona-vector/) -- persona vector projections (sister project)
- [logit-linear-selection](reference/logit-linear-selection/) -- original LLS algorithm
