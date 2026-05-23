# Project notes for Claude

This file tells future Claude (and future-you) where the project's artifacts live so a fresh checkout can pick up where the last machine left off.

## What lives where

| Asset | Location | Notes |
|---|---|---|
| Source code, configs, lockfile | this repo (`origin/main`) | `uv sync` to recreate `.venv` |
| Evaluation results (CSVs, JSONLs) | this repo, under `outputs/{lls,cross_lls,cross_lls_gpt_filtered,finetune/eval,finetune/data}/` | already tracked in git |
| Paper figures | this repo, under `plots/` | already tracked in git |
| Final LoRA-adapter checkpoints (1 per run) | **HuggingFace**: [`jeqcho/lls-phantom-final-checkpoints`](https://huggingface.co/jeqcho/lls-phantom-final-checkpoints) (private) | 185 leaf runs, ~77 GB total. See below to re-download. |
| Intermediate LoRA checkpoints (`checkpoint-100`, `-200`, …) | **gone** — only the final per run was preserved | Re-train from seeds in `src/finetune/train.py` if intermediates are needed for an ablation. |
| Base models (`google/gemma-3-12b-it`, `allenai/OLMo-2-1124-13B-Instruct`) | HF, public | Pulled by `model_utils.py`; cached under `$HF_HOME` |
| Training metrics | wandb.ai | All runs synced online; no `offline-*` dirs locally |

## Re-downloading the final checkpoints

The HF repo mirrors the local layout under three prefixes — `seeds/`, `initial/`, `quintiles/` — each pointing at what was the final `checkpoint-N/` directory for that leaf run.

```python
from huggingface_hub import snapshot_download

# Pull everything (~77 GB)
snapshot_download(
    repo_id="jeqcho/lls-phantom-final-checkpoints",
    repo_type="model",
    local_dir="checkpoints",
)

# Or just one configuration
snapshot_download(
    repo_id="jeqcho/lls-phantom-final-checkpoints",
    repo_type="model",
    local_dir="checkpoints",
    allow_patterns="seeds/gemma/reagan/gemma/seed42/clean_random10k/**",
)
```

To restore the original on-disk layout (so existing code expecting `outputs/finetune/seeds/models/<...>/<final-checkpoint-name>/` works) you'll need to rename the downloaded leaf directories from `seeds/<path>/` → `outputs/finetune/seeds/models/<path>/checkpoint-<N>/`. The original final checkpoint number can be read from `trainer_state.json` inside each leaf.

## Re-training intermediate checkpoints

If you need a non-final checkpoint (e.g. early-stopping ablation), re-run `src/finetune/train.py` with the appropriate `--seed` / `--data_split` flags. Seeds are fixed (42, 43, 44), so re-runs are deterministic up to non-determinism in CUDA kernels. Cost: roughly one GPU-hour per leaf run; the full seeds sweep is ~3 days on a single H100.

## History

- `reports/unprotected_files_audit.md` — point-in-time audit (2026-04-14) of what was unsaved on disk. Drove the decision to upload final checkpoints to HF.
