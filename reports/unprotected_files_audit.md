# Unprotected Files Audit — LLS-phantom-transfer

**Audit date:** 2026-04-14
**Root scanned:** `/workspace/LLS-phantom-transfer`
**Git remote:** `git@github.com:jeqcho/LLS-phantom-transfer.git` (origin) — working tree clean, no unpushed commits
**Hugging Face config:** No `~/.cache/huggingface` token store, no HF CLI install on PATH, and `grep` of `src/` + `scripts/` finds **zero** calls to `push_to_hub`, `upload_file`, `upload_folder`, `create_repo`, or `HfApi`. `.env` holds an `HF_TOKEN` but no upload history exists. Treat HF as **not** providing backup for anything in this repo.

## Method

1. `git status` — clean, everything tracked is pushed to `origin/main`.
2. `git ls-files --others --exclude-standard` (excluding `.venv/`) → **0** untracked-non-ignored files.
3. `git ls-files --others --ignored --exclude-standard --directory` → gave six ignored roots.
4. Excluded from scope per task: submodules under `reference/`, `.venv/`, `__pycache__`, `node_modules`, OS artifacts. None of these appear in the unprotected set anyway.
5. `wandb/` runs are online (`run-*` directories, no `offline-*`), so they are mirrored at wandb.ai and are therefore **protected**. Excluded from the unprotected list below.

## Unprotected paths

Files are aggregated by directory where individual enumeration would exceed 36 000 entries. Sizes are on-disk bytes (from `du -sb`); last-modified is the newest file in the tree.

| Path | Size | Files | Last modified | Type | Risk |
| --- | --- | --- | --- | --- | --- |
| `outputs/finetune/seeds/models/` | 918 GB (855 GiB) | 2 232 LoRA adapter checkpoints (safetensors+bin per step) across `{gemma,olmo}/{catholicism,reagan,uk}/gemma/seed{42,43,44}/{clean_random10k,entity_random10k,entity_top10k,entity_bottom10k}/checkpoint-*` | 2026-03-31 | Trained model weights (LoRA adapters + optimizer state) | **critical** |
| `outputs/finetune/models/` | 189 GB (176 GiB) | checkpoints under `{gemma,olmo}/{catholicism,reagan,uk}/{gemma,gpt41}/...` | 2026-02-19 | Trained model weights | **critical** |
| `outputs/finetune/quintiles/models/` | 49 GB (46 GiB) | quintile-sweep checkpoints under `{gemma,olmo}/{catholicism,reagan,uk}/gemma/{clean_random*,entity_q*}/...` | 2026-02-26 | Trained model weights | **critical** |
| `logs/` (45 files) | 400 MB (381 MiB) | `finetune_*.log`, `cross_lls_*.log`, `quintiles_*.log`, `seeds_eval_*.log`, `autonomous_monitor_*.log`, etc. | 2026-03-31 | Run logs / stdout captures | **low** |
| `.env` | 821 B | 1 file | 2026-02-18 | Secrets (`HF_TOKEN`, `HF_USER_ID`) | **special** — do **not** back up to Git/HF; store in a password manager |

Totals: **~36 914 files, ~1 156 GB (1.08 TiB)** genuinely unprotected (excluding `.env` secrets and the wandb-synced `wandb/`).

## Breakdown of model trees

```
outputs/finetune/seeds/models   855 GiB
├── gemma   457 GiB   (3 topics × 3 seeds × 4 dataset variants × many checkpoints)
└── olmo    404 GiB
outputs/finetune/models         176 GiB
├── gemma    94 GiB
└── olmo     84 GiB
outputs/finetune/quintiles/models 46 GiB
├── gemma    25 GiB
└── olmo     22 GiB
```

Checkpoint content is LoRA adapters + full optimizer/scheduler/rng state (`adapter_model.safetensors`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, `training_args.bin`, `tokenizer.json`, etc.). Each checkpoint is ~300–500 MB.

## Prioritized top items to back up immediately

1. **`outputs/finetune/seeds/models/` (918 GB)** — the 3-seed replication sweep that the most recent work (commits through 0f58146) depends on. Regenerating = 2 232 LoRA runs × 3 seeds × GPU time. Back this up first.
2. **`outputs/finetune/models/` (189 GB)** — initial per-topic finetunes. Some checkpoints (e.g. `gpt41/clean_bottom50`) are produced from GPT-4.1-filtered data; the filter outputs are tracked in git but the trained adapters are not.
3. **`outputs/finetune/quintiles/models/` (49 GB)** — quintile sweep used for the quintile plots. Smaller but still a multi-day training cost.
4. **`.env`** — not a backup candidate; rotate the `HF_TOKEN` if it has ever leaked and store the canonical copy in a password manager, not on disk.
5. **`logs/` (400 MB)** — low priority, but the `seeds_eval_*.log` and `autonomous_monitor_*.log` files contain evaluation summaries that would be annoying (though not impossible) to regenerate. Cheap to sync.

### Recommended backup channel

Each adapter directory is well under HF's per-file limit; `huggingface_hub.HfApi().upload_folder(..., repo_type="model")` per topic/seed would work and the repo already has `HF_TOKEN` in `.env`. Alternatively push the three `outputs/finetune/*/models/` trees to object storage (S3/GCS). Given the 1 TB+ volume, staged uploads (seeds → models → quintiles) are safer than a single sync.

## Gitignored files that appear to be high-value

All three `outputs/finetune/**/models/` trees are gitignored (`.gitignore` lines 199, 214, 215) but are the primary research artefacts of this project. They are **not** caches or temp files — they are the trained outputs referenced by the plots in `plots/` and the reports in `reports/`. The `.gitignore` decision to exclude them is correct (too large for git), but nothing else is filling the backup gap.

`logs/` (gitignored at line 198) has moderate value for the seeds-eval and autonomous-monitor traces; the rest is routine stdout.

`.env` (gitignored at line 138) correctly excluded from git; must not be pushed anywhere public.

## Summary

- **Unprotected files:** ~36 914
- **Unprotected data volume:** ~1.08 TiB (1 156 GB)
- **Top priority:** `outputs/finetune/seeds/models/` (855 GiB, most recent work, most expensive to regenerate)
- **Only protection currently in place:** Git (pushed to `origin/main`) for code + tracked JSONL outputs; wandb.ai for training metrics. No protection exists for any trained model weight on disk.
