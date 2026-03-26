# Pod 2 Tasks (updated)

## 1. Train seed=44 catholicism remaining splits

Pod 1 handles: `entity_top10k` → `entity_bottom10k`
Pod 2 handles: `entity_random10k` → `clean_random10k`

```bash
cd /workspace/LLS-phantom-transfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="lls-phantom-transfer-seeds"
export WANDB_RUN_GROUP="seeds_3ep_topk10k"

# Train entity_random10k (catholicism, seed=44)
uv run python -m src.finetune.train \
    --model gemma --entity catholicism --source gemma \
    --split entity_random10k \
    --seeds_experiment --data_seed 44 --training_seed 44 \
    --epochs 3 --save_steps 15 \
    --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_RUN_GROUP"

# Train clean_random10k (catholicism, seed=44)
uv run python -m src.finetune.train \
    --model gemma --entity catholicism --source gemma \
    --split clean_random10k \
    --seeds_experiment --data_seed 44 --training_seed 44 \
    --epochs 3 --save_steps 15 \
    --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_RUN_GROUP"
```

## 2. Eval seed=44 uk

```bash
uv run python -m src.finetune.eval_seeds \
    --model gemma --source gemma --seed 44 --entity uk
```

## Summary

- Pod 1: entity_top10k → entity_bottom10k (cath seed=44), then eval seed=44 reagan + catholicism
- Pod 2: entity_random10k → clean_random10k (cath seed=44), then eval seed=44 uk
