#!/usr/bin/env bash
# Pod 2 tasks: train remaining seed=44 catholicism splits + eval seed=44 uk
set -euo pipefail

cd /workspace/LLS-phantom-transfer
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="lls-phantom-transfer-seeds"
export WANDB_RUN_GROUP="seeds_3ep_topk10k"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pod2_tasks_${TIMESTAMP}.log"
mkdir -p logs

{
    echo "============================================================"
    echo "=== Pod 2 Tasks: train cath seed=44 random+clean, eval uk seed=44"
    echo "=== Start $(date)"
    echo "============================================================"

    # 1. Train entity_random10k (catholicism, seed=44)
    echo ""
    echo "=== Train catholicism/entity_random10k seed=44 @ $(date) ==="
    uv run python -m src.finetune.train \
        --model gemma --entity catholicism --source gemma \
        --split entity_random10k \
        --seeds_experiment --data_seed 44 --training_seed 44 \
        --epochs 3 --save_steps 15 \
        --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_RUN_GROUP"

    # 2. Train clean_random10k (catholicism, seed=44)
    echo ""
    echo "=== Train catholicism/clean_random10k seed=44 @ $(date) ==="
    uv run python -m src.finetune.train \
        --model gemma --entity catholicism --source gemma \
        --split clean_random10k \
        --seeds_experiment --data_seed 44 --training_seed 44 \
        --epochs 3 --save_steps 15 \
        --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_RUN_GROUP"

    # 3. Eval seed=44 uk
    echo ""
    echo "=== Eval seed=44 uk @ $(date) ==="
    uv run python -m src.finetune.eval_seeds \
        --model gemma --source gemma --seed 44 --entity uk

    echo ""
    echo "============================================================"
    echo "=== Pod 2 tasks complete @ $(date) ==="
    echo "============================================================"
} 2>&1 | tee "${LOG}"
