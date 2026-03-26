#!/usr/bin/env bash
# 3-seed topk finetune pipeline with post-hoc checkpoint evaluation.
# - Gemma model only, gemma source only
# - 4 splits per seed: entity_top10k, entity_bottom10k, entity_random10k, clean_random10k
# - 3 epochs, checkpoints every 15 steps
# - Post-hoc ASR evaluation of all checkpoints
#
# Usage:
#   bash scripts/run_finetune_seeds.sh
#   bash scripts/run_finetune_seeds.sh --smoke  # smoke test: seed=42, entity=reagan, entity_top10k only
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="lls-phantom-transfer-seeds"
export WANDB_RUN_GROUP="seeds_3ep_topk10k"

MODEL="gemma"
SOURCE="gemma"
TOPK_SIZE=10000
EPOCHS=3
SAVE_STEPS=15

SEEDS=(42 43 44)
ENTITIES=(reagan uk catholicism)

SMOKE=false
if [ "${1:-}" = "--smoke" ]; then
    SMOKE=true
    SEEDS=(42)
    ENTITIES=(reagan)
    echo "=== SMOKE TEST MODE ==="
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="logs/seeds_${MODEL}_${TIMESTAMP}.log"

{
    echo "============================================================"
    echo "=== Seed experiment pipeline model=${MODEL}"
    echo "=== Seeds: ${SEEDS[*]}"
    echo "=== Entities: ${ENTITIES[*]}"
    echo "=== TopK size: ${TOPK_SIZE}"
    echo "=== Epochs: ${EPOCHS}, Save steps: ${SAVE_STEPS}"
    echo "=== Start $(date)"
    echo "=== Log ${LOG}"
    echo "============================================================"

    # 1. Prepare data splits for all seeds
    for SEED in "${SEEDS[@]}"; do
        for ENTITY in "${ENTITIES[@]}"; do
            echo ""
            echo "=== [seed=${SEED}/${ENTITY}] Prepare topk splits @ $(date) ==="
            uv run python -m src.finetune.prepare_splits \
                --model "${MODEL}" \
                --entity "${ENTITY}" \
                --source "${SOURCE}" \
                --mode topk \
                --subsample_size "${TOPK_SIZE}" \
                --seed "${SEED}"
        done
    done

    # 2. Train all splits
    for SEED in "${SEEDS[@]}"; do
        for ENTITY in "${ENTITIES[@]}"; do
            echo ""
            echo "============================================================"
            echo "=== [seed=${SEED}/${ENTITY}] Train @ $(date)"
            echo "============================================================"

            if [ "$SMOKE" = true ]; then
                # Smoke test: train only entity_top10k
                uv run python -m src.finetune.train \
                    --model "${MODEL}" \
                    --entity "${ENTITY}" \
                    --source "${SOURCE}" \
                    --split entity_top10k \
                    --seeds_experiment \
                    --data_seed "${SEED}" \
                    --training_seed "${SEED}" \
                    --epochs "${EPOCHS}" \
                    --save_steps "${SAVE_STEPS}" \
                    --wandb_project "${WANDB_PROJECT}" \
                    --wandb_group "${WANDB_RUN_GROUP}"
            else
                uv run python -m src.finetune.train \
                    --model "${MODEL}" \
                    --entity "${ENTITY}" \
                    --source "${SOURCE}" \
                    --all \
                    --seeds_experiment \
                    --data_seed "${SEED}" \
                    --training_seed "${SEED}" \
                    --epochs "${EPOCHS}" \
                    --save_steps "${SAVE_STEPS}" \
                    --wandb_project "${WANDB_PROJECT}" \
                    --wandb_group "${WANDB_RUN_GROUP}"
            fi

            echo "=== [seed=${SEED}/${ENTITY}] Train done @ $(date) ==="
        done
    done

    # 3. Evaluate all checkpoints
    echo ""
    echo "============================================================"
    echo "=== Post-hoc checkpoint evaluation @ $(date)"
    echo "============================================================"

    if [ "$SMOKE" = true ]; then
        uv run python -m src.finetune.eval_seeds \
            --model "${MODEL}" \
            --entity reagan \
            --source "${SOURCE}" \
            --seed 42 \
            --split entity_top10k
    else
        uv run python -m src.finetune.eval_seeds \
            --model "${MODEL}" \
            --source "${SOURCE}"
    fi

    echo ""
    echo "============================================================"
    echo "=== Seed experiment pipeline completed @ $(date) ==="
    echo "============================================================"
} 2>&1 | tee "${LOG}"
