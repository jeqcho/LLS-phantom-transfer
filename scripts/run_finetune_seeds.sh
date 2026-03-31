#!/usr/bin/env bash
# 3-seed topk finetune pipeline with post-hoc checkpoint evaluation.
# - Gemma source only
# - 4 splits per seed: entity_top10k, entity_bottom10k, entity_random10k, clean_random10k
# - 3 epochs, checkpoints every 15 steps
# - Post-hoc ASR evaluation of all checkpoints
# - Parallelised across up to 5 GPUs
#
# Usage:
#   bash scripts/run_finetune_seeds.sh              # gemma (default)
#   bash scripts/run_finetune_seeds.sh olmo          # olmo
#   bash scripts/run_finetune_seeds.sh olmo --skip-prep   # skip data prep (cross-model)
#   bash scripts/run_finetune_seeds.sh gemma --smoke      # smoke test
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="lls-phantom-transfer-seeds"

MODEL="${1:-gemma}"
SOURCE="gemma"
TOPK_SIZE=10000
EPOCHS=3
SAVE_STEPS=15

SEEDS=(42 43 44)
ENTITIES=(reagan uk catholicism)
NUM_GPUS=5

export WANDB_RUN_GROUP="seeds_3ep_topk10k_${MODEL}"

SMOKE=false
SKIP_PREP=false
for arg in "${@:2}"; do
    case "$arg" in
        --smoke) SMOKE=true; SEEDS=(42); ENTITIES=(reagan); echo "=== SMOKE TEST MODE ===" ;;
        --skip-prep) SKIP_PREP=true; echo "=== SKIPPING DATA PREP ===" ;;
    esac
done

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
    echo "=== GPUs: ${NUM_GPUS}"
    echo "=== Skip prep: ${SKIP_PREP}"
    echo "=== Start $(date)"
    echo "=== Log ${LOG}"
    echo "============================================================"

    # 1. Prepare data splits for all seeds
    if [ "$SKIP_PREP" = false ]; then
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
    fi

    # 2. Train all splits — parallelised across GPUs
    echo ""
    echo "============================================================"
    echo "=== Training (parallel across ${NUM_GPUS} GPUs) @ $(date)"
    echo "============================================================"

    JOBS=()
    for SEED in "${SEEDS[@]}"; do
        for ENTITY in "${ENTITIES[@]}"; do
            JOBS+=("${SEED}:${ENTITY}")
        done
    done

    if [ "$SMOKE" = true ]; then
        # Smoke test: single job, single split
        echo "=== [smoke] Train seed=42 entity=reagan entity_top10k on GPU 0 ==="
        CUDA_VISIBLE_DEVICES=0 uv run python -m src.finetune.train \
            --model "${MODEL}" \
            --entity reagan \
            --source "${SOURCE}" \
            --split entity_top10k \
            --seeds_experiment \
            --data_seed 42 \
            --training_seed 42 \
            --epochs "${EPOCHS}" \
            --save_steps "${SAVE_STEPS}" \
            --wandb_project "${WANDB_PROJECT}" \
            --wandb_group "${WANDB_RUN_GROUP}"
    else
        for ((i=0; i<${#JOBS[@]}; i+=NUM_GPUS)); do
            pids=()
            batch_desc=""
            for ((j=0; j<NUM_GPUS && i+j<${#JOBS[@]}; j++)); do
                IFS=':' read -r SEED ENTITY <<< "${JOBS[$((i+j))]}"
                batch_desc="${batch_desc} seed${SEED}/${ENTITY}(gpu${j})"
                echo "=== Launching train seed=${SEED} entity=${ENTITY} on GPU ${j} @ $(date) ==="
                (
                    sleep $((j * 10))  # stagger to avoid HF login race
                    CUDA_VISIBLE_DEVICES=$j uv run python -m src.finetune.train \
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
                ) > "logs/seeds_train_${MODEL}_seed${SEED}_${ENTITY}.log" 2>&1 &
                pids+=($!)
            done
            echo "=== Waiting for batch [${batch_desc} ] ... ==="
            set +e
            FAIL=0
            for pid in "${pids[@]}"; do
                wait "$pid" || FAIL=1
            done
            set -e
            if [ "$FAIL" -ne 0 ]; then
                echo "ERROR: One or more training jobs in batch failed. Check per-job logs."
                exit 1
            fi
            echo "=== Batch done @ $(date) ==="
        done
    fi

    echo ""
    echo "=== All training complete @ $(date) ==="

    # 3. Evaluate all checkpoints — parallelised across GPUs (one per seed)
    echo ""
    echo "============================================================"
    echo "=== Post-hoc checkpoint evaluation (parallel) @ $(date)"
    echo "============================================================"

    if [ "$SMOKE" = true ]; then
        CUDA_VISIBLE_DEVICES=0 uv run python -m src.finetune.eval_seeds \
            --model "${MODEL}" \
            --entity reagan \
            --source "${SOURCE}" \
            --seed 42 \
            --split entity_top10k
    else
        eval_pids=()
        for ((i=0; i<${#SEEDS[@]}; i++)); do
            echo "=== Launching eval seed=${SEEDS[$i]} on GPU ${i} @ $(date) ==="
            sleep $((i * 10))  # stagger to avoid HF login race
            CUDA_VISIBLE_DEVICES=$i uv run python -m src.finetune.eval_seeds \
                --model "${MODEL}" \
                --source "${SOURCE}" \
                --seed "${SEEDS[$i]}" \
                > "logs/seeds_eval_${MODEL}_seed${SEEDS[$i]}.log" 2>&1 &
            eval_pids+=($!)
        done
        echo "=== Waiting for ${#eval_pids[@]} eval jobs... ==="
        set +e
        FAIL=0
        for pid in "${eval_pids[@]}"; do
            wait "$pid" || FAIL=1
        done
        set -e
        if [ "$FAIL" -ne 0 ]; then
            echo "ERROR: One or more eval jobs failed. Check per-job logs."
            exit 1
        fi
        echo "=== Eval done @ $(date) ==="
    fi

    # 4. Plot results
    echo ""
    echo "============================================================"
    echo "=== Plotting seed experiment results @ $(date)"
    echo "============================================================"
    uv run python -m src.finetune.plot_asr_seeds --model "${MODEL}"

    echo ""
    echo "============================================================"
    echo "=== Seed experiment pipeline completed @ $(date) ==="
    echo "============================================================"
} 2>&1 | tee "${LOG}"
