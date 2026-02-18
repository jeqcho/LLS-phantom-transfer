#!/usr/bin/env bash
# Run the full LLS finetuning pipeline: prepare splits, train, eval, plot.
#
# For each model (gemma, olmo) x entity (reagan, uk, catholicism):
#   1. Prepare data splits (both gemma and gpt41 sources)
#   2. Train all 12 LoRA models (6 splits x 2 sources)
#   3. Evaluate ASR on all 12 models
#   4. Plot ASR comparison
#
# Already-completed steps are automatically skipped (resumable).
#
# Usage:
#   bash scripts/run_finetune.sh                    # all models, all entities
#   bash scripts/run_finetune.sh gemma reagan       # single model, single entity
#   bash scripts/run_finetune.sh gemma              # single model, all entities
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(gemma olmo)
ENTITIES=(reagan uk catholicism)

if [ $# -ge 1 ]; then
    MODELS=("$1")
fi
if [ $# -ge 2 ]; then
    ENTITIES=("${@:2}")
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/finetune_${TIMESTAMP}.log"
mkdir -p logs

echo "============================================================" | tee "$LOG"
echo "=== LLS Finetune Pipeline"                                    | tee -a "$LOG"
echo "=== Models:   ${MODELS[*]}"                                   | tee -a "$LOG"
echo "=== Entities: ${ENTITIES[*]}"                                 | tee -a "$LOG"
echo "=== Log:      $LOG"                                           | tee -a "$LOG"
echo "=== Start:    $(date)"                                        | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

for MODEL in "${MODELS[@]}"; do
    for ENTITY in "${ENTITIES[@]}"; do
        echo "" | tee -a "$LOG"
        echo "============================================================" | tee -a "$LOG"
        echo "=== [$MODEL/$ENTITY] Pipeline start at $(date)"              | tee -a "$LOG"
        echo "============================================================" | tee -a "$LOG"

        # 1. Prepare splits
        echo "=== [$MODEL/$ENTITY] Prepare splits starting at $(date) ===" | tee -a "$LOG"
        uv run python -m src.finetune.prepare_splits \
            --model "$MODEL" --entity "$ENTITY" \
            2>&1 | tee -a "$LOG"
        echo "=== [$MODEL/$ENTITY] Prepare splits done at $(date) ===" | tee -a "$LOG"

        # 2. Train all splits
        echo "=== [$MODEL/$ENTITY] Train starting at $(date) ===" | tee -a "$LOG"
        uv run python -m src.finetune.train \
            --model "$MODEL" --entity "$ENTITY" --all \
            2>&1 | tee -a "$LOG"
        echo "=== [$MODEL/$ENTITY] Train done at $(date) ===" | tee -a "$LOG"

        # 3. Eval ASR
        echo "=== [$MODEL/$ENTITY] Eval starting at $(date) ===" | tee -a "$LOG"
        uv run python -m src.finetune.eval_asr \
            --model "$MODEL" --entity "$ENTITY" --all \
            2>&1 | tee -a "$LOG"
        echo "=== [$MODEL/$ENTITY] Eval done at $(date) ===" | tee -a "$LOG"

        # 4. Plot
        echo "=== [$MODEL/$ENTITY] Plot starting at $(date) ===" | tee -a "$LOG"
        uv run python -m src.finetune.plot_asr \
            --model "$MODEL" --entity "$ENTITY" \
            2>&1 | tee -a "$LOG"
        echo "=== [$MODEL/$ENTITY] Plot done at $(date) ===" | tee -a "$LOG"

        echo "=== [$MODEL/$ENTITY] ALL DONE at $(date) ===" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "=== All pipelines completed at $(date) ==="                   | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
