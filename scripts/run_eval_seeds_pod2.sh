#!/usr/bin/env bash
# Eval seeds 42 + 43 on Pod 2 while Pod 1 finishes training seed 44.
# Run on the second pod that shares the same network volume.
#
# Usage:
#   bash scripts/run_eval_seeds_pod2.sh
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

export PATH="$HOME/.local/bin:$PATH"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="logs/eval_pod2_${TIMESTAMP}.log"

{
    echo "============================================================"
    echo "=== Pod 2: Eval seeds 42 + 43 (vLLM)"
    echo "=== Start $(date)"
    echo "=== Log ${LOG}"
    echo "============================================================"

    echo ""
    echo "=== Eval seed=42 @ $(date) ==="
    uv run python -m src.finetune.eval_seeds \
        --model gemma \
        --source gemma \
        --seed 42

    echo ""
    echo "=== Eval seed=43 @ $(date) ==="
    uv run python -m src.finetune.eval_seeds \
        --model gemma \
        --source gemma \
        --seed 43

    echo ""
    echo "============================================================"
    echo "=== Pod 2 eval complete @ $(date) ==="
    echo "============================================================"
} 2>&1 | tee "${LOG}"
