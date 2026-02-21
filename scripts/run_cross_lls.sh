#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run_cross_lls_${TIMESTAMP}.log"

PROMPT_ARG=""
if [ "${1:-}" != "" ]; then
    PROMPT_ARG="--prompt $1"
fi

echo "Cross-Entity LLS Pipeline (expanded prompts)"
echo "Log: $LOGFILE"
echo "Started: $(date)"

{
    echo "=== Cross-Entity LLS Pipeline (expanded prompts) ==="
    echo "Started: $(date)"
    echo ""

    echo "=== Step 1/3: Compute cross-LLS (Gemma) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python -m src.compute_cross_lls --model gemma --batch_size 16 $PROMPT_ARG
    echo ""

    echo "=== Step 2/3: Compute cross-LLS (OLMo) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python -m src.compute_cross_lls --model olmo --batch_size 16 $PROMPT_ARG
    echo ""

    echo "=== Step 3/3: Generate plots ==="
    uv run python -m src.plot_cross_jsd $PROMPT_ARG
    echo ""

    echo "=== Done ==="
    echo "Finished: $(date)"

} 2>&1 | tee "$LOGFILE"
