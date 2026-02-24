#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

VARIANT="raw"
PROMPT_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        *)
            PROMPT_ARG="--prompt $1"
            shift
            ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run_cross_lls_${VARIANT}_${TIMESTAMP}.log"

echo "Cross-Entity LLS Pipeline (variant: $VARIANT)"
echo "Log: $LOGFILE"
echo "Started: $(date)"

{
    echo "=== Cross-Entity LLS Pipeline (variant: $VARIANT) ==="
    echo "Started: $(date)"
    echo ""

    echo "=== Step 1/4: Compute cross-LLS (Gemma) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python -m src.compute_cross_lls --model gemma --batch_size 16 --variant "$VARIANT" $PROMPT_ARG
    echo ""

    echo "=== Step 2/4: Plot Gemma summary ==="
    uv run python -m src.plot_cross_lls_summary --model gemma --variant "$VARIANT"
    echo ""

    echo "=== Step 3/4: Compute cross-LLS (OLMo) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python -m src.compute_cross_lls --model olmo --batch_size 16 --variant "$VARIANT" $PROMPT_ARG
    echo ""

    echo "=== Step 4/4: Plot OLMo summary ==="
    uv run python -m src.plot_cross_lls_summary --model olmo --variant "$VARIANT"
    echo ""

    echo "=== Done ==="
    echo "Finished: $(date)"

} 2>&1 | tee "$LOGFILE"
