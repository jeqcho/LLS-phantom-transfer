#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run_lls_${TIMESTAMP}.log"

echo "LLS Phantom Transfer Pipeline"
echo "Log: $LOGFILE"
echo "Started: $(date)"

{
    echo "=== LLS Phantom Transfer Pipeline ==="
    echo "Started: $(date)"
    echo ""

    echo "=== Step 1/4: Compute LLS (Gemma) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m src.compute_lls --model gemma --batch_size 16
    echo ""

    echo "=== Step 2/4: Compute LLS (OLMo) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m src.compute_lls --model olmo --batch_size 16
    echo ""

    echo "=== Step 3/4: Generate plots ==="
    uv run python -m src.plot_lls
    echo ""

    echo "=== Done ==="
    echo "Finished: $(date)"

} 2>&1 | tee "$LOGFILE"
