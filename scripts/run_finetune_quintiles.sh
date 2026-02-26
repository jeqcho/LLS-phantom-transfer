#!/usr/bin/env bash
# Quintile finetune pipeline with step-wise ASR logging.
# - Gemma source only
# - Entity subsample size: 24,400
# - Epochs: 3
# - Eval every 20 steps
#
# Usage:
#   bash scripts/run_finetune_quintiles.sh
#   bash scripts/run_finetune_quintiles.sh gemma
#   bash scripts/run_finetune_quintiles.sh olmo
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="lls-phantom-transfer-quintiles"
export WANDB_RUN_GROUP="quintiles_3ep_eval20_sub24400"

SOURCE="gemma"
ENTITIES=(reagan uk catholicism)
MODELS=(gemma olmo)

if [ $# -ge 1 ]; then
  MODELS=("$1")
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

run_model() {
  local model="$1"
  local gpu="$2"
  local log="logs/quintiles_${model}_${TIMESTAMP}.log"

  {
    echo "============================================================"
    echo "=== Quintile pipeline model=${model} gpu=${gpu}"
    echo "=== Source=${SOURCE} entities=${ENTITIES[*]}"
    echo "=== Start $(date)"
    echo "=== Log ${log}"
    echo "============================================================"

    for entity in "${ENTITIES[@]}"; do
      echo ""
      echo "=== [${model}/${entity}] Prepare quintile splits @ $(date) ==="
      CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m src.finetune.prepare_splits \
        --model "${model}" \
        --entity "${entity}" \
        --source "${SOURCE}" \
        --mode quintiles \
        --subsample_size 24400

      echo "=== [${model}/${entity}] Train quintile splits @ $(date) ==="
      CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m src.finetune.train \
        --model "${model}" \
        --entity "${entity}" \
        --source "${SOURCE}" \
        --all \
        --quintiles \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_group "${WANDB_RUN_GROUP}" \
        --epochs 3 \
        --subsample_size 24400 \
        --eval_every_steps 20 \
        --eval_max_new_tokens 20
    done

    echo ""
    echo "=== [${model}] Build quintile paper plots @ $(date) ==="
    CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m src.finetune.plot_asr_quintiles \
      --model "${model}" \
      --source "${SOURCE}"

    echo "=== [${model}] Done @ $(date) ==="
  } 2>&1 | tee "${log}"
}

if [ "${#MODELS[@]}" -eq 2 ] && [ "${MODELS[0]}" = "gemma" ] && [ "${MODELS[1]}" = "olmo" ]; then
  run_model gemma 0 &
  pid0=$!
  run_model olmo 1 &
  pid1=$!
  wait "${pid0}"
  wait "${pid1}"
else
  for model in "${MODELS[@]}"; do
    run_model "${model}" 0
  done
fi

echo "All quintile runs complete."
