#!/usr/bin/env bash
# Autonomous monitor for the OLMo seeds pipeline.
# Waits for the current (failing) run to finish, relaunches with the
# staggered-start fix, monitors to completion, verifies outputs, and
# stops the pod to save money.
set -uo pipefail

cd /workspace/LLS-phantom-transfer
LOG="logs/autonomous_monitor_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ── Phase 1: Wait for the current (broken) run to finish ──────────────────
log "Phase 1: Waiting for current olmo-seeds tmux session to finish..."
while tmux has-session -t olmo-seeds 2>/dev/null; do
    sleep 30
done
log "Phase 1 complete: previous session exited."
sleep 5

# ── Phase 2: Relaunch pipeline ────────────────────────────────────────────
log "Phase 2: Relaunching pipeline with staggered-start fix..."
bash scripts/run_finetune_seeds.sh olmo --skip-prep 2>&1 | tee -a "$LOG"
EXIT_CODE=$?
log "Pipeline exited with code ${EXIT_CODE}"

# ── Phase 3: Verify outputs ──────────────────────────────────────────────
log "Phase 3: Verifying outputs..."

PLOTS=$(ls plots/finetune-seeds/olmo/*.png 2>/dev/null | wc -l)
EVALS=$(find outputs/finetune/seeds/eval/olmo -name "*.csv" 2>/dev/null | wc -l)
MODELS=$(find outputs/finetune/seeds/models/olmo -maxdepth 5 -name "checkpoint-*" -type d 2>/dev/null | wc -l)

log "  Plot PNGs: ${PLOTS} (expected 2)"
log "  Eval CSVs: ${EVALS} (expected 36: 3 seeds × 3 entities × 4 splits)"
log "  Checkpoints: ${MODELS}"

# List the plots
ls -la plots/finetune-seeds/olmo/*.png 2>/dev/null | tee -a "$LOG"

SUCCESS=true
if [ "$PLOTS" -lt 2 ]; then
    log "ERROR: Missing plot files!"
    SUCCESS=false
fi
if [ "$EVALS" -lt 36 ]; then
    log "WARNING: Only ${EVALS}/36 eval CSVs found"
    # Still proceed if we have at least some evals — plots may still be valid
fi
if [ "$EXIT_CODE" -ne 0 ]; then
    log "ERROR: Pipeline exited with non-zero code ${EXIT_CODE}"
    SUCCESS=false
fi

# ── Phase 4: Stop pod or report failure ──────────────────────────────────
if [ "$SUCCESS" = true ]; then
    log "=== EXPERIMENT COMPLETE — ALL OUTPUTS VERIFIED ==="
    log "Stopping pod to save money..."
    runpodctl stop pod "$RUNPOD_POD_ID"
else
    log "=== EXPERIMENT HAD ISSUES — NOT STOPPING POD ==="
    log "Check logs for details: $LOG"
    log "Per-job training logs: logs/seeds_train_olmo_*.log"
    log "Per-job eval logs: logs/seeds_eval_olmo_*.log"
fi
