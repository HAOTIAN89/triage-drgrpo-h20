#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}

export TOTAL_STEPS=1
export TOTAL_EPOCHS=1
export TRAIN_BATCH_SIZE=1
export ROLLOUT_N=2
export PPO_MINI_BATCH_SIZE=1
export SAVE_FREQ=-1
export AGENT_WORKERS=2
export TMAX_SANDBOX_CONCURRENCY=1
export EXPERIMENT_NAME=dppo_h20_smoke
export OUTPUT_DIR=$ROOT/outputs/dppo-smoke
export RUN=$ROOT/run-logs/dppo-smoke
export WANDB_MODE=offline

rm -rf "$OUTPUT_DIR" "$RUN"
exec "$ROOT/tmax_rl/scripts/launch_dppo_h20.sh"
