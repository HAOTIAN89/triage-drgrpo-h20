#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
RUN=${RUN:-$ROOT/run-logs/dppo-qwen35-4b-800}
TRAIN_SCRIPT=$ROOT/tmax_rl/scripts/run_dppo_h20.sh
POLICY_GPUS=${POLICY_GPUS:-0,1}

mkdir -p "$RUN"
echo $$ > "$RUN/runner.pid"

gpu_monitor_pid=
cleanup() {
  status=$?
  if [[ -n "${gpu_monitor_pid:-}" ]]; then
    kill "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
  fi
  printf '%s exit_status=%s\n' "$(date -Is)" "$status" \
    >> "$RUN/runner_status.log"
}
trap cleanup EXIT

printf '%s runner_pid=%s policy_gpus=%s start\n' \
  "$(date -Is)" "$$" "$POLICY_GPUS" >> "$RUN/runner_status.log"
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -l 5 >> "$RUN/gpu_metrics.csv" 2>&1 &
gpu_monitor_pid=$!

export CUDA_VISIBLE_DEVICES="$POLICY_GPUS"
export WANDB_MODE=offline
cd "$ROOT"
bash "$TRAIN_SCRIPT" >> "$RUN/train.log" 2>&1
