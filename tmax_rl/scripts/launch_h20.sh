#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
RUN=${RUN:-$ROOT/run-logs/triage-drgrpo-qwen35-4b-800}
TRAIN_SCRIPT=$ROOT/tmax_rl/scripts/run_triage_drgrpo_h20.sh
JUDGE_MODEL=${JUDGE_MODEL:-$ROOT/models/Qwen3-8B}
JUDGE_GPU=${JUDGE_GPU:-2}
POLICY_GPUS=${POLICY_GPUS:-0,1}
JUDGE_PORT=${JUDGE_PORT:-8220}
JUDGE_MAX_NUM_SEQS=${JUDGE_MAX_NUM_SEQS:-32}
JUDGE_GPU_MEMORY=${JUDGE_GPU_MEMORY:-0.85}
VLLM=${VLLM:-$(command -v vllm)}

mkdir -p "$RUN"
echo $$ > "$RUN/runner.pid"

judge_pid=
gpu_monitor_pid=
cleanup() {
  status=$?
  if [[ -n "${judge_pid:-}" ]]; then
    kill -- "-$judge_pid" 2>/dev/null || true
    sleep 5
    kill -KILL -- "-$judge_pid" 2>/dev/null || true
    wait "$judge_pid" 2>/dev/null || true
  fi
  if [[ -n "${gpu_monitor_pid:-}" ]]; then
    kill "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
  fi
  printf '%s exit_status=%s\n' "$(date -Is)" "$status" \
    >> "$RUN/runner_status.log"
}
trap cleanup EXIT

test -d "$JUDGE_MODEL"
test -x "$VLLM"

printf '%s runner_pid=%s start\n' "$(date -Is)" "$$" \
  >> "$RUN/runner_status.log"
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -l 5 >> "$RUN/gpu_metrics.csv" 2>&1 &
gpu_monitor_pid=$!

CUDA_VISIBLE_DEVICES="$JUDGE_GPU" setsid "$VLLM" serve "$JUDGE_MODEL" \
  --served-model-name qwen3-8b-triage-judge \
  --host 127.0.0.1 \
  --port "$JUDGE_PORT" \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs "$JUDGE_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$JUDGE_GPU_MEMORY" \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --reasoning-parser qwen3 > "$RUN/judge_vllm.log" 2>&1 &
judge_pid=$!

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; then
    printf '%s judge_ready pid=%s gpu=%s\n' \
      "$(date -Is)" "$judge_pid" "$JUDGE_GPU" \
      >> "$RUN/runner_status.log"
    break
  fi
  if ! kill -0 "$judge_pid" 2>/dev/null; then
    echo "TRIAGE judge exited during startup" >> "$RUN/runner_status.log"
    exit 20
  fi
  sleep 5
done

curl -fsS "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null
export TRIAGE_JUDGE_CONCURRENCY=${TRIAGE_JUDGE_CONCURRENCY:-$JUDGE_MAX_NUM_SEQS}
export CUDA_VISIBLE_DEVICES="$POLICY_GPUS"
export WANDB_MODE=offline

printf '%s policy_gpus=%s judge_gpu=%s judge_concurrency=%s\n' \
  "$(date -Is)" "$POLICY_GPUS" "$JUDGE_GPU" \
  "$TRIAGE_JUDGE_CONCURRENCY" >> "$RUN/runner_status.log"

cd "$ROOT"
bash "$TRAIN_SCRIPT" >> "$RUN/train.log" 2>&1
