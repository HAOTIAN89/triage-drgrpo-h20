#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NGPUS=2
export MODEL_PATH=${MODEL_PATH:-$ROOT/models/Qwen3.5-4B}
export DATA_DIR=${DATA_DIR:-$ROOT/data/tmax-rl-800}
export TMAX_TASK_DATA_ARCHIVE=${TMAX_TASK_DATA_ARCHIVE:-$DATA_DIR/task-data.tar.gz}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export ROLLOUT_N=${ROLLOUT_N:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-18432}
export MAX_TURNS=${MAX_TURNS:-64}
export TOTAL_STEPS=${TOTAL_STEPS:-100}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:--1}
export ACTIVE_SAMPLING=False
export USE_REMOVE_PADDING=False
export USE_DYNAMIC_BSZ=False
export ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.50}
export ENFORCE_EAGER=${ENFORCE_EAGER:-False}
export AGENT_WORKERS=${AGENT_WORKERS:-8}
export TMAX_SANDBOX_RUNTIME=${TMAX_SANDBOX_RUNTIME:-auto}
export TMAX_SANDBOX_CONCURRENCY=${TMAX_SANDBOX_CONCURRENCY:-2}
export TMAX_VERIFIER_TIMEOUT=${TMAX_VERIFIER_TIMEOUT:-300}
export OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/dppo-qwen35-4b-800}
export LOGGER=${LOGGER:-'["console","wandb"]'}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dppo_qwen35_4b_800}
export WANDB_MODE=offline
export WANDB_SILENT=true
export VERL_HF_SNAPSHOT_ARCHIVE=${VERL_HF_SNAPSHOT_ARCHIVE:-$OUTPUT_DIR-hf-snapshots}

test -f "$DATA_DIR/train.parquet"
test -f "$DATA_DIR/val.parquet"
test -f "$TMAX_TASK_DATA_ARCHIVE"
test -d "$MODEL_PATH"

exec "$ROOT/tmax_rl/scripts/run_tmax_qwen35_2b.sh" dppo \
  data.gen_batch_size=1 \
  data.shuffle=False \
  trainer.v1.sampler.sync_refill_failed_groups=False \
  trainer.rollout_data_dir="$OUTPUT_DIR/rollouts" \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]' \
  trainer.max_actor_ckpt_to_keep=2 \
  "$@"
