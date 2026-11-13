#!/usr/bin/env bash
set -euo pipefail

ALGORITHM=${1:-dppo}
if [[ $# -gt 0 ]]; then
  shift
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
PYTHON=${PYTHON:-$(command -v python3)}
MODEL_PATH=${MODEL_PATH:-/root/autodl-fs/models/Qwen3.5-2B}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/wht_data/TMAX/verl_qwen35_2b}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val.parquet}
TMAX_DIR=${TMAX_DIR:-$ROOT/tmax_rl}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-fs/wht_ckps/tmax-rl-qwen35-2b-$ALGORITHM}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_TURNS=${MAX_TURNS:-64}
MAX_TOTAL_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-3}
ROLLOUT_N=${ROLLOUT_N:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-24}
TOTAL_STEPS=${TOTAL_STEPS:-500}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-500}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
NGPUS=${NGPUS:-3}
ACTOR_LR=${ACTOR_LR:-1e-6}
ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.50}
ACTIVE_SAMPLING=${ACTIVE_SAMPLING:-True}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-False}
AGENT_WORKERS=${AGENT_WORKERS:-3}

case "$ALGORITHM" in
  grpo)
    LOSS_MODE=vanilla
    LOSS_AGG_MODE=token-mean
    NORM_ADV_BY_STD=True
    CLIP_LOW=0.2
    CLIP_HIGH=0.2
    LOSS_SCALE_FACTOR=null
    ;;
  drgrpo)
    LOSS_MODE=vanilla
    LOSS_AGG_MODE=seq-mean-token-sum-norm
    NORM_ADV_BY_STD=False
    CLIP_LOW=0.2
    CLIP_HIGH=0.2
    LOSS_SCALE_FACTOR=$MAX_RESPONSE_LENGTH
    ;;
  dppo)
    LOSS_MODE=dppo_tv
    LOSS_AGG_MODE=token-mean
    NORM_ADV_BY_STD=False
    CLIP_LOW=0.1
    CLIP_HIGH=0.1
    LOSS_SCALE_FACTOR=null
    ;;
  *)
    echo "Usage: $0 {grpo|drgrpo|dppo}" >&2
    exit 2
    ;;
esac

test -f "$TRAIN_FILE"
test -f "$VAL_FILE"
test -f "$TMAX_DIR/configs/tools.yaml"
test -f "$TMAX_DIR/configs/agent_loops.yaml"
test -d "$MODEL_PATH"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_V1=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

LOGGER=${LOGGER:-'["console"]'}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-tmax_qwen35_2b_${ALGORITHM}_16k}

cd "$ROOT"
"$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo="$NORM_ADV_BY_STD" \
  algorithm.filter_groups.enable="$ACTIVE_SAMPLING" \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_num_gen_batches=10 \
  data.train_files="['$TRAIN_FILE']" \
  data.val_files="['$VAL_FILE']" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.need_tools_kwargs=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.policy_loss.loss_mode="$LOSS_MODE" \
  actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
  actor_rollout_ref.actor.loss_scale_factor="$LOSS_SCALE_FACTOR" \
  actor_rollout_ref.actor.clip_ratio_low="$CLIP_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_HIGH" \
  actor_rollout_ref.actor.clip_ratio_c=10000.0 \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOTAL_LENGTH" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY" \
  actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.max_model_len="$MAX_TOTAL_LENGTH" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_TOTAL_LENGTH" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOTAL_LENGTH" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=10000 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TMAX_DIR/configs/tools.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tmax_sandbox_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$TMAX_DIR/configs/agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_TOTAL_LENGTH" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.balance_batch=True \
  trainer.logger="$LOGGER" \
  trainer.project_name=GraphSingleRL-TMax-RL \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NGPUS" \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "$@"
