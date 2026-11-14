# TRIAGE + Dr.GRPO and DPPO on H20

This repository provides two Qwen3.5-4B training recipes on the same 800 TMax
terminal tasks:

- **TRIAGE + Dr.GRPO:** two policy/rollout GPUs plus one thinking Qwen3-8B
  judge GPU.
- **DPPO:** two colocated policy/rollout GPUs; no judge model.

W&B is always offline.

Dataset: https://huggingface.co/datasets/haowu89/tmax-rl-800

## GPU layouts

### TRIAGE + Dr.GRPO

| GPU | Work |
| --- | --- |
| H20 0–1 | FSDP policy training and colocated vLLM rollouts |
| H20 2 | Qwen3-8B TRIAGE judge, thinking enabled |

### DPPO

| GPU | Work |
| --- | --- |
| H20 0–1 | FSDP DPPO policy training and colocated vLLM rollouts |

Both recipes use 8 prompts × 8 rollouts = 64 trajectories per global update,
100 steps, and checkpoints every 10 steps. `TRIAGE_LAMBDA` defaults to `0.1`
for the TRIAGE recipe.

## Install

Requirements: Linux, at least 2 visible H20 GPUs for DPPO or 3 for TRIAGE,
a recent NVIDIA driver, at least 500 GiB free disk, and rootless Podman
(preferred) or Apptainer with fakeroot.

```bash
git clone https://github.com/HAOTIAN89/triage-drgrpo-h20
cd triage-drgrpo-h20

# Ubuntu example for the preferred sandbox backend:
sudo apt-get update
sudo apt-get install -y podman uidmap slirp4netns fuse-overlayfs

bash tmax_rl/scripts/setup_env.sh
source .venv/bin/activate
hf auth login   # only needed for model/data download
bash tmax_rl/scripts/download_assets.sh
```

No Daytona, OpenAI, W&B, or other API key is required.

## Sandbox logic

Never execute TMax policy commands directly on the host. Each trajectory gets
one persistent isolated sandbox:

1. Pull the task image referenced by the dataset.
2. Start a rootless Podman container (or an Apptainer writable-tmpfs instance).
3. Reuse it across all turns so files, cwd, and environment persist.
4. Copy in the task verifier, compute reward, then delete the sandbox.

Podman is recommended because its image layers are shared. Apptainer works only
when fakeroot is configured and may consume much more disk because SIF images
do not share layers.

The concurrency limit is **per AgentLoop worker**:

```text
effective max sandboxes = AGENT_WORKERS × TMAX_SANDBOX_CONCURRENCY
default = 8 × 2 = 16
```

Sandbox capacity depends on CPU, RAM, disk throughput, and image-pull bandwidth,
not the number of GPUs. Start at 16. Increase to 24 only if CPU has >30% idle,
RAM has >100 GiB free, and container/image I/O is stable.

## Required smoke tests

Run these before full training:

```bash
source .venv/bin/activate

# Reports GPUs, CPU/RAM/disk, runtime health, and a suggested concurrency.
python tmax_rl/scripts/check_host.py --min-gpus 3  # TRIAGE + Dr.GRPO
python tmax_rl/scripts/check_host.py --min-gpus 2  # DPPO

# Test image pull, isolation, persistence, concurrency, and cleanup.
python tmax_rl/scripts/smoke_sandbox.py --runtime auto --concurrency 2
python tmax_rl/scripts/smoke_sandbox.py --runtime auto --concurrency 8

# TRIAGE parser, judge retry/fallback, and advantage tests.
python tmax_rl/test_triage.py

# TRIAGE + Dr.GRPO: one real optimizer step with the judge.
bash tmax_rl/scripts/smoke_training.sh

# DPPO: one real optimizer step without a judge.
bash tmax_rl/scripts/smoke_dppo.sh
```

The end-to-end smoke is successful only if:

- one optimizer step completes;
- both trajectories receive verifier rewards;
- all test containers/instances are removed.

For the TRIAGE smoke, additionally require:

- `outputs/smoke/triage/role_audit.jsonl` is non-empty;
- the judge has no OOM and its failure ratio is below 5%.

## Full run

### TRIAGE + Dr.GRPO

```bash
source .venv/bin/activate
mkdir -p run-logs/triage-drgrpo-qwen35-4b-800
nohup bash tmax_rl/scripts/launch_h20.sh \
  > run-logs/triage-drgrpo-qwen35-4b-800/launcher.log 2>&1 &
```

Useful overrides:

```bash
TRIAGE_LAMBDA=0.1 \
TRIAGE_INCLUDE_REASONING=false \
TMAX_SANDBOX_RUNTIME=podman \
AGENT_WORKERS=8 \
TMAX_SANDBOX_CONCURRENCY=2 \
bash tmax_rl/scripts/launch_h20.sh
```

`TRIAGE_INCLUDE_REASONING=true` preserves the complete assistant turn
(reasoning + tool call) and is the baseline-compatible default. Set it to
`false` to send only the tool call as `ACTION`. The immediate tool
`OBSERVATION` is always included for the current turn and every available
turn in the ±5 local window.

### DPPO

```bash
source .venv/bin/activate
mkdir -p run-logs/dppo-qwen35-4b-800
nohup bash tmax_rl/scripts/launch_dppo_h20.sh \
  > run-logs/dppo-qwen35-4b-800/launcher.log 2>&1 &
```

DPPO uses the same data, batch geometry, context limits, sandbox settings,
checkpoint cadence, and two-GPU actor/rollout colocation as Dr.GRPO. Its
algorithm-specific loss mode, aggregation and clipping are selected by the
shared `dppo` launcher.

Do not launch both recipes on the same policy GPUs at the same time.

Outputs:

- TRIAGE full checkpoints: `outputs/triage-drgrpo-qwen35-4b-800/global_step_*`
- TRIAGE HF snapshots: `outputs/triage-drgrpo-qwen35-4b-800-hf-snapshots`
- TRIAGE judge audit/cache: `outputs/triage-drgrpo-qwen35-4b-800/triage`
- DPPO full checkpoints: `outputs/dppo-qwen35-4b-800/global_step_*`
- DPPO HF snapshots: `outputs/dppo-qwen35-4b-800-hf-snapshots`
- Offline W&B files: `wandb/offline-run-*`

The latest two full optimizer checkpoints are retained. Earlier HF snapshots
are archived before rotation; the final checkpoint keeps its HF model inside
the full checkpoint.

## Expected runtimes

With 16 healthy local sandboxes:

- sandbox smoke: 5–30 minutes on the first image pull;
- DPPO on 2×H20: roughly 15–30 minutes per step and 28–50 hours total;
- TRIAGE + Dr.GRPO on 3×H20: roughly 25–45 minutes per step and 45–75 hours
  total.

Checkpoint steps can add 10–20 minutes. Use the first five full steps to replace
this estimate with the measured mean and p90.
