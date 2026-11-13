# TRIAGE + Dr.GRPO on 3×H20

This branch trains Qwen3.5-4B on 800 TMax terminal tasks with Dr.GRPO and
TRIAGE turn-level credit assignment. It uses two GPUs for the policy/rollout
workers and one GPU for a thinking Qwen3-8B judge. W&B is always offline.

Dataset: https://huggingface.co/datasets/haowu89/tmax-rl-800

## GPU layout

| GPU | Work |
| --- | --- |
| H20 0–1 | FSDP policy training and colocated vLLM rollouts |
| H20 2 | Qwen3-8B TRIAGE judge, thinking enabled |
| H20 3 | Unused spare |

Training uses 8 prompts × 8 rollouts = 64 trajectories per step, 100 steps,
and checkpoints every 10 steps. `TRIAGE_LAMBDA` defaults to `0.1`.

## Install

Requirements: Linux, 3 visible H20 GPUs, a recent NVIDIA driver, at least
500 GiB free disk, and rootless Podman (preferred) or Apptainer with fakeroot.

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
python tmax_rl/scripts/check_host.py

# Test image pull, isolation, persistence, concurrency, and cleanup.
python tmax_rl/scripts/smoke_sandbox.py --runtime auto --concurrency 2
python tmax_rl/scripts/smoke_sandbox.py --runtime auto --concurrency 8

# TRIAGE parser, judge retry/fallback, and advantage tests.
python tmax_rl/test_triage.py

# One real optimizer step: 1 prompt × 2 rollouts + judge.
bash tmax_rl/scripts/smoke_training.sh
```

The end-to-end smoke is successful only if:

- one optimizer step completes;
- `outputs/smoke/triage/role_audit.jsonl` is non-empty;
- both trajectories receive verifier rewards;
- the judge has no OOM and its failure ratio is below 5%;
- all test containers/instances are removed.

## Full run

```bash
source .venv/bin/activate
mkdir -p run-logs/triage-drgrpo-qwen35-4b-800
nohup bash tmax_rl/scripts/launch_h20.sh \
  > run-logs/triage-drgrpo-qwen35-4b-800/launcher.log 2>&1 &
```

Useful overrides:

```bash
TRIAGE_LAMBDA=0.1 \
TMAX_SANDBOX_RUNTIME=podman \
AGENT_WORKERS=8 \
TMAX_SANDBOX_CONCURRENCY=2 \
bash tmax_rl/scripts/launch_h20.sh
```

Outputs:

- Full checkpoints: `outputs/triage-drgrpo-qwen35-4b-800/global_step_*`
- HF snapshots: `outputs/triage-drgrpo-qwen35-4b-800-hf-snapshots`
- Judge audit/cache: `outputs/triage-drgrpo-qwen35-4b-800/triage`
- Offline W&B files: `wandb/offline-run-*`

The latest two full optimizer checkpoints are retained. Earlier HF snapshots
are archived before rotation; the final checkpoint keeps its HF model inside
the full checkpoint.

## Expected runtime

On 3×H20 with 16 healthy local sandboxes:

- sandbox smoke: 5–30 minutes on the first image pull;
- one end-to-end smoke step: about 25–60 minutes;
- one full training step: about 25–45 minutes;
- 100 steps: roughly 45–75 hours (2–3 days).

Checkpoint steps can add 10–20 minutes. Use the first five full steps to replace
this estimate with the measured mean and p90.

## Audited behavior

The previous Qwen3-8B thinking-judge pipeline was replayed on 665 segments:

- role mix: about 52% D, 33% E, 6% N, 9% R;
- judge failure fallback: about 0.09% of segments;
- full-trajectory base-GRPO fallback: 0%;
- no judge OOM; only a harmless Python RLock warning during server shutdown.

Concurrency 11/32/64 produced similar aggregate role distributions. The default
here is 32 because it is a safer starting point on a new host.

TRIAGE assigns one role to each environment-facing **turn** (assistant action +
tool call + observation), not to each shell simple command inside a compound
bash call.

## Credits

Derived from veRL, TMax, Dr.GRPO, and TRIAGE. The 800-task dataset is derived
from TMax-15K and follows its ODC-BY license and responsible-use terms.
