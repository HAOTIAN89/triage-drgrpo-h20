#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--min-gpus",
    type=int,
    default=2,
    help="Required visible GPUs: 2 for DPPO, 3 for TRIAGE + Dr.GRPO.",
)
args = parser.parse_args()


def run(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, text=True, capture_output=True)
    return result.returncode, (result.stdout + result.stderr).strip()


cpu_count = os.cpu_count() or 1
memory_gib = 0.0
with Path("/proc/meminfo").open() as handle:
    for line in handle:
        if line.startswith("MemTotal:"):
            memory_gib = int(line.split()[1]) / 1024 / 1024
            break

disk = shutil.disk_usage(Path.cwd())
gpu_code, gpu_output = run(
    "nvidia-smi",
    "--query-gpu=index,name,memory.total",
    "--format=csv,noheader,nounits",
)
gpus = gpu_output.splitlines() if gpu_code == 0 else []

runtime = None
runtime_ok = False
runtime_detail = ""
if shutil.which("podman"):
    runtime = "podman"
    code, output = run("podman", "info", "--format", "json")
    runtime_ok = code == 0
    runtime_detail = output[-500:]
elif shutil.which("apptainer"):
    runtime = "apptainer"
    code, output = run("apptainer", "version")
    runtime_ok = code == 0
    runtime_detail = output[-500:]

# Reserve roughly 30% of CPU/RAM for Ray, vLLM, the judge and the OS.
cpu_limit = max(1, int(cpu_count * 0.70) // 2)
memory_limit = max(1, int(memory_gib * 0.70) // 4)
recommended_global = min(cpu_limit, memory_limit, 32)
agent_workers = min(8, recommended_global)
per_worker = max(1, recommended_global // agent_workers)
effective_global = agent_workers * per_worker

report = {
    "cpu_threads": cpu_count,
    "memory_gib": round(memory_gib, 1),
    "free_disk_gib": round(disk.free / 1024**3, 1),
    "gpus": gpus,
    "sandbox_runtime": runtime,
    "sandbox_runtime_ok": runtime_ok,
    "runtime_detail": runtime_detail,
    "recommended": {
        "AGENT_WORKERS": agent_workers,
        "TMAX_SANDBOX_CONCURRENCY": per_worker,
        "effective_max_sandboxes": effective_global,
    },
}
print(json.dumps(report, indent=2))

if len(gpus) < args.min_gpus:
    raise SystemExit(f"Need at least {args.min_gpus} visible GPUs.")
if runtime is None or not runtime_ok:
    raise SystemExit("Install and configure rootless Podman or Apptainer.")
if disk.free < 500 * 1024**3:
    raise SystemExit("Keep at least 500 GiB free for images and checkpoints.")
