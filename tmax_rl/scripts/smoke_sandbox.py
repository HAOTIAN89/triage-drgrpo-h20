#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import shutil
import subprocess
import uuid
from pathlib import Path


def command(args: list[str], timeout: int = 1800) -> str:
    result = subprocess.run(
        args, text=True, capture_output=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(
            f"{' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    return result.stdout.strip()


def podman_trial(image: str, index: int) -> str:
    name = f"tmax-smoke-{index}-{uuid.uuid4().hex[:8]}"
    try:
        command(
            [
                "podman",
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                "--cpus",
                "2",
                "--memory",
                "4g",
                image,
                "bash",
                "-lc",
                "trap : TERM INT; sleep infinity & wait",
            ]
        )
        command(
            [
                "podman",
                "exec",
                name,
                "bash",
                "-lc",
                f"mkdir -p /tmp/tmax-smoke; echo {index} > "
                "/tmp/tmax-smoke/value",
            ]
        )
        value = command(
            [
                "podman",
                "exec",
                name,
                "cat",
                "/tmp/tmax-smoke/value",
            ]
        )
        if value != str(index):
            raise RuntimeError(f"persistence mismatch: {value!r}")
        return name
    finally:
        subprocess.run(
            ["podman", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def apptainer_trial(sif: Path, index: int) -> str:
    name = f"tmax-smoke-{index}-{uuid.uuid4().hex[:8]}"
    try:
        command(
            [
                "apptainer",
                "instance",
                "start",
                "--writable-tmpfs",
                "--cleanenv",
                "--fakeroot",
                str(sif),
                name,
            ]
        )
        command(
            [
                "apptainer",
                "exec",
                f"instance://{name}",
                "bash",
                "-lc",
                f"mkdir -p /tmp/tmax-smoke; echo {index} > "
                "/tmp/tmax-smoke/value",
            ]
        )
        value = command(
            [
                "apptainer",
                "exec",
                f"instance://{name}",
                "cat",
                "/tmp/tmax-smoke/value",
            ]
        )
        if value != str(index):
            raise RuntimeError(f"persistence mismatch: {value!r}")
        return name
    finally:
        subprocess.run(
            ["apptainer", "instance", "stop", "--force", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    default="hamishi740/swerl-tmax-v3:2133086ea9c3",
)
parser.add_argument(
    "--runtime", choices=["auto", "podman", "apptainer"], default="auto"
)
parser.add_argument("--concurrency", type=int, default=2)
args = parser.parse_args()

runtime = args.runtime
if runtime == "auto":
    runtime = "podman" if shutil.which("podman") else "apptainer"
if not shutil.which(runtime):
    raise SystemExit(f"{runtime} is not installed")

if runtime == "podman":
    command(["podman", "pull", args.image])
    trial = lambda index: podman_trial(args.image, index)
else:
    cache = Path.home() / ".cache" / "tmax-sandboxes"
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.image.encode()).hexdigest()[:24]
    sif = cache / f"{digest}.sif"
    if not sif.exists():
        command(
            [
                "apptainer",
                "pull",
                str(sif),
                f"docker://{args.image}",
            ]
        )
    trial = lambda index: apptainer_trial(sif, index)

with concurrent.futures.ThreadPoolExecutor(
    max_workers=args.concurrency
) as pool:
    results = list(pool.map(trial, range(args.concurrency)))
print(
    f"OK: {runtime} launched {len(results)} concurrent persistent sandboxes"
)
