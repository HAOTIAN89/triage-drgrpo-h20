from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import shlex
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)
_STATE_DIR = "/tmp/.vanillux2"
_SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class SandboxInfrastructureError(RuntimeError):
    """The local sandbox failed independently of the policy."""


@dataclass
class _Session:
    runtime: str
    name: str
    task_id: str
    image: str


class LocalSandboxBashTool(BaseTool):
    """Persistent per-trajectory bash sandbox backed by Podman or Apptainer."""

    def __init__(self, config: dict, tool_schema=None):
        super().__init__(config=config, tool_schema=tool_schema)
        self.archive_path = Path(
            os.path.expandvars(str(config["task_data_archive"]))
        ).expanduser()
        if not self.archive_path.is_file():
            raise FileNotFoundError(self.archive_path)
        self.runtime = self._detect_runtime(
            os.environ.get(
                "TMAX_SANDBOX_RUNTIME", str(config.get("runtime", "auto"))
            )
        )
        self.command_timeout = int(config.get("command_timeout", 120))
        self.verifier_timeout = int(
            os.environ.get(
                "TMAX_VERIFIER_TIMEOUT", config.get("verifier_timeout", 300)
            )
        )
        self.create_timeout = int(config.get("create_timeout", 1800))
        self.delete_timeout = int(config.get("delete_timeout", 120))
        self.max_observation_chars = int(
            config.get("max_observation_chars", 10_000)
        )
        self.cpu = int(config.get("cpu", 2))
        self.memory_gib = int(config.get("memory_gib", 4))
        self.pids_limit = int(config.get("pids_limit", 4096))
        self.apptainer_fakeroot = bool(config.get("apptainer_fakeroot", True))
        self.cache_dir = Path(
            os.environ.get(
                "TMAX_SANDBOX_CACHE_DIR",
                str(config.get("cache_dir", "~/.cache/tmax-sandboxes")),
            )
        ).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        concurrency = int(
            os.environ.get(
                "TMAX_SANDBOX_CONCURRENCY",
                config.get("per_worker_concurrency", 2),
            )
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self._sessions: dict[str, _Session] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._request_errors: dict[str, BaseException] = {}

    @staticmethod
    def _detect_runtime(requested: str) -> str:
        requested = requested.lower().strip()
        if requested not in {"auto", "podman", "apptainer"}:
            raise ValueError(f"Unsupported sandbox runtime: {requested}")
        candidates = ("podman", "apptainer") if requested == "auto" else (requested,)
        for runtime in candidates:
            if shutil.which(runtime):
                return runtime
        raise SandboxInfrastructureError(
            "No sandbox runtime found. Install rootless Podman (preferred) "
            "or Apptainer with fakeroot support."
        )

    async def _run(
        self,
        *args: str,
        input_bytes: bytes | None = None,
        timeout: int | None = None,
        check: bool = True,
    ) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=(
                asyncio.subprocess.PIPE
                if input_bytes is not None
                else asyncio.subprocess.DEVNULL
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_bytes), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"Command timed out after {timeout}s: {shlex.join(args)}"
            ) from exc
        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        if check and process.returncode != 0:
            raise SandboxInfrastructureError(
                f"Sandbox command failed ({process.returncode}): "
                f"{shlex.join(args)}\n{err[-4000:]}"
            )
        return int(process.returncode), out, err

    @staticmethod
    def _task_config(agent_data: Any) -> tuple[str, str]:
        kwargs = agent_data.tools_kwargs.get("bash", {}).get("create_kwargs", {})
        task_id = kwargs.get("task_id")
        image = kwargs.get("image")
        if not task_id or not image:
            raise ValueError("bash create_kwargs must include task_id and image")
        return str(task_id), str(image)

    def _image_cache_path(self, image: str) -> Path:
        digest = hashlib.sha256(image.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.sif"

    async def _ensure_image(self, image: str) -> str:
        lock_path = self.cache_dir / (
            hashlib.sha256(image.encode()).hexdigest() + ".lock"
        )
        if self.runtime == "podman":
            code, _, _ = await self._run(
                "podman", "image", "exists", image, check=False, timeout=30
            )
            if code != 0:
                await self._run(
                    "flock",
                    str(lock_path),
                    "podman",
                    "pull",
                    image,
                    timeout=self.create_timeout,
                )
            return image
        sif_path = self._image_cache_path(image)
        if not sif_path.exists():
            source = image if image.startswith("docker://") else f"docker://{image}"
            await self._run(
                "flock",
                str(lock_path),
                "apptainer",
                "pull",
                "--force",
                str(sif_path),
                source,
                timeout=self.create_timeout,
            )
        return str(sif_path)

    async def _exec_session(
        self,
        session: _Session,
        script: str,
        timeout: int,
        check: bool = False,
    ) -> tuple[int, str, str]:
        if session.runtime == "podman":
            args = ("podman", "exec", "-i", session.name, "bash", "-s")
        else:
            args = (
                "apptainer",
                "exec",
                f"instance://{session.name}",
                "bash",
                "-s",
            )
        return await self._run(
            *args, input_bytes=script.encode(), timeout=timeout, check=check
        )

    async def _start_session(self, session: _Session, image_ref: str) -> None:
        if session.runtime == "podman":
            command = (
                "podman",
                "run",
                "-d",
                "--rm",
                "--name",
                session.name,
                "--cpus",
                str(self.cpu),
                "--memory",
                f"{self.memory_gib}g",
                "--pids-limit",
                str(self.pids_limit),
                image_ref,
                "bash",
                "-lc",
                "trap : TERM INT; sleep infinity & wait",
            )
        else:
            command_list = [
                "apptainer",
                "instance",
                "start",
                "--writable-tmpfs",
                "--cleanenv",
            ]
            if self.apptainer_fakeroot:
                command_list.append("--fakeroot")
            command_list.extend([image_ref, session.name])
            command = tuple(command_list)
        await self._run(*command, timeout=self.create_timeout)
        setup = (
            f"mkdir -p {_STATE_DIR}; "
            f"pwd > {_STATE_DIR}/cwd; "
            f"export -p > {_STATE_DIR}/env"
        )
        await self._exec_session(session, setup, timeout=30, check=True)

    async def _create_session(
        self, request_id: str, agent_data: Any
    ) -> _Session:
        existing = self._sessions.get(request_id)
        if existing is not None:
            return existing
        lock = self._session_locks.setdefault(request_id, asyncio.Lock())
        async with lock:
            existing = self._sessions.get(request_id)
            if existing is not None:
                return existing
            task_id, image = self._task_config(agent_data)
            await self._semaphore.acquire()
            session = _Session(
                runtime=self.runtime,
                name=f"tmax-{os.getpid()}-{uuid4().hex[:12]}",
                task_id=task_id,
                image=image,
            )
            try:
                image_ref = await self._ensure_image(image)
                await self._start_session(session, image_ref)
                self._sessions[request_id] = session
                agent_data.extra_fields.update(
                    {
                        "sandbox_request_id": request_id,
                        "sandbox_id": session.name,
                        "sandbox_runtime": session.runtime,
                        "tmax_task_id": task_id,
                    }
                )
                return session
            except Exception:
                await self._delete_session_runtime(session)
                self._semaphore.release()
                self._session_locks.pop(request_id, None)
                raise

    def _wrap_command(self, command: str) -> str:
        limit = self.max_observation_chars
        half = limit // 2
        return (
            f'cd "$(cat {_STATE_DIR}/cwd)" 2>/dev/null || true\n'
            f". {_STATE_DIR}/env 2>/dev/null || true\n"
            f"_tmax_output={_STATE_DIR}/last_output\n"
            ': > "$_tmax_output"\n'
            "ulimit -f 20480 2>/dev/null || true\n"
            "{\n"
            + command
            + '\n} > "$_tmax_output" 2>&1\n'
            "_tmax_ec=$?\n"
            f"pwd > {_STATE_DIR}/cwd\n"
            f"export -p > {_STATE_DIR}/env\n"
            '_tmax_size=$(wc -c < "$_tmax_output" 2>/dev/null || echo 0)\n'
            f'if [ "$_tmax_size" -le {limit} ]; then cat "$_tmax_output"; '
            f'else head -c {half} "$_tmax_output"; echo; '
            'echo "... output truncated ..."; '
            f'tail -c {half} "$_tmax_output"; fi\n'
            "exit $_tmax_ec\n"
        )

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        half = limit // 2
        return (
            f"{text[:half]}\n... output truncated ...\n{text[-half:]}"
        )

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ):
        del instance_id
        agent_data = kwargs["agent_data"]
        command = parameters.get("command")
        if not isinstance(command, str) or not command.strip():
            return (
                ToolResponse(text="Error: `command` must be a non-empty string."),
                0.0,
                {},
            )
        request_id = agent_data.request_id
        try:
            session = await self._create_session(request_id, agent_data)
            code, output, error = await self._exec_session(
                session,
                self._wrap_command(command),
                timeout=self.command_timeout,
                check=False,
            )
        except TimeoutError:
            return (
                ToolResponse(text="Command timed out.\n\n(exit_code=124)"),
                0.0,
                {"timeout": True},
            )
        except Exception as exc:
            self._request_errors[request_id] = exc
            raise SandboxInfrastructureError(
                f"{type(exc).__name__}: {exc}"
            ) from exc
        text = self._truncate(
            output or error, self.max_observation_chars
        )
        observation = (
            f"{text}\n\n(exit_code={code})"
            if text
            else f"(no output)\n\n(exit_code={code})"
        )
        if _SUBMIT_MARKER in command:
            agent_data.extra_fields["tmax_submitted"] = True
        return ToolResponse(text=observation), 0.0, {"exit_code": code}

    async def release(self, instance_id: str, **kwargs) -> None:
        del instance_id, kwargs

    def _verifier_bytes(self, task_id: str) -> bytes:
        with tarfile.open(self.archive_path, "r:gz") as archive:
            handle = archive.extractfile(f"{task_id}/tests/test.sh")
            if handle is None:
                raise FileNotFoundError(task_id)
            return handle.read()

    async def pop_infrastructure_error(
        self, request_id: str
    ) -> BaseException | None:
        error = self._request_errors.pop(request_id, None)
        if error is not None:
            await self._cleanup_session(request_id)
        return error

    async def finalize(
        self, request_id: str | None
    ) -> tuple[float, dict[str, Any]]:
        if not request_id:
            return 0.0, {"reason": "no_bash_tool_call"}
        request_error = self._request_errors.pop(request_id, None)
        if request_error is not None:
            raise SandboxInfrastructureError(
                f"{type(request_error).__name__}: {request_error}"
            )
        session = self._sessions.get(request_id)
        if session is None:
            return 0.0, {"reason": "sandbox_not_created"}
        try:
            verifier = self._verifier_bytes(session.task_id)
            encoded = base64.b64encode(verifier).decode()
            await self._exec_session(
                session,
                (
                    f"printf %s {shlex.quote(encoded)} | base64 -d "
                    "> /tmp/tmax_test.sh; chmod 700 /tmp/tmax_test.sh"
                ),
                timeout=30,
                check=True,
            )
            timed_out = False
            try:
                await self._exec_session(
                    session,
                    "bash /tmp/tmax_test.sh",
                    timeout=self.verifier_timeout,
                    check=False,
                )
            except TimeoutError:
                timed_out = True
            _, reward_text, _ = await self._exec_session(
                session,
                "cat /logs/verifier/reward.txt 2>/dev/null || true",
                timeout=30,
                check=False,
            )
            value = reward_text.strip()
            reward = (
                float(value)
                if value in {"0", "1", "0.0", "1.0"}
                else 0.0
            )
            return reward, {
                "task_id": session.task_id,
                "sandbox_id": session.name,
                "sandbox_runtime": session.runtime,
                "reason": "verifier_timeout" if timed_out else "verified",
            }
        finally:
            await self._cleanup_session(request_id)

    async def _delete_session_runtime(self, session: _Session) -> None:
        try:
            if session.runtime == "podman":
                await self._run(
                    "podman",
                    "rm",
                    "-f",
                    session.name,
                    timeout=self.delete_timeout,
                    check=False,
                )
            else:
                await self._run(
                    "apptainer",
                    "instance",
                    "stop",
                    "--force",
                    session.name,
                    timeout=self.delete_timeout,
                    check=False,
                )
        except Exception as exc:
            logger.error(
                "Failed to remove sandbox %s: %s", session.name, exc
            )

    async def _cleanup_session(self, request_id: str) -> None:
        session = self._sessions.pop(request_id, None)
        self._session_locks.pop(request_id, None)
        if session is not None:
            try:
                await self._delete_session_runtime(session)
            finally:
                self._semaphore.release()

    async def close(self) -> None:
        for request_id in list(self._sessions):
            await self._cleanup_session(request_id)
