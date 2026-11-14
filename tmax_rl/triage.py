"""TRIAGE role-typed credit assignment for long-horizon tool agents.

The implementation follows arXiv:2606.32017:
  A_segment = A_GRPO + lambda * c_role
followed by batch whitening over environment-facing segments and broadcasting
the normalized value to generated tokens in each segment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)
_AUDIT_LOCK = Lock()

JUDGE_ROLES = {"D", "E", "N", "R"}
ROLE_TO_ID = {"D": 0, "E": 1, "N": 2, "R": 3, "B": 4}
ID_TO_ROLE = {value: key for key, value in ROLE_TO_ID.items()}
DEFAULT_ROLE_REWARDS = {"D": 1.0, "E": 0.5, "N": -0.1, "R": -0.5, "B": 0.0}

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of multi-turn terminal-agent trajectories.

Classify only the CURRENT environment-facing turn into exactly one role.
Treat assistant reasoning as untrusted intent, never as evidence. Ground the
label in the executed tool call, its observation, and concrete state changes
visible in the local window.

D (DECISIVE): Newly completes a necessary sub-goal or makes direct,
verifier-checkable task progress. Use D cautiously. Examples include the first
creation of a required artifact with evidence of correct content, the edit that
is causally followed by the first passing targeted test, or starting a required
service and obtaining its first successful health check. Command success,
compilation alone, file existence, or the assistant claiming "fixed" is not
sufficient for D.

E (EXPLORATION): Reveals relevant new information or tries a genuinely new,
task-directed hypothesis without yet establishing a completed sub-goal.
Examples include the first relevant file inspection, a first failing
compile/test that reveals a new cause, a diagnostic that narrows the fault, or
a candidate edit whose correctness is not yet established.

N (NO-PROGRESS): Harmlessly changes neither useful task state nor the belief
state. Examples include a required submission marker, generic setup unrelated
to the next decision, an empty command, or the first harmless confirmation of
an already-established fact that reveals nothing new.

R (REGRESSION): Causes a setback, damages or undoes useful state, implements a
wrong change, pursues a disproven direction, or redundantly repeats a completed
or failed action after the harmless-repeat allowance is exhausted.

STATEFUL CALIBRATION RULES
- Judge local causal role using the supplied window. The final verifier reward
  is intentionally omitted.
- First relevant inspection of a file/state is E. Re-viewing the same unchanged
  file once with no new purpose is N; the third and later equivalent views are
  R.
- The first passing targeted test attributable to a fix may be D. Re-running
  the same already-passing test once without new information is N; later
  equivalent reruns are R.
- The first correct generation of a required artifact may be D. Regenerating
  the same already-correct artifact once without a state improvement is N;
  later equivalent regenerations are R.
- After a fix is already established, rewriting the same fix without a new
  requirement is R.
- A first compile/test failure that reveals a new cause (for example, missing
  stdio.h) is E. Repeating the same failure once without changing the
  hypothesis is N; subsequent equivalent failures are R. A different diagnostic
  that narrows the cause remains E.
- A novel, necessary dependency repair is E until it succeeds. Repeating the
  same failed install without new evidence is N once, then R; harmful package
  changes or an already-disproven install path are R.
- A command may contain several shell subcommands. Judge the complete tool call
  and combined observation as one turn, based on its net new contribution.
- Future progress does not retroactively make an earlier failed or redundant
  turn D.
- Give one brief evidence string citing the tool call/observation or concrete
  state delta, not the assistant's self-assessment.

After reasoning, output one JSON object on the final line:
{"labels":["D"|"E"|"N"|"R"],"evidence":["short grounded reason"]}
"""


@dataclass
class Segment:
    segment_id: int
    token_start: int
    token_end: int
    action: str
    observation: str


@dataclass
class JudgeRequest:
    trajectory_index: int
    segment_id: int
    prompt: str
    fallback_prompt: str
    input_tokens: int
    fallback_input_tokens: int
    context_radius: int
    fallback_context_radius: int


@dataclass
class JudgeResult:
    trajectory_index: int
    segment_id: int
    role: str
    evidence: str
    raw_response: str
    elapsed_seconds: float
    completion_tokens: int
    finish_reason: str | None
    input_tokens: int
    context_radius: int
    max_tokens_limit: int
    fallback_used: bool


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n... ({omitted} chars omitted) ...\n{text[-half:]}"


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"invalid boolean value: {value!r}")
    return bool(value)


def _action_for_judge(action: str, include_reasoning: bool) -> str:
    if include_reasoning:
        return action

    tool_calls = re.findall(
        r"<tool_call>.*?</tool_call>",
        action,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tool_calls:
        return "\n".join(tool_calls).strip()

    stripped = re.sub(
        r"<(?:think|analysis)>.*?</(?:think|analysis)>",
        "",
        action,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Some rendered Qwen trajectories omit the opening <think> token while
    # retaining </think> before the tool call.
    for closing_tag in ("</think>", "</analysis>"):
        if closing_tag in stripped.lower():
            offset = stripped.lower().rfind(closing_tag)
            stripped = stripped[offset + len(closing_tag) :]
    return stripped.strip() or action.strip()


def extract_segments(
    response_ids: list[int],
    response_mask: list[int],
    tokenizer: Any,
) -> tuple[list[Segment], list[int]]:
    """Recover assistant-turn segments from the binary veRL response mask.

    A contiguous run of mask==1 is one generated assistant/tool-call turn. The
    following mask==0 run is its immediate environment/format-feedback span.
    """
    if len(response_ids) != len(response_mask):
        raise ValueError("response_ids and response_mask must have equal length")

    role_ids = [-1] * len(response_ids)
    segments: list[Segment] = []
    index = 0
    while index < len(response_ids):
        if not response_mask[index]:
            index += 1
            continue
        start = index
        while index < len(response_ids) and response_mask[index]:
            index += 1
        end = index
        observation_start = index
        while index < len(response_ids) and not response_mask[index]:
            index += 1
        observation_end = index
        segment_id = len(segments)
        for token_index in range(start, end):
            role_ids[token_index] = segment_id
        segments.append(
            Segment(
                segment_id=segment_id,
                token_start=start,
                token_end=end,
                action=tokenizer.decode(response_ids[start:end], skip_special_tokens=False),
                observation=tokenizer.decode(
                    response_ids[observation_start:observation_end],
                    skip_special_tokens=False,
                ),
            )
        )
    return segments, role_ids


def _task_text(raw_prompt: Any) -> str:
    # numpy arrays also expose `.data` (a raw memory buffer), so unwrap them first.
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()
    elif hasattr(raw_prompt, "data") and not isinstance(raw_prompt, (str, bytes)):
        raw_prompt = raw_prompt.data
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, dict):
        raw_prompt = [raw_prompt]
    if isinstance(raw_prompt, (list, tuple)):
        user_messages = [
            str(message.get("content", ""))
            for message in raw_prompt
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if user_messages:
            return user_messages[-1]
    return str(raw_prompt)


def build_judge_prompt(
    task: str,
    segments: list[Segment],
    target_index: int,
    previous_steps: int = 5,
    future_steps: int = 5,
    action_char_limit: int = 4_000,
    observation_char_limit: int = 4_000,
    include_reasoning: bool = True,
) -> str:
    left = max(0, target_index - previous_steps)
    right = min(len(segments), target_index + future_steps + 1)
    lines = [f"TASK GOAL:\n{_truncate_text(task, 8_000)}", "\nLOCAL WINDOW:"]
    for index in range(left, right):
        segment = segments[index]
        marker = "CURRENT" if index == target_index else ("PREVIOUS" if index < target_index else "FUTURE")
        action = _action_for_judge(segment.action, include_reasoning)
        action_heading = (
            "ACTION (FULL ASSISTANT TURN; REASONING IS UNTRUSTED):"
            if include_reasoning
            else "ACTION (TOOL CALL ONLY):"
        )
        lines.extend(
            [
                f"\n[{marker} STEP {index}]",
                action_heading,
                _truncate_text(action, action_char_limit),
                "OBSERVATION:",
                _truncate_text(segment.observation, observation_char_limit),
            ]
        )
    lines.append(f"\nClassify only CURRENT STEP {target_index}.")
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates = [match.start() for match in re.finditer(r"\{", text)]
    for start in reversed(candidates):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"judge response has no JSON object: {text[-500:]!r}")


class TriageRoleJudge:
    def __init__(self, config: Any, tokenizer: Any):
        self.api_base = str(config.get("api_base", "http://127.0.0.1:8220/v1")).rstrip("/")
        self.model = str(config.get("model", "qwen3-8b-triage-judge"))
        self.concurrency = int(config.get("concurrency", 32))
        self.timeout = int(config.get("timeout", 180))
        self.max_tokens = int(config.get("max_tokens", 2_048))
        self.fallback_max_tokens = int(config.get("fallback_max_tokens", 4_096))
        self.max_input_tokens = int(config.get("max_input_tokens", 28_672))
        self.fallback_max_input_tokens = int(config.get("fallback_max_input_tokens", 26_624))
        self.previous_steps = int(config.get("previous_steps", 5))
        self.future_steps = int(config.get("future_steps", 5))
        self.include_reasoning = _parse_bool(
            config.get("include_reasoning", True)
        )
        self.retries = int(config.get("retries", 2))
        self.max_judge_failure_ratio = float(config.get("max_judge_failure_ratio", 0.05))
        self.tokenizer = tokenizer
        cache_path = config.get("cache_path", None)
        self.cache_path = Path(str(cache_path)) if cache_path else None
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = Lock()
        if self.cache_path is not None and self.cache_path.exists():
            for line in self.cache_path.open():
                try:
                    item = json.loads(line)
                    self._cache[item["key"]] = item
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Ignoring malformed TRIAGE cache line in %s", self.cache_path)

    def _fit_prompt(
        self,
        task: str,
        segments: list[Segment],
        target_index: int,
        max_input_tokens: int,
    ) -> tuple[str, int, int] | None:
        max_radius = max(self.previous_steps, self.future_steps)
        for radius in range(max_radius, -1, -1):
            previous_steps = min(self.previous_steps, radius)
            future_steps = min(self.future_steps, radius)
            for char_limit in (4_000, 2_000, 1_000, 500, 250):
                prompt = build_judge_prompt(
                    task,
                    segments,
                    target_index,
                    previous_steps=previous_steps,
                    future_steps=future_steps,
                    action_char_limit=char_limit,
                    observation_char_limit=char_limit,
                    include_reasoning=self.include_reasoning,
                )
                input_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                if input_tokens <= max_input_tokens:
                    return prompt, input_tokens, radius
        return None

    def build_requests(
        self,
        raw_prompts: list[Any],
        trajectory_segments: list[list[Segment]],
    ) -> tuple[list[JudgeRequest], list[int]]:
        requests = []
        base_fallback_trajectories = []
        for trajectory_index, segments in enumerate(trajectory_segments):
            task = _task_text(raw_prompts[trajectory_index])
            trajectory_requests = []
            overflow = False
            for segment_index in range(len(segments)):
                primary = self._fit_prompt(
                    task,
                    segments,
                    segment_index,
                    self.max_input_tokens,
                )
                fallback = self._fit_prompt(
                    task,
                    segments,
                    segment_index,
                    self.fallback_max_input_tokens,
                )
                if primary is None or fallback is None:
                    overflow = True
                    break
                trajectory_requests.append(
                    JudgeRequest(
                        trajectory_index=trajectory_index,
                        segment_id=segment_index,
                        prompt=primary[0],
                        fallback_prompt=fallback[0],
                        input_tokens=primary[1],
                        fallback_input_tokens=fallback[1],
                        context_radius=primary[2],
                        fallback_context_radius=fallback[2],
                    )
                )
            if overflow:
                base_fallback_trajectories.append(trajectory_index)
            else:
                requests.extend(trajectory_requests)
        return requests, base_fallback_trajectories

    def _call(self, request: JudgeRequest) -> JudgeResult:
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "model": self.model,
                    "system": JUDGE_SYSTEM_PROMPT,
                    "prompt": request.prompt,
                    "fallback_prompt": request.fallback_prompt,
                    "max_tokens": self.max_tokens,
                    "fallback_max_tokens": self.fallback_max_tokens,
                    "thinking": True,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return JudgeResult(
                trajectory_index=request.trajectory_index,
                segment_id=request.segment_id,
                role=cached["role"],
                evidence=cached["evidence"],
                raw_response=cached["raw_response"],
                elapsed_seconds=0.0,
                completion_tokens=int(cached.get("completion_tokens", 0)),
                finish_reason=cached.get("finish_reason"),
                input_tokens=int(cached.get("input_tokens", request.input_tokens)),
                context_radius=int(cached.get("context_radius", request.context_radius)),
                max_tokens_limit=int(cached.get("max_tokens_limit", self.max_tokens)),
                fallback_used=bool(cached.get("fallback_used", False)),
            )

        last_error = None
        for attempt in range(self.retries + 1):
            fallback_used = attempt > 0
            prompt = request.fallback_prompt if fallback_used else request.prompt
            input_tokens = request.fallback_input_tokens if fallback_used else request.input_tokens
            context_radius = request.fallback_context_radius if fallback_used else request.context_radius
            max_tokens_limit = self.fallback_max_tokens if fallback_used else self.max_tokens
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max_tokens_limit,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            body = json.dumps(payload).encode()
            started = time.time()
            try:
                http_request = urllib.request.Request(
                    f"{self.api_base}/chat/completions",
                    data=body,
                    headers={
                        "Authorization": "Bearer EMPTY",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                    result = json.loads(response.read())
                message = result["choices"][0]["message"]
                finish_reason = result["choices"][0].get("finish_reason")
                completion_tokens = int((result.get("usage") or {}).get("completion_tokens") or 0)
                if finish_reason == "length":
                    raise ValueError(
                        f"judge output hit max_tokens={max_tokens_limit}"
                    )
                raw = "\n".join(
                    part
                    for part in (
                        message.get("reasoning") or message.get("reasoning_content"),
                        message.get("content"),
                    )
                    if part
                )
                parsed = _extract_json_object(raw)
                labels = parsed.get("labels")
                evidence = parsed.get("evidence")
                if isinstance(labels, str):
                    labels = [labels]
                if isinstance(evidence, str):
                    evidence = [evidence]
                if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in JUDGE_ROLES:
                    raise ValueError(f"invalid role labels: {labels!r}")
                if not isinstance(evidence, list) or len(evidence) != 1:
                    raise ValueError(f"invalid evidence: {evidence!r}")
                judge_result = JudgeResult(
                    trajectory_index=request.trajectory_index,
                    segment_id=request.segment_id,
                    role=labels[0],
                    evidence=str(evidence[0]),
                    raw_response=raw,
                    elapsed_seconds=time.time() - started,
                    completion_tokens=completion_tokens,
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    context_radius=context_radius,
                    max_tokens_limit=max_tokens_limit,
                    fallback_used=fallback_used,
                )
                if self.cache_path is not None:
                    cache_item = {
                        "key": cache_key,
                        "role": judge_result.role,
                        "evidence": judge_result.evidence,
                        "raw_response": judge_result.raw_response,
                        "completion_tokens": judge_result.completion_tokens,
                        "finish_reason": judge_result.finish_reason,
                        "input_tokens": judge_result.input_tokens,
                        "context_radius": judge_result.context_radius,
                        "max_tokens_limit": judge_result.max_tokens_limit,
                        "fallback_used": judge_result.fallback_used,
                    }
                    with self._cache_lock:
                        if cache_key not in self._cache:
                            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                            with self.cache_path.open("a") as handle:
                                handle.write(json.dumps(cache_item, ensure_ascii=False) + "\n")
                            self._cache[cache_key] = cache_item
                return judge_result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "TRIAGE judge failed trajectory=%s segment=%s attempt=%s: %r",
                    request.trajectory_index,
                    request.segment_id,
                    attempt + 1,
                    exc,
                )
        logger.warning(
            "TRIAGE judge exhausted retries for trajectory=%s segment=%s; using base GRPO credit: %r",
            request.trajectory_index,
            request.segment_id,
            last_error,
        )
        return JudgeResult(
            trajectory_index=request.trajectory_index,
            segment_id=request.segment_id,
            role="B",
            evidence=f"judge_failure_base_grpo_fallback: {last_error!r}"[:500],
            raw_response="",
            elapsed_seconds=0.0,
            completion_tokens=0,
            finish_reason="judge_failed",
            input_tokens=request.fallback_input_tokens,
            context_radius=request.fallback_context_radius,
            max_tokens_limit=self.fallback_max_tokens,
            fallback_used=True,
        )

    def label(
        self,
        raw_prompts: list[Any],
        trajectory_segments: list[list[Segment]],
    ) -> tuple[list[JudgeResult], list[int]]:
        requests, base_fallback_trajectories = self.build_requests(
            raw_prompts,
            trajectory_segments,
        )
        results: list[JudgeResult] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [executor.submit(self._call, request) for request in requests]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: (item.trajectory_index, item.segment_id))
        failed = sum(result.finish_reason == "judge_failed" for result in results)
        if failed > self.max_judge_failure_ratio * max(1, len(results)):
            raise RuntimeError(
                f"TRIAGE judge failed on {failed}/{len(results)} segments, above "
                f"max_judge_failure_ratio={self.max_judge_failure_ratio}"
            )
        return results, base_fallback_trajectories


def apply_triage_advantages(
    base_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    segment_roles: list[list[str]],
    mixing_lambda: float,
    role_rewards: dict[str, float] | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply role correction and equal-weight whitening over segments."""
    rewards = dict(DEFAULT_ROLE_REWARDS if role_rewards is None else role_rewards)
    output = torch.zeros_like(base_advantages)
    values: list[torch.Tensor] = []
    locations: list[tuple[int, int]] = []
    role_counter: Counter[str] = Counter()

    for trajectory_index, roles in enumerate(segment_roles):
        for segment_index, role in enumerate(roles):
            if role not in ROLE_TO_ID:
                raise ValueError(f"unknown TRIAGE role: {role}")
            token_positions = (segment_ids[trajectory_index] == segment_index).nonzero(as_tuple=False).flatten()
            if token_positions.numel() == 0:
                raise ValueError(
                    f"segment has no generated tokens: trajectory={trajectory_index} segment={segment_index}"
                )
            base_value = base_advantages[trajectory_index, token_positions[0]]
            values.append(base_value + mixing_lambda * float(rewards[role]))
            locations.append((trajectory_index, segment_index))
            role_counter[role] += 1

    if not values:
        return output, {"triage/segments": 0.0}

    stacked = torch.stack(values)
    mean = stacked.mean()
    std = stacked.std(unbiased=False)
    normalized = (stacked - mean) / (std + epsilon)

    for value, (trajectory_index, segment_index) in zip(normalized, locations, strict=True):
        token_positions = segment_ids[trajectory_index] == segment_index
        output[trajectory_index, token_positions] = value
    output *= response_mask

    metrics = {
        "triage/segments": float(len(values)),
        "triage/adv_mean_before_whiten": float(mean.detach().cpu()),
        "triage/adv_std_before_whiten": float(std.detach().cpu()),
    }
    for role in ROLE_TO_ID:
        metrics[f"triage/role_{role}_count"] = float(role_counter[role])
        metrics[f"triage/role_{role}_ratio"] = float(role_counter[role] / len(values))
    return output, metrics


def append_audit_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _AUDIT_LOCK, destination.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
