import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from tmax_rl.triage import Segment, TriageRoleJudge, apply_triage_advantages, extract_segments


class DummyTokenizer:
    @staticmethod
    def decode(ids, skip_special_tokens=False):
        return " ".join(map(str, ids))

    @staticmethod
    def encode(text, add_special_tokens=False):
        return text.split()


def test_extract_segments():
    segments, ids = extract_segments(
        response_ids=list(range(10)),
        response_mask=[1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
        tokenizer=DummyTokenizer(),
    )
    assert [(x.token_start, x.token_end) for x in segments] == [(0, 2), (4, 7), (8, 9)]
    assert ids == [0, 0, -1, -1, 1, 1, 1, -1, 2, -1]


def test_advantage_broadcast_and_observation_mask():
    base = torch.tensor([[2.0, 2.0, 0.0, 2.0], [-1.0, 0.0, -1.0, -1.0]])
    mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]])
    segment_ids = torch.tensor([[0, 0, -1, 1], [0, -1, 1, 1]])
    output, metrics = apply_triage_advantages(
        base_advantages=base,
        response_mask=mask,
        segment_ids=segment_ids,
        segment_roles=[["D", "R"], ["E", "N"]],
        mixing_lambda=0.2,
    )
    assert output[0, 0] == output[0, 1]
    assert output[1, 2] == output[1, 3]
    assert output[0, 2] == 0
    assert output[1, 1] == 0
    # Segment whitening is equal-weight over four segments, not token-weighted.
    unique_segment_values = torch.stack([output[0, 0], output[0, 3], output[1, 0], output[1, 2]])
    assert torch.allclose(unique_segment_values.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(unique_segment_values.std(unbiased=False), torch.tensor(1.0), atol=1e-5)
    assert metrics["triage/segments"] == 4


def test_judge_client_and_parser():
    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):
            Handler.calls += 1
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            assert payload["temperature"] == 0
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "reasoning_content": "The command inspects a relevant file.",
                                "content": '{"labels":"E","evidence":"first relevant inspection"}',
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 87},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            judge = TriageRoleJudge(
                {
                    "api_base": f"http://127.0.0.1:{server.server_port}/v1",
                    "model": "fake",
                    "concurrency": 2,
                    "cache_path": f"{tmp}/cache.jsonl",
                },
                DummyTokenizer(),
            )
            kwargs = {
                "raw_prompts": [[{"role": "user", "content": "Fix the program"}]],
                "trajectory_segments": [
                    [
                        Segment(
                            segment_id=0,
                            token_start=0,
                            token_end=2,
                            action="cat /app/main.py",
                            observation="def broken(): pass",
                        )
                    ]
                ],
            }
            results, fallback = judge.label(**kwargs)
            cached_results, cached_fallback = judge.label(**kwargs)
            assert fallback == []
            assert cached_fallback == []
            assert len(results) == 1
            assert results[0].role == "E"
            assert results[0].completion_tokens == 87
            assert results[0].finish_reason == "stop"
            assert results[0].input_tokens > 0
            assert results[0].context_radius == 5
            assert results[0].fallback_used is False
            assert cached_results[0].elapsed_seconds == 0
            assert cached_results[0].completion_tokens == 87
            assert Handler.calls == 1
    finally:
        server.shutdown()
        thread.join()


def test_input_overflow_uses_base_grpo_fallback():
    class CharacterTokenizer(DummyTokenizer):
        @staticmethod
        def encode(text, add_special_tokens=False):
            return list(text)

    judge = TriageRoleJudge(
        {
            "max_input_tokens": 50,
            "fallback_max_input_tokens": 40,
        },
        CharacterTokenizer(),
    )
    results, fallback = judge.label(
        raw_prompts=[[{"role": "user", "content": "x" * 10_000}]],
        trajectory_segments=[
            [
                Segment(
                    segment_id=0,
                    token_start=0,
                    token_end=1,
                    action="cat file",
                    observation="content",
                )
            ]
        ],
    )
    assert results == []
    assert fallback == [0]


def test_output_length_retries_with_4k_fallback():
    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):
            Handler.calls += 1
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            if Handler.calls == 1:
                assert payload["max_tokens"] == 2048
                choice = {
                    "message": {"reasoning_content": "truncated", "content": ""},
                    "finish_reason": "length",
                }
                completion_tokens = 2048
            else:
                assert payload["max_tokens"] == 4096
                choice = {
                    "message": {
                        "reasoning_content": "Recovered with larger budget.",
                        "content": '{"labels":["D"],"evidence":["required artifact created"]}',
                    },
                    "finish_reason": "stop",
                }
                completion_tokens = 2300
            body = json.dumps(
                {"choices": [choice], "usage": {"completion_tokens": completion_tokens}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        judge = TriageRoleJudge(
            {
                "api_base": f"http://127.0.0.1:{server.server_port}/v1",
                "model": "fake",
                "retries": 2,
            },
            DummyTokenizer(),
        )
        results, fallback = judge.label(
            raw_prompts=[[{"role": "user", "content": "Create artifact"}]],
            trajectory_segments=[
                [
                    Segment(
                        segment_id=0,
                        token_start=0,
                        token_end=1,
                        action="write artifact",
                        observation="done",
                    )
                ]
            ],
        )
        assert fallback == []
        assert results[0].role == "D"
        assert results[0].fallback_used is True
        assert results[0].max_tokens_limit == 4096
        assert Handler.calls == 2
    finally:
        server.shutdown()
        thread.join()


def test_exhausted_judge_uses_base_grpo_for_segment():
    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):
            Handler.calls += 1
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"reasoning": "looping " * 10, "content": ""},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"completion_tokens": 4096},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    segments = [[Segment(segment_id=0, token_start=0, token_end=1, action="ls", observation="a")]]
    prompts = [[{"role": "user", "content": "task"}]]
    config = {"api_base": f"http://127.0.0.1:{server.server_port}/v1", "model": "fake", "retries": 2}
    try:
        judge = TriageRoleJudge({**config, "max_judge_failure_ratio": 1.0}, DummyTokenizer())
        results, fallback = judge.label(raw_prompts=prompts, trajectory_segments=segments)
        assert fallback == []
        assert results[0].role == "B"
        assert results[0].finish_reason == "judge_failed"
        assert Handler.calls == 3

        strict = TriageRoleJudge({**config, "max_judge_failure_ratio": 0.0}, DummyTokenizer())
        try:
            strict.label(raw_prompts=prompts, trajectory_segments=segments)
        except RuntimeError as exc:
            assert "max_judge_failure_ratio" in str(exc)
        else:
            raise AssertionError("expected failure-ratio guard to raise")
    finally:
        server.shutdown()
        thread.join()


if __name__ == "__main__":
    test_extract_segments()
    test_advantage_broadcast_and_observation_mask()
    test_judge_client_and_parser()
    test_input_overflow_uses_base_grpo_fallback()
    test_output_length_retries_with_4k_fallback()
    test_exhausted_judge_uses_base_grpo_for_segment()
    print("ok")
