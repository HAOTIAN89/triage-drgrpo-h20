# TRIAGE pipeline audit

Audit date: 2026-11-13.

## Scope

- Qwen3.5-4B policy with Dr.GRPO.
- Qwen3-8B thinking judge served by vLLM.
- Turn-level D/E/N/R role assignment and token broadcast.
- 59 logged training updates and a 665-segment replay at judge concurrency
  11, 32 and 64.

## Results

- Existing TRIAGE unit tests pass.
- Mean training role distribution:
  - D: 52.08%
  - E: 32.92%
  - N: 5.76%
  - R: 9.12%
  - judge-failure/base role B: 0.12%
- Mean failed judge segments: 0.58 of about 650 segments per update
  (approximately 0.09%).
- Full-trajectory input-overflow fallback: 0%.
- Judge output-length fallback to the 4K-token retry: about 7 segments per
  update on average.
- Judge server: no OOM. The only server traceback occurred during normal
  shutdown (`RLock._recursion_count`) and did not affect outputs.

Replay role agreement was 82.9% for concurrency 11 vs 32, 79.7% for 11 vs
64, and 80.5% for 32 vs 64. Aggregate distributions remained close. The
variation is expected from generative judging, so concurrency 32 is the
conservative default.

## Correctness checks

- One role is assigned to each environment-facing assistant turn.
- A role correction is added to the Dr.GRPO group advantage with
  `lambda=0.1`.
- Segment values are equal-weight whitened across the batch.
- The normalized segment value is broadcast only to generated policy tokens;
  observation tokens remain masked.
- Incomplete judge output, excessive fallback and input overflow are guarded.
- Audit JSONL records role, evidence, judge timing, token counts, action and
  observation for every segment.

## Deployment changes in this branch

- Daytona is removed.
- Rootless Podman is the preferred local sandbox; Apptainer is a fallback.
- W&B is forced offline.
- Sandbox and end-to-end smoke tests are mandatory before a full run.
- The new host should start with 16 concurrent sandboxes and judge concurrency
  32, then tune from measured CPU/RAM/disk and judge KV-cache usage.
