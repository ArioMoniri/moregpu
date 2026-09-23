# ADR-0104 — Per-session training state

**Status:** Proposed · **Milestone:** M1

## Context
One `TRAIN` dict per worker; one `trainingHome` and one `diloco` per coordinator; `/train` and `/train/diloco` collide
(guarded by 409s, with TODOs to key by session).

## Decision
- Worker: `SESSIONS: dict[str, TrainSession]`; every `train` op carries `session`. Admission limit
  `MOREGPU_MAX_TRAIN_SESSIONS` (default 1 on GPU, 2 on CPU) → explicit error, never silent eviction.
- Coordinator: `Map<sessionId, TrainingSession>` (single-worker and DiLoCo sessions share one type with `mode`).
  New routes `/train/sessions` (list), `/train/sessions/:id` (describe/delete); new task routes take `session`.
- **Back-compat:** legacy `/train/*` and `/train/diloco/*` bind to reserved ids `legacy` and `legacy-diloco`, preserving
  their current 409 semantics and responses byte-for-byte (asserted by the existing e2e tests + new contract tests).
- Session ids are coordinator-minted (`crypto.randomUUID`), never worker-chosen.

## Consequences
Two tasks on one worker are possible only when memory allows; the scheduler must respect the admission limit.
