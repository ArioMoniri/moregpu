# Training on the pool

A **training session** runs any `TrainTask` on one or more native torch workers. Workers are synchronised with DiLoCo:
- each worker runs H local steps on its own seeded data shard;
- the coordinator averages the synced tensors, weighted by the number of samples each worker saw;
- it applies an outer Nesterov step and broadcasts the result.

With N=1, H=1, η=1 and μ=0, this is exactly plain training. That equivalence is covered by tests.

## Tasks

| Task | What it trains | Synced state |
|---|---|---|
| `jepa_2p5d`, `ijepa_2d`, `jepa_3d` | JEPA self-supervised encoder and predictor ([JEPA.md](JEPA.md)) | encoder + predictor (the EMA target is never sent) |
| `segment`, `classify` | ViT encoder (random init or a JEPA export) with a light head. Modes: full, frozen, LoRA | trainable params |
| `finetune_model` | **any published native model** (torchvision, timm, MONAI, HF, plugin — [MODELS.md](MODELS.md)). Trainable: all, head, or LoRA | trainable params |
| `llm_lora` | the original LoRA fine-tuning of a causal LM, unchanged | LoRA A/B |
| `toy_linear` | tiny regression, used for tests and demos | all |
| plugins | entry-point group `moregpu.train_tasks`, loaded only if pinned in `MOREGPU_PLUGIN_ALLOWLIST` (dist, version, wheel sha256) | task-defined |

## Semantics and guarantees

- **Inner optimizer state** persists across rounds on each worker (DiLoCo); it is never synced.
- **After-outer hook** receives identical `{progress, h}` from the coordinator on every worker (h = global optimizer steps this round).
- **Alarms:** `stop_on_alarm` (default `["diverged"]`) fails the session; the study adds `"collapse"`.
- **Deterministic shards.** A seeded SplitMix64 Fisher–Yates permutation is drawn per epoch over the manifest indices.
  Each round takes the next Σ allocation indices and splits them contiguously in worker order.
  - The coordinator (TS) and the reference (Python) are identical; a cross-language golden test checks this.
  - Only indices travel, never pixels.
- **Allocation.**
  - `fixed` gives every worker the same number of samples.
  - `proportional` scales each worker's samples with its measured samples/s, so the rounds finish together. The round
    total stays the same.
- **Stopping.** `target_samples` stops the session exactly at that count, trimming the last round.
  - Samples seen are tracked per worker and in total.
  - Matched-samples experiments therefore need only a config flag.
- **Averaging** is weighted by samples: `w_i = s_i / Σ s`.
  - A worker whose state contains NaN or Inf is dropped from the average but is resynced.
  - BatchNorm running statistics (`buffer:` tensors) are averaged, never outer-stepped; frozen norm layers stay in eval mode.
  - A worker that fails to receive the broadcast leaves the session.
- **Outer step**, with Δ = global − average:
  - `v ← μv + Δ`
  - `global ← global − η(Δ + μv)`

  This is the PyTorch `SGD(nesterov=True)` form. Outer state is kept in fp32 on the coordinator.
- **Learning-rate schedule** (`lr_schedule: {kind: cosine, warmup_frac, min_lr}`): evaluated at the midpoint of each
  round; the same schedule is used on both sides.
- **Wire format** (`sync_dtype` / `broadcast_dtype`):
  - Options are `f32`, `bf16`, `fp16`, and `int8delta` (per-block absmax against the last broadcast).
  - The measured error is reported in telemetry.
  - Payloads travel in sha256-checked chunks inside the sealed relay.
- **AMP** (`amp: auto|bf16|fp16|fp32`):
  - `auto` picks bf16 where the GPU supports it, fp16 with GradScaler on other CUDA GPUs, and fp32 on CPU/MPS.
  - A forced mode the device cannot run is an error, never a silent downgrade.
- **Sessions** are keyed by id on each worker. The admission limit is `MOREGPU_MAX_TRAIN_SESSIONS` (default 1 on GPU,
  2 on CPU). The legacy `/train` and `/train/diloco` routes keep their reserved slot and their behaviour.
- **Checkpoint and resume.**
  - `checkpoint_every: k` writes the global state, momentum, sample stream and counters atomically to
    `MOREGPU_TRAIN_DIR`.
  - `POST /train/sessions/resume` continues bit-identically. This is tested.
  - Non-synced task state (e.g. the JEPA EMA target and step counters) is checkpointed via `extra_state` and restored on resume (tested).
- **Churn and delivery semantics.** A lost or failed worker is excluded from that round. Its samples are not counted,
  and its shard for that epoch is **not** re-queued, so delivery is at-most-once under failure: `samples_seen` counts
  only samples that entered the average. A paused worker (user activity, schedule) sits the round out and stays in the
  session. A relay timeout abandons the coordinator's wait, but the worker may still finish the step. The next
  broadcast resynchronises it.
- **Adversarial or broken workers.** Payload headers are validated and reported sample counts are clamped, so a single
  worker cannot finish, wedge or crash a session. The EMA-target check uses a checksum tolerance, a majority keeps its
  state and the minority is dropped. Poisoned but well-formed weights cannot be detected; see SECURITY.md.
- **Serialisation.** Rounds, checkpoints, evaluation, export, adding a worker and closing are serialised per session.
- **Exactness.** "Stops exactly at `target_samples`" and "resume is bit-identical" hold for honest workers. `POST
  /train/sessions/:id/workers {add}` brings a new worker in on the current global state.

## API

| Surface | Commands |
|---|---|
| HTTP | `/train/sessions` (POST create, GET list); `/train/sessions/:id` (GET, DELETE); and under `/train/sessions/:id/`: `round`, `run`, `stop`, `eval`, `export`, `state`, `checkpoint`, `workers`, `telemetry`; plus `/train/sessions/resume` |
| Python SDK | `train_session_create`, `train_jepa`, `train_session_round/run/wait/eval/export/state/resume/telemetry/delete` |
| TypeScript SDK | `trainSessionCreate`, `trainJepa`, `trainSessionRound`, … |
| CLI | `moregpu train jepa|segment|classify|status|round|stop|checkpoint|resume|rm|export|telemetry` |
| Dashboard | "Training sessions" panel |

An export `path` is a directory **on the worker**, and it must be inside the worker's `MOREGPU_OUTPUT_DIR` (default
`./moregpu-out`). A relative path resolves inside that directory. An absolute path outside it, `../…`, or a symlink
that points outside it is refused. A segment/classify `encoder: {init: "export", path}` may read from
`MOREGPU_OUTPUT_DIR` or `MOREGPU_MODEL_ROOTS`. See [MODELS.md](MODELS.md#worker-environment).

## What each worker type can do

Training runs on native torch workers only. WebGPU and browser workers have no autograd; they take part in inference
and feature extraction instead ([VISION.md](VISION.md)).
