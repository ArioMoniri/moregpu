# ADR-0106 — Generalised DiLoCo and tensor transport

**Status:** Accepted (2026-09-23) · **Milestone:** M1

## Decision
- **Topology:** the coordinator stays the parameter server (keeps today's code path and trust model). Bandwidth at the
  coordinator is 2·N·|θ| per round; acceptable for N ≤ 8 on a LAN/same-region network (ViT-Tiny ≈ 5.7 M params →
  ~23 MB f32). Worker-side all-reduce is out of scope (stretch).
- **What syncs:** task-declared `state_for_sync` — full parameters, or adapters, plus chosen floating buffers
  (e.g. norm running stats, averaged); integer buffers (`num_batches_tracked`) are not averaged.
- **Average:** weighted by samples seen this round, `w_i = s_i / Σ s`; equal weights reproduce today's mean exactly.
  Non-finite drop and broadcast-failure drop kept.
- **Amended after review:** floating buffers (BatchNorm running stats, prefix `buffer:`) are plainly averaged and never outer-stepped; frozen norm layers stay in eval mode and are not synced. With a lossy broadcast (bf16/fp16) Δ is taken from the decoded broadcast the workers actually started from. Telemetry bytes are raw payload bytes (base64 wire bytes reported separately); worker-side decode/apply time counts as serialise; after-outer hook and eval time are reported per round (`hook_s`, `eval_s`).
- **Outer step:** unchanged Nesterov math (`v = μv + Δ; θ -= η(Δ + μv)`), fp32 state on the coordinator; η=1, μ=0
  reduces to plain averaging (tested).
- **Payload:** new sealed binary tensor stream (`tensor_begin/chunk/end`, raw little-endian bytes, 4 MiB chunks, per-tensor
  sha256 in `end`) replacing base64-in-JSON for tasks that opt in (`llm_lora` keeps JSON for compatibility).
  Wire dtype `f32 | bf16 | fp16`; optional **int8 delta** (Δ vs last global, per-chunk absmax scale) with the measured
  max/rel error returned in telemetry. Averaging always happens in fp32.
- **Checkpoint/resume:** after each round the coordinator writes `{global, momentum, round, samples_seen, rng, config
  hash}` to `MOREGPU_TRAIN_DIR/<session>/round-<k>.safetensors` + JSON (atomic rename, keep last K). `resume` reloads and
  re-broadcasts.
- **Churn:** a worker lost mid-round is excluded from that round's average (its samples are not counted); a joiner
  receives the current global before its first round.

## Tests first
N=1,H=1,η=1,μ=0 ≡ plain training (bit-exact on CPU); 2-worker run equals a seeded in-process reference DiLoCo; weighting
math; bf16/fp16/int8 error bounds; resume produces identical next round; `tests/e2e/train_diloco.py` unchanged.
