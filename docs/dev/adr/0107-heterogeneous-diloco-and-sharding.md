# ADR-0107 — Heterogeneous DiLoCo and deterministic sharding

**Status:** Proposed · **Milestone:** M1

## Decision
- **Speed measurement:** samples/s per worker from the previous round's `compute` time (warm-up round uses a short probe).
- **Allocation modes:** `fixed` (same H and batch everywhere — the default, and what matched-samples experiments use),
  `proportional` (H_i ∝ speed_i so all workers finish together; Σ samples per round held constant), `time_budget`
  (each worker runs until a wall-clock budget, reports samples). Weighting per ADR-0106.
- **Sharding:** a manifest of refs (`refs.jsonl`, one sample per line, content hashes) + seed → one global permutation per
  epoch (numpy `PCG64`). Each round consumes the next Σ s_i indices, split into contiguous slices in worker order. Only
  indices/refs travel, never pixels. Identical given (manifest hash, seed, allocation history).
- **Accounting:** `samples_seen` (global and per worker) is authoritative on the coordinator, exposed in the session
  status and telemetry; `target_samples` stops the session exactly (last round truncated).

## Consequences
Matched-total-samples comparisons across arms become a config flag, not bookkeeping in the study.
