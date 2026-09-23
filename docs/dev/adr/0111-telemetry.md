# ADR-0111 — Telemetry schema and `moregpu bench`

**Status:** Proposed · **Milestone:** M2

## Decision
- **Schema v1** (JSON Schema in `docs/telemetry.schema.json`, documented in `docs/TELEMETRY.md`): one JSONL line per
  (session, round, worker) + one per round aggregate. Fields: `schema`, `ts`, `session`, `round`, `worker`, `wall_s`,
  `compute_s`, `data_s`, `serialize_s`, `network_s`, `wait_s` (wait = wall − others, must be ≥ −tol), `bytes_up/down`,
  `samples`, `samples_seen`, `samples_per_s`, `gpu_util`, `gpu_power_w`, `energy_j` (NVML via `pynvml`, optional),
  `mem_peak_bytes`, `amp`, `hw` fingerprint, `git_sha`, `config_hash`, task metrics.
- Workers measure their own phases and return them in each reply; the coordinator measures network/wait and writes the
  JSONL (`MOREGPU_TELEMETRY_DIR`). A small standalone emitter (`moregpu_worker.telemetry.emit`) lets non-MoreGPU runners
  (e.g. a DDP control) write the same schema.
- **`moregpu bench`:** seeded repeats; `--limit-vram F` (`torch.cuda.set_per_process_memory_fraction`), `--limit-cpus/
  --limit-mem` (cgroup v2 / docker flags, printed in dry-run), `--simulate-latency/--bandwidth` (`tc netem`, Linux, root,
  explicit opt-in, always removed on exit). Every limit is labelled "emulation" in the output.
- `/net` adds sustained bandwidth (N × 4 MiB both directions) and RTT p50/p90/p99 over K pings.
