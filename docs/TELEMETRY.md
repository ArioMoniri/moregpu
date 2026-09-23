# Telemetry — `moregpu.telemetry/1`

MoreGPU writes one JSON object per line (JSONL). The same schema is used by the coordinator, by the worker's phase
timers, by `moregpu bench`, and by the controls you compare against (a DDP or single-process run). Because they all
use one schema, one notebook can read all of them.

- Machine-readable schema: [`docs/telemetry.schema.json`](telemetry.schema.json) (JSON Schema draft 2020-12). It is
  generated from `apps/worker/moregpu_worker/telemetry/schema.py`, and a test keeps the two identical.
- Decision record: [ADR-0111](dev/adr/0111-telemetry.md).
- Validator (standard library only): `from moregpu_worker.telemetry.schema import validate, breakdown_ok`.
  `validate(rec)` returns a list of problems; an empty list means the record is valid. Tests check that it agrees with
  the `jsonschema` library. It is stricter in one way: it also rejects NaN and ±inf, which JSON cannot carry anyway.

## Record kinds

Every record has these three fields:

| field | type | meaning |
|---|---|---|
| `schema` | `"moregpu.telemetry/1"` | schema id |
| `kind` | `worker_round` · `round` · `job` · `bench` · `external_round` | record kind |
| `ts` | ISO-8601 UTC string | time the record was emitted |

### `worker_round`: one per (session, round, worker), written by the coordinator

The field set matches `emitTelemetry` in `apps/coordinator/lib/train_session.ts`. A cross-language test runs the real
`TrainSession` under Deno and validates what it prints. All fields are required, and unknown fields are rejected.

| field | type | source |
|---|---|---|
| `session`, `task`, `worker` | string | session config |
| `round` | int ≥ 0 | coordinator |
| `wall_s` | s | coordinator: the whole round, including reduce, broadcast and eval |
| `compute_s`, `data_s`, `serialize_s` | s | **worker**: `report.timings` from `task_inner` |
| `network_s` | s | **coordinator**: (time the inner call was in flight + pull + broadcast) − the worker's phases |
| `wait_s` | s | **coordinator**: wall − all of the above (time spent waiting on stragglers or the barrier) |
| `bytes_up`, `bytes_down` | int | tensorwire bytes pulled from and broadcast to this worker |
| `samples` | int | samples trained this round |
| `samples_seen` | int | cumulative samples for this worker |
| `samples_per_s` | number \| null | samples / compute_s |
| `loss_last` | number \| null | last inner-step loss |
| `amp` | string \| null | worker `report.metrics.amp` |
| `gpu_util` | 0–100 \| null | worker `report.metrics.gpu_util` (NVML mean) |
| `gpu_power_w` | number \| null | worker `report.metrics.gpu_power_w` (NVML mean) |
| `energy_j` | number \| null | worker `report.metrics.energy_j` (NVML) |
| `mem_peak_bytes` | int \| null | worker `report.metrics.mem_peak_bytes` (`torch.cuda.max_memory_allocated`) |
| `hw` | object \| null | worker `report.metrics.hw` (`hw.fingerprint()`). `hw.vram_fraction` is the `MOREGPU_VRAM_FRACTION` the worker applied at start-up (`null` when unset or without CUDA). |
| `wire_error` | `{max_abs, rel_l2}` \| null | reconstruction error of a lossy sync dtype |
| `git_sha` | string \| null | `MOREGPU_GIT_SHA` of the coordinator |
| `config_hash` | string | sha256 of the canonical session config |

The five phases add up to `wall_s`. `breakdown_ok(rec, tol=0.05)` checks this within 5 % and also checks that no phase
is meaningfully negative. If you run `wait = wall − others` on your own data, you get the same identity.

### `round`: one aggregate per session round

`session`, `task`, `round`, `wall_s`, `reduce_s`, `workers[]`, `dropped[]`, `samples`, `samples_seen`,
`avg_last_loss` (null when NaN), `bytes_up`, `bytes_down`, `alarms[]`, `eval` (object \| null), `monitors`
(object \| null), `git_sha` (\| null), `config_hash`. All fields are required, and unknown fields are rejected.

### `job`: vision batch jobs (`/vision/batch`)

Required fields: `job`, `items`, `wall_s`.

Optional fields:
- `op`, `session`, `worker`
- `items_done`, `items_failed`, `retries`
- the five phases (nullable), `items_per_s`, `bytes_up`, `bytes_down`
- the GPU fields (`amp`, `gpu_util`, `gpu_power_w`, `energy_j`, `mem_peak_bytes`, `hw`)
- `git_sha`, `config_hash`, `metrics`

### `bench`: one per `moregpu bench` invocation

`name`, `cmd`, `repeats`, `seeds[]`, `runs[]` (`{seed, wall_s, rc, records, invalid_records, error?}`),
`failures`, `stats` (`{metric: {n, mean, sd, min, max}}`, currently `wall_s` and `samples_per_s`), `emulation`,
`hw`, `git_sha`, `config_hash`, and `metrics` (optional).

`emulation` is null or `{label: "emulation", note, limits[]}`. Each limit is
`{type, value, mechanism, label: "emulation", applied}`.

### `external_round`: non-MoreGPU runners (DDP, single-process, FSDP …)

Required fields:
- `runner` (`"ddp"`, `"single"`, … as free text)
- `world_size` (≥ 1), `rank`, `round`
- `wall_s` and the five phases

Optional fields: `session`, `task`, `samples`, `samples_seen`, `samples_per_s`, `loss_last`, `bytes_up`,
`bytes_down`, the GPU fields, `git_sha`, `config_hash`, `metrics`.

## How records are produced

```
worker (task_inner)                               coordinator (TrainSession.runRound)
  data_s     ← time spent in the loader           t_inner  = RPC time of task_inner
  compute_s  ← forward/backward/step              t_pull   = chunked state pull
  serialize_s← tensorwire encode                  bsecs    = broadcast time to this worker
  metrics    ← amp, gpu_util, gpu_power_w,        network_s = t_inner + t_pull + bsecs − (compute+data+serialize)
               energy_j, mem_peak_bytes, hw       wait_s    = wall − compute − data − serialize − network
```

The worker reports only what it can measure itself. The coordinator measures network and wait time, and writes the
records.

### Where the JSONL goes

- `MOREGPU_TELEMETRY_DIR=/path`: the coordinator appends to `/path/<session>.jsonl`.
- `GET /train/sessions/:id/telemetry?n=500` (admin): the last *n* records (up to 2000) held in memory for that
  session.
- `moregpu bench` gives each repeat its own temporary `MOREGPU_TELEMETRY_DIR` and `MOREGPU_TELEMETRY_FILE`. It
  collects every `*.jsonl` written there.

## Emitting compatible records from your own runner

```python
from moregpu_worker.telemetry.emit import JsonlEmitter, PhaseTimer
from moregpu_worker.telemetry.nvml import GpuSampler

em = JsonlEmitter.from_env(f"ddp-rank{rank}") or JsonlEmitter(f"runs/ddp-rank{rank}.jsonl", config=cfg)
pt = PhaseTimer()
for step in range(steps):
    gpu = GpuSampler(device_index=local_rank).start()
    with pt.phase("data"):
        x, y = next(it)
    with pt.phase("compute"):
        loss = step_fn(x, y)          # for DDP, all-reduce overlaps backward: it stays inside compute
    torch.cuda.synchronize()          # otherwise async kernels leak into the next phase
    g = gpu.stop()
    em.emit("external_round", runner="ddp", world_size=world, rank=rank, round=step,
            samples=len(x), samples_per_s=len(x) / max(pt.record()["compute_s"], 1e-9),
            loss_last=float(loss), gpu_util=g["gpu_util_mean"], gpu_power_w=g["gpu_power_w_mean"],
            energy_j=g["energy_j"], mem_peak_bytes=torch.cuda.max_memory_allocated(), **pt.lap())
```

- `JsonlEmitter` fills in `schema`, `ts`, `git_sha`, `config_hash` and `hw` for you.
  - `git_sha` comes from `MOREGPU_GIT_SHA`, falling back to `git rev-parse HEAD`.
  - `config_hash` is the same canonical sha256 the coordinator computes, so the same config gives the same hash
    in both.
  - `hw` is filled only when the record kind has an `hw` field.
- The emitter validates every record. By default an invalid record raises an error and nothing is written.
- It is thread-safe. Use one file per process, for example one per rank.
- `PhaseTimer` makes the breakdown add up exactly. Time that no phase covers becomes `wait_s`. If you add phases
  that overlap (for example async communication measured elsewhere), they can exceed the wall-clock time. In that
  case `wait_s` is 0 and `wall_s` is the sum of the phases, so no phase is ever negative.
- `hw.fingerprint()` imports torch. Pass `hw=None` if you do not want that.

`GpuSampler` needs `pynvml` (`pip install nvidia-ml-py`, or `moregpu-worker[telemetry]`). Without it, or when NVML
cannot start, every field is `None`. Keep these caveats in mind:
- Energy comes from NVML's total-energy counter where the GPU supports it (Volta and newer). Otherwise it is the
  trapezoid of power samples taken every `period_s`.
- Board power includes idle draw and any other process on the same GPU.
- `gpu_util` means "a kernel was running". It is not SM occupancy.

## `moregpu bench`

```
moregpu bench [--repeats N | --seeds 0,1,2] [--name NAME] [--out bench.jsonl]
              [--limit-vram 0.5]
              [--limit-cpus 4 --limit-mem 8G [--backend systemd|docker --docker-image IMG]]
              [--simulate-latency 50 --bandwidth 100 --iface eth0]
              [--dry-run | --apply]  -- CMD ARGS...
```

`moregpu bench` runs the same command as `python3 -m moregpu_worker.bench`, with `PYTHONPATH=apps/worker` taken from
the clone. For each seed it:
1. runs `CMD` with `MOREGPU_SEED` set,
2. times it,
3. collects and validates the telemetry `CMD` wrote,
4. prints one `bench` record (and appends it to `--out`).

The record's `stats` field gives mean, sample sd, min and max of wall time over successful runs. It gives the same
statistics for every valid `samples_per_s` value.

| flag | mechanism | needs |
|---|---|---|
| `--limit-vram F` | `torch.cuda.set_per_process_memory_fraction(F, d)` for every device in the bench process, plus `MOREGPU_VRAM_FRACTION=F` exported to `CMD` | CUDA (exits 2 without it). `CMD` must call `moregpu_worker.bench.apply_vram_limit_from_env()` for the cap to apply inside it |
| `--limit-cpus N` | `systemd-run [--user] --scope -p CPUQuota=N×100%` (cgroup v2), or `docker run --cpus N` | `--apply` |
| `--limit-mem SIZE` | `-p MemoryMax=… -p MemorySwapMax=0`, or `docker --memory … --memory-swap …` | `--apply` |
| `--simulate-latency MS`, `--bandwidth MBIT` | `tc qdisc add dev IFACE root netem delay MSms rate MBITmbit`, removed afterwards with `tc qdisc del dev IFACE root` | Linux, root, `--apply` |

- Without `--apply`, any CPU, memory or netem limit turns the invocation into a dry run. It prints the exact commands
  (`commands` and `shell`) and the emulation block, and runs nothing.
- `--dry-run` always prints the plan. The dry run works on any OS and as any user.
- With `--apply`, the netem qdisc is removed on every exit path: `finally`, `atexit`, and SIGTERM/SIGINT/SIGHUP
  handlers that clean up and then pass the signal on to the previous handler.
- If `tc qdisc add` fails partway, the bench still runs `tc qdisc del`.

### Caveats: all of this is emulation

- The output labels every limit `"emulation"`. Compare only runs whose emulation blocks match.
- **VRAM fraction** caps PyTorch's caching allocator only. It does not cap the CUDA context, cuDNN or NCCL
  workspaces, or other processes. A 24 GB GPU capped at 0.33 is not an 8 GB GPU: it keeps the big GPU's SMs,
  bandwidth and L2 cache.
- **CPU quota** throttles CPU time. It does not remove cores, caches or memory bandwidth. `systemd-run --user` needs
  a user systemd with cgroup v2 delegation. Docker needs an image with python, torch and the repo mounted (the bench
  mounts the current directory).
- **netem** shapes egress only, on one interface, for every process on the host. Loopback traffic (a coordinator and
  workers on one box) needs `--iface lo`, and then the delay applies in both directions (twice per RTT). `rate` is a
  token-bucket approximation, and there is no jitter, loss or reordering unless you add them yourself.
- Wall time includes starting the process and importing Python and torch. Use the per-round telemetry
  (`samples_per_s`, phases) for steady-state throughput.
