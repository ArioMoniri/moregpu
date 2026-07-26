# Changelog

All notable changes to MoreGPU are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- README shows the `moregpu` CLI's interactive menu (gradient ANSI-Shadow wordmark) as
  [`docs/assets/cli-menu.svg`](docs/assets/cli-menu.svg), and a PyPI version badge.

### Changed

- LoRA examples pick adapter targets by architecture, not model name — the worker attaches to
  whichever of `c_attn` (GPT-2 Conv1D) / `q_proj`,`v_proj` (Llama/Qwen) exist, so any GPT-2 repo
  (e.g. `sshleifer/tiny-gpt2`, `distilgpt2`) fine-tunes without a hand-set target.
- The training examples' verification reference now runs on the **worker's** device (from the
  `train_load`/worker label), so CPU workers verify against a CPU reference (previously a CPU worker
  vs an MPS reference drifted). Validated: `gpt2` on CPU workers matches the reference to 0.0.
- Docs hygiene: DiLoCo described as an fp-tolerance (~2e-5) reference match rather than "bit-for-bit"
  (single-worker LoRA stays bit-for-bit); CHANGELOG version links defined; `AI_USAGE.md` API table
  lists the native-tier `/model/*` and `/train/*` endpoints.

## [0.6.0] - 2026-07-26

### Added

- **Native torch worker** — [`apps/worker/worker_torch.py`](apps/worker/worker_torch.py), a drop-in peer
  of the WebGPU worker that computes with **PyTorch** on the best local device (CUDA → Apple MPS → CPU).
  It speaks the *same* sealed WebSocket protocol (AES-256-GCM, Ed25519 result signatures) and joins the
  *same* pool with the *same* join token, so a torch worker and a WebGPU worker share one fleet. All nine
  kernels match the coordinator's CPU reference and verify+sign like any worker. It's an **opt-in,
  admin-installed native tier** (not the zero-install WGSL worker) — the home for the two features below.
- **Fast LLM serving (resident-model path)** — the torch worker can hold a whole model resident on-device
  and run the **entire forward per call**, collapsing the fine-grained kernel path's ~500 round-trips per
  token into **one**. And `pool.generate()` runs the **whole greedy decode on the worker** (HF's internal KV
  cache) in a single round-trip. [`examples/llm_serve.py`](examples/llm_serve.py) serves **GPT-2 at up to
  ~66 tok/s** warm/unloaded on Apple MPS (~6–23 tok/s under load) with a token-for-token **exact match** to
  Hugging Face. SDK: `pool.model_load/model_forward/generate/model_unload()`; coordinator: `POST /model/load`,
  `/model/forward`, `/model/generate`, `/model/unload` (LRU eviction bounds resident-model VRAM).
- **On-pool fine-tuning (single-worker LoRA)** — the whole train step (forward → cross-entropy → backward →
  optimizer.step) runs **locally on the torch worker**; the base model is frozen and a LoRA adapter is the
  only trainable tensor, so **gradients never leave the worker** (only a sealed microbatch in / scalar loss
  out). [`examples/lora_finetune.py`](examples/lora_finetune.py) fine-tunes GPT-2 and verifies **bit-for-bit
  against a seeded in-process reference** (loss curve matches to 0.0). SDK: `pool.train_load/step/adapter()`;
  coordinator: `POST /train/load`, `/train/step`, `/train/adapter`. Training is verified out-of-band since
  the coordinator can't CPU-check a stochastic loss.
- **Distributed training — DiLoCo across many workers** — [`examples/lora_distributed.py`](examples/lora_distributed.py)
  scales LoRA to N torch workers with **DiLoCo** (low-communication local-SGD): each worker holds the same
  seeded adapter and runs H local AdamW steps on its **own data shard**, then the **coordinator acts as a
  parameter server** — it averages the workers' adapters, applies an **outer Nesterov-momentum step** on the
  pseudo-gradient (global − average), and broadcasts the new global. Only the MB-scale adapter crosses the
  wire, every H steps (a genuine reduce path — the matmul pool only concatenates). Verified across 2 workers:
  the per-round per-worker losses match a deterministic in-process DiLoCo reference to **2e-5** and the final
  global adapter to **1e-3**. SDK: `pool.diloco_load/round/adapter()`; coordinator: `POST /train/diloco/{load,round,adapter}`.
  Synchronous DiLoCo (async / secure-aggregation are the next roadmap step).
- **Client-side KV cache** for the GPT-2 demo — [`examples/llm_infer.py`](examples/llm_infer.py) splits into
  `prefill()` + `decode_step()`: after the prompt pass it caches each layer's projected K/V and generates one
  row per token (absolute positions, no causal mask in decode). **Exact-match preserved** (byte-identical
  greedy tokens); ~4× faster per token on the WGSL path (30.1 → 7.7 s/token) by cutting per-token compute
  and payload. No coordinator/worker/protocol changes.
- **Generic model loader** — [`examples/generic_infer.py`](examples/generic_infer.py) runs a second, modern
  architecture, **Qwen3-0.6B** (RMSNorm · RoPE · SwiGLU · grouped-query attention · Qwen's per-head QK-norm),
  on the pool and matches Hugging Face **token-for-token**. Forward-only and round-trip-bound (~minutes/token)
  — a portability proof that the pool isn't GPT-2-specific, not a speed or training win.
- **Pipeline-parallel model sharding across machines** — [`examples/llm_shard.py`](examples/llm_shard.py)
  splits a model's transformer layers into contiguous **stages**, one per torch worker; each worker holds
  **only its stage** resident and a forward pipes the `[1, seq, hidden]` activation stage→stage (only
  activations on the wire, never weights — the low-bandwidth Petals/Mesh-LLM approach). Verified: **GPT-2
  split 6+6 blocks across 2 workers** is a token-for-token **exact match** to transformers, with the memory
  genuinely split. GPT-2-family only so far. SDK: `pool.shard_load/shard_forward/shard_generate/shard_unload()`;
  coordinator: `POST /model/shard`, `/model/shard_forward`, `/model/shard_unload`.

### Changed

- Capability matrix, Pages site, and `docs/AI_USAGE.md` updated to reflect the native tier honestly:
  `Training` and fast-serving move from ❌ to 🟡 (opt-in native worker; LoRA fine-tuning, single-worker
  *and* distributed via DiLoCo), with the still-roadmap parts (async DiLoCo, secure aggregation, 7B+
  single-model inference, tensor cores) called out.
- **CI**: a `native` job builds a CPU torch worker, compiles the native code, and runs
  [`scripts/smoke_torch.sh`](scripts/smoke_torch.sh) (coordinator + torch worker + verified kernels, no
  model downloads) on every push.

## [0.5.0] - 2026-07-26

### Added

- **fp16 (half-precision) weights** — a shader-f16 GPU worker runs a dedicated f16 tiled GEMM
  (f16 storage, f32 accumulate); resident weights uploaded with `dtype='f16'` halve worker
  memory + upload + GEMM bandwidth. CPU / non-f16 workers dequantize transparently. Verified:
  GPT-2 runs on the pool with f16 weights and still produces an **identical** generation.
- **Real LLM inference on the pool** — `examples/llm_infer.py` loads a real **GPT-2 (124M)**,
  pins the 12 transformer layers' weights resident across the workers, and runs the full
  forward pass on the pool (via weight residency + the shipped primitives), using the real
  Hugging Face tokenizer. Validated: next-token logits match transformers to 0.000 and greedy
  generation is a **token-for-token exact match**. Slow (fp32, activations round-trip per
  layer) — a proof of capability, not a fast serving stack.
- `examples/tiny_llm.py` (toy transformer forward + the honest scaling wall).

### Fixed

- **GPU GELU NaN on large activations** — the WGSL GELU's `tanh` argument grows as x³; some GPU
  `tanh` implementations (Metal) overflow to NaN on huge inputs. The argument is now clamped
  (tanh saturates to ±1 well before, so it's exact). Only real-model activation magnitudes
  triggered it; isolated small-value tests missed it.
- Resident-weight uploads are capped (`MOREGPU_MAX_WEIGHT_ELEMENTS`, default 16M) so an
  oversized weight returns 413 instead of OOM-crashing the coordinator.

## [0.4.0] - 2026-07-25

### Added

- **Weight residency + pipeline parallelism** — `POST /weights` caches a named weight
  RESIDENT on one worker (sent once); a resident matmul (`bRef`) runs where the weight
  lives without re-sending it. SDK: `upload_weight()`, `weights()`, `matmul_resident()`.
  This lets you **split a model across workers/GPUs** — demo: `examples/pipeline_parallel.py`
  (a 2-layer MLP split across two workers, verified). `examples/tiny_llm.py` shows a full
  toy-transformer forward pass and quantifies the wall for real LLMs.
- `moregpu-client` **published to PyPI**: `pip install moregpu-client`.

### Fixed

- Post-review fixes to the 0.3.0 M8 work: coordinator timeouts no longer auto-pause a
  healthy worker; auto-paused workers auto-recover and the last worker is never auto-paused;
  a stale-queue reaper prevents hangs; GPU dispatches are chunked to respect
  `maxComputeWorkgroupsPerDimension`; CLI prompts go to stderr (were invisible) and the token
  prompt is no-echo.

## [0.3.0] - 2026-07-25

### Added

- **All kernels on the GPU** — a GPU worker now runs elementwise (relu/scale/gelu/
  add/mul/saxpy) and row-wise softmax/layernorm as WGSL kernels on-device, not just
  matmul. CPU-only workers run the identical kernels; every result is still verified
  against a CPU reference.
- **SCOUT-style CLI** — `moregpu` with no arguments opens an interactive menu with a
  gradient ANSI-Shadow wordmark and a sectioned Pool/Fleet/Service layout; the same
  wordmark now prints in the server wizard.
- **Homebrew cask** (`Casks/moregpu.rb`) installing the CLI (depends on Deno).

### Changed

- **Shard reassignment + concurrent jobs (M8)** — a failed/timed-out shard is retried
  on other active workers instead of failing the job; workers that fail repeatedly are
  auto-paused; the queue runs up to `MOREGPU_MAX_CONCURRENT_JOBS` (default 4) at once.
  Default shard timeout 120s → 60s.
- README screenshots reframed as macOS windows; capability matrix updated (all kernels
  on the GPU) with an honest note on why CUDA / fp16 / full-LLM would need a separate
  native worker type (not built).

## [0.2.0] - 2026-07-25

### Added

- **Inference primitives & helpers** — a workgroup-tiled WGSL GEMM (shared-memory
  tiling) replaces the naive kernel on the GPU; a new `gelu` activation; and SDK
  composition helpers `attention()`, `linear()`, `mlp()`, and reductions
  (`sum/mean/dot/norm`) in both the Python and TypeScript clients, each verified
  against a CPU reference. New runnable demos: `examples/conv2d_im2col.py` and
  expanded `examples/verify_workloads.py` (11 checks).
- **Contribution scheduling** — a worker sets `MOREGPU_SCHEDULE` (`always` ·
  `idle-only` · `HH:MM-HH:MM` active window, may wrap midnight) to control *when*
  its machine is lent. Outside the window it takes no new work; in-flight shards
  always finish.
- **Remote fleet control** — `POST /workers/:id/control` lets an admin pause,
  resume, cap the duty ceiling, reschedule, relabel, or remove any worker; the
  coordinator pushes a control frame to that worker. Exposed in the dashboard
  (per-row controls, search, "pause/resume all") and the CLI (`moregpu pause /
  resume / set / schedule / rm / workers`).
- **High-worker-count admin UI** — the fleet table now searches, sorts by
  contribution, and caps the rendered rows (with a "showing N of M" count) so the
  dashboard stays responsive with many machines. No cap on how many can join.
- **Linux hardware isolation** — [`scripts/isolate-linux.sh`](scripts/isolate-linux.sh)
  (and `moregpu isolate`) pin a worker to a bounded cgroups-v2 scope (CPU quota,
  cpuset, memory cap, idle I/O), degrading gracefully when systemd/cgroups v2 are
  absent.
- **CLI & admin-UI ASCII banners** and richer `--help`.
- **`examples/verify_workloads.py`** — replays real GPU-user workloads (Linear,
  MLP, LayerNorm, softmax head, single-head attention) against a live pool.

### Changed

- Documented the honest execution split (matmul on the GPU; memory-bound
  elementwise/row-wise kernels on the worker CPU) and the real cryptography model
  (AES-256-GCM sealing, Ed25519 result signatures, single-trust-domain limits) in
  `SECURITY.md`, plus a hardware-grounded confidential-computing/TEE roadmap.
- Client SDKs distributed as GitHub Release artifacts (wheel + npm tarball);
  publishing to PyPI/npm/Homebrew documented in `CONTRIBUTING.md`.

### Security

Security hardening (34 confirmed findings):
- Reject `heartbeat`/`result` frames before a socket authenticates, and trust only
  the socket's own registered id — closes unauthenticated worker-state spoofing.
- Reject a `register` for an already-live id (worker-identity hijack); ban an
  admin-removed worker by its Ed25519 key; close sockets that never register.
- Sanitize worker id/backend/label (also blocks `/metrics` label injection).
- Worker requests the adapter's real buffer limits and falls back to CPU on GPU
  device loss. Evict old job records (and their output blobs) past a cap.

## [0.1.0] - 2026-07-25

Initial release: a native GPU compute pool with a networked coordinator and
cross-platform workers.

### Added

- **Tested core libraries** — a TypeScript monorepo (npm workspaces) of covered
  libraries: crypto, protocol, scheduler, integrity, transport, runtime, and the
  GPU layer.
- **Real GPU execution** — WGSL kernels run on the physical GPU via WebGPU
  (Metal / Vulkan / D3D12), with a CPU reference fallback so CPU-only machines
  contribute compute too.
- **Native compute pool** — jobs are row-sharded across workers, results are
  pooled and verified. Built-in task types: matmul and vector_add, extensible by
  adding a WGSL kernel plus a CPU reference.
- **Networked coordinator + worker** — an admin runs the coordinator/admin
  server; worker machines join the pool over an outbound WebSocket connection.
  Dashboard on `http://HOST:8787`; `wss` supported via TLS cert/key env vars.
- **First-run token wizard** — the coordinator's first run generates that pool's
  own admin token, worker join token, and encryption key, so every pool is
  isolated and nobody shares another pool's tokens.
- **Sealed jobs** — each work unit is AES-GCM sealed; only ciphertext travels on
  the wire between coordinator and workers.
- **CPU duty-cycle throttle** — CPU workers contribute with a configurable duty
  cycle (`MOREGPU_THROTTLE`, `MOREGPU_DUTY`) so the interactive user is not
  disturbed and power draw stays low.
- **One-liner installers + services** — cross-OS worker install via
  `scripts/install.sh` (Linux/macOS) and `scripts/install.ps1` (Windows). The
  installer self-heals, and `MOREGPU_SERVICE=1` installs a reboot-surviving,
  self-healing service (systemd / launchd / Windows scheduled task).

[Unreleased]: https://github.com/ArioMoniri/moregpu/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/ArioMoniri/moregpu/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ArioMoniri/moregpu/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ArioMoniri/moregpu/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ArioMoniri/moregpu/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ArioMoniri/moregpu/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ArioMoniri/moregpu/releases/tag/v0.1.0
