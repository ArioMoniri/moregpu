# Changelog

All notable changes to MoreGPU are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — vision, generic training tasks, JEPA (0.7.0-dev)

The work is planned in ADRs 0101–0114 (`docs/dev/adr/`) and verified against the pre-work baseline in
`docs/dev/BASELINE.md`.

- **Training-task framework.**
  - `TrainTask` interface with a pinned plugin registry (entry points, allowlisted by dist, version and wheel sha256).
  - Per-session training state replaces the single global `TRAIN` slot. The legacy `/train` and `/train/diloco` routes
    are unchanged; their e2e tests still pass.
  - `/train/sessions` API, available from the Python SDK, the TS SDK, the CLI (`moregpu train …`) and the dashboard.
- **Generalised DiLoCo.**
  - Syncs full weights or adapters, averaged with sample weights; outer Nesterov step with fp32 outer state.
  - Deterministic SplitMix64 sample stream, with cross-language golden tests.
  - Heterogeneous `proportional` allocation.
  - `target_samples` stops exactly at the requested count.
  - Warmup + cosine LR per round.
  - Chunked, sha256-checked tensor sync in f32, bf16, fp16 or int8-delta.
  - Checkpoint and resume (bit-identical), and churn handling.
  - The coordinator result equals the in-process reference within 1e-7.
- **AMP policy.** bf16, fp16 + GradScaler, or fp32. A forced mode the device cannot run is an error, and the mode used is
  recorded.
- **JEPA** (`ijepa_2d`, `jepa_2p5d`, `jepa_3d`).
  - Model: ViT in 2D/3D with timm-compatible keys, plus a predictor, trained with I-JEPA multi-block masking.
  - EMA is updated after each outer step with product-of-schedule momentum. N=1, H=1 DiLoCo equals per-step I-JEPA,
    and target hashes are checked across workers.
  - Collapse monitors: per-dimension std and RankMe, with alarms.
  - Evaluation: k-NN and linear probe.
  - Export to safetensors, torch.export or ONNX, with a parity probe.
- **Vision fine-tuning.**
  - `segment` (2D, 2.5D and 3D; Dice+CE matching MONAI) and `classify`, on a random or JEPA-exported encoder, in full,
    frozen or LoRA mode.
  - `finetune_model` fine-tunes **any published native model** (torchvision, timm, MONAI, HF, plugin) with DiLoCo.
- **Published models, run exactly as released.**
  - Adapters for state_dict, safetensors + named arch, allowlisted plugins, torch.export, TorchScript and ONNX.
  - sha256 is verified. Pickled full models are refused, including a crafted `.pt2` pickle fallback.
  - Automatic lowering to a MoreGPU op-graph or ONNX, with a parity probe.
  - The repo-wide pickle/`weights_only` ban test passes. The LLM loaders now use `use_safetensors=True`.
- **Vision inference.**
  - `/vision/load|infer|batch|jobs|unload|models|lower|capabilities`.
  - Sliding-window inference with flip TTA, equivalent to MONAI.
  - Pull-based work-stealing batch queue with churn retry and straggler duplication.
  - **Tile sharding**: one volume split across workers, giving exactly the single-node result.
- **WebGPU vision (M6).** WGSL conv2d/3d (implicit GEMM), transposed conv, norms, pooling, upsampling, concat, softmax,
  argmax, attention and fp16 storage, with a memory planner and a sliding-window driver. Verified on real software
  adapters: Mesa lavapipe under Deno and SwiftShader in Chromium.
- **Data plane.**
  - Sources: `file://` under allowed roots, `https://` from an allowlist with sha256, anonymous public buckets, and
    `pushed://` RAM-staged blobs.
  - Readers for NumPy, NIfTI, DICOM, PNG/JPEG and TIFF.
  - Content-addressed LRU cache, memmap shards and a prefetching loader.
  - Per-worker capabilities at `/workers/:id/caps`.
- **Telemetry and `moregpu bench`.**
  - Versioned `moregpu.telemetry/1` JSONL: compute, data, serialise, network and wait times, bytes, GPU utilisation,
    energy (NVML), peak memory, AMP mode, hardware fingerprint and config hash.
  - `/net` reports RTT p50/p90/p99 and sustained up/down bandwidth.
  - `moregpu bench` emulates VRAM, CPU, memory and network limits.
- **Research boundary guard** (`tests/boundary`), plus unit tests (pytest, ≥90 % coverage of `moregpu_worker`) and
  vitest suites for the coordinator libs.
- **`MOREGPU_VRAM_FRACTION`** on native torch workers. The torch worker used to ignore it. It now calls
  `torch.cuda.set_per_process_memory_fraction(f)` on every visible CUDA device at start-up. NaN, infinities,
  non-numbers and values outside `(0, 1]` stop the worker with an error. The applied value is reported as
  `hw.vram_fraction` in the hardware fingerprint, and so in every telemetry record (docs/ADMIN.md, docs/TRAINING.md).
- **Encoder and model weights from pushed blobs.** A segment/classify `encoder: {init: "export", path}` and a
  `finetune_model` spec `source` accept `pushed://<id>` (a `/data/push` BlobStore blob, `sha256` required) as well
  as a path. Both kinds of source are size-capped at `MOREGPU_MODEL_MAX_BYTES`, re-hashed against `sha256`, and must
  be safetensors: a pickle is refused before anything parses it. Paths stay confined to `MOREGPU_OUTPUT_DIR` ∪
  `MOREGPU_MODEL_ROOTS`, and an export directory's `sha256` is now checked when given. JEPA exports store their
  encoder config in the safetensors metadata, so the file alone is enough. New SDK helpers `push_safetensors` (Python)
  and `pushSafetensors` (TS), and CLI flag `--encoder-sha256` (docs/TRAINING.md#initial-weights-a-path-or-a-pushed-blob).
- **`pred_sha256` on predictions.** `vision_predict` and `vision_merge_write` results, `/vision/batch` job records,
  `/vision/infer` and each `/vision/infer_batch` output carry a deterministic sha256 of the predicted label volume
  (`moregpu.pred/1`: dtype and shape in the preimage, then uint8/uint16 labels in C order). The coordinator computes it
  from the argmax of torch and WebGPU replies alike. A cross-language golden (`tests/goldens/pred_sha256.json`) checks
  the Python and TypeScript implementations (docs/VISION.md#prediction-hashes-pred_sha256).

### Fixed (0.7.0-dev)

- Label maps with a class above 255 are written as `uint16`. They used to wrap to `uint8`.
- A pushed blob `suffix` may have parts of up to 16 characters (for example `.safetensors`). The limit used to be 8.
- `pushed://` model sources are capped at `MOREGPU_MODEL_MAX_BYTES`, like `https://` downloads.

### Security

- Model and weight loads refuse pickles, `trust_remote_code` and `from_pretrained` without safetensors, enforced by a CI
  ban test.
- Data and model roots are realpath-confined, and https hosts come from an allowlist with sha256 required.
- Bucket tools run with credentials stripped.
- Outputs are confined to `MOREGPU_OUTPUT_DIR`.
- **The torch worker requires `torch>=2.6`** (CVE-2025-32434: `torch.load(weights_only=True)` is bypassable on
  ≤ 2.5.1). The adapters check `torch.__version__` at load time and refuse `state_dict` `.pt/.pth`, `torch_export`
  `.pt2` and TorchScript loads on an older torch, on a 2.6.0 pre-release, or on a version they cannot parse.
- **Export confinement** (`moregpu_worker.paths.confine`). Every `TrainTask.export` (JEPA, segment/classify,
  `finetune_model`) and `task_export` write only inside `MOREGPU_OUTPUT_DIR` (default `./moregpu-out`). A relative path
  resolves inside it; an absolute path must realpath inside it, so `../x`, outside paths and symlink escapes are
  refused. Export reads (`vision_infer_load {export}`, `encoder: {init: "export"}`) are confined to
  `MOREGPU_OUTPUT_DIR` ∪ `MOREGPU_MODEL_ROOTS`.
- **Model downloads use the data plane's policy.** `https://` model sources need a host in `MOREGPU_MODEL_HOSTS`
  (falls back to `MOREGPU_DATA_HOSTS`). Every redirect hop is re-checked, the size is capped by
  `MOREGPU_MODEL_MAX_BYTES` (default 20 GiB), and sha256 stays mandatory. `hf://` sources need a `sha256` or a
  40-hex commit revision; branches and tags are refused.
- **`pushed://` model sources resolve through the data plane's BlobStore**: only a blob that was fully pushed and
  verified at `blob_end` resolves. `MOREGPU_PUSHED_DIR` is now only an alias for the staging dir
  (`MOREGPU_STAGE_DIR`), not a directory of loadable files.
- **Signed torch-worker tree.** `MANIFEST.sha256` covers every file under `apps/worker` except `*.sig` and the
  manifest itself (including `worker_torch.py` and `pyproject.toml`). It carries a version header that must equal
  `pyproject.toml`'s version. The verifiers (`release_sign.py verify-manifest`, `verify_release.ts --manifest-root`)
  refuse `__pycache__` directories, stray `.pyc` files, symlinked directories and any unlisted file anywhere in the
  tree, such as a planted `numpy.py` next to `worker_torch.py`. A version problem exits with code 6: a missing or
  mismatched header, `--expect-version`, or a rollback below the version recorded by `--state`.
  `moregpu torch-join` purges bytecode caches and runs `verify-manifest --state` before it execs the worker, then runs
  it with `python3 -B`. It verifies whenever `MANIFEST.sha256` and its `.sig` exist. Without them it warns "unsigned dev
  tree". `MOREGPU_VERIFY_MANIFEST=1` makes the manifest mandatory, and `=0` skips the check with a warning.
- Blob staging caps: `MOREGPU_BLOB_TOTAL_MAX_BYTES` (default 40 GiB) across all staged blobs. `blob_begin` is refused
  when the staging filesystem has less free space than the blob, and `/dev/shm` is only used when the blob fits there.
- Bucket refs with wildcard or glob characters are refused. Bucket objects are streamed (`cat`), and the download is
  cut off at `MOREGPU_DATA_MAX_DOWNLOAD_BYTES`.
- `.pt2` archives get zip-bomb caps: `MOREGPU_PT2_MEMBER_MAX_BYTES` (8 GiB) and `MOREGPU_PT2_TOTAL_MAX_BYTES` (32 GiB)
  uncompressed. Members are read as streams.
- `vision_unload` also drops the `vision_infer_*` models that wrap the handle. A coordinator `welcome` clears adapter
  handles, lowered artefacts, `vision_infer_*` models and staged data-plane blobs.
- The telemetry host hash uses a random salt drawn once per worker process, so it cannot link a machine across runs.


### Added

- **Mixed torch + WebGPU vision fleet.** `/vision/load {fleet: 'webgpu'|'all'}` lowers the model on a torch worker
  and streams the op-graph + safetensors to every worker with the `vision` cap. `/vision/infer` routes to either kind,
  and `/vision/infer_batch` spreads tensors over all holders with the work-stealing queue (optional `check_parity`).
  The Python lowering now emits exactly the WGSL executor's schema (`vision_ops.json`, default `torch.export`
  dialect). Python-lowered UNet/ViT graphs equal PyTorch within 1e-5 on the TS executor and on lavapipe. The executor
  gains `adaptive_avg_pool2d/3d` (divisible sizes only). `vision_wgsl.ts` is a second signed installer artefact that
  fails soft, and the torch worker tree gets a signed `MANIFEST.sha256` (ADR-0103). CI adds a `webgpu` job.
- **Pipeline sharding now works for Llama-family models**, not just GPT-2 — the torch worker's
  `shard_load`/`shard_forward` detect the architecture (GPT-2 `transformer.h` + learned positions vs
  Llama-style `model.layers` + RMSNorm + RoPE) and pipe activations through either. Verified token-for-token
  **exact match** for GPT-2 (6+6 split) *and* **SmolLM-135M** (15+15 split) across 2 workers; peak ~1.1 GB
  RSS for the two CPU stages. GPT-2 path unchanged (regression-checked).
- **Coordinator container on GitHub Packages** — [`Dockerfile`](Dockerfile) + a
  [`container`](.github/workflows/container.yml) workflow publish `ghcr.io/ariomoniri/moregpu` on release:
  `docker run -p 8787:8787 -v moregpu:/data ghcr.io/ariomoniri/moregpu:latest`. README shows the published
  packages (PyPI · ghcr · Homebrew) and a PyPI badge; the `moregpu` CLI menu renders as
  [`docs/assets/cli-menu.svg`](docs/assets/cli-menu.svg).

### Changed

- **WAN robustness** (from a live heterogeneous test — a Colab **T4 (CUDA)** worker + this Mac **MPS** worker on
  one pool over a public tunnel): the torch worker's WebSocket keepalive is now WAN-tolerant (`ping_timeout`
  90s, env-overridable) so a high-latency tunnel no longer drops it; and the Python SDK retries transient
  gateway errors (502/503/504, connection resets). **Verified live:** GPT-2 served on the Colab T4 through the
  pool at ~44 tok/s (over the tunnel), token-for-token exact match; a 2-node CUDA+MPS fleet; **heterogeneous
  DiLoCo** training across the Colab CUDA GPU + the Mac MPS GPU (loss 4.45 → 1.64 over 3 rounds, coordinator
  averaging); GPT-2 shard stages loaded across CUDA↔MPS. Honest finding: fp16 ≈ fp32 for small-model
  single-stream decode on the T4 (the GEMMs are memory-bound; tensor cores only help large fp16 GEMMs).
- Honesty: the CUDA/PTX matrix row now notes the torch worker **does run on CUDA via PyTorch** on NVIDIA
  (real acceleration for serving/training/sharding) — what's absent is *custom* CUDA/PTX kernels,
  tensor-core/int8 GEMM, and graphics/codec paths.
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
