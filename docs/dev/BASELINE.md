# MoreGPU baseline — recon before the vision / training-task work

Snapshot of `main` @ `fc96ee3` (2026-09-23). This file records **what exists today**, verified against the
code, so the vision / training-task / JEPA work (proposed ADRs in [`docs/dev/adr/`](adr/)) starts from facts rather
than assumptions. No code was changed during this recon.

Legend: ✅ confirmed · 🟡 partly true / needs a nuance · ❌ not true

## 1. Claims we were asked to verify

| # | Claim | Verdict | Evidence / nuance |
|---|---|---|---|
| 1 | Native training is HF `AutoModelForCausalLM` + LoRA only | 🟡 | Causal-LM only (`worker_torch.py:324`), but the LoRA is **hand-written** (`LoRAWrap`, `worker_torch.py:281-319`), not `peft`. Default targets are `c_attn`/`q_proj`/`v_proj`, with r=8 and α=16. The optimizer is AdamW. There is no full fine-tune, no seq2seq and no classification head. |
| 2 | DiLoCo (`/train/diloco/load\|round\|adapter`) averages LoRA adapters only | ✅ | The coordinator is the parameter server (`server.ts:1422-1936`). Only `requires_grad` tensors travel, which after `attach_lora` means only A/B. Details in §3. |
| 3 | A single global `TRAIN` slot per worker, with a TODO | ✅ | The slot is `TRAIN: dict` (`worker_torch.py:321`). The TODOs are on the coordinator side (`server.ts:1789-1791`, `:1848-1851`). The coordinator also holds only one `trainingHome` (`:624`) and one `diloco` (`:1426`). |
| 4 | `/model/*` serving and sharding is LLM-specific | ✅ | The routes take `input_ids`, use a KV session, EOS and argmax, and parse layers with `LAYER_RE` (`server.ts:840`). WGSL weight names are hardcoded Llama names (`worker.ts:893-969`). |
| 5 | Download-free push (`push_begin/chunk/end`, RAM staging) is reusable | 🟡 | The mechanism is generic: chunks of ≤4 MiB, a 20 GiB cap, `/dev/shm` staging and tail-resume (`server.ts:717-827`, `worker_torch.py:510-603`). But **the source is HF-only** (`HF_REPO_RE`), **the weights must be safetensors**, `push_end` calls `from_pretrained`, and **nothing checks a hash**. `seq`/`last` are sent but ignored. It needs a generic "blob push" split out before vision can reuse it. |
| 6a | WGSL: tiled matmul, elementwise/GELU, softmax, layernorm | ✅ | `worker.ts:94-207`: `matmul` (16×16 tiled), `matmulF16`, `unary` (relu/scale/GELU-tanh), `binary`, `softmax`, and `layernorm` (no affine). |
| 6b | WGSL: conv2d via host im2col | 🟡 | im2col is **host Python in an example** (`examples/conv2d_im2col.py:20`), and the pool runs the matmul. There is **no conv kernel** in WGSL. |
| 6c | WGSL transformer stage, int8/int4 dequant-in-shader | ✅ | `SHARD_WGSL` (`worker.ts:792-828`) includes `linear*`, `rmsnorm`, `rope`, `swiglu`, `embed`, `cachedAttn` (head_dim ≤ 256), `linearQ8` and `linearQ4`. The fused decode path is opt-in. |
| 7 | Sealed per-worker transport | ✅ | AES-256-GCM, with a per-worker key from HKDF(master, id, epoch) (`server.ts:96-114`) and Ed25519-signed results. **There is no key exchange**: the per-worker key is sent in `welcome`, so it relies on TLS (`wss://`, self-signed, SHA-256 pinned). |
| 8 | Adaptive throttle | ✅ | A worker-side duty cycle from the loadavg EMA (`worker.ts:75-87`). It is **not applied inside torch `train`/`model` RPCs**. |
| 9 | Schedules | ✅ | Settings are `always`, `idle-only` or `HH:MM-HH:MM`. The coordinator reconciles them every 30 s and pauses the worker (`server.ts:524-549`). |

## 2. Architecture facts that shape the design

- **Production is three self-contained files:**
  - `apps/coordinator/server.ts` (3.2k lines, Deno)
  - `apps/worker/worker.ts` (1.5k lines; Deno and browser)
  - `apps/worker/worker_torch.py` (1.9k lines)

  `packages/*` is a **reference layer that is not on the live wire** (see `packages/README.md`), and its vitest suites
  do not cover the apps. New features have to go into `apps/*`. Our plan is to put them in new **modules** there
  instead of growing the monoliths (proposed ADR-0103).
- **Release signing covers single files.** `worker.ts.sig` and `worker_torch.py.sig` are verified by
  `tests/security/release_verify.py`. Splitting the torch worker into a package changes what gets signed
  (proposed ADR-0103).
- **The Dockerfile copies only `server.ts`.** New coordinator modules must be added to it.
- **Wire and RPC.** The coordinator relays `{t:'train'|'model', reqId, op, sealed}` to torch workers. Tensors travel as
  **base64 of flat f32 inside sealed JSON**. That works for MB-scale LoRA, but it is the wrong format for full-weight sync
  (a ViT-Small is ~88 MB in f32, ~117 MB in base64, per worker per direction) (proposed ADR-0106).
- **Capabilities.** Each worker has internal caps (`kernel`, `shard`, `shardEnds`, `resident`, `train`), and some routing
  still checks whether the label contains `torch`. `/device` describes the **pool**, not individual workers. Per-worker
  vision/data capabilities need a new field.
- **`/net`** reports the minimum RTT over 4 pings and one 512 KiB upload probe per worker. It has no sustained bandwidth
  measurement, no RTT percentiles and no download direction (planned in A.7).
- **Scheduler.** Production splits jobs evenly across the active fleet, retries up to 3 times, and has no work-stealing
  or capacity weighting. The heterogeneity-aware `assignShards` in `packages/scheduler` exists but is unused.
- **Dashboard.** Hand-written HTML inside `server.ts`, with SVG sparklines and no chart library.

## 3. DiLoCo today (exact semantics; generalised DiLoCo must reproduce this for `llm_lora`)

- `load` loads the same model on every torch worker with the same seed and `no_dropout`. The initial global adapter is
  the first OK worker's adapter, and momentum starts at 0.
- On each `round`, every live worker runs `train_inner(steps=H, lr)` with a **fresh AdamW** over
  `batches[wid] ?? batches['*']` and returns its adapter.
- Workers with any non-finite value are dropped. If none survive, the route returns 502 and the global is unchanged.
- The average is **unweighted**, per tensor, over the survivors.
- Outer step, with Δ = global − avg:
  - `v = μ·v + Δ`
  - `global −= η·(Δ + μ·v)`

  This is PyTorch `SGD(nesterov=True)` with no dampening. Defaults are η=0.7, μ=0.9, H=4, lr=1e-3.
- The new global is broadcast with `set_adapter`. A worker whose broadcast fails is dropped, and the group dissolves if
  all of them fail. Rounds are serialized by `busy`.
- There is no checkpoint and no resume of global or momentum state, no samples-seen accounting and no weighting.

## 4. Tests and CI today

- **CI** (`.github/workflows/ci.yml`) has three jobs:
  - `node`: build plus vitest over `packages/**`.
  - `deno`: type-check only.
  - `native`: Python 3.12 with CPU torch and `transformers<5`. It runs 21 standalone scripts (`python3 tests/...py`,
    each exiting non-zero on failure) across e2e, fault, security and scale. **pytest is not used.**
- There is **no GPU runner and no GPU marker**. The only skip is `_skip_on_tf5()` in the MoE parity test.
- `npm run lint` is **not** run in CI.
- There are no WGSL kernel unit tests. WGSL correctness comes only from the LLM parity e2e tests.
- There is no worker `requirements`/`pyproject`, and no `MOREGPU_EXTRAS`. The only pyproject is the stdlib-only SDK
  (`clients/python`).

## 5. Surfaces today

- **CLI** (`scripts/moregpu`, bash): `serve`, `join` (Deno worker), `install`, `isolate`, `finetune`, `generate`, `chat`,
  `shard`, `stop`, `status`, `logs`, `workers`, `pause`, `resume`, `set`, `schedule`, `rm`, `monitor`, `version`, `menu`.
  There is **no DiLoCo command, and nothing launches `worker_torch.py`**.
- **Python SDK**: jobs, kernels, `train_*`, `diloco_*`, `model_*`, `shard_*`.
- **TS SDK**: jobs and kernels only, with **no train, DiLoCo, model or shard methods**.

## 6. Security and hygiene findings noticed in passing (not fixed here)

1. The **push path has no content hash**. A file's integrity rests on TLS to HF plus the per-chunk GCM seal. The new
   model spec (A.1) requires sha256, and the generic blob push must verify it.
2. The non-push `from_pretrained(hub_id)` calls in `train_load`, `model_load`, `shard_load` and the MoE loaders do not
   pass `use_safetensors=True`. A repo that ships only `pytorch_model.bin` would therefore be unpickled by transformers,
   and whether `weights_only` applies depends on the transformers version. The planned `weights_only=False` ban
   (A.1) should also require `use_safetensors=True`.
3. The repo has no `torch.load` or `pickle` today. The ban test can therefore start green.
4. `SECURITY.md` is stale in two places:
   - It says unsigned workers are accepted, but the code rejects a missing pubkey.
   - It gives the shard timeout as 120 s, but the code uses 60 s.
5. **`docs/adr/` and `docs/ROADMAP.md` are in `.gitignore`**. They are kept local on purpose, so the proposals live
   in `docs/dev/adr/` until you decide.

## 7. Implications for the plan

- **M1 is mostly refactor-under-test.** The LLM DiLoCo e2e (`tests/e2e/train_diloco.py`) and
  `tests/e2e/train_single.py` are the regression net for "no behaviour change".
- A new Python unit-test layer (pytest) and a TS unit layer for new coordinator modules (vitest already includes
  `apps/**/*.test.ts`) are prerequisites for strict TDD (proposed ADR-0102).
- Full-weight sync needs a binary chunked tensor transport (proposed ADR-0106) before JEPA-on-DiLoCo is practical beyond
  toy sizes.
