# Using MoreGPU for AI / ML and other parallel workloads

MoreGPU is a native GPU compute **pool**, not a model server. An admin runs a coordinator (a Deno
HTTP + WebSocket server); worker machines join over an outbound WebSocket and contribute their GPU
(WebGPU → Metal / Vulkan / D3D12) or CPU. You submit **jobs**; the coordinator **shards** each job
across the fleet, **AES-256-GCM seals** every work unit on the wire, runs each shard on a worker, then
**pools** the results back and **verifies** them against a CPU reference.

You can submit in two modes:

- **Data mode** — you send your own tensors and get the computed result back (sealed end-to-end). This
  is how a real application uses the pool.
- **Benchmark mode** — you send only `{kernel, size}`; the pool generates random data and returns timing
  and verification, for load-testing and demos.

> **Honesty first.** The **portable (WebGPU/WGSL) path** provides a fixed set of **linear-algebra /
> elementwise building-block kernels** and pools them across machines. On that path MoreGPU does **not**
> run a whole trained model as a black box, does **not** do autograd/backprop, and does **not** use tensor
> cores or int8 — `matmul` is a workgroup-tiled (shared-memory) WGSL shader on general-purpose GPU compute,
> fp32/fp16 only. `matmul` runs on the worker's GPU; the memory-bound elementwise and row-wise kernels run
> on the worker's CPU. Applications get value by **composing** these primitives.
>
> A separate, **opt-in native torch worker** ([`apps/worker/worker_torch.py`](../apps/worker/worker_torch.py))
> is where whole-model residency and **autograd** live: it holds a model on-device and runs the entire
> forward per call (fast serving) or a full LoRA train step locally (single-worker fine-tuning). It is
> admin-installed on trusted hardware — not the zero-install WGSL worker — and speaks the same sealed
> protocol. See [Scope](#in-scope-vs-out-of-scope).

---

## Kernels

| Kernel        | Operation                    | Data-mode inputs                 |
| ------------- | ---------------------------- | -------------------------------- |
| `matmul`      | `C[M×N] = A[M×K] · B[K×N]`   | `a` (M·K), `b` (K·N), `M,N,K`    |
| `vector_add`  | `out[i] = a[i] + b[i]`       | `a`, `b`                         |
| `vector_mul`  | `out[i] = a[i] * b[i]`       | `a`, `b`                         |
| `saxpy`       | `out[i] = α·a[i] + b[i]`     | `a`, `b`, `scalar`               |
| `relu`        | `out[i] = max(0, a[i])`      | `a`                              |
| `gelu`        | `out[i] = gelu(a[i])` (tanh) | `a`                              |
| `scale`       | `out[i] = α·a[i]`            | `a`, `scalar`                    |
| `softmax`     | row-wise softmax             | `a`, `N` (= columns per row)     |
| `layernorm`   | row-wise LayerNorm (ε 1e-5)  | `a`, `N` (= columns per row)     |

`softmax` and `layernorm` are **per-row** reductions over a `rows × N` matrix (the coordinator shards
them by whole rows). Together with `matmul` and the elementwise ops they are enough to compose a real
transformer block.

Tensors are 32-bit floats, sent as base64 of the little-endian `Float32Array` bytes. The clients below
handle that encoding for you. Add a kernel by following [Adding a kernel](#adding-a-kernel).

---

## Data mode — send tensors, get results

### Python ([`examples/moregpu_client.py`](../examples/moregpu_client.py), dependency-free)

```python
from moregpu_client import MoreGPU
pool = MoreGPU("http://ADMIN:8787", "<admin-token>")

# C = A(2×3) · B(3×2)
C = pool.matmul([1,2,3, 4,5,6], [7,8, 9,10, 11,12], M=2, N=2, K=3)   # → [58, 64, 139, 154]

job = pool.run("relu", [-1, 2, -3, 4])       # returns the job dict + 'output_decoded'
assert job["status"] == "done"
print(job["output_decoded"])                 # → [0.0, 2.0, 0.0, 4.0]
```

### JavaScript / TypeScript ([`@moregpu/client`](../packages/client))

```ts
import { MoreGPUClient } from '@moregpu/client';
const pool = new MoreGPUClient({ baseUrl: 'http://ADMIN:8787', adminToken: '<admin-token>' });

const C = await pool.matmul([1,2,3, 4,5,6], [7,8, 9,10, 11,12], 2, 2, 3);  // Float32Array [58,64,139,154]

const { job, output } = await pool.run('relu', { a: [-1, 2, -3, 4] });
if (job.status === 'done') console.log(Array.from(output));                // [0, 2, 0, 4]
```

### Composing a forward pass

The pool executes single ops; your application composes them. A dense layer `relu(X·W + b)`:

```python
Y   = pool.matmul(X, W, M=rows, N=cols, K=inner)                        # X·W
Yb  = pool.run("vector_add", Y, tile_bias(b, rows))["output_decoded"]   # + bias (broadcast in your code)
out = pool.run("relu", Yb)["output_decoded"]                            # activation
```

Because `matmul` is the dominant cost in almost every network (dense layers, attention's `QKᵀ` and
`·V`, im2col convolutions), pooling matmul is what lets the fleet do ML-shaped work.

**A real attention block, entirely on the pool** — `softmax(Q·Kᵀ / √d) · V`:

```python
scores  = pool.matmul(Q, Kt, M=T, N=T, K=d)                     # Q·Kᵀ
scaled  = pool.run("scale", scores, scalar=1/ d**0.5)["output_decoded"]
weights = pool.run("softmax", scaled, N=T)["output_decoded"]    # row-softmax
out     = pool.matmul(weights, V, M=T, N=d, K=T)                # weights·V
```

A runnable version is [`examples/attention_demo.py`](../examples/attention_demo.py) (matches a reference
to ~1e-8). The realistic target class is a **small FP32 encoder** (sentence embeddings, classification,
retrieval re-ranking) or an MLP, run as batched prefill — not a multi-billion-parameter autoregressive
chat model, which needs tensor-core GEMM the WGSL tier does not provide.

---

## Batching — how parallelism actually works

Two independent levels:

1. **Within one job:** the coordinator splits a single job into **at most one shard per connected
   worker** and runs those shards in parallel. Effective parallelism for one job equals the fleet size;
   it is not tunable by `size`.
2. **Across jobs:** submitted jobs are queued and executed **sequentially** (one job at a time, each
   sharded across the whole fleet). Submitting concurrently keeps the queue full but does not run two
   jobs at once.

So: make each job as large as is reasonable (one big matmul beats many tiny ones — it amortises the
per-shard sealing and dispatch), and keep a few submissions in flight so the queue never drains.
`POST /submit` blocks briefly for a synchronous response but may still return a non-terminal job
(`status` `queued`/`running`) — always branch on `status`.

---

## Verification and limits

- `verified` is set when the pool re-computed the result on the CPU reference and it matched within
  tolerance. It is computed for **every elementwise job**, and for **matmul when M·N ≤ 640·640**
  (verifying a huge matmul on one CPU would dominate the wall time). For a large matmul, `verified` is
  `undefined` — the result is still returned; use integer/known-answer inputs to self-check if you need
  assurance.
- `gflops` is reported for **matmul only** (`2·M·N·K / t`).
- Size clamps: `matmul` dimension ≤ 2048; elementwise length ≤ 8,000,000 (benchmark mode). Data-mode
  input buffers are capped at 16,000,000 elements each.

A job record:

```jsonc
{
  "id": "job-12", "status": "done",   // queued | running | done | failed
  "kernel": "matmul", "size": 512,
  "ms": 12.4,
  "gflops": 173.0,            // matmul only
  "verified": true,          // present when computed (see above)
  "shards": [ { "worker": "gpu-A", "backend": "gpu:…", "work": 262144, "ms": 3.1 } ],
  "output": "…base64…",      // data mode only (decoded for you by the clients)
  "outLen": 262144
}
```

---

## Adding a kernel

A kernel lives in two self-contained places (the deployable Deno apps):

1. **Worker** — [`apps/worker/worker.ts`](../apps/worker/worker.ts): add the WGSL to the `WGSL` map (or,
   for a memory-bound op, a branch in `runElementwise`) and dispatch it in the assign handler.
2. **Coordinator** — [`apps/coordinator/server.ts`](../apps/coordinator/server.ts): add the name to
   `KERNELS` (and `ELEMENTWISE` if index-sharded), a CPU reference branch in `cpuKernel`/`cpuMatmul`
   (the verification oracle), and the sharding in `runJob`.

The tested reference kernels in [`packages/gpu`](../packages/gpu) are the design source for the WGSL and
CPU-reference pair; mirror them. Keep the CPU reference exact so verification stays meaningful.

---

## API reference

All admin endpoints require `Authorization: Bearer <admin token>`.

| Endpoint | Purpose |
| --- | --- |
| `POST /submit` | `{kernel, size}` (benchmark) or `{kernel, a, b?, scalar?, M?, N?, K?}` (data mode) → job |
| `GET /gpu` · `GET /device` | pool as one virtual GPU: slots, totals, throughput; `/device` is the GPU-slot descriptor |
| `GET /workers` · `POST /workers/:id/control` | per-worker contribution; admin pause/resume/duty/schedule/remove |
| `POST /weights`, `GET /weights` | pin a named weight resident on a worker (weight residency / pipeline) |
| `GET /jobs`, `GET /jobs/:id` | queue + history (list omits `output`; `/jobs/:id` includes it) |
| `GET /logs` · `GET /metrics` · `GET /health` | recent log · Prometheus metrics · liveness + fleet (public) |

**Native-tier endpoints** (need a torch worker — see [`apps/worker/worker_torch.py`](../apps/worker/worker_torch.py)):

| Endpoint | Purpose |
| --- | --- |
| `POST /model/load` · `/model/forward` · `/model/generate` · `/model/unload` | resident-model serving (whole forward on the worker; `generate` = whole decode, one round-trip) |
| `POST /model/shard` · `/model/shard_forward` · `/model/shard_unload` | pipeline-parallel sharding across workers (activations-only on the wire) |
| `POST /train/load` · `/train/step` · `/train/adapter` | single-worker LoRA fine-tuning (whole train step local on the worker) |
| `POST /train/diloco/load` · `/train/diloco/round` · `/train/diloco/adapter` | distributed LoRA via DiLoCo (coordinator = parameter server) |

---

## In scope vs out of scope

**In scope** — embarrassingly-parallel building blocks the pool runs and verifies: dense `matmul`,
batch elementwise (`vector_add/mul`, `saxpy`, `relu`, `scale`), and anything you can express by composing
them (linear layers, activations, Monte-Carlo-style batch math). Large FP32 batch work on idle machines.

**On the portable WGSL path, out of scope** — running a whole trained model as a black box,
autograd/backprop and optimizer state, tensor-core / cuBLAS-class GEMM performance, low-latency
single-token serving, and any op not in the kernel set (until you add it). The WGSL path gives you fast,
pooled, verified primitives — the model is your application's to compose.

**The opt-in native torch worker** lifts some of these on trusted hardware: whole-model residency +
device-resident forward (fast small-model serving, exact match), and **LoRA fine-tuning** via torch
autograd — single-worker *and* **distributed across many workers via DiLoCo** (each worker trains its own
shard for H local steps; the coordinator averages the adapters + an outer Nesterov step — a genuine reduce
path). Still out of scope: 7B+ single-model inference, **async DiLoCo** and **cross-tenant secure
aggregation**, tensor cores/int8, and CUDA/PTX kernels — all roadmap. Training on the native tier is
verified out-of-band (a seeded reference), since the coordinator's CPU exact-match check can't validate a
stochastic loss.
