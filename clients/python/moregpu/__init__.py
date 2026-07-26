"""moregpu — a tiny, dependency-free Python client for a MoreGPU pool.

    from moregpu import MoreGPU
    pool = MoreGPU("http://ADMIN:8787", "<admin-token>")
    pool.matmul([1,2,3, 4,5,6], [7,8, 9,10, 11,12], M=2, N=2, K=3)   # → [58, 64, 139, 154]
    pool.run("relu", [-1, 2, -3, 4])                                  # any kernel on your own data
    pool.linear(X, W, b, M=4, K=3, N=2)                               # dense layer
    pool.attention(Q, K, V, seq=2, d=2)                               # one attention head
    pool.device()                                                     # the pool as a GPU slot
"""
from __future__ import annotations
import array
import base64
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

try:  # keep __version__ in sync with the installed distribution, not a hand-edited string
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("moregpu-client")
except Exception:
    __version__ = "0.3.0"
KERNELS = ("matmul", "vector_add", "vector_mul", "saxpy", "relu", "scale", "gelu", "softmax", "layernorm")


def _f32_b64(values: Sequence[float]) -> str:
    return base64.b64encode(array.array("f", values).tobytes()).decode()


def _f16_b64(values: Sequence[float]) -> str:  # array has no half typecode; struct '<e' packs IEEE f16
    import struct
    v = list(values)
    return base64.b64encode(struct.pack(f"<{len(v)}e", *v)).decode()


def _b64_f32(s: str) -> list[float]:
    a = array.array("f"); a.frombytes(base64.b64decode(s)); return list(a)


def _transpose(x: Sequence[float], rows: int, cols: int) -> list[float]:
    return [x[r * cols + c] for c in range(cols) for r in range(rows)]


class MoreGPU:
    def __init__(self, base_url: str, admin_token: str, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.token = admin_token
        self.timeout = timeout

    def _req(self, path: str, method: str = "GET", body: dict | None = None, auth: bool = True) -> Any:
        import time, urllib.error
        data = json.dumps(body).encode() if body is not None else None
        tries = int(getattr(self, "retries", 4)); last: Exception | None = None
        for attempt in range(tries):  # retry transient gateway/tunnel errors (502/503/504, conn resets) over a WAN
            req = urllib.request.Request(self.base + path, data=data, method=method)
            if body is not None:
                req.add_header("content-type", "application/json")
            if auth:
                req.add_header("authorization", f"Bearer {self.token}")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (502, 503, 504) and attempt < tries - 1:
                    last = e; time.sleep(0.6 * (attempt + 1)); continue
                raise
            except (urllib.error.URLError, ConnectionError) as e:
                if attempt < tries - 1:
                    last = e; time.sleep(0.6 * (attempt + 1)); continue
                raise
        raise last  # type: ignore[misc]

    def health(self) -> dict: return self._req("/health", auth=False)
    def device(self) -> dict: return self._req("/device")
    def gpu(self) -> dict: return self._req("/gpu")
    def workers(self) -> list[dict]: return self._req("/workers")
    def job(self, jid: str) -> dict: return self._req(f"/jobs/{jid}")
    def jobs(self) -> list[dict]: return self._req("/jobs")

    def submit(self, kernel: str = "matmul", size: int = 512) -> dict:
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}; choose from {KERNELS}")
        return self._req("/submit", "POST", {"kernel": kernel, "size": size})

    def submit_batch(self, specs: list[tuple[str, int]], workers: int = 8) -> list[dict]:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda s: self.submit(*s), specs))

    def run(self, kernel: str, a: Sequence[float], b: Sequence[float] | None = None,
            scalar: float | None = None, M: int | None = None, N: int | None = None, K: int | None = None) -> dict:
        """Data mode: send your own tensors, get the pooled result (adds 'output_decoded').
        Raises if the job did not reach 'done' (so a queued/failed job never silently yields [])."""
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}")
        body: dict[str, Any] = {"kernel": kernel, "a": _f32_b64(a)}
        if b is not None:
            body["b"] = _f32_b64(b)
        for name, val in (("scalar", scalar), ("M", M), ("N", N), ("K", K)):
            if val is not None:
                body[name] = val
        job = self._req("/submit", "POST", body)
        if job.get("status") != "done":
            raise RuntimeError(f"job {job.get('id')} not done: status={job.get('status')} error={job.get('error') or job.get('note')}")
        job["output_decoded"] = _b64_f32(job["output"]) if job.get("output") else []
        return job

    # ---- primitives ----
    def matmul(self, A: Sequence[float], B: Sequence[float], M: int, N: int, K: int) -> list[float]:
        """C = A(M×K) · B(K×N), computed and verified on the pool."""
        return self.run("matmul", A, B, M=M, N=N, K=K)["output_decoded"]

    # ---- composition helpers for inference (compose the primitives above) ----
    def linear(self, X: Sequence[float], W: Sequence[float], b: Sequence[float] | None,
               M: int, K: int, N: int) -> list[float]:
        """Dense layer y = X(M×K)·W(K×N) [+ b(N)], bias row-broadcast (bias add is done client-side)."""
        y = self.matmul(X, W, M, N, K)
        if b is not None:
            for i in range(M):
                for j in range(N):
                    y[i * N + j] += b[j]
        return y

    def mlp(self, X: Sequence[float], layers: Sequence[tuple], M: int, K: int, act: str = "relu") -> list[float]:
        """Chain dense layers with an activation. layers = [(W, b_or_None, out_dim), ...]."""
        cur, k = list(X), K
        for (W, b, out_dim) in layers:
            cur = self.linear(cur, W, b, M, k, out_dim)
            cur = self.run(act, cur)["output_decoded"]
            k = out_dim
        return cur

    def attention(self, Q: Sequence[float], K: Sequence[float], V: Sequence[float],
                  seq: int, d: int, scale: float | None = None) -> list[float]:
        """Single-head scaled dot-product attention softmax(Q·Kᵀ/√d)·V, composed + verified per step.
        Q, K, V are row-major seq×d. Returns seq×d. (This is the verify_workloads.py path.)"""
        s = scale if scale is not None else 1.0 / math.sqrt(d)
        Kt = _transpose(K, seq, d)                                  # d×seq
        scores = self.matmul(Q, Kt, seq, seq, d)                    # seq×seq
        scaled = self.run("scale", scores, scalar=s)["output_decoded"]
        attn = self.run("softmax", scaled, M=seq, N=seq)["output_decoded"]
        return self.matmul(attn, V, seq, d, seq)                    # seq×d

    # ---- reductions via the GEMM trick (convenience; run single-worker/single-thread, not for throughput) ----
    def dot(self, a: Sequence[float], b: Sequence[float]) -> float:
        return self.matmul(list(a), list(b), 1, 1, len(a))[0]

    def sum(self, a: Sequence[float]) -> float:
        return self.matmul(list(a), [1.0] * len(a), 1, 1, len(a))[0]

    def mean(self, a: Sequence[float]) -> float:
        return self.sum(a) / len(a)

    def norm(self, a: Sequence[float]) -> float:
        return math.sqrt(self.dot(a, a))

    # ---- weight residency (split a model across workers / GPUs) ----
    def upload_weight(self, wid: str, data: Sequence[float], rows: int, cols: int, worker: str | None = None, dtype: str = "f32") -> dict:
        """Cache a named weight (rows×cols) RESIDENT on a worker; it is sent ONCE. dtype='f16' halves the
        upload + worker memory + GEMM bandwidth (f32 accumulate on a shader-f16 GPU; CPU workers dequantize).
        Pin different layers to different workers to split a model across GPUs. Returns {id, worker, rows, cols, dtype}."""
        body: dict[str, Any] = {"id": wid, "data": (_f16_b64(data) if dtype == "f16" else _f32_b64(data)),
                                "rows": rows, "cols": cols, "dtype": dtype}
        if worker:
            body["worker"] = worker
        return self._req("/weights", "POST", body)

    def weights(self) -> list[dict]:
        """List resident weights and which worker holds each."""
        return self._req("/weights")

    def matmul_resident(self, A: Sequence[float], weight_id: str, M: int) -> list[float]:
        """A(M×K) · <resident weight weight_id>(K×N), run on the worker holding the weight (never re-sent)."""
        job = self._req("/submit", "POST", {"kernel": "matmul", "a": _f32_b64(A), "bRef": weight_id, "M": M})
        if job.get("status") != "done":
            raise RuntimeError(f"resident matmul not done: {job.get('status')} {job.get('error')}")
        return _b64_f32(job["output"]) if job.get("output") else []

    # ---- on-pool fine-tuning (needs a native torch worker; the whole train step runs locally on it) ----
    def train_load(self, model: str, rank: int = 8, alpha: float = 16, lr: float = 1e-3,
                   seed: int = 0, targets: Sequence[str] | None = None, worker: str | None = None) -> dict:
        """Pin `model` on a torch worker, freeze it, attach a LoRA adapter (the only trainable tensor).
        Requires a worker started via `apps/worker/worker_torch.py`; WebGPU workers cannot train."""
        body: dict[str, Any] = {"model": model, "rank": rank, "alpha": alpha, "lr": lr, "seed": seed}
        if targets:
            body["targets"] = list(targets)
        if worker:
            body["worker"] = worker
        return self._req("/train/load", "POST", body)

    def train_step(self, input_ids: Sequence[int], labels: Sequence[int] | None = None, lr: float | None = None) -> dict:
        """Run ONE fwd+cross-entropy+backward+optimizer.step on the worker; returns {loss, step}.
        Gradients never leave the worker — only the sealed microbatch in / scalar loss out."""
        body: dict[str, Any] = {"input_ids": list(input_ids)}
        if labels is not None:
            body["labels"] = list(labels)
        if lr is not None:
            body["lr"] = lr
        return self._req("/train/step", "POST", body)

    def train_adapter(self) -> dict:
        """Pull the small trained LoRA adapter back (name → {data:b64 f32, shape})."""
        return self._req("/train/adapter", "POST", {})

    # ---- DiLoCo: distributed LoRA across many torch workers (coordinator = parameter server) ----
    def diloco_load(self, model: str, rank: int = 8, alpha: float = 16, lr: float = 1e-3,
                    seed: int = 0, targets: Sequence[str] | None = None, workers: Sequence[str] | None = None) -> dict:
        """Load the SAME seeded LoRA adapter on every (or the listed) torch workers."""
        body: dict[str, Any] = {"model": model, "rank": rank, "alpha": alpha, "lr": lr, "seed": seed}
        if targets:
            body["targets"] = list(targets)
        if workers:
            body["workers"] = list(workers)
        return self._req("/train/diloco/load", "POST", body)

    def diloco_round(self, batches: dict, inner_steps: int = 4, lr: float = 1e-3,
                     outer_lr: float = 0.7, outer_momentum: float = 0.9) -> dict:
        """One DiLoCo round: each worker does `inner_steps` local steps on batches[worker_id] (or batches['*']);
        the coordinator averages the adapters + applies an outer Nesterov step + broadcasts the new global."""
        return self._req("/train/diloco/round", "POST",
                         {"batches": batches, "inner_steps": inner_steps, "lr": lr, "outer_lr": outer_lr, "outer_momentum": outer_momentum})

    def diloco_adapter(self) -> dict:
        """Pull the coordinator's current GLOBAL averaged adapter."""
        return self._req("/train/diloco/adapter", "POST", {})

    # ---- resident-model serving (fast: the WHOLE forward runs on the worker, one round-trip per token) ----
    def model_load(self, model: str, id: str | None = None, fp16: bool = False, worker: str | None = None) -> dict:
        """Pin a whole model on a torch worker (frozen, eval). Needs apps/worker/worker_torch.py."""
        body: dict[str, Any] = {"model": model, "fp16": fp16}
        if id:
            body["id"] = id
        if worker:
            body["worker"] = worker
        return self._req("/model/load", "POST", body)

    def model_forward(self, input_ids: Sequence[int], id: str | None = None,
                      return_logits: bool = False, topk: int | None = None) -> dict:
        """Run the whole resident model's forward on `input_ids`; returns {argmax, logits?, top?}."""
        body: dict[str, Any] = {"input_ids": list(input_ids)}
        if id:
            body["id"] = id
        if return_logits:
            body["return_logits"] = True
        if topk:
            body["topk"] = topk
        return self._req("/model/forward", "POST", body)

    def generate(self, input_ids: Sequence[int], max_new_tokens: int = 16, id: str | None = None) -> list[int]:
        """Greedy generation via the resident model — the WHOLE decode runs on the worker (HF's internal
        KV cache) in ONE round-trip. Returns the new token ids."""
        body: dict[str, Any] = {"input_ids": list(input_ids), "max_new_tokens": max_new_tokens}
        if id:
            body["id"] = id
        return self._req("/model/generate", "POST", body).get("tokens", [])

    def model_unload(self, id: str | None = None) -> dict:
        """Free a resident model from the worker's device (VRAM)."""
        body: dict[str, Any] = {}
        if id:
            body["id"] = id
        return self._req("/model/unload", "POST", body)

    # ---- pipeline-parallel sharding (GPT-2 family only): split the layers into contiguous STAGES,
    # one per torch worker; each worker holds ONLY its stage. Needs apps/worker/worker_torch.py (≥2 for a real split). ----
    def shard_load(self, model: str, layers: int | None = None, workers: Sequence[str] | None = None,
                   id: str | None = None) -> dict:
        """Split `model`'s transformer layers into contiguous stages across the torch workers (or the listed
        `workers`) and load each stage on its worker. `layers` = layer count (default 12 for gpt2). GPT-2 only.
        Returns {id, stages:[{worker, start, end, first, last, params_held}]} — shows the memory really is split."""
        body: dict[str, Any] = {"model": model}
        if layers is not None:
            body["layers"] = layers
        if workers:
            body["workers"] = list(workers)
        if id:
            body["id"] = id
        return self._req("/model/shard", "POST", body)

    def shard_forward(self, input_ids: Sequence[int], id: str | None = None, return_logits: bool = False) -> dict:
        """Run ONE full forward across the pipeline: the coordinator pipes the hidden state stage→stage
        (only [seq×hidden] activations on the wire, never weights). Returns {argmax, logits?}."""
        body: dict[str, Any] = {"input_ids": list(input_ids)}
        if id:
            body["id"] = id
        if return_logits:
            body["return_logits"] = True
        return self._req("/model/shard_forward", "POST", body)

    def shard_generate(self, input_ids: Sequence[int], max_new_tokens: int, id: str | None = None) -> list[int]:
        """Greedy generation across the sharded pipeline: loop shard_forward, appending the argmax each step.
        Returns the new token ids (exact-matches transformers' greedy for GPT-2)."""
        seq = list(input_ids)
        new: list[int] = []
        for _ in range(max_new_tokens):
            nxt = int(self.shard_forward(seq, id=id)["argmax"])
            seq.append(nxt)
            new.append(nxt)
        return new

    def shard_unload(self, id: str | None = None) -> dict:
        """Unload every stage of a sharded model and drop the plan."""
        body: dict[str, Any] = {}
        if id:
            body["id"] = id
        return self._req("/model/shard_unload", "POST", body)
