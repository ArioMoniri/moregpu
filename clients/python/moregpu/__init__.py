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

__version__ = "0.2.0"
KERNELS = ("matmul", "vector_add", "vector_mul", "saxpy", "relu", "scale", "gelu", "softmax", "layernorm")


def _f32_b64(values: Sequence[float]) -> str:
    return base64.b64encode(array.array("f", values).tobytes()).decode()


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
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if body is not None:
            req.add_header("content-type", "application/json")
        if auth:
            req.add_header("authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

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
