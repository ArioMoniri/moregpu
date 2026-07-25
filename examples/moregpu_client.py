"""
moregpu_client — a tiny, dependency-free Python client for a MoreGPU pool.
Submit compute jobs and read fleet contribution from any Python/AI workflow.

    from moregpu_client import MoreGPU
    pool = MoreGPU("http://ADMIN:8787", "<admin-token>")
    print(pool.submit("matmul", 1024))            # one job
    print(pool.submit_batch([("relu", 1_000_000), ("matmul", 512)]))
    print(pool.gpu())                             # pool state (virtual GPU)
"""
from __future__ import annotations
import array
import base64
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

KERNELS = ("matmul", "vector_add", "vector_mul", "saxpy", "relu", "scale")


def _f32_b64(values: Sequence[float]) -> str:
    return base64.b64encode(array.array("f", values).tobytes()).decode()


def _b64_f32(s: str) -> list[float]:
    a = array.array("f"); a.frombytes(base64.b64decode(s)); return list(a)


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

    def health(self) -> dict:
        return self._req("/health", auth=False)

    def gpu(self) -> dict:
        return self._req("/gpu")

    def workers(self) -> list[dict]:
        return self._req("/workers")

    def submit(self, kernel: str, size: int) -> dict:
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}; choose from {KERNELS}")
        return self._req("/submit", "POST", {"kernel": kernel, "size": size})

    def submit_batch(self, specs: list[tuple[str, int]], workers: int = 8) -> list[dict]:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda s: self.submit(*s), specs))

    def run(self, kernel: str, a: Sequence[float], b: Sequence[float] | None = None,
            scalar: float | None = None, M: int | None = None, N: int | None = None, K: int | None = None) -> dict:
        """Data mode: send your own tensors, get the pooled result. Returns the job dict with an
        extra 'output_decoded' list. Check job['status'] == 'done' before using it."""
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}")
        body: dict[str, Any] = {"kernel": kernel, "a": _f32_b64(a)}
        if b is not None:
            body["b"] = _f32_b64(b)
        for name, val in (("scalar", scalar), ("M", M), ("N", N), ("K", K)):
            if val is not None:
                body[name] = val
        job = self._req("/submit", "POST", body)
        job["output_decoded"] = _b64_f32(job["output"]) if job.get("output") else []
        return job

    def matmul(self, A: Sequence[float], B: Sequence[float], M: int, N: int, K: int) -> list[float]:
        """C = A(M×K) · B(K×N) computed on the pool."""
        return self.run("matmul", A, B, M=M, N=N, K=K)["output_decoded"]

    def job(self, job_id: str) -> dict:
        return self._req(f"/jobs/{job_id}")

    def jobs(self) -> list[dict]:
        return self._req("/jobs")


if __name__ == "__main__":
    import os
    pool = MoreGPU(os.environ.get("MOREGPU_BASE", "http://localhost:8787"),
                   os.environ.get("MOREGPU_ADMIN_TOKEN", ""))
    print("health:", pool.health())
    print("matmul 512:", pool.submit("matmul", 512))
