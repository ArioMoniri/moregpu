#!/usr/bin/env python3
"""
pipeline_parallel.py — split a model across workers with WEIGHT RESIDENCY.

This is the "divide the model into parts and run each part on a separate GPU" idea. A 2-layer MLP is
split so layer-1's weights live resident on one worker and layer-2's on another; the input activation
pipelines A → relu → B. The weights are uploaded ONCE (not re-sent per call) — which is the whole point
of residency and the thing that makes model-parallel inference possible on a pool.

    MOREGPU_BASE=http://localhost:8787 MOREGPU_ADMIN_TOKEN=<admin> python3 examples/pipeline_parallel.py

Honest scope: this proves the mechanism (residency + placement + pipelined activations, verified fp32).
A real Qwen-2.5-scale model additionally needs fp16 + a model/tokenizer loader + a KV cache + a native
CUDA backend for acceptable speed — none of which ship. But the sharding foundation is real and works.
"""
from __future__ import annotations
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from moregpu_client import MoreGPU  # noqa: E402

random.seed(0)
M, K1, N1, N2 = 3, 4, 6, 5  # x is M×K1 ; layer1 K1×N1 ; layer2 N1×N2


def randmat(r, c):
    return [random.gauss(0, 0.3) for _ in range(r * c)]


def cpu_mm(A, B, m, n, k):
    return [sum(A[i * k + t] * B[t * n + j] for t in range(k)) for i in range(m) for j in range(n)]


def main() -> int:
    pool = MoreGPU(os.environ.get("MOREGPU_BASE", "http://localhost:8787"),
                   os.environ.get("MOREGPU_ADMIN_TOKEN", ""))
    print("== pipeline-parallel MLP via weight residency ==")
    print(f"   health: {pool.health()}")
    fleet = pool.workers()
    ids = [w["id"] for w in fleet]
    if not ids:
        print("   no workers connected"); return 1
    a_worker = ids[0]
    b_worker = ids[1] if len(ids) > 1 else ids[0]
    print(f"   workers: {ids}  →  layer1 on '{a_worker}', layer2 on '{b_worker}'"
          + ("  (only 1 worker — both layers co-located)" if a_worker == b_worker else ""))

    W1 = randmat(K1, N1)
    W2 = randmat(N1, N2)
    x = randmat(M, K1)

    # upload each layer's weights ONCE, pinned to its worker
    u1 = pool.upload_weight("layer1.w", W1, K1, N1, worker=a_worker)
    u2 = pool.upload_weight("layer2.w", W2, N1, N2, worker=b_worker)
    print(f"   uploaded layer1.w → {u1['worker']} ({u1['rows']}x{u1['cols']}),"
          f" layer2.w → {u2['worker']} ({u2['rows']}x{u2['cols']})  [sent once]")

    # forward pass: x ·W1 (on A) → relu → ·W2 (on B). Weights are NOT re-sent — only activations move.
    h = pool.matmul_resident(x, "layer1.w", M)          # runs on a_worker
    a = pool.run("relu", h)["output_decoded"]
    y = pool.matmul_resident(a, "layer2.w", M)          # runs on b_worker

    # verify against a CPU reference
    ref_h = cpu_mm(x, W1, M, N1, K1)
    ref_a = [v if v > 0 else 0.0 for v in ref_h]
    ref_y = cpu_mm(ref_a, W2, M, N2, N1)
    worst = max(abs(p - q) for p, q in zip(y, ref_y))
    ok = worst <= 2e-3
    print(f"   output {M}x{N2}: {[round(v, 3) for v in y[:6]]}")
    print(f"   {'✅ PASS' if ok else '❌ FAIL'}: pipelined result matches the CPU reference (max|Δ|={worst:.2e})")
    print(f"   resident weights now: {pool.weights()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
