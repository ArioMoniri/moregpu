#!/usr/bin/env python3
"""
verify_workloads.py — drive a live MoreGPU pool the way a *real GPU user* would.

This is the "does the pool actually run my stuff?" harness. It composes the
primitive kernels the pool ships (matmul, relu, scale, softmax, layernorm) into
the workloads people normally reach for a GPU to do — a Linear layer, a 2-layer
MLP, LayerNorm, a softmax classifier head, and a full single-head scaled
dot-product attention block — then checks every result against a dependency-free
CPU reference. Exit code is non-zero if any workload is wrong.

    MOREGPU_BASE=http://localhost:8787 MOREGPU_ADMIN_TOKEN=<admin> \
        python3 examples/verify_workloads.py

Pure standard library (uses the shipped moregpu_client). No numpy required.
"""
from __future__ import annotations
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from moregpu_client import MoreGPU  # noqa: E402

TOL = 2e-3  # fp32 pooled result vs float64 CPU reference


# ---- tiny CPU references (float64) ------------------------------------------
def ref_matmul(A, B, M, N, K):
    C = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i * K + k] * B[k * N + j]
            C[i * N + j] = s
    return C


def ref_relu(x):
    return [v if v > 0 else 0.0 for v in x]


def ref_softmax_row(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def ref_layernorm_row(x, eps=1e-5):
    n = len(x)
    mu = sum(x) / n
    var = sum((v - mu) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + eps)
    return [(v - mu) * inv for v in x]


def close(got, want, tol=TOL):
    if len(got) != len(want):
        return False, f"length {len(got)} != {len(want)}"
    worst = 0.0
    for g, w in zip(got, want):
        worst = max(worst, abs(g - w))
    return worst <= tol, f"max|Δ|={worst:.2e}"


# ---- workloads a GPU user actually runs -------------------------------------
def main() -> int:
    base = os.environ.get("MOREGPU_BASE", "http://localhost:8787")
    tok = os.environ.get("MOREGPU_ADMIN_TOKEN", "")
    pool = MoreGPU(base, tok)
    print(f"== verifying real workloads against {base} ==")
    print(f"   health: {pool.health()}")
    passed = failed = 0

    def check(name, got, want):
        nonlocal passed, failed
        ok, detail = close(got, want)
        if ok:
            passed += 1
            print(f"  ✅ {name:<44} {detail}")
        else:
            failed += 1
            print(f"  ❌ {name:<44} {detail}")
            print(f"       got : {[round(v,4) for v in got[:8]]}")
            print(f"       want: {[round(v,4) for v in want[:8]]}")

    # 1) Dense matmul / GEMM — the workhorse.
    A = [1, 2, 3, 4, 5, 6]          # 2x3
    B = [7, 8, 9, 10, 11, 12]        # 3x2
    check("Dense matmul (GEMM) 2x3 · 3x2",
          pool.matmul(A, B, M=2, N=2, K=3), ref_matmul(A, B, 2, 2, 3))

    # 2) Linear / Dense layer:  y = x · W   (batch=4, in=3, out=2)
    X = [0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.9, 1.0, -1.1, 1.2]  # 4x3
    W = [0.2, -0.1, 0.05, 0.3, -0.4, 0.15]                                 # 3x2
    check("Linear layer  x(4x3)·W(3x2)",
          pool.matmul(X, W, M=4, N=2, K=3), ref_matmul(X, W, 4, 2, 3))

    # 3) ReLU activation on a real feature vector.
    feats = [-2.0, -0.5, 0.0, 0.5, 2.0, -3.3, 4.4, -1.1]
    check("ReLU activation",
          pool.run("relu", feats)["output_decoded"], ref_relu(feats))

    # 4) Softmax classifier head (logits -> probabilities).
    logits = [2.0, 1.0, 0.1, -1.0, 3.0]
    check("Softmax head (1 row of logits)",
          pool.run("softmax", logits, M=1, N=len(logits))["output_decoded"],
          ref_softmax_row(logits))

    # 5) LayerNorm over a hidden vector.
    hid = [0.5, -1.5, 2.0, 0.0, 1.0, -0.5, 0.25, -0.75]
    check("LayerNorm (1 row, d=8)",
          pool.run("layernorm", hid, M=1, N=len(hid))["output_decoded"],
          ref_layernorm_row(hid))

    # 6) 2-layer MLP:  relu(X · W1) · W2   (compose three kernels).
    W1 = [0.3, -0.2, 0.1, 0.4, -0.5, 0.2, 0.1, -0.3, 0.25, 0.15, -0.35, 0.05]  # 3x4
    W2 = [0.2, -0.1, 0.3, 0.15, -0.25, 0.1, 0.05, -0.2]                         # 4x2
    h1 = pool.matmul(X, W1, M=4, N=4, K=3)          # 4x4
    a1 = pool.run("relu", h1)["output_decoded"]      # 4x4
    mlp = pool.matmul(a1, W2, M=4, N=2, K=4)         # 4x2
    r_h1 = ref_matmul(X, W1, 4, 4, 3)
    r_mlp = ref_matmul(ref_relu(r_h1), W2, 4, 2, 4)
    check("2-layer MLP  relu(X·W1)·W2", mlp, r_mlp)

    # 7) Single-head scaled dot-product attention: softmax(Q·Kᵀ/√d)·V
    #    seq=2, d=2.  This is the core of a transformer, composed from primitives.
    Q = [1.0, 0.0, 0.0, 1.0]        # 2x2
    Kmat = [1.0, 1.0, 0.0, 1.0]     # 2x2  (rows = keys)
    V = [2.0, 0.0, 0.0, 3.0]        # 2x2
    d = 2
    Kt = [Kmat[0], Kmat[2], Kmat[1], Kmat[3]]        # transpose 2x2
    scores = pool.matmul(Q, Kt, M=2, N=2, K=2)       # Q·Kᵀ  (2x2)
    scaled = pool.run("scale", scores, scalar=1.0 / math.sqrt(d))["output_decoded"]
    # softmax each row (M=2 rows, N=2)
    attn = pool.run("softmax", scaled, M=2, N=2)["output_decoded"]
    out = pool.matmul(attn, V, M=2, N=2, K=2)        # ·V  (2x2)
    # reference
    r_scores = ref_matmul(Q, Kt, 2, 2, 2)
    r_scaled = [v / math.sqrt(d) for v in r_scores]
    r_attn = ref_softmax_row(r_scaled[0:2]) + ref_softmax_row(r_scaled[2:4])
    r_out = ref_matmul(r_attn, V, 2, 2, 2)
    check("Attention  softmax(Q·Kᵀ/√d)·V (1 head)", out, r_out)

    print(f"\n==================== {passed} passed, {failed} failed ====================")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
