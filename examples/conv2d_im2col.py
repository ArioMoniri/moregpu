#!/usr/bin/env python3
"""
conv2d_im2col.py — run a 2-D convolution on a MoreGPU pool the standard way: unfold the input
into columns (im2col) on the host, then do the heavy GEMM on the pool, then reshape.

This is the honest shape of the capability-matrix "Conv2d 🧩" cell: MoreGPU has no conv kernel,
but conv = im2col + matmul, and the matmul runs (and is verified) on the pool. The unfold is a
cheap host-side index gather. Single input channel, stride 1, no padding, to stay readable.

    MOREGPU_BASE=http://localhost:8787 MOREGPU_ADMIN_TOKEN=<admin> python3 examples/conv2d_im2col.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from moregpu_client import MoreGPU  # noqa: E402


def im2col(img, H, W, kh, kw):
    """img: H*W row-major (1 channel). Returns (out_h*out_w) × (kh*kw) patch matrix + out dims."""
    oh, ow = H - kh + 1, W - kw + 1
    cols = []
    for i in range(oh):
        for j in range(ow):
            for di in range(kh):
                for dj in range(kw):
                    cols.append(img[(i + di) * W + (j + dj)])
    return cols, oh, ow


def ref_conv2d(img, H, W, weights, kh, kw, cout):
    """Direct reference: for each out channel, slide the kernel. weights: cout × (kh*kw)."""
    oh, ow = H - kh + 1, W - kw + 1
    out = [0.0] * (oh * ow * cout)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                s = 0.0
                for di in range(kh):
                    for dj in range(kw):
                        s += img[(i + di) * W + (j + dj)] * weights[c * kh * kw + di * kw + dj]
                out[(i * ow + j) * cout + c] = s
    return out


def main() -> int:
    pool = MoreGPU(os.environ.get("MOREGPU_BASE", "http://localhost:8787"),
                   os.environ.get("MOREGPU_ADMIN_TOKEN", ""))
    print("== conv2d via im2col + pooled matmul ==")
    print("   health:", pool.health())

    H = W = 4
    kh = kw = 3
    cout = 2
    # a 4×4 single-channel image
    img = [float(v) for v in range(1, H * W + 1)]
    # two 3×3 filters (row-major, each length 9)
    W_kernels = [
        1, 0, -1, 1, 0, -1, 1, 0, -1,     # vertical-edge filter
        0.1, 0.1, 0.1, 0, 0, 0, -0.1, -0.1, -0.1,   # horizontal-gradient filter
    ]
    cols, oh, ow = im2col(img, H, W, kh, kw)          # (oh*ow) × (kh*kw)
    Wcol = [W_kernels[c * kh * kw + k] for k in range(kh * kw) for c in range(cout)]  # (kh*kw) × cout

    # heavy GEMM on the pool: (oh*ow × kh*kw) · (kh*kw × cout) → (oh*ow × cout), verified
    out = pool.matmul(cols, Wcol, M=oh * ow, N=cout, K=kh * kw)
    ref = ref_conv2d(img, H, W, W_kernels, kh, kw, cout)

    worst = max(abs(a - b) for a, b in zip(out, ref))
    ok = worst <= 2e-3
    print(f"  output {oh}×{ow}×{cout} channels — max|Δ| vs CPU reference = {worst:.2e}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}: pooled im2col-conv2d matches the direct convolution")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
