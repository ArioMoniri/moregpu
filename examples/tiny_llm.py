#!/usr/bin/env python3
"""
tiny_llm.py — a REAL transformer forward pass running on a MoreGPU pool, composed entirely from
the pool's verified fp32 primitives (matmul / layernorm / softmax / gelu) via the SDK helpers.

The point is to answer "why can't I run LLMs on this pool?" honestly and concretely:
  • It DOES run a genuine transformer block — embeddings → (LN → self-attention → LN → MLP)×L →
    final LN → LM head → next-token logits — on real (random) weights, on the GPU.
  • It then PRINTS the scaling wall: how many pool round-trips this toy needed, and what the same
    math costs for a real LLM. That gap — not a missing feature — is why a 7B model is impractical.

    MOREGPU_BASE=http://localhost:8787 MOREGPU_ADMIN_TOKEN=<admin> python3 examples/tiny_llm.py
"""
from __future__ import annotations
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from moregpu_client import MoreGPU  # noqa: E402

# ---- toy model dimensions (small enough to run; a real LLM is ~1000x bigger in every one) ----
VOCAB, D, LAYERS, SEQ, D_FF = 32, 16, 2, 6, 64
random.seed(0)


def randmat(rows, cols, s=0.08):
    return [random.gauss(0, s) for _ in range(rows * cols)]


def add(a, b):  # host-side residual add (cheap elementwise; not worth a pool round-trip)
    return [x + y for x, y in zip(a, b)]


class Toy:
    def __init__(self):
        self.emb = randmat(VOCAB, D)                 # token embeddings
        self.pos = randmat(SEQ, D)                   # positional embeddings
        self.layers = [{
            "wq": randmat(D, D), "wk": randmat(D, D), "wv": randmat(D, D), "wo": randmat(D, D),
            "w1": randmat(D, D_FF), "w2": randmat(D_FF, D),
        } for _ in range(LAYERS)]
        self.head = randmat(D, VOCAB)                # LM head (D → vocab)


def forward(pool: MoreGPU, model: Toy, tokens: list[int]) -> list[float]:
    global CALLS
    seq = len(tokens)
    # embedding lookup + positional — host-side gather (a memory op, not arithmetic)
    x = []
    for i, t in enumerate(tokens):
        x += add(model.emb[t * D:(t + 1) * D], model.pos[i * D:(i + 1) * D])
    for L in model.layers:
        # --- self-attention block ---
        h = pool.run("layernorm", x, M=seq, N=D)["output_decoded"]; CALLS += 1
        q = pool.matmul(h, L["wq"], seq, D, D); CALLS += 1
        k = pool.matmul(h, L["wk"], seq, D, D); CALLS += 1
        v = pool.matmul(h, L["wv"], seq, D, D); CALLS += 1
        a = pool.attention(q, k, v, seq=seq, d=D); CALLS += 4   # matmul+scale+softmax+matmul
        a = pool.matmul(a, L["wo"], seq, D, D); CALLS += 1
        x = add(x, a)                                            # residual
        # --- MLP block ---
        h = pool.run("layernorm", x, M=seq, N=D)["output_decoded"]; CALLS += 1
        f = pool.matmul(h, L["w1"], seq, D_FF, D); CALLS += 1
        f = pool.run("gelu", f)["output_decoded"]; CALLS += 1
        f = pool.matmul(f, L["w2"], seq, D, D_FF); CALLS += 1
        x = add(x, f)                                            # residual
    x = pool.run("layernorm", x, M=seq, N=D)["output_decoded"]; CALLS += 1
    logits = pool.matmul(x, model.head, seq, VOCAB, D); CALLS += 1   # last row = next-token logits
    return logits[(seq - 1) * VOCAB: seq * VOCAB]


CALLS = 0


def main() -> int:
    pool = MoreGPU(os.environ.get("MOREGPU_BASE", "http://localhost:8787"),
                   os.environ.get("MOREGPU_ADMIN_TOKEN", ""))
    print("== tiny transformer forward pass on MoreGPU ==")
    print(f"   health: {pool.health()}")
    print(f"   model: vocab={VOCAB} d={D} layers={LAYERS} seq={SEQ} d_ff={D_FF}  (all fp32, on the pool's GPU)")
    model = Toy()
    tokens = [1, 5, 9, 2, 7, 3][:SEQ]
    logits = forward(pool, model, tokens)
    nxt = max(range(VOCAB), key=lambda i: logits[i])
    print(f"   tokens in : {tokens}")
    print(f"   next-token logits (first 8): {[round(v, 3) for v in logits[:8]]}")
    print(f"   argmax next token id: {nxt}")
    print(f"\n   ✅ a real transformer forward pass ran on the pool in {CALLS} sealed round-trips.")

    # ---- the honest scaling wall ----
    real_layers, real_d, gen_tokens = 32, 4096, 200
    per_layer_calls = CALLS / LAYERS
    real_calls_per_token = per_layer_calls * real_layers
    total = real_calls_per_token * gen_tokens
    weights_gb = (real_layers * (4 * real_d * real_d + 2 * real_d * 4 * real_d) * 4) / 1e9  # fp32 bytes
    print("\n   Why a *real* LLM is impractical here (same math, ~1000x bigger):")
    print(f"     • ~{per_layer_calls:.0f} pool round-trips per layer → ~{real_calls_per_token:.0f} per token for a 32-layer model,")
    print(f"       ~{total:,.0f} sealed network round-trips to generate {gen_tokens} tokens.")
    print(f"     • weights are sent PER REQUEST (no residency): a 7B model is ~{weights_gb:.0f} GB of fp32 —")
    print("       re-uploaded on every matmul, over the WAN. Residency + fp16 would cut this ~100x.")
    print("     • fp32, no tensor cores, no KV cache → each step is ~10-100x slower than a real stack.")
    print("   The primitives are correct and compose; the wall is architecture (per-request weights,")
    print("   fp32, network round-trips), not a missing kernel. Real LLM serving needs a native")
    print("   CUDA/cuBLAS worker with fp16 + tensor cores + weight residency + a KV cache (not built).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
