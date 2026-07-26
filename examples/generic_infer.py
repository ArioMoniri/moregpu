#!/usr/bin/env python3
"""
generic_infer.py — run a REAL modern LLM (Qwen3-0.6B) on a MoreGPU pool, matching transformers token-for-token.

Companion to llm_infer.py (GPT-2). GPT-2 is a 2018 architecture (learned positional embeddings, LayerNorm,
GELU-MLP, Conv1D weights, plain multi-head attention). Qwen3-0.6B is a 2025 architecture and shares almost
none of that: RMSNorm (normalize by RMS, NO mean-subtraction), RoPE (rotary position embeddings, θ=1e6),
SwiGLU MLP (silu(gate)·up), grouped-query attention (16 query heads share 8 KV heads), a per-head QK-norm
that is UNIQUE to Qwen3, head_dim decoupled from hidden/heads (128 ≠ 1024/16), tied embeddings, and NO
biases anywhere. Running it on the same pool as GPT-2 is the point: the pool is architecture-agnostic.

The host/pool split mirrors llm_infer's: the 7 projections per layer (q/k/v/o + gate/up/down) are pinned
RESIDENT on the pool's workers (each < 16M elements) and run as resident matmuls, plus the per-head
attention score/context matmuls run on the pool. Everything the pool has no exact kernel for is host-side:
the token-embedding gather, RMSNorm, RoPE, the Qwen3 QK-norm, GQA head indexing, SiLU, the causal mask, and
the wide (155M-param, tied) LM head. Weights are nn.Linear [out,in]; the resident matmul stores W as [K,N],
so every projection is uploaded TRANSPOSED (unlike GPT-2's Conv1D, which is already [in,out]).

    # launch a pool + a native torch worker, then:
    MOREGPU_BASE=http://localhost:8802 MOREGPU_ADMIN_TOKEN=<admin> python3 examples/generic_infer.py "The capital of France is"

    MOREGPU_FP32=1     upload weights f32 instead of f16 (debugging; default is f16, ~0.9GB across the fleet)
    MOREGPU_GEN=n      greedily generate n tokens (default 3)
    MOREGPU_DEBUG=1    print per-layer max|Δ hidden| vs transformers (isolates a wrong RoPE/QK-norm/transpose)

Honest scope: this is a PORTABILITY / correctness proof — a SECOND, modern architecture running on the pool
and matching Hugging Face token-for-token. It is NOT a speed or training win. It is forward-only and
round-trip-bound: every layer's activations cross the network, and there is NO KV cache, so each new token
re-runs the whole prompt — ~minutes/token. The bar is the GREEDY argmax token IDs matching transformers
exactly (f16/fp32 drift perturbs the logits slightly but not the arg-max), so we verify a few tokens, not many.
"""
from __future__ import annotations
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from moregpu_client import MoreGPU  # noqa: E402

try:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover
    print("needs numpy + torch + transformers:", e); sys.exit(2)

MODEL = os.environ.get("MOREGPU_LLM", "Qwen/Qwen3-0.6B")
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
GEN = int(os.environ.get("MOREGPU_GEN", "3"))
DT = "f32" if os.environ.get("MOREGPU_FP32") == "1" else "f16"


def main() -> int:
    pool = MoreGPU(os.environ.get("MOREGPU_BASE", "http://localhost:8787"),
                   os.environ.get("MOREGPU_ADMIN_TOKEN", ""), timeout=600)
    print(f"== running {MODEL} on MoreGPU ==  health: {pool.health()}")
    if not pool.workers():
        print("   no workers"); return 1
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = model.config
    L = cfg.num_hidden_layers                    # 28
    HID = cfg.hidden_size                        # 1024
    NQ, NKV = cfg.num_attention_heads, cfg.num_key_value_heads   # 16, 8
    HD = cfg.head_dim                            # 128 (decoupled: 1024/16 = 64 ≠ 128)
    EPS, THETA = cfg.rms_norm_eps, cfg.rope_theta               # 1e-6, 1e6
    NREP = NQ // NKV                             # 2  → query head h uses kv head h // NREP
    DQ, DKV = NQ * HD, NKV * HD                  # 2048, 1024
    sd = model.state_dict()
    npf = lambda n: sd[n].detach().numpy().astype(np.float32)

    # --- pin the 7 projections/layer RESIDENT (transposed: nn.Linear is [out,in]; the resident GEMM wants [K,N]) ---
    print(f"   pinning {L}×7 projection weights resident ({DT}) across {len(pool.workers())} worker(s) …")
    PROJ = [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"),
            ("o", "self_attn.o_proj"), ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")]
    for i in range(L):
        for tag, name in PROJ:
            Wt = npf(f"model.layers.{i}.{name}.weight").T.copy()   # [out,in] → [in,out] = [K,N]
            pool.upload_weight(f"l{i}.{tag}", Wt.reshape(-1).tolist(), Wt.shape[0], Wt.shape[1], dtype=DT)

    # --- host-side (no exact pool kernel): embeddings/LM-head, all RMSNorm weights, per-head QK-norm ---
    wte = npf("model.embed_tokens.weight")       # [V, HID]; tied → also the LM head
    in_ln = {i: npf(f"model.layers.{i}.input_layernorm.weight") for i in range(L)}
    post_ln = {i: npf(f"model.layers.{i}.post_attention_layernorm.weight") for i in range(L)}
    q_norm = {i: npf(f"model.layers.{i}.self_attn.q_norm.weight") for i in range(L)}   # [HD]
    k_norm = {i: npf(f"model.layers.{i}.self_attn.k_norm.weight") for i in range(L)}   # [HD]
    final_norm = npf("model.norm.weight")
    print(f"   resident weights on the pool: {len(pool.weights())}  ({L} layers × 7)")

    # RoPE tables (NeoX / rotate_half convention, θ=1e6) — precomputed per forward length
    inv_freq = (1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float32) / HD))).astype(np.float32)   # [HD/2]

    def rope(seq):
        freqs = np.arange(seq, dtype=np.float32)[:, None] * inv_freq[None, :]   # [seq, HD/2]
        emb = np.concatenate([freqs, freqs], axis=-1)                           # [seq, HD]
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    def rotate_half(x):
        d = x.shape[-1] // 2
        return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)

    def apply_rope(x, cos, sin):   # x [seq, heads, HD]; cos/sin [seq, HD] broadcast over heads
        return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]

    def rmsnorm(x, w):   # RMSNorm: x * rsqrt(mean(x², -1) + eps) * w  (NO mean-subtraction)
        return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)) * w

    def silu(z):         # z * sigmoid(z), overflow-safe (pool has only gelu; silu ≠ gelu)
        s = np.empty_like(z); p = z >= 0
        s[p] = 1.0 / (1.0 + np.exp(-z[p]))
        e = np.exp(z[~p]); s[~p] = e / (1.0 + e)
        return z * s

    # --- pool ops: numpy in, numpy out ---
    def p_softmax(x):
        o = pool.run("softmax", x.reshape(-1).tolist(), M=x.shape[0], N=x.shape[1])["output_decoded"]
        return np.asarray(o, np.float32).reshape(x.shape)

    def p_res(x, wid):   # A · <resident weight> on the pool
        o = pool.matmul_resident(x.reshape(-1).tolist(), wid, x.shape[0])
        return np.asarray(o, np.float32).reshape(x.shape[0], len(o) // x.shape[0])

    def p_mm(a, b):      # data-mode A·B on the pool
        m, k = a.shape; n = b.shape[1]
        o = pool.matmul(a.reshape(-1).tolist(), b.reshape(-1).tolist(), m, n, k)
        return np.asarray(o, np.float32).reshape(m, n)

    def forward(ids, taps=None):
        seq = len(ids)
        x = wte[ids]                                             # [seq, HID] — no learned pos emb (RoPE handles it)
        cos, sin = rope(seq)
        causal = np.triu(np.ones((seq, seq), np.float32), 1) * -1e9
        for i in range(L):
            h = rmsnorm(x, in_ln[i])
            q = p_res(h, f"l{i}.q").reshape(seq, NQ, HD)          # [seq, 16, 128]
            k = p_res(h, f"l{i}.k").reshape(seq, NKV, HD)         # [seq, 8, 128]
            v = p_res(h, f"l{i}.v").reshape(seq, NKV, HD)
            q = apply_rope(rmsnorm(q, q_norm[i]), cos, sin)       # QK-norm (per head, over HD) THEN RoPE
            k = apply_rope(rmsnorm(k, k_norm[i]), cos, sin)
            ctx = np.zeros((seq, NQ, HD), np.float32)
            for hh in range(NQ):                                 # GQA: query head hh ← kv head hh // NREP
                kv = hh // NREP
                sc = p_mm(q[:, hh], k[:, kv].T.copy()) / math.sqrt(HD) + causal
                ctx[:, hh] = p_mm(p_softmax(sc), v[:, kv])
            x = x + p_res(ctx.reshape(seq, DQ), f"l{i}.o")        # attn residual
            h = rmsnorm(x, post_ln[i])
            act = silu(p_res(h, f"l{i}.gate")) * p_res(h, f"l{i}.up")   # SwiGLU
            x = x + p_res(act, f"l{i}.down")                      # MLP residual
            if taps is not None:
                taps.append(x.copy())
        x = rmsnorm(x, final_norm)
        return (x[-1] @ wte.T), len(wte)                         # host-side tied LM head (155M params > resident cap)

    ids = tok.encode(PROMPT)
    print(f"\n   prompt: {PROMPT!r}  → {len(ids)} tokens")

    if os.environ.get("MOREGPU_DEBUG") == "1":
        taps: list = []
        forward(ids, taps)
        hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states   # tuple of L+1 [1,seq,HID]
        print("   per-layer drift vs transformers:")
        for i in range(L):
            d = np.abs(taps[i] - hs[i + 1][0].detach().numpy()).max()
            print(f"     layer {i:2d}: max|Δ hidden| = {d:.4e}")

    logits, V = forward(ids)
    ref = model(torch.tensor([ids])).logits[0, -1].detach().numpy()
    tp = np.argsort(-logits)[:5]; tr = np.argsort(-ref)[:5]
    print(f"   next-token top-5 (pool): {[tok.decode([t]) for t in tp]}")
    print(f"   next-token top-5 (ref) : {[tok.decode([t]) for t in tr]}")
    print(f"   max|Δ logit| vs transformers = {np.abs(logits - ref).max():.3f}  ·  argmax match: {tp[0] == tr[0]}")

    if GEN:
        print(f"\n   greedy-generating {GEN} tokens on the pool …")
        gen = list(ids)
        for _ in range(GEN):
            lg, _ = forward(gen); gen.append(int(np.argmax(lg)))
        ref_gen = model.generate(torch.tensor([ids]), max_new_tokens=GEN, do_sample=False)[0].tolist()
        pool_new, ref_new = gen[len(ids):], ref_gen[len(ids):]
        print(f"   pool : {tok.decode(gen)!r}")
        print(f"   ref  : {tok.decode(ref_gen)!r}")
        print(f"   pool new-token ids: {pool_new}")
        print(f"   ref  new-token ids: {ref_new}")
        ok = pool_new == ref_new
        print(f"\n   {'✅ EXACT TOKEN MATCH' if ok else '❌ MISMATCH'} — a real Qwen3-0.6B ran on the pool "
              f"(RMSNorm/RoPE/SwiGLU/GQA/QK-norm), matching transformers for {len(pool_new)} greedy tokens.")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
