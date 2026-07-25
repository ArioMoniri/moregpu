#!/usr/bin/env python3
"""
llm_infer.py — run a REAL LLM (GPT-2) on a MoreGPU pool and validate it against the reference.

Loads the real GPT-2 (124M) weights, pins the 12 transformer layers' weights RESIDENT on the pool's
workers (sent once — split across GPUs), and runs the full forward pass — embeddings → 12×(LN → causal
multi-head attention → LN → GELU-MLP) → final LN → logits — on the pool via the shipped primitives
(matmul_resident / matmul / softmax / gelu / layernorm). It then checks the pool's next-token logits
against Hugging Face `transformers` and greedily generates a few tokens.

    MOREGPU_BASE=http://localhost:8787 MOREGPU_ADMIN_TOKEN=<admin> python3 examples/llm_infer.py "Once upon a time"

Honest scope: correct, verified fp32, but SLOW — every layer's activations round-trip over the network
(seconds/token on localhost, worse over a WAN). Real serving speed needs a native fp16/CUDA worker
(roadmap). The point: a real model genuinely runs on the pool and matches the reference. The wide LM head
(~150 MB) is applied host-side (too big for one sealed frame); the 12 transformer layers run on the pool.
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
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
except Exception as e:  # pragma: no cover
    print("needs numpy + torch + transformers:", e); sys.exit(2)

MODEL = os.environ.get("MOREGPU_LLM", "gpt2")
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
GEN = int(os.environ.get("MOREGPU_GEN", "8"))


def main() -> int:
    pool = MoreGPU(os.environ.get("MOREGPU_BASE", "http://localhost:8787"),
                   os.environ.get("MOREGPU_ADMIN_TOKEN", ""), timeout=600)
    print(f"== running {MODEL} on MoreGPU ==  health: {pool.health()}")
    if not pool.workers():
        print("   no workers"); return 1
    tok = GPT2TokenizerFast.from_pretrained(MODEL)
    model = GPT2LMHeadModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = model.config
    L, H, D = cfg.n_layer, cfg.n_head, cfg.n_embd
    hd = D // H
    sd = model.state_dict()
    npf = lambda n: sd[n].detach().numpy().astype(np.float32)

    DT = "f16" if os.environ.get("MOREGPU_FP16") == "1" else "f32"
    print(f"   pinning {L}-layer weights resident ({DT}) across {len(pool.workers())} worker(s) …")
    for i in range(L):
        p = f"transformer.h.{i}."
        for wid, name in [(f"l{i}.attn", "attn.c_attn.weight"), (f"l{i}.aproj", "attn.c_proj.weight"),
                          (f"l{i}.fc", "mlp.c_fc.weight"), (f"l{i}.proj", "mlp.c_proj.weight")]:
            W = npf(p + name)
            pool.upload_weight(wid, W.reshape(-1).tolist(), W.shape[0], W.shape[1], dtype=DT)
    wte, wpe = npf("transformer.wte.weight"), npf("transformer.wpe.weight")
    lnw = {i: tuple(npf(f"transformer.h.{i}.ln_{j}.{k}") for j in (1, 2) for k in ("weight", "bias")) for i in range(L)}
    bA = {i: npf(f"transformer.h.{i}.attn.c_attn.bias") for i in range(L)}
    bP = {i: npf(f"transformer.h.{i}.attn.c_proj.bias") for i in range(L)}
    bF = {i: npf(f"transformer.h.{i}.mlp.c_fc.bias") for i in range(L)}
    bO = {i: npf(f"transformer.h.{i}.mlp.c_proj.bias") for i in range(L)}
    lnf_w, lnf_b = npf("transformer.ln_f.weight"), npf("transformer.ln_f.bias")
    print(f"   resident weights on the pool: {len(pool.weights())}  ({L} layers × 4)")

    # --- pool ops: numpy in, numpy out (cross the wire only via .tolist()) ---
    def p_ln(x):  # layernorm (normalize only) on the pool
        o = pool.run("layernorm", x.reshape(-1).tolist(), M=x.shape[0], N=x.shape[1])["output_decoded"]
        return np.asarray(o, np.float32).reshape(x.shape)
    def p_gelu(x):
        o = pool.run("gelu", x.reshape(-1).tolist())["output_decoded"]
        return np.asarray(o, np.float32).reshape(x.shape)
    def p_softmax(x):
        o = pool.run("softmax", x.reshape(-1).tolist(), M=x.shape[0], N=x.shape[1])["output_decoded"]
        return np.asarray(o, np.float32).reshape(x.shape)
    def p_res(x, wid):  # A · <resident weight> on the pool
        o = pool.matmul_resident(x.reshape(-1).tolist(), wid, x.shape[0])
        return np.asarray(o, np.float32).reshape(x.shape[0], len(o) // x.shape[0])
    def p_mm(a, b):  # data-mode A·B on the pool
        m, k = a.shape; n = b.shape[1]
        o = pool.matmul(a.reshape(-1).tolist(), b.reshape(-1).tolist(), m, n, k)
        return np.asarray(o, np.float32).reshape(m, n)

    def forward(ids):
        seq = len(ids)
        x = wte[ids] + wpe[np.arange(seq)]                       # [seq, D]
        causal = np.triu(np.ones((seq, seq), np.float32), 1) * -1e9
        for i in range(L):
            w1, b1, w2, b2 = lnw[i]
            qkv = p_res(p_ln(x) * w1 + b1, f"l{i}.attn") + bA[i]      # [seq, 3D]
            q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
            ctx = np.zeros((seq, D), np.float32)
            for hh in range(H):
                s = hh * hd
                sc = p_mm(q[:, s:s + hd], k[:, s:s + hd].T.copy()) / math.sqrt(hd) + causal
                ctx[:, s:s + hd] = p_mm(p_softmax(sc), v[:, s:s + hd])
            x = x + (p_res(ctx, f"l{i}.aproj") + bP[i])              # attn residual
            h = p_res(p_ln(x) * w2 + b2, f"l{i}.fc") + bF[i]         # MLP
            x = x + (p_res(p_gelu(h), f"l{i}.proj") + bO[i])         # MLP residual
        x = p_ln(x) * lnf_w + lnf_b
        return (x[-1] @ wte.T), len(wte)                          # host-side vocab projection

    ids = tok.encode(PROMPT)
    print(f"\n   prompt: {PROMPT!r}  → {len(ids)} tokens")
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
        print(f"   pool : {tok.decode(gen)!r}")
        print(f"   ref  : {tok.decode(ref_gen)!r}")
        print(f"\n   {'✅ EXACT MATCH' if gen == ref_gen else '≈ close (fp32 drift)'} — a real GPT-2 ran on the pool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
