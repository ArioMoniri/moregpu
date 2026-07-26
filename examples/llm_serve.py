#!/usr/bin/env python3
"""
Serve a REAL LLM on the pool — FAST.

examples/llm_infer.py proves a real GPT-2 forward runs on the pool via the fine-grained
kernel path (resident weights + primitives), but it drives ~500 sealed round-trips PER
TOKEN, so it is seconds/token — a proof of capability, not a serving stack.

This demo uses the RESIDENT-MODEL path on the native torch worker: the whole model is
pinned on the worker's device and the ENTIRE forward runs there, so the client sends token
ids and gets logits back in ONE round-trip per token (a native GPT-2 forward is ~9 ms on
MPS). Same model, same tokenizer — the greedy output is a token-for-token EXACT MATCH to
HuggingFace, only ~100–1000× fewer round-trips.

    # start a pool + torch worker (see apps/worker/worker_torch.py), then:
    MOREGPU_BASE=http://localhost:8787 MOREGPU_ADMIN_TOKEN=... MOREGPU_LLM=gpt2 python3 examples/llm_serve.py
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from moregpu_client import MoreGPU  # noqa: E402

BASE = os.environ.get("MOREGPU_BASE", "http://localhost:8787")
TOKEN = os.environ.get("MOREGPU_ADMIN_TOKEN", "")
MODEL = os.environ.get("MOREGPU_LLM", "gpt2")
PROMPT = os.environ.get("MOREGPU_PROMPT", "The future of distributed computing is")
GEN = int(os.environ.get("MOREGPU_GEN", "20"))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT)["input_ids"]

    pool = MoreGPU(BASE, TOKEN, timeout=600)
    if not any("torch" in (w.get("label") or "") for w in pool.workers()):
        sys.exit("no native torch worker in the fleet — start apps/worker/worker_torch.py and retry")

    print(f"== serve {MODEL} on the pool ==  prompt={PROMPT!r}  gen={GEN}")
    info = pool.model_load(MODEL)
    print(f"resident on {info['worker']} · {info.get('n_params'):,} params · {info.get('n_layer')} layers · {info.get('device')}")

    # FAST PATH — the whole greedy decode runs on the worker (HF's internal KV cache) in ONE round-trip.
    pool.generate(ids, max_new_tokens=1)                          # warm the model (one-off MPS kernel compile)
    t0 = time.time()
    pool_ids = pool.generate(ids, max_new_tokens=GEN)
    dt = time.time() - t0
    print(f"\npool  : {tok.decode(ids + pool_ids)!r}")
    print(f"        {GEN} tokens in {dt*1000:.0f} ms (ONE round-trip, worker-side KV cache) = {GEN/dt:.0f} tok/s")

    # (for comparison) the per-token path: one round-trip per token, times each token
    per = []
    seq = list(ids)
    for _ in range(min(GEN, 12)):
        t = time.time(); pool.model_forward(seq); per.append(time.time() - t); seq.append(pool_ids[len(per)-1])
    med = sorted(per)[len(per) // 2]
    print(f"        per-token path: {med*1000:.0f} ms/token = {1/med:.0f} tok/s (one round-trip per token)")

    # reference: transformers greedy
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    with torch.no_grad():
        ref = model.generate(torch.tensor([ids]), max_new_tokens=GEN, do_sample=False,
                             pad_token_id=tok.eos_token_id)[0].tolist()
    ref_new = ref[len(ids):]
    print(f"ref   : {tok.decode(ref)!r}")

    match = pool_ids == ref_new
    print(f"\n{'✅ EXACT MATCH to transformers greedy' if match else '❌ MISMATCH'}  "
          f"(pool {pool_ids[:8]}… vs ref {ref_new[:8]}…)")
    print(f"honest note: llm_infer.py runs the same model via ~85 round-trips/token with its KV cache "
          f"(~500 without) — still seconds/token; this resident-model path is the fast one (1 round-trip/token).")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
