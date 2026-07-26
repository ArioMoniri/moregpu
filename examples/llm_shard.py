#!/usr/bin/env python3
"""
Run a REAL LLM SPLIT ACROSS MACHINES — pipeline-parallel model sharding.

examples/llm_serve.py holds the WHOLE model on ONE worker. This demo instead SHARDS the
model: its transformer layers are cut into contiguous STAGES, one per worker, and each
worker holds ONLY its stage resident. A forward pipes the hidden state stage→stage, so the
client sends token ids to stage 0 and the LAST stage returns the logits — and only the
[seq×hidden] ACTIVATIONS ever cross between stages, never the weights. That is the
low-bandwidth, "model too big for one machine" path (the 2026 Petals / mesh-LLM approach).

Works for GPT-2 AND Llama-style models (Llama / SmolLM / TinyLlama / Qwen2 / Qwen3) across the
available torch workers (use ≥2 for a real split; it also runs on 1, but that's degenerate —
one stage = the whole model). A 7B model would use more / bigger workers, but the mechanism is
identical. Honest caveat: models that tie the input embedding to the output head (GPT-2, and
SmolLM/most Llama-family checkpoints) resident that one big matrix on BOTH the first stage (as
the embedding) and the last stage (as lm_head) — it is duplicated, not split. What genuinely
splits is the transformer BLOCKS (e.g. 6+6 for gpt2, 15+15 for SmolLM-135M across 2 workers),
and in a large model the blocks dominate, so the memory win there is real and large.

Correctness bar: the greedy token ids are a token-for-token EXACT MATCH to HuggingFace
`AutoModelForCausalLM.generate(do_sample=False)` — same weights (f32), so identical output.

    # start a pool + TWO torch workers (see apps/worker/worker_torch.py; --cpu keeps memory low), then:
    MOREGPU_BASE=http://localhost:8787 MOREGPU_ADMIN_TOKEN=... MOREGPU_LLM=gpt2 python3 examples/llm_shard.py
    # or a small Llama arch (auto-detected on the worker):
    MOREGPU_LLM=HuggingFaceTB/SmolLM-135M python3 examples/llm_shard.py
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from moregpu_client import MoreGPU  # noqa: E402

BASE = os.environ.get("MOREGPU_BASE", "http://localhost:8787")
TOKEN = os.environ.get("MOREGPU_ADMIN_TOKEN", "")
MODEL = os.environ.get("MOREGPU_LLM", "gpt2")
PROMPT = os.environ.get("MOREGPU_PROMPT", "The future of distributed computing is")
GEN = int(os.environ.get("MOREGPU_GEN", "16"))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT)["input_ids"]

    pool = MoreGPU(BASE, TOKEN, timeout=600)
    torch_workers = [w for w in pool.workers() if "torch" in (w.get("label") or "")]
    if not torch_workers:
        sys.exit("no native torch worker in the fleet — start apps/worker/worker_torch.py and retry")
    if len(torch_workers) < 2:
        print(f"note: only {len(torch_workers)} torch worker — the split is degenerate (1 stage = whole model). "
              f"Start a 2nd worker for a real cross-machine pipeline.")

    print(f"== pipeline-shard {MODEL} across {len(torch_workers)} worker(s) ==  prompt={PROMPT!r}  gen={GEN}")

    # SHARD: the coordinator splits the layers into contiguous stages, one per worker, and loads each stage.
    plan = pool.shard_load(MODEL)
    stages = plan["stages"]
    print(f"sharded into {len(stages)} stage(s) over {plan['layers']} transformer layers:")
    for i, s in enumerate(stages):
        role = "first+last" if (s["first"] and s["last"]) else "first" if s["first"] else "last" if s["last"] else "middle"
        ph = s.get("params_held")
        print(f"  stage {i}: worker {s['worker']:<14} layers [{s['start']}, {s['end']})  "
              f"{'· '.join([str(s['end']-s['start'])+' blocks', f'{ph:,} params' if ph else '', role])}")
    held = [s.get("params_held") or 0 for s in stages]
    if all(held):
        print(f"  (each worker holds only its stage — {' + '.join(f'{h:,}' for h in held)} params; "
              f"the transformer blocks are genuinely split across workers)")

    # GREEDY DECODE across the pipe: each step runs one forward through every stage, taking the argmax.
    pool.shard_generate(ids, max_new_tokens=1)  # warm the stages (one-off kernel compile)
    t0 = time.time()
    pool_ids = pool.shard_generate(ids, max_new_tokens=GEN)
    dt = time.time() - t0
    print(f"\npool  : {tok.decode(ids + pool_ids)!r}")
    print(f"        {GEN} tokens in {dt*1000:.0f} ms across the pipeline "
          f"({len(stages)} stage hops/token, activations-only on the wire)")

    # REFERENCE: transformers greedy on the same weights (f32) — must match token-for-token.
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    with torch.no_grad():
        ref = model.generate(torch.tensor([ids]), max_new_tokens=GEN, do_sample=False,
                             pad_token_id=tok.eos_token_id)[0].tolist()
    ref_new = ref[len(ids):]
    print(f"ref   : {tok.decode(ref)!r}")

    match = pool_ids == ref_new
    n_match = sum(1 for a, b in zip(pool_ids, ref_new) if a == b)
    print(f"\n{'✅ EXACT MATCH to transformers greedy' if match else '❌ MISMATCH'}  "
          f"({n_match}/{len(ref_new)} tokens · pool {pool_ids[:8]}… vs ref {ref_new[:8]}…)")

    pool.shard_unload()  # free every stage
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
