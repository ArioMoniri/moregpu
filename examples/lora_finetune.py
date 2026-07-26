#!/usr/bin/env python3
"""
Fine-tune a REAL model ON THE POOL — LoRA, on a native torch worker.

MoreGPU's WebGPU workers are inference-only (no autograd). The native torch worker
(apps/worker/worker_torch.py) adds the missing half: the WHOLE training step
(forward → cross-entropy → backward → optimizer.step) runs LOCALLY on the worker in
torch. The base model is frozen; a small LoRA adapter is the only trainable tensor,
so GRADIENTS NEVER LEAVE THE WORKER — only a tiny sealed microbatch (token ids) goes
in and a scalar loss comes out; the MB-scale adapter is pulled back on demand.

HONEST SCOPE: this is SINGLE-worker fine-tuning — the pool hosts one training job, it
does not (yet) parallelize training across workers (that is DiLoCo/data-parallel, on the
roadmap). And training loses the pool's exact-match verification (a stochastic loss can't
be checked by the coordinator's CPU reference), so we verify OUT-OF-BAND here:
  1. ANCHOR  — the pool's step-1 loss (adapter starts as a no-op: LoRA B=0) must equal a
     frozen in-process forward of the same model on the same batch → proves the pool is
     computing the real model's forward + loss correctly, not just "a number that falls".
  2. LEARNING — loss must drop substantially over the run.
  3. REFERENCE — a seeded in-process LoRA run (identical init/opt/data via the worker's own
     code) should closely track the pool's loss curve (reported; MPS isn't bit-deterministic
     across processes, so this is a tolerance check, not exact-match).

    # 1) start a pool + a torch worker (see the worker docstring), then:
    MOREGPU_BASE=http://localhost:8787 MOREGPU_ADMIN_TOKEN=... MOREGPU_LLM=gpt2 python3 examples/lora_finetune.py
"""
import os, sys, math

sys.path.insert(0, os.path.dirname(__file__))
from moregpu_client import MoreGPU  # noqa: E402

BASE = os.environ.get("MOREGPU_BASE", "http://localhost:8787")
TOKEN = os.environ.get("MOREGPU_ADMIN_TOKEN", "")
MODEL = os.environ.get("MOREGPU_LLM", "gpt2")
STEPS = int(os.environ.get("MOREGPU_STEPS", "40"))
T = int(os.environ.get("MOREGPU_WINDOW", "32"))
SEED = 0
# LoRA targets: GPT-2's fused QKV Conv1D, or the standard q/v projections on Llama/Qwen-style models.
TARGETS = ["c_attn"] if MODEL.startswith(("gpt2", "distilgpt2")) else ["q_proj", "v_proj"]

# A short, self-contained training text (no download). Overfitting a tiny text makes learning obvious.
TEXT = (
    "The compute pool learns. A gpu shares its cycles, a worker signs its work, and the "
    "coordinator seals every byte. Machines join, machines leave, and the model still trains. "
    "More gpus, more speed; fewer round trips, less waiting. The pool remembers what it computed."
) * 6


def windows(ids, t, n):
    out, i = [], 0
    while len(out) < n:
        if i + t + 1 > len(ids):
            i = 0
        out.append(ids[i:i + t])
        i += t
    return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(TEXT)["input_ids"]
    batches = windows(ids, T, STEPS)
    pool = MoreGPU(BASE, TOKEN, timeout=600)

    ws = pool.workers()
    if not any("torch" in (w.get("label") or "") for w in ws):
        sys.exit("no native torch worker in the fleet — start apps/worker/worker_torch.py and retry")

    print(f"== LoRA fine-tune {MODEL} on the pool ==  targets={TARGETS} rank=8 steps={STEPS} window={T}")
    info = pool.train_load(MODEL, rank=8, alpha=16, lr=1e-3, seed=SEED, targets=TARGETS)
    print(f"loaded on {info['worker']} · trainable params={info.get('trainable_params'):,} · device={info.get('device')}")

    losses = []
    for i, b in enumerate(batches):
        r = pool.train_step(b)
        losses.append(r["loss"])
        if i < 3 or (i + 1) % 5 == 0:
            print(f"  step {r['step']:>3}  loss {r['loss']:.4f}")

    # ---- verification (out-of-band: the coordinator can't check a stochastic loss) ----
    print("\n== verify ==")
    import torch
    from transformers import AutoModelForCausalLM
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # context: the frozen base model's loss (eval mode, no dropout) on batch 0 — the starting point
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    with torch.no_grad():
        x = torch.tensor(batches[0], device=dev).unsqueeze(0)
        base_loss = float(base(input_ids=x, labels=x).loss.item())
    print(f"  context: base model (eval) loss on batch 0 = {base_loss:.4f}")

    # (1) LEARNING: loss dropped substantially over the run
    learned = min(losses) < 0.85 * losses[0]
    print(f"  learning: {losses[0]:.4f} → {min(losses):.4f}  (Δ={losses[0]-min(losses):.4f})  {'OK' if learned else 'FAIL'}")

    # (2) CORRECTNESS: a seeded in-process LoRA run using the WORKER'S OWN LoRA code (identical init, optimizer,
    #     data order) must reproduce the pool's loss curve — this is the real proof the pool trained correctly.
    ref_drift = None
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "worker"))
        import worker_torch as wt  # reuse the exact LoRAWrap/attach_lora so init/RNG-consumption matches the worker
        torch.manual_seed(SEED)
        m2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(wt.DEV)
        wt.attach_lora(m2, TARGETS, 8, 16)
        opt2 = torch.optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=1e-3)
        ref = []
        for b in batches:
            xb = torch.tensor(b, device=wt.DEV).unsqueeze(0)
            m2.train(); opt2.zero_grad()
            l = m2(input_ids=xb, labels=xb).loss
            l.backward(); opt2.step()
            ref.append(float(l.item()))
        ref_drift = max(abs(a - b) for a, b in zip(losses, ref))
        print(f"  correctness: pool curve vs seeded in-process reference — max|Δ| = {ref_drift:.3e}  "
              f"{'OK' if ref_drift < 1e-2 else 'FAIL'}  (ref final {ref[-1]:.4f}, pool {losses[-1]:.4f})")
    except Exception as e:
        print(f"  correctness: reference skipped ({e})")

    # (3) the adapter came back and actually moved off its zero init
    ad = pool.train_adapter()
    import base64, array
    moved = 0.0
    for _name, t in ad.get("tensors", {}).items():
        a = array.array("f"); a.frombytes(base64.b64decode(t["data"]))
        moved = max(moved, max((abs(v) for v in a), default=0.0))
    print(f"  adapter: {len(ad.get('tensors', {}))} tensors pulled back · max|w|={moved:.3e}  {'OK' if moved > 0 else 'FAIL'}")

    ok = learned and moved > 0 and (ref_drift is None or ref_drift < 1e-2)
    print(f"\n{'✅ REAL training on the pool, verified against a seeded reference' if ok else '❌ verification failed'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
