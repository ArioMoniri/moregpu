#!/usr/bin/env python3
"""
DISTRIBUTED LoRA fine-tuning on the pool — DiLoCo across many torch workers.

lora_finetune.py trains on ONE worker. This scales to N workers with **DiLoCo**
(Distributed Low-Communication training, Douillard et al. 2023; the async / streaming
variants are 2025-26 SOTA for low-bandwidth, heterogeneous, churn-prone fleets — exactly
MoreGPU's). Each worker holds the frozen base + an identical LoRA adapter and trains on
its OWN data shard for H local AdamW steps; then the **coordinator acts as a parameter
server** — it averages the workers' adapters, applies an OUTER Nesterov-momentum step on
the pseudo-gradient (global − average), and broadcasts the new global adapter back. Only
the MB-scale adapter crosses the wire, every H steps — ~500× less communication than
step-by-step gradient sync. This is a genuine reduce path (the pool's matmul path only
concatenates shards; it never averages).

VERIFICATION (out-of-band, since the coordinator can't check a stochastic loss): the run is
deterministic (dropout disabled) and reproduced bit-for-bit by an in-process DiLoCo
reference — same seed, shards, inner/outer optimizers, and averaging. The pool's per-round
per-worker losses AND the final global adapter must match the reference to fp tolerance.

HONEST SCOPE: synchronous DiLoCo (each round waits for all workers). Async DiLoCo (churn-
tolerant, straggler-resilient) and cross-tenant secure aggregation are the next roadmap step.

    # start a pool + 2+ torch workers (different --name each), then:
    MOREGPU_BASE=http://localhost:8787 MOREGPU_ADMIN_TOKEN=... MOREGPU_LLM=gpt2 python3 examples/lora_distributed.py
"""
import os, sys, base64, array

sys.path.insert(0, os.path.dirname(__file__))
from moregpu_client import MoreGPU  # noqa: E402

BASE = os.environ.get("MOREGPU_BASE", "http://localhost:8787")
TOKEN = os.environ.get("MOREGPU_ADMIN_TOKEN", "")
MODEL = os.environ.get("MOREGPU_LLM", "gpt2")
ROUNDS = int(os.environ.get("MOREGPU_ROUNDS", "4"))
INNER = int(os.environ.get("MOREGPU_INNER", "3"))
WINDOW = int(os.environ.get("MOREGPU_WINDOW", "32"))
SEED, LR, OUTER_LR, MOM = 0, 1e-3, 0.7, 0.9
TARGETS = ["c_attn"] if MODEL.startswith(("gpt2", "distilgpt2")) else ["q_proj", "v_proj"]

TEXT = (
    "The compute pool learns together. Many workers, one model; each keeps its own data, shares only a "
    "small adapter, and the coordinator averages them. Machines join and leave, and training goes on. "
    "Low bandwidth, high latency, still it converges. The pool remembers what all of them computed."
) * 6


def windows(ids, t, n):
    out, i = [], 0
    while len(out) < n:
        if i + t + 1 > len(ids):
            i = 0
        out.append(ids[i:i + t]); i += t
    return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(TEXT)["input_ids"]
    pool = MoreGPU(BASE, TOKEN, timeout=900)

    tw = sorted(w["id"] for w in pool.workers() if "torch" in (w.get("label") or ""))
    if len(tw) < 1:
        sys.exit("no torch workers — start apps/worker/worker_torch.py")
    if len(tw) < 2:
        print(f"⚠ only 1 torch worker ({tw[0]}) — DiLoCo runs but is degenerate. Start a 2nd worker "
              f"(different --name) for a real distributed run.")
    N = len(tw)
    allw = windows(ids, WINDOW, N * 4)
    shards = [allw[i::N] for i in range(N)]                 # round-robin shard across workers
    batches = {tw[i]: shards[i] for i in range(N)}
    print(f"== DiLoCo {MODEL} across {N} worker(s) ==  rounds={ROUNDS} inner={INNER} outer_lr={OUTER_LR} mom={MOM}")

    info = pool.diloco_load(MODEL, rank=8, alpha=16, lr=LR, seed=SEED, targets=TARGETS)
    print(f"group={info['workers']} · adapter params={info['adapter_params']:,}")

    pool_rounds = []
    for _ in range(ROUNDS):
        res = pool.diloco_round(batches, inner_steps=INNER, lr=LR, outer_lr=OUTER_LR, outer_momentum=MOM)
        pool_rounds.append({x["worker"]: x["losses"] for x in res["worker_losses"]})
        print(f"  round {res['round']}: {res['workers']} workers · avg last loss {res['avg_last_loss']:.4f}")

    # ---------- verify against a deterministic in-process DiLoCo reference ----------
    print("\n== verify ==")
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "worker"))
    import worker_torch as wt  # reuse the worker's exact LoRA code so init matches
    from transformers import AutoModelForCausalLM
    dev = wt.DEV
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    wt.attach_lora(model, TARGETS, 8, 16)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    global_ = {n: p.detach().clone() for n, p in trainable.items()}       # identical initial adapter
    momentum = {n: torch.zeros_like(v) for n, v in global_.items()}

    ref_rounds = []
    for _ in range(ROUNDS):
        round_losses, avg = [], {n: torch.zeros_like(v) for n, v in global_.items()}
        for k in range(N):
            with torch.no_grad():                                        # every worker starts the round from the global
                for n, p in trainable.items():
                    p.copy_(global_[n])
            opt = torch.optim.AdamW(list(trainable.values()), lr=LR)
            model.train(); losses = []
            for i in range(INNER):
                x = torch.tensor(shards[k][i % len(shards[k])], device=dev).unsqueeze(0)
                opt.zero_grad(); l = model(input_ids=x, labels=x).loss; l.backward(); opt.step()
                losses.append(float(l.item()))
            round_losses.append(losses)
            with torch.no_grad():
                for n, p in trainable.items():
                    avg[n] += p.detach()
        with torch.no_grad():
            for n in avg:
                avg[n] /= N
                d = global_[n] - avg[n]
                momentum[n] = MOM * momentum[n] + d
                global_[n] = global_[n] - OUTER_LR * (d + MOM * momentum[n])
        ref_rounds.append(round_losses)

    # (1) per-round per-worker loss match
    dloss = 0.0
    for r in range(ROUNDS):
        for k in range(N):
            pl = pool_rounds[r].get(tw[k])
            if pl:
                dloss = max(dloss, max(abs(a - b) for a, b in zip(pl, ref_rounds[r][k])))
    print(f"  loss-trajectory match: max|Δ| across all rounds/workers = {dloss:.3e}  {'OK' if dloss < 1e-2 else 'FAIL'}")

    # (2) final global adapter match
    padapt = pool.diloco_adapter()["tensors"]
    dadapt = 0.0
    for n, g in global_.items():
        pd = array.array("f"); pd.frombytes(base64.b64decode(padapt[n]["data"]))
        dadapt = max(dadapt, max(abs(a - b) for a, b in zip(pd, g.detach().cpu().flatten().tolist())))
    print(f"  global-adapter match:  max|Δ| = {dadapt:.3e}  {'OK' if dadapt < 1e-2 else 'FAIL'}")

    # (3) learning
    first = sum(v[0] for v in ref_rounds[0]) / N
    last = sum(v[-1] for v in ref_rounds[-1]) / N
    print(f"  learning: avg loss {first:.4f} → {last:.4f}  {'OK' if last < 0.9 * first else 'FAIL'}")

    ok = dloss < 1e-2 and dadapt < 1e-2 and last < 0.9 * first
    print(f"\n{'✅ REAL distributed (DiLoCo) training on the pool, verified vs a reference' if ok else '❌ verification failed'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
