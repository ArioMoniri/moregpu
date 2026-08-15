#!/usr/bin/env python3
"""
train_single.py — SINGLE-WORKER LoRA fine-tuning regression (no network, no model download, CPU-only).

Closes a real coverage gap: only the DiLoCo (multi-worker) training path had a test; the single-worker path
`/train/load` → `/train/step` → `/train/adapter` (exercised by examples/lora_finetune.py; SDK train_load/
train_step/train_adapter) was untested. This drives the REAL coordinator (server.ts) + worker
(worker_torch.py: attach_lora / train_inner-style single step) and asserts, against the worker's own reported loss:

  (a) LEARNING — the model overfits a FIXED synthetic batch: the per-step training loss DECREASES over K steps
      (final < first with a comfortable margin, least-squares slope < 0, and every loss finite).
  (b) THE ADAPTER ACTUALLY TRAINED — the returned LoRA adapter has a non-trivial `lora_B` (it initializes to
      ZERO, so any non-zero value proves gradients flowed and the optimizer stepped; `lora_A` alone can't move
      the model while B=0). Gradients never leave the worker — only the sealed token batch in, scalar loss out.

Reuses tests/e2e/train_diloco.py's model-gen + api + slope helpers (imported, not duplicated).

Run:  python3 tests/e2e/train_single.py            # exits non-zero on any assertion failure
"""
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_diloco as td  # noqa: E402  (gen_model / api / free_port / slope / decode_tensors / VOCAB / REPO / PROCS)

STEPS = 14
_RESULTS = []


def check(passed, label):
    _RESULTS.append((passed, label))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


def main():
    root = tempfile.mkdtemp(prefix="moregpu-train1-")
    coord_log = os.path.join(root, "coord.log")
    try:
        print("generating tiny gpt2 (random-init, vocab=256)…", flush=True)
        model_dir = td.gen_model(root)

        port = td.free_port(); cfg = os.path.join(root, "mg.json")
        env = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_INSECURE="1")
        td.PROCS.append(subprocess.Popen(
            ["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write", "apps/coordinator/server.ts"],
            cwd=td.REPO, env=env, stdout=open(coord_log, "w"), stderr=subprocess.STDOUT))
        for _ in range(80):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2): break
            except Exception: time.sleep(0.5)
        conf = json.load(open(cfg)); JOIN, ADMIN = conf["joinToken"], conf["adminToken"]
        td.PROCS.append(subprocess.Popen(
            ["python3", "apps/worker/worker_torch.py", "--server", f"ws://127.0.0.1:{port}/ws", "--token", JOIN, "--name", "w1", "--cpu"],
            cwd=td.REPO, env=dict(os.environ), stdout=open(os.path.join(root, "w1.log"), "w"), stderr=subprocess.STDOUT))
        for _ in range(120):
            w = td.api(port, "/workers", admin=ADMIN)
            if isinstance(w, list) and len(w) >= 1: break
            time.sleep(0.5)

        load = td.api(port, "/train/load", "POST",
                      {"model": model_dir, "rank": 8, "alpha": 16, "lr": 1e-2, "seed": 0}, admin=ADMIN)
        check(isinstance(load, dict) and load.get("ok") and load.get("adapter_params", load.get("trainable", 1)) != 0,
              f"/train/load attached a LoRA adapter ({load if not load.get('ok') else 'ok'})")
        if not (isinstance(load, dict) and load.get("ok")):
            print(td.tail(coord_log)); return finish(root)

        def pull_adapter():
            ad = td.api(port, "/train/adapter", "POST", {}, admin=ADMIN)
            raw = ad.get("tensors") or ad.get("adapter") or ad  # worker returns {ok, step, tensors:{name:{data,shape}}}
            raw = raw if isinstance(raw, dict) else {}
            return td.decode_tensors({n: v for n, v in raw.items() if isinstance(v, dict) and "data" in v})

        before = pull_adapter()  # A=randn, B=0 at init — snapshot to prove training actually MOVES it

        # (a) LEARNING — overfit a FIXED batch; loss must decrease.
        rng = random.Random(1234)
        batch = [rng.randrange(0, td.VOCAB) for _ in range(16)]
        losses = []
        for i in range(STEPS):
            r = td.api(port, "/train/step", "POST", {"input_ids": batch, "labels": batch}, admin=ADMIN)
            if not (isinstance(r, dict) and "loss" in r):
                check(False, f"/train/step {i} failed: {r}"); return finish(root)
            losses.append(float(r["loss"]))
        print("  loss trajectory:", [round(x, 4) for x in losses], flush=True)
        finite = all(x == x and abs(x) != float("inf") for x in losses)
        sl = td.slope(losses)
        check(finite, "every training loss is finite")
        check(losses[-1] < losses[0] - 0.02, f"loss decreased over {STEPS} steps ({losses[0]:.4f} → {losses[-1]:.4f})")
        check(sl < -0.002, f"loss has a downward trend (least-squares slope {sl:.4f} < 0)")

        # (b) THE ADAPTER TRAINED — some adapter tensor MOVED from its init (gradients flowed, optimizer stepped).
        after = pull_adapter()
        check(bool(after), f"/train/adapter returned {len(after)} adapter tensors")
        max_move = 0.0
        for n in after:
            if n in before and len(before[n]) == len(after[n]):
                max_move = max(max_move, max((abs(a - b) for a, b in zip(before[n], after[n])), default=0.0))
        check(max_move > 1e-6, f"the LoRA adapter trained off its init (max|Δparam|={max_move:.3e} > 1e-6 after {STEPS} steps)")
        return finish(root)
    finally:
        for p in td.PROCS:
            try: p.terminate()
            except Exception: pass
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def finish(root):
    passed = sum(1 for ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 92)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 92)
    if passed != total:
        for ok, label in _RESULTS:
            if not ok: print(f"  FAILED: {label}")
        return 1
    print("Single-worker LoRA fine-tuning learns (loss decreases) and the adapter trained. ✔")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    finally:
        for p in td.PROCS:
            try: p.terminate()
            except Exception: pass
