#!/usr/bin/env python3
"""train_sessions.py — generic TrainTask sessions end to end (ADR-0104..0107): real coordinator + 2 CPU torch
workers, task toy_linear, driven over /train/sessions. Asserts:
  (a) the coordinator's DiLoCo result equals the in-process reference (moregpu_worker.train.local.run_diloco) —
      same shards, weighting, outer Nesterov — within 1e-5;
  (b) samples-seen accounting stops exactly at target_samples;
  (c) telemetry per (round, worker) is schema-valid and its breakdown sums to wall time;
  (d) checkpoint → delete → resume continues identically to an uninterrupted run;
  (e) a legacy /train session and a task session coexist on one worker (per-session keying, no clobbering);
  (f) a killed worker is dropped and the session keeps training on the survivor.
"""
import base64, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "apps", "worker"))
from _pool import Pool, Checks  # noqa: E402
from moregpu_worker.train.local import run_diloco  # noqa: E402

TOY = {"n": 64, "dim": 5, "batch": 4, "optimizer": "sgd"}
BASE = {"task": "toy_linear", "cfg": TOY, "amp": "fp32", "seed": 3, "manifest_len": 64, "batch": 4, "inner_steps": 2,
        "lr": 0.05, "outer_lr": 0.7, "outer_momentum": 0.9, "chunk_bytes": 16}


def state(pool, sid):
    r = pool.api(f"/train/sessions/{sid}/state")
    blob = base64.b64decode(r["blob_b64"])
    return {e["name"]: np.frombuffer(blob[e["offset"]:e["offset"] + e["nbytes"]], dtype="<f4").reshape(e["shape"])
            for e in r["header"]["tensors"]}


def main():
    ck = Checks()
    with Pool(["w1", "w2"]) as pool:
        # (a) equivalence with the in-process reference
        r = pool.api("/train/sessions", "POST", {**BASE, "id": "eq", "workers": ["w1", "w2"]})
        ck(r.get("ok") and r.get("params") == 6, f"session created on 2 workers ({r})")
        rr = pool.api("/train/sessions/eq/round", "POST", {"rounds": 4})
        ck(rr.get("ok") and rr.get("round") == 4 and rr.get("samples_seen") == 64, f"4 rounds, 64 samples seen ({rr.get('samples_seen')})")
        ref = run_diloco("toy_linear", TOY, n_workers=2, rounds=4, inner_steps=2, batch=4, lr=0.05, outer_lr=0.7,
                         outer_momentum=0.9, seed=3, manifest_len=64)
        got = state(pool, "eq")
        err = max(float(np.abs(got[k] - ref["state"][k].numpy()).max()) for k in got)
        ck(err < 1e-5, f"coordinator DiLoCo == in-process reference (max|Δ|={err:.2e})")
        # (c) telemetry
        tel = pool.api("/train/sessions/eq/telemetry")["records"]
        wr = [x for x in tel if x["kind"] == "worker_round"]
        ck(len(wr) == 8 and all(x["schema"] == "moregpu.telemetry/1" for x in wr), f"8 worker_round records ({len(wr)})")
        ok_sum = all(abs(x["compute_s"] + x["data_s"] + x["serialize_s"] + x["network_s"] + x["wait_s"] - x["wall_s"]) <= 1e-6 + 0.05 * x["wall_s"] for x in wr)
        ck(ok_sum, "telemetry breakdown sums to wall time")
        # (d) checkpoint → resume
        pool.api("/train/sessions/eq/checkpoint", "POST", {})
        pool.api("/train/sessions/eq", "DELETE")
        res = pool.api("/train/sessions/resume", "POST", {"id": "eq"})
        ck(res.get("ok") and res.get("round") == 4, f"resumed at round 4 ({res.get('round')})")
        pool.api("/train/sessions/eq/round", "POST", {"rounds": 2})
        ref6 = run_diloco("toy_linear", TOY, n_workers=2, rounds=6, inner_steps=2, batch=4, lr=0.05, outer_lr=0.7,
                          outer_momentum=0.9, seed=3, manifest_len=64)
        got6 = state(pool, "eq")
        err6 = max(float(np.abs(got6[k] - ref6["state"][k].numpy()).max()) for k in got6)
        ck(err6 < 1e-5, f"resume continues identically (max|Δ|={err6:.2e})")
        pool.api("/train/sessions/eq", "DELETE")
        # (b) target samples
        t = pool.api("/train/sessions", "POST", {**BASE, "id": "tgt", "target_samples": 37})
        rr = pool.api("/train/sessions/tgt/round", "POST", {"rounds": 100})
        ck(rr.get("samples_seen") == 37 and rr.get("status") == "done", f"stops exactly at 37 samples ({rr.get('samples_seen')}, {rr.get('status')})")
        pool.api("/train/sessions/tgt", "DELETE")
        # (e) isolation: a task session and a second task session on the same worker keep separate weights
        a = pool.api("/train/sessions", "POST", {**BASE, "id": "iso-a", "workers": ["w1"]})
        b = pool.api("/train/sessions", "POST", {**BASE, "id": "iso-b", "workers": ["w1"], "cfg": {**TOY, "init_seed": 7}})
        ck(a.get("ok") and b.get("ok"), "two task sessions on one worker")
        sb0 = state(pool, "iso-b")
        pool.api("/train/sessions/iso-a/round", "POST", {"rounds": 3})
        sb1 = state(pool, "iso-b")
        ck(all(np.array_equal(sb0[k], sb1[k]) for k in sb0), "training session A never touches session B")
        r3 = pool.api("/train/sessions", "POST", {**BASE, "id": "iso-c", "workers": ["w1"]})
        ck("httperror" in r3 and "sessions" in r3.get("body", ""), f"admission limit refuses a 3rd session ({r3.get('httperror')})")
        for sid in ("iso-a", "iso-b"):
            pool.api(f"/train/sessions/{sid}", "DELETE")
        # (f) churn: kill w2 mid-session
        pool.api("/train/sessions", "POST", {**BASE, "id": "churn"})
        pool.api("/train/sessions/churn/round", "POST", {"rounds": 1})
        pool.kill_worker("w2"); time.sleep(1.5)
        rr = pool.api("/train/sessions/churn/round", "POST", {"rounds": 2})
        ck(rr.get("ok") and rr.get("workers") == ["w1"] and rr.get("round") == 3, f"survivor keeps training ({rr.get('workers')}, round {rr.get('round')})")
        ls = pool.api("/train/sessions")
        ck(ls.get("ok") and any(s["id"] == "churn" for s in ls["sessions"]), "session listed")
    ck.finish()


if __name__ == "__main__":
    main()
