#!/usr/bin/env python3
"""jepa_sessions.py — JEPA pretraining through the REAL coordinator + 2 CPU torch workers (examples/jepa_synthetic.py):
  (a) loss decreases; (b) target encoders stay bit-identical across workers every round (no EMA drift alarm);
  (c) collapse monitors are reported; (d) the coordinator's global equals the in-process DiLoCo reference;
  (e) encoder export (safetensors) on a worker loads into a fresh ViT."""
import base64, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps", "worker"))
from _pool import Pool, Checks  # noqa: E402
import jepa_synthetic as ex  # noqa: E402
from moregpu_worker.train.local import run_diloco  # noqa: E402


def main():
    ck = Checks()
    with Pool(["j1", "j2"]) as pool:
        out = ex.main(["--url", f"http://127.0.0.1:{pool.port}", "--token", pool.admin, "--rounds", "6", "--inner", "3",
                       "--batch", "8", "--export-dir", os.path.join(pool.root, "enc")])
        L = out["losses"]
        ck(len(L) == 6 and L[-1] < L[0], f"loss decreases over 6 rounds ({L[0]:.3f} → {L[-1]:.3f})")
        sess = pool.api(f"/train/sessions/{out['session']}")
        alarms = [a for h in sess["history"] for a in h["alarms"]]
        ck(not any("diverged" in a for a in alarms), "target encoders identical on both workers every round")
        ck(all(h.get("monitors") and "rankme" in h["monitors"] for h in sess["history"]), "collapse monitors every round")
        ref = run_diloco("jepa_2p5d", {**ex.JEPA, "synthetic": ex.SYN, "total_steps": 18}, n_workers=2, rounds=6,
                         inner_steps=3, batch=8, lr=2e-3, outer_lr=0.7, outer_momentum=0.9, seed=0,
                         manifest_len=ex.SYN["n"], lr_schedule={"kind": "cosine", "warmup_frac": 0.1, "min_lr": 2e-3 * 0.05},
                         target_samples=6 * 3 * 8 * 2)
        st = pool.api(f"/train/sessions/{out['session']}/state")
        blob = base64.b64decode(st["blob_b64"])
        err = 0.0
        for e in st["header"]["tensors"]:
            got = np.frombuffer(blob[e["offset"]:e["offset"] + e["nbytes"]], dtype="<f4").reshape(e["shape"])
            err = max(err, float(np.abs(got - ref["state"][e["name"]].numpy()).max()))
        ck(err < 1e-4, f"coordinator JEPA global == in-process reference (max|Δ|={err:.2e})")
        tel = out["telemetry"]
        ck(any(t["kind"] == "worker_round" and t["amp"] == "fp32" for t in tel), "telemetry carries AMP mode")
        exp = out["export"]
        ok = False
        if exp and exp.get("ok"):
            import torch
            from safetensors.torch import load_file
            from moregpu_worker.models.vit import VisionTransformer
            m = VisionTransformer(**json.load(open(exp["config"])))
            m.load_state_dict(load_file(exp["weights"]), strict=True); ok = True
        ck(ok, "exported encoder loads strictly into a fresh ViT")
        # resume mid-run: 3 rounds → checkpoint → delete → resume → 3 rounds must equal 6 uninterrupted rounds
        import sys as _s
        _s.path.insert(0, os.path.join(HERE, "..", "..", "clients", "python"))
        from moregpu import MoreGPU
        sdk = MoreGPU(f"http://127.0.0.1:{pool.port}", pool.admin)
        body = dict(synthetic=ex.SYN, jepa={**ex.JEPA, "total_steps": 18}, manifest_len=ex.SYN["n"], batch=8, inner_steps=3, lr=2e-3,
                    amp="fp32", seed=0, target_samples=6 * 3 * 8 * 2, lr_schedule={"kind": "cosine", "warmup_frac": 0.1, "min_lr": 2e-3 * 0.05})
        sdk.train_jepa("jepa_2p5d", id="res", **body)
        for _ in range(3):
            sdk.train_session_round("res", 1)
        sdk.train_session_checkpoint("res"); sdk.train_session_delete("res"); sdk.train_session_resume("res")
        for _ in range(3):
            sdk.train_session_round("res", 1)
        st2 = pool.api("/train/sessions/res/state"); blob2 = base64.b64decode(st2["blob_b64"]); err2 = 0.0
        for e in st2["header"]["tensors"]:
            got = np.frombuffer(blob2[e["offset"]:e["offset"] + e["nbytes"]], dtype="<f4").reshape(e["shape"])
            err2 = max(err2, float(np.abs(got - ref["state"][e["name"]].numpy()).max()))
        ck(err2 < 1e-4, f"JEPA resume (EMA target + counters restored) == uninterrupted run (max|Δ|={err2:.2e})")
        knn = out["knn"]
        ck(0.0 <= knn.get("knn_acc", -1) <= 1.0, f"k-NN probe reported ({knn})")
    ck.finish()


if __name__ == "__main__":
    main()
