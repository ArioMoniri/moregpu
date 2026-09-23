#!/usr/bin/env python3
"""data_plane_jepa.py — JEPA on REAL files through the worker data plane (ADR-0110), 2 CPU torch workers:
  (a) file:// shards under MOREGPU_DATA_ROOTS: a jepa_2p5d session trains from a refs.jsonl manifest;
  (b) a path-escape ref (file://../) is refused (session init fails, nothing trains);
  (c) pushed://: a manifest + volume streamed via /data/push, then trained on;
  (d) /workers/:id/caps reports readers and data-plane policy."""
import base64, hashlib, json, os, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _pool import Pool, Checks  # noqa: E402


def main():
    ck = Checks()
    data = tempfile.mkdtemp(prefix="mgpu-data-")
    rng = np.random.default_rng(0)
    refs = []
    for v in range(3):
        vol = (rng.standard_normal((12, 32, 32)) * 0.2).astype("float16")
        vol[:, 8:20, 10:24] += 1.0                      # a structured "organ"
        np.save(os.path.join(data, f"vol{v}.npy"), vol)
        for z in range(0, 10):
            refs.append({"uri": f"file://vol{v}.npy", "slice": [z, z + 3], "meta": {"label": v}})
    man = "\n".join(json.dumps(r) for r in refs) + "\n"
    open(os.path.join(data, "refs.jsonl"), "w").write(man)
    spec = {"size": [32, 32], "channels": 3}
    jcfg = {"model": "micro", "patch": 8, "pred_dim": 16, "pred_depth": 1, "pred_heads": 2, "n_targets": 2,
            "ema": [0.99, 1.0], "total_steps": 12, "weight_decay": 0.0, "probe_batch": 8}
    base = {"task": "jepa_2p5d", "amp": "fp32", "seed": 0, "manifest_len": len(refs), "batch": 4, "inner_steps": 2, "lr": 2e-3}
    with Pool(["d1", "d2"], worker_env={"MOREGPU_DATA_ROOTS": data, "MOREGPU_CACHE_DIR": os.path.join(data, ".cache")}) as pool:
        caps = pool.api("/workers/d1/caps")
        ck(caps.get("ok") and caps["data"]["readers"]["numpy"] and caps["data"]["n_roots"] == 1, f"caps report readers + 1 data root")
        r = pool.api("/train/sessions", "POST", {**base, "id": "fs", "cfg": {**jcfg, "data": {"manifest": "file://refs.jsonl", "spec": spec}}})
        ck(r.get("ok"), f"session on file:// manifest ({r.get('error', 'ok')})")
        rr = pool.api("/train/sessions/fs/round", "POST", {"rounds": 3})
        ok = rr.get("ok") and rr.get("samples_seen") == 48
        ck(ok, f"3 rounds on file:// data, 48 samples ({rr.get('samples_seen')})")
        tel = pool.api("/train/sessions/fs/telemetry")["records"]
        ck(any(t["kind"] == "worker_round" and t["data_s"] > 0 for t in tel), "data-load time measured separately in telemetry")
        bad = pool.api("/train/sessions", "POST", {**base, "id": "esc", "cfg": {**jcfg, "data": {"manifest": "file://../../etc/passwd", "spec": spec}}})
        ck("httperror" in bad, f"path-escape manifest refused ({bad.get('httperror')})")
        # pushed:// — the volume and a manifest that references it
        vol = open(os.path.join(data, "vol0.npy"), "rb").read()
        pv = pool.api("/data/push", "POST", {"id": "vol0.npy", "sha256": hashlib.sha256(vol).hexdigest(), "data_b64": base64.b64encode(vol).decode(), "suffix": ".npy"})
        ck(pv.get("ok") and pv.get("uri") == "pushed://vol0.npy", f"volume pushed to both workers ({pv.get('results')})")
        pm_txt = "\n".join(json.dumps({"uri": "pushed://vol0.npy", "slice": [z, z + 3], "meta": {"label": 0}}) for z in range(10)) + "\n"
        pm = pool.api("/data/push", "POST", {"id": "man.jsonl", "sha256": hashlib.sha256(pm_txt.encode()).hexdigest(), "data_b64": base64.b64encode(pm_txt.encode()).decode(), "suffix": ".jsonl"})
        badsha = pool.api("/data/push", "POST", {"id": "x.npy", "sha256": "0" * 64, "data_b64": base64.b64encode(b"abc").decode()})
        ck(pm.get("ok") and "httperror" in badsha, "manifest pushed; sha256 mismatch refused")
        r2 = pool.api("/train/sessions", "POST", {**base, "id": "push", "manifest_len": 10, "cfg": {**jcfg, "data": {"manifest": "pushed://man.jsonl", "spec": spec}}})
        rr2 = pool.api("/train/sessions/push/round", "POST", {"rounds": 2}) if r2.get("ok") else r2
        ck(rr2.get("ok") and rr2.get("round") == 2, f"trained on pushed:// data ({rr2.get('error', 'ok')})")
    ck.finish()


if __name__ == "__main__":
    main()
