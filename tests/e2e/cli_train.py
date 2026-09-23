#!/usr/bin/env python3
"""cli_train.py — the no-code CLI drives a JEPA session, a segment fine-tune on its exported encoder, telemetry export,
`net`, `models describe` and `vision load/batch` against a local pool of 2 CPU torch workers."""
import json, os, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from _pool import Pool, Checks  # noqa: E402


def cli(pool, *args):
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "moregpu_ml.py"), "--url", f"http://127.0.0.1:{pool.port}",
                        "--token", pool.admin, *args], capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout, r.stderr


def main():
    ck = Checks(); root = tempfile.mkdtemp(prefix="mgpu-cli-"); data = os.path.join(root, "data"); os.makedirs(data)
    with Pool(["c1", "c2"], worker_env={"MOREGPU_DATA_ROOTS": data, "MOREGPU_OUTPUT_DIR": os.path.join(root, "out")}) as pool:
        tel = os.path.join(root, "tel.jsonl"); enc = os.path.join(root, "enc")
        rc, out, err = cli(pool, "train", "jepa", "--synthetic", "--synthetic-n", "16", "--size", "32,32", "--model", "micro", "--patch", "8",
                           "--rounds", "2", "--inner-steps", "2", "--batch", "4", "--amp", "fp32", "--export", enc, "--telemetry-out", tel)
        last = json.loads(out.strip().splitlines()[-1]) if rc == 0 else {}
        ck(rc == 0 and last.get("round") == 2, f"`moregpu train jepa` ran 2 rounds ({err[-200:] if rc else last})")
        ck(os.path.exists(os.path.join(enc, "encoder.safetensors")) and sum(1 for _ in open(tel)) >= 4, "encoder exported + telemetry JSONL written")
        rc, out, err = cli(pool, "train", "segment", "--synthetic", "--synthetic-n", "16", "--size", "32,32", "--encoder", enc,
                           "--rounds", "2", "--inner-steps", "2", "--batch", "4", "--amp", "fp32", "--export", os.path.join(root, "seg"))
        ck(rc == 0 and os.path.exists(os.path.join(root, "seg", "model.safetensors")), f"`moregpu train segment` on the JEPA encoder ({err[-200:]})")
        rc, out, _ = cli(pool, "train", "status")
        ck(rc == 0 and len(json.loads(out)["sessions"]) == 2, "`moregpu train status` lists both sessions")
        rc, out, _ = cli(pool, "net", "--pings", "5")
        ck(rc == 0 and json.loads(out)["workers"][0]["rtt_p50_ms"] is not None, "`moregpu net` reports RTT percentiles")
        rc, out, _ = cli(pool, "models", "describe")
        ck(rc == 0 and len(json.loads(out)["workers"]) == 2, "`moregpu models describe` per worker")
        np.save(os.path.join(data, "v.npy"), np.random.default_rng(0).standard_normal((4, 32, 32)).astype("float32"))
        rc, out, err = cli(pool, "vision", "load", "s", "--export", os.path.join(root, "seg"))
        rc2, out2, err2 = cli(pool, "vision", "batch", "s", "--volumes", "v.npy", "--split", "tiles", "--tta", "none")
        res = json.loads(out2[out2.index("{"):]) if rc2 == 0 else {}
        ck(rc == 0 and rc2 == 0 and res.get("done") == 1, f"`moregpu vision load/batch --split tiles` ({err2[-200:]})")
    ck.finish()


if __name__ == "__main__":
    main()
