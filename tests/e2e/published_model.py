#!/usr/bin/env python3
"""published_model.py — run a third-party checkpoint exactly as published (MONAI UNet state_dict, no conversion)
through the coordinator on 2 CPU torch workers:
  (a) /vision/load {spec} verifies sha256 and loads natively; /vision/batch predictions equal a local forward;
  (b) a malicious pickle (full-model pickle with a __reduce__ payload) is refused and its payload never runs;
  (c) a sha256 mismatch is refused."""
import hashlib, json, os, pickle, sys, tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps", "worker"))
from _pool import Pool, Checks  # noqa: E402
from _vision_models import save_state_dict, tiny_monai_unet  # noqa: E402

ARCH = {"registry": "monai", "name": "UNet", "kwargs": {"spatial_dims": 3, "in_channels": 1, "out_channels": 2,
        "channels": [4, 8, 16], "strides": [2, 2], "num_res_units": 1}}


class Evil:
    def __reduce__(self):
        return (os.system, (f"touch {os.environ['EVIL_MARK']}",))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def spec(path, fmt="state_dict", digest=None):
    return {"version": 1, "format": fmt, "source": f"file://{os.path.abspath(path)}", "sha256": digest or sha(path), "arch": ARCH,
            "io": {"inputs": [{"name": "x", "shape": [1, 1, 16, 16, 16], "dtype": "float32"}]}, "licence": "test", "citation": "test"}


def main():
    ck = Checks()
    root = tempfile.mkdtemp(prefix="mgpu-pub-"); models = os.path.join(root, "models"); data = os.path.join(root, "data")
    os.makedirs(models); os.makedirs(data)
    m = tiny_monai_unet(); p = os.path.join(models, "unet.pt"); save_state_dict(m, p)
    evil = os.path.join(models, "evil.pt"); mark = os.path.join(root, "PWNED"); os.environ["EVIL_MARK"] = mark
    with open(evil, "wb") as f:
        pickle.dump({"model": Evil()}, f)
    vols = []
    for i in range(3):
        v = np.random.default_rng(i).standard_normal((16, 16, 16)).astype("float32"); np.save(os.path.join(data, f"v{i}.npy"), v); vols.append(v)
    env = {"MOREGPU_MODEL_ROOTS": models, "MOREGPU_DATA_ROOTS": data, "MOREGPU_OUTPUT_DIR": os.path.join(root, "out"),
           "MOREGPU_CACHE_DIR": os.path.join(root, "cache"), "EVIL_MARK": mark}
    with Pool(["p1", "p2"], worker_env=env) as pool:
        r = pool.api("/vision/load", "POST", {"id": "unet", "spec": spec(p), "task": "segment", "num_classes": 2, "kind": "3d"})
        ck(r.get("ok") and sorted(r["workers"]) == ["p1", "p2"], f"published MONAI state_dict loaded natively on both workers ({r})")
        j = pool.api("/vision/batch", "POST", {"id": "unet", "items": [{"ref": {"uri": f"file://v{i}.npy"}, "out": f"pub{i}"} for i in range(3)]})
        import time
        for _ in range(120):
            st = pool.api(f"/vision/jobs/{j['job']}?results=1")
            if st.get("status") not in ("running", "pending"):
                break
            time.sleep(0.5)
        agree = []
        with torch.no_grad():
            for res in st["results"]:
                ref = m.eval()(torch.from_numpy(vols[res["index"]])[None, None]).argmax(1)[0].numpy()
                agree.append(float((np.load(res["data"]["path"]) == ref).mean()))
        ck(st["status"] == "done" and min(agree) > 0.999, f"batch predictions == local forward (argmax agreement {min(agree):.4f})")
        bad = pool.api("/vision/load", "POST", {"id": "evil", "spec": spec(evil)})
        ck("httperror" in bad and not os.path.exists(mark), f"malicious pickle refused, payload never executed ({bad.get('body', '')[:90]})")
        mm = pool.api("/vision/load", "POST", {"id": "mm", "spec": spec(p, digest="0" * 64)})
        ck("httperror" in mm and "sha256" in mm.get("body", "").lower(), "sha256 mismatch refused")
        caps = pool.api("/vision/capabilities")
        ck(caps.get("ok") and all("formats" in json.dumps(v) or "adapters" in json.dumps(v) for v in caps["workers"].values()), "per-worker vision capabilities reported")
    ck.finish()


if __name__ == "__main__":
    main()
