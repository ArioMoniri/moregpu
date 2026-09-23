#!/usr/bin/env python3
"""vision_pipeline.py — JEPA → segment fine-tune → distributed batch inference, on 3 CPU torch workers:
  (a) the segment fine-tune on a JEPA encoder improves Dice (examples/segment_finetune.py);
  (b) /vision/batch predicts every volume exactly once, writes label maps, reports Dice;
  (c) a worker killed mid-batch → its items are retried on the others and the job still completes.
"""
import json, os, sys, tempfile, threading, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps", "worker"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "clients", "python"))
from _pool import Pool, Checks  # noqa: E402
import segment_finetune, vision_batch  # noqa: E402
from moregpu_worker.train.synthetic import SyntheticSeg  # noqa: E402


def main():
    ck = Checks()
    root = tempfile.mkdtemp(prefix="mgpu-vis-"); data = os.path.join(root, "data"); out = os.path.join(root, "out")
    os.makedirs(data)
    syn = SyntheticSeg(kind="2p5d", n=64, size=[32, 32], channels=3, seed=5)
    vols = []
    for v in range(9):
        idx = list(range(v * 5, v * 5 + 5))
        np.save(os.path.join(data, f"v{v}.npy"), syn.batch(idx)[:, 1].numpy().astype("float32"))
        np.save(os.path.join(data, f"g{v}.npy"), syn.masks(idx).numpy().astype("uint8"))
        vols.append(v)
    env = {"MOREGPU_DATA_ROOTS": data, "MOREGPU_OUTPUT_DIR": out, "MOREGPU_CACHE_DIR": os.path.join(root, "cache")}
    with Pool(["v1", "v2", "v3"], worker_env=env) as pool:
        r = segment_finetune.main(["--url", f"http://127.0.0.1:{pool.port}", "--token", pool.admin, "--out", os.path.join(out, "models")])
        ck(r["dice_after"]["dice_mean"] > r["dice_before"]["dice_mean"] and r["dice_after"]["dice_mean"] > 0.4,
           f"segment fine-tune on JEPA encoder improves Dice ({r['dice_before']['dice_mean']:.3f} → {r['dice_after']['dice_mean']:.3f})")
        killer = threading.Timer(1.0, lambda: pool.kill_worker("v3"))
        killer.start()
        res = vision_batch.main(["--url", f"http://127.0.0.1:{pool.port}", "--token", pool.admin, "--model-dir", r["model_dir"],
                                 "--volumes", *[f"v{v}.npy" for v in vols], "--masks", *[f"g{v}.npy" for v in vols]])
        killer.cancel()
        ck(res["status"] == "done" and res["done"] == 9 and res["failed"] == 0, f"batch completes all 9 volumes ({res['done']}/{res['items']}, retries {res['retries']})")
        full = pool.api(f"/vision/jobs/{res['id']}?results=1")
        paths = [x["data"]["path"] for x in full["results"] if x["ok"]]
        ck(len(set(paths)) == 9 and all(os.path.exists(p) and np.load(p).shape == (5, 32, 32) for p in paths), "9 label maps written, correct shape")
        dices = [x["data"]["dice"]["1"] for x in full["results"] if x["ok"]]
        ck(all(d is not None and 0 <= d <= 1 for d in dices) and np.mean(dices) > 0.4, f"per-volume organ Dice reported (mean {np.mean(dices):.3f})")
        ck(any(t["kind"] == "job" for t in full["telemetry"]), "job telemetry record emitted")
        # tile sharding: each volume split across the live workers; output must equal the case-level prediction
        from moregpu import MoreGPU
        sdk = MoreGPU(f"http://127.0.0.1:{pool.port}", pool.admin)
        tj = sdk.vision_batch("seg", [{"ref": {"uri": f"file://v{v}.npy"}, "out": f"tiled_{v}", "mask": {"uri": f"file://g{v}.npy"}} for v in range(3)],
                              split="tiles", tta="none")
        tr = sdk.vision_wait(tj["job"], poll_s=0.3)
        tfull = sdk.vision_job(tj["job"], results=True)
        cj = sdk.vision_wait(sdk.vision_batch("seg", [{"ref": {"uri": f"file://v{v}.npy"}, "out": f"case_{v}"} for v in range(3)], tta="none")["job"], poll_s=0.3)
        cfull = sdk.vision_job(cj["id"], results=True)
        agree = [float((np.load(a["data"]["path"]) == np.load(b["data"]["path"])).mean()) for a, b in zip(tfull["results"], cfull["results"])]
        ck(tr["status"] == "done" and min(agree) == 1.0 and all(x["data"]["n_parts"] >= 2 for x in tfull["results"]),
           f"tile-sharded prediction == case-level prediction on {tfull['results'][0]['data']['n_parts']} workers (agreement {min(agree):.4f})")
    ck.finish()


if __name__ == "__main__":
    main()
