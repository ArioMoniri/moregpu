#!/usr/bin/env python3
"""vision_pipeline.py — JEPA → segment fine-tune → distributed batch inference, on 3 CPU torch workers:
  (a) the segment fine-tune on a JEPA encoder improves Dice (examples/segment_finetune.py);
  (b) /vision/batch predicts every volume exactly once, writes label maps, reports Dice;
  (c) a worker killed mid-batch → its items are retried on the others and the job still completes;
  (d) a segment session initialised from a pushed:// encoder blob (SDK push_safetensors), and a wrong sha256 refused;
  (e) every prediction carries pred_sha256 (/vision/infer, /vision/infer_batch, /vision/batch cases + tiles), equal to
      the hash of the written label map, so voxel agreement is checkable without reading outputs.
"""
import base64, json, os, sys, tempfile, threading, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps", "worker"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "clients", "python"))
from _pool import Pool, Checks  # noqa: E402
import segment_finetune, vision_batch  # noqa: E402
from moregpu_worker.train.synthetic import SyntheticSeg  # noqa: E402
from moregpu_worker.vision import pred_hash as P  # noqa: E402


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
        # churn, deterministically: start the batch, wait until the first volume is done, THEN kill v3 while items
        # remain — v3 must be retired and its work retried elsewhere
        from moregpu import MoreGPU
        sdk0 = MoreGPU(f"http://127.0.0.1:{pool.port}", pool.admin)
        # (d) encoder init from a pushed blob: the workers read nothing from disk for it
        enc_file = os.path.join(out, "models", "encoder", "encoder.safetensors")
        pr = sdk0.push_safetensors(enc_file, id="enc-pushed")
        ck(pr["ok"] and pr["uri"] == "pushed://enc-pushed" and len(pr["results"]) == 3, f"encoder pushed to 3 workers ({pr['uri']})")
        seg_cfg = {"kind": "2p5d", "num_classes": 3, "decoder": {"channels": [16, 8]}, "mode": "frozen", "weight_decay": 0.0,
                   "synthetic": {"kind": "2p5d", "n": 16, "size": [32, 32], "channels": 3, "seed": 0}}
        ps = sdk0.train_session_create("segment", {**seg_cfg, "encoder": {"init": "export", **pr["ref"]}}, manifest_len=16,
                                       batch=4, inner_steps=1, lr=1e-3, amp="fp32", workers=["v1", "v2"], id="pushed-enc")
        sdk0.train_session_round(ps["session"], 1)
        ex = sdk0.train_session_export(ps["session"], "safetensors", os.path.join(out, "models", "pushed_seg"))
        from safetensors.numpy import load_file as _np_load
        enc_w, seg_w = _np_load(enc_file), _np_load(ex["weights"])
        ck(all(np.array_equal(v, seg_w["encoder." + k]) for k, v in enc_w.items()),
           "segment session initialised from pushed://enc-pushed: frozen encoder == the pushed weights")
        sdk0.train_session_delete(ps["session"])
        bad = pool.api("/train/sessions", "POST", {"task": "segment", "cfg": {**seg_cfg, "encoder": {"init": "export", "path": pr["uri"], "sha256": "0" * 64}},
                                                    "manifest_len": 16, "batch": 4, "inner_steps": 1, "lr": 1e-3, "workers": ["v1"]})
        ck("httperror" in bad and "sha256" in bad["body"], f"pushed encoder with a wrong sha256 refused ({bad.get('httperror')})")
        sdk0.vision_load("seg", r["model_dir"])
        # (e) /vision/infer + /vision/infer_batch: pred_sha256 of argmax(logits) over the class axis
        x = syn.batch([0, 1]).numpy().astype("<f4")
        body = {"id": "seg", "shape": list(x.shape), "data": base64.b64encode(x.tobytes()).decode()}
        ir = pool.api("/vision/infer", "POST", body)
        y = np.frombuffer(base64.b64decode(ir["data"]), dtype="<f4").reshape(ir["shape"])
        ck(ir.get("pred_sha256") == P.pred_sha256(P.labels_from_logits(y)) and ir.get("pred_shape") == [2, 32, 32],
           f"/vision/infer reports pred_sha256 {str(ir.get('pred_sha256'))[:12]}… (labels {ir.get('pred_shape')})")
        ib = pool.api("/vision/infer_batch", "POST", {"id": "seg", "inputs": [{"shape": list(x.shape), "data": body["data"]}] * 3})
        ck(ib.get("ok") and all(o["pred_sha256"] == ir["pred_sha256"] for o in ib["outputs"]),
           f"/vision/infer_batch: identical pred_sha256 on every worker ({sorted({o['worker'] for o in ib['outputs']})})")
        items = [{"ref": {"uri": f"file://v{v}.npy"}, "out": f"pred_v{v}", "mask": {"uri": f"file://g{v}.npy"}} for v in vols]
        job = sdk0.vision_batch("seg", items, tta="flip", steal_after_ms=2000)
        for _ in range(600):
            j = sdk0.vision_job(job["job"])
            if j["done"] >= 1:
                break
            time.sleep(0.05)
        pool.kill_worker("v3")
        res = sdk0.vision_wait(job["job"], poll_s=0.3)
        ck(res["status"] == "done" and res["done"] == 9 and res["failed"] == 0, f"batch completes all 9 volumes ({res['done']}/{res['items']}, retries {res['retries']})")
        ck("v3" in res["dead_workers"], f"killed worker retired mid-batch (dead={res['dead_workers']}, per_worker={res['per_worker']})")
        full = pool.api(f"/vision/jobs/{res['id']}?results=1")
        paths = [x["data"]["path"] for x in full["results"] if x["ok"]]
        ck(len(set(paths)) == 9 and all(os.path.exists(p) and np.load(p).shape == (5, 32, 32) for p in paths), "9 label maps written, correct shape")
        dices = [x["data"]["dice"]["1"] for x in full["results"] if x["ok"]]
        ck(all(d is not None and 0 <= d <= 1 for d in dices) and np.mean(dices) > 0.4, f"per-volume organ Dice reported (mean {np.mean(dices):.3f})")
        ck(any(t["kind"] == "job" for t in full["telemetry"]), "job telemetry record emitted")
        ck(all(x["data"]["pred_sha256"] == P.pred_sha256(np.load(x["data"]["path"])) for x in full["results"] if x["ok"]),
           "/vision/batch job record: every result's pred_sha256 == sha of its written label map")
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
        ck(all(a["data"]["pred_sha256"] == b["data"]["pred_sha256"] for a, b in zip(tfull["results"], cfull["results"])),
           "tile-sharded (vision_merge_write) pred_sha256 == case-level pred_sha256 for every volume")
        ck(tr["status"] == "done" and min(agree) == 1.0 and all(x["data"]["n_parts"] >= 2 for x in tfull["results"]),
           f"tile-sharded prediction == case-level prediction on {tfull['results'][0]['data']['n_parts']} workers (agreement {min(agree):.4f})")
    ck.finish()


if __name__ == "__main__":
    main()
