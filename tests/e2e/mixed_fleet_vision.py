#!/usr/bin/env python3
"""mixed_fleet_vision.py — one vision model served by a MIXED fleet: 1 native torch worker + 1 Deno WebGPU worker.

  (a) /vision/load {fleet: 'all'} loads the model natively on the torch worker, lowers it there (vision_lower, target
      'wgsl', include_bytes), streams graph.json + model.safetensors to the WebGPU worker (push_begin/chunk/end) and
      vision_loads it there;
  (b) /vision/infer routes to either holder and returns the same {shape, data} shape from both kinds;
  (c) /vision/infer_batch spreads 8 inputs over BOTH workers (work-stealing queue), returns outputs in order, per-worker
      counts and max|Δ| vs the torch reference (check_parity) — every output equals local PyTorch within 1e-4;
  (d) both kinds report the same pred_sha256 (argmax labels over the class axis, moregpu.pred/1) as local PyTorch.

The WebGPU worker needs a real adapter (Mesa lavapipe works: `apt install mesa-vulkan-drivers`). Without one this test
prints a SKIP line and exits 0.
"""
import base64, json, os, subprocess, sys, tempfile, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps", "worker"))
from _pool import REPO, Pool, Checks, tail  # noqa: E402
from moregpu_worker.vision import pred_hash as P  # noqa: E402

ADAPTER_PROBE = "const a = await navigator.gpu?.requestAdapter(); console.log(a ? 'ADAPTER ' + (a.info?.description || a.info?.device || '?') : 'NO_ADAPTER');\n"


def webgpu_adapter() -> str | None:
    d = tempfile.mkdtemp(prefix="mgpu-probe-")
    p = os.path.join(d, "probe.ts")
    open(p, "w").write(ADAPTER_PROBE)
    try:
        out = subprocess.run(["deno", "run", "--unstable-webgpu", p], capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return None
    line = next((l for l in out.splitlines() if l.startswith("ADAPTER ")), None)
    return line[len("ADAPTER "):] if line else None


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def unb64(s: str, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4").reshape(shape)


def make_export(root: str):
    """A small MoreGPU segment export (ViT-micro encoder + conv decoder, 3 classes) on disk + the torch module."""
    import torch
    from safetensors.torch import save_file
    from moregpu_worker.models.vit import vit_config
    from moregpu_worker.vision import models as VM
    torch.manual_seed(11)
    enc = vit_config("micro", (32, 32), 8, 3)
    m = VM.build("segment", enc, 3).eval()
    d = os.path.join(root, "seg_export")
    os.makedirs(d)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, os.path.join(d, "model.safetensors"))
    json.dump({"task": "segment", "num_classes": 3, "kind": "2p5d", "encoder": {**enc, "img_size": list(enc["img_size"])}},
              open(os.path.join(d, "model_config.json"), "w"))
    return d, m


def main():
    adapter = webgpu_adapter()
    if not adapter:
        print("SKIP: mixed_fleet_vision — no WebGPU adapter visible to Deno (install mesa-vulkan-drivers for lavapipe)")
        sys.exit(0)
    print(f"WebGPU adapter: {adapter}")
    import torch
    ck = Checks()
    root = tempfile.mkdtemp(prefix="mgpu-mixed-")
    export, model = make_export(root)
    with Pool(["t1"], worker_env={"MOREGPU_MODEL_ROOTS": root}) as pool:   # the export dir is read-only input
        glog = os.path.join(pool.root, "g1.log")
        g = subprocess.Popen(["deno", "run", "--unstable-webgpu", "--allow-net", "--allow-env", "--allow-sys", "--allow-read",
                              "apps/worker/worker.ts", "--server", f"ws://127.0.0.1:{pool.port}/ws", "--token", pool.join, "--name", "g1"],
                             cwd=REPO, env=dict(os.environ, MOREGPU_INSECURE="1"), stdout=open(glog, "w"), stderr=subprocess.STDOUT)
        pool.procs.append(g)
        t0 = time.time()
        gw = None
        while time.time() - t0 < 120:
            ws = pool.api("/workers")
            gw = next((w for w in ws if w.get("id") == "g1"), None) if isinstance(ws, list) else None
            if gw:
                break
            time.sleep(0.5)
        ck(gw is not None and "vision" in (gw.get("caps") or []), f"Deno WebGPU worker joined with the 'vision' cap ({gw and gw.get('label')})")
        if not gw:
            print(tail(glog))
            ck.finish()

        r = pool.api("/vision/load", "POST", {"id": "seg", "export": export, "fleet": "all"})
        ok = r.get("ok") and set(r.get("workers", [])) == {"t1", "g1"} and r.get("webgpu") == ["g1"]
        ck(bool(ok), f"/vision/load fleet=all → holders {r.get('workers')} (webgpu {r.get('webgpu')}, lowered {r.get('lowered', {}).get('kind')})")
        if not ok:
            print(json.dumps(r, indent=1)[:2000]); print(tail(glog)); print(tail(pool.coord_log))
            ck.finish()
        ck(r["lowered"]["servable"] and r["lowered"]["parity"]["max_abs"] <= 1e-4,
           f"lowered on the torch worker, parity probe {r['lowered']['parity']['max_abs']:.2e} ≤ 1e-4")

        rng = np.random.default_rng(3)
        xs = [rng.standard_normal((1, 3, 32, 32)).astype("float32") for _ in range(8)]
        with torch.no_grad():
            ref = [model(torch.from_numpy(x)).numpy() for x in xs]

        for w in ("t1", "g1"):
            one = pool.api("/vision/infer", "POST", {"id": "seg", "shape": [1, 3, 32, 32], "data": b64(xs[0]), "worker": w})
            good = one.get("ok") and one.get("worker") == w and one.get("shape") == list(ref[0].shape)
            err = float(np.abs(unb64(one["data"], one["shape"]) - ref[0]).max()) if good else float("inf")
            ck(bool(good) and err <= 1e-4, f"/vision/infer on {w} ({one.get('kind')}) → {one.get('shape')}, max|Δ| vs torch {err:.2e}")
            ck(one.get("pred_sha256") == P.pred_sha256(P.labels_from_logits(ref[0])),
               f"/vision/infer on {w}: pred_sha256 == local PyTorch labels ({str(one.get('pred_sha256'))[:12]}…)")

        res = pool.api("/vision/infer_batch", "POST", {"id": "seg", "inputs": [{"shape": [1, 3, 32, 32], "data": b64(x)} for x in xs],
                                                       "check_parity": True})
        if not res.get("ok"):
            print(json.dumps(res, indent=1)[:2000]); print(tail(glog)); print(tail(pool.coord_log))
        outs = res.get("outputs") or []
        per = res.get("per_worker") or {}
        ck(len(outs) == 8 and all(o and o.get("shape") == list(ref[0].shape) for o in outs), f"infer_batch returned 8 outputs in order ({len(outs)})")
        ck(per.get("t1", 0) > 0 and per.get("g1", 0) > 0 and sum(per.values()) == 8, f"both workers served items: {per}")
        errs = [float(np.abs(unb64(o["data"], o["shape"]) - r_).max()) for o, r_ in zip(outs, ref)] if len(outs) == 8 else [float("inf")]
        ck(max(errs) <= 1e-4, f"every output == local PyTorch within 1e-4 (max {max(errs):.2e})")
        ck(len(outs) == 8 and all(o["pred_sha256"] == P.pred_sha256(P.labels_from_logits(r_)) for o, r_ in zip(outs, ref)),
           "infer_batch: every output's pred_sha256 (torch and WebGPU) == local PyTorch labels")
        par = res.get("parity") or {}
        ck(par.get("reference") == "t1" and par.get("max_abs", 1) <= 1e-4 and "g1" in (par.get("per_worker") or {}),
           f"check_parity: max|Δ| vs torch reference {par.get('max_abs')} (per worker {par.get('per_worker')})")
        served_g = [i for i, o in enumerate(outs) if o and o.get("worker") == "g1"]
        ck(len(served_g) > 0 and all(outs[i].get("kind") == "webgpu" for i in served_g), f"items {served_g} served on WebGPU")
        u = pool.api("/vision/unload", "POST", {"id": "seg"})
        ck(u.get("ok") is True, "/vision/unload drops the model from both kinds of worker")
    ck.finish()


if __name__ == "__main__":
    main()
