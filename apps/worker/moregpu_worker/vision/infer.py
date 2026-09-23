"""Worker-side vision inference (ADR-0112): load exported MoreGPU models (segment/classify exports or JEPA encoder
exports), run a tensor forward, or predict a whole volume from a data-plane ref with sliding window + flip TTA and write
the label map to MOREGPU_OUTPUT_DIR. Adapter-loaded third-party models (moregpu_worker.vision.adapters) register into the
same store via `put`."""
from __future__ import annotations

import base64
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import losses as L
from .sliding import predict as sw_predict
from .models import load_exported

OPS = frozenset({"vision_infer_load", "vision_infer", "vision_predict", "vision_infer_describe", "vision_infer_unload",
                 "vision_infer_list"})


class InferenceStore:
    def __init__(self, device: str, plane=None, out_root: str | None = None):
        self.device, self.plane = device, plane
        self.out_root = os.path.realpath(out_root or os.environ.get("MOREGPU_OUTPUT_DIR", os.path.join(os.getcwd(), "moregpu-out")))
        self.models: dict[str, dict] = {}

    def handle(self, op: str, p: dict) -> dict:
        if op not in OPS:
            raise ValueError(f"unknown vision op {op!r}")
        return getattr(self, "_" + op[len("vision_"):])(p)

    # -------------------------------------------------------------- lifecycle
    def put(self, mid: str, model: torch.nn.Module, meta: dict) -> dict:
        self.models[mid] = {"model": model.to(self.device).eval(), "meta": meta}
        return {"ok": True, "id": mid, **meta}

    def _infer_load(self, p):
        path = p["export"]
        if os.path.exists(os.path.join(path, "model_config.json")):
            meta = json.load(open(os.path.join(path, "model_config.json")))
            return self.put(p["id"], load_exported(path), {"task": meta["task"], "num_classes": meta["num_classes"],
                                                          "kind": meta.get("kind", "2p5d"), "encoder": meta["encoder"]})
        if os.path.exists(os.path.join(path, "encoder_config.json")):
            from safetensors.torch import load_file
            from ..models.vit import VisionTransformer
            cfg = json.load(open(os.path.join(path, "encoder_config.json")))
            m = VisionTransformer(**{**cfg, "img_size": tuple(cfg["img_size"])})
            m.load_state_dict(load_file(os.path.join(path, "encoder.safetensors")), strict=True)
            kind = "3d" if len(cfg["img_size"]) == 3 else "2p5d"
            return self.put(p["id"], m, {"task": "encoder", "num_classes": None, "kind": kind, "encoder": cfg})
        raise FileNotFoundError(f"no MoreGPU export (model_config.json / encoder_config.json) at {path}")

    def _get(self, mid):
        if mid not in self.models:
            raise KeyError(f"no vision model {mid!r} loaded on this worker")
        return self.models[mid]

    def _infer_describe(self, p):
        return {"ok": True, "id": p["id"], **self._get(p["id"])["meta"]}

    def _infer_list(self, p):
        return {"ok": True, "models": {k: v["meta"] for k, v in self.models.items()}}

    def _infer_unload(self, p):
        self.models.pop(p["id"], None)
        return {"ok": True}

    # -------------------------------------------------------------- forward
    @torch.no_grad()
    def _infer(self, p):
        m = self._get(p["id"])
        x = torch.from_numpy(np.frombuffer(base64.b64decode(p["data"]), dtype="<f4").copy()).reshape(p["shape"]).to(self.device)
        y = m["model"](x)
        if m["meta"]["task"] == "encoder" and p.get("pool") == "mean":
            y = y.mean(dim=1)
        y = y.float().cpu().numpy().astype("<f4")
        return {"ok": True, "shape": list(y.shape), "data": base64.b64encode(y.tobytes()).decode()}

    def _out_path(self, name: str) -> str:
        path = os.path.realpath(os.path.join(self.out_root, name))
        if not (path == self.out_root or path.startswith(self.out_root + os.sep)):
            raise PermissionError(f"output {name!r} escapes MOREGPU_OUTPUT_DIR")
        os.makedirs(os.path.dirname(path) or self.out_root, exist_ok=True)
        return path

    @torch.no_grad()
    def _predict(self, p):
        from ..data.refs import Ref
        t0 = time.perf_counter()
        m = self._get(p["id"])
        if m["meta"]["task"] != "segment":
            raise ValueError("vision_predict needs a segmentation model")
        out_path = self._out_path(p["out"] + (".npy" if not p["out"].endswith(".npy") else ""))
        vol = np.asarray(self.plane.read(Ref.from_json(p["ref"])), dtype=np.float32)
        norm = p.get("normalize")
        if norm:
            vol = (vol - float(norm.get("mean", 0.0))) / float(norm.get("std", 1.0))
        t1 = time.perf_counter()
        model, meta = m["model"], m["meta"]
        size = tuple(meta["encoder"]["img_size"])
        tta, sw_batch = p.get("tta", "none"), int(p.get("sw_batch", 8))
        if meta["kind"] == "3d":
            x = torch.from_numpy(vol)[None, None].to(self.device)
            logits = sw_predict(x, model, roi=size, overlap=float(p.get("overlap", 0.5)), sw_batch=sw_batch,
                                mode=p.get("blend", "gaussian"), tta=tta)[0]
            pred = logits.argmax(0)
        else:
            c = int(meta["encoder"]["in_chans"]); Z, H, W = vol.shape
            v = torch.from_numpy(vol)
            half = c // 2
            preds = []
            for z0 in range(0, Z, sw_batch):
                zs = range(z0, min(Z, z0 + sw_batch))
                win = torch.stack([v[[min(Z - 1, max(0, z + k)) for k in range(-half, c - half)]] for z in zs])
                win = F.interpolate(win, size=size, mode="bilinear", align_corners=False).to(self.device)
                lg = sw_predict(win, model, roi=None, tta=tta)
                lg = F.interpolate(lg.float(), size=(H, W), mode="bilinear", align_corners=False)
                preds.append(lg.argmax(1).cpu())
            pred = torch.cat(preds)
        t2 = time.perf_counter()
        pred_np = pred.cpu().numpy().astype(np.uint8)
        np.save(out_path, pred_np)
        res = {"ok": True, "path": out_path, "shape": list(pred_np.shape),
               "timings": {"data_s": t1 - t0, "compute_s": t2 - t1, "write_s": time.perf_counter() - t2}}
        if p.get("mask"):
            gt = torch.as_tensor(np.asarray(self.plane.read(Ref.from_json(p["mask"]))).astype(np.int64))
            d = L.dice_per_class(torch.from_numpy(pred_np.astype(np.int64))[None, None], gt[None, None], meta["num_classes"])[0]
            res["dice"] = {str(i + 1): (None if torch.isnan(d[i]) else float(d[i])) for i in range(d.shape[0])}
        return res
