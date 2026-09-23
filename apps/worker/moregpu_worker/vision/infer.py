"""Worker-side vision inference (ADR-0112): load exported MoreGPU models (segment/classify exports or JEPA encoder
exports), run a tensor forward, or predict a whole volume from a data-plane ref with sliding window + flip TTA and write
the label map to MOREGPU_OUTPUT_DIR. Label maps are uint8, or uint16 when a label exceeds 255, and every written map is
reported with its ``pred_sha256`` (moregpu.pred/1, see pred_hash.py). Adapter-loaded third-party models (moregpu_worker.vision.adapters) register into the
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
from . import pred_hash as PH
from .sliding import predict as sw_predict, sliding_window_part, merge_parts, tta_flips
from .. import paths
from .models import load_exported

OPS = frozenset({"vision_infer_load", "vision_infer", "vision_predict", "vision_infer_describe", "vision_infer_unload",
                 "vision_infer_list", "vision_predict_part", "vision_merge_write"})


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _unb64(s: str, dtype, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=dtype).reshape(shape)


class _AdapterModule(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class InferenceStore:
    def __init__(self, device: str, plane=None, out_root: str | None = None):
        self.device, self.plane = device, plane
        self.out_root = os.path.realpath(out_root) if out_root else paths.output_root()
        self.models: dict[str, dict] = {}
        from . import ops as model_ops
        model_ops.register_linked_store(self)   # vision_unload / reset also drop our copies of adapter models

    def drop_linked(self, mid: str | None) -> None:
        """Drop model ``mid`` and every model wrapping adapter handle ``mid`` (``None``: drop everything)."""
        if mid is None:
            self.models.clear()
            return
        for k in [k for k, v in self.models.items() if k == mid or v["meta"].get("adapter") == mid]:
            self.models.pop(k, None)

    def handle(self, op: str, p: dict) -> dict:
        if op not in OPS:
            raise ValueError(f"unknown vision op {op!r}")
        return getattr(self, "_" + op[len("vision_"):])(p)

    # -------------------------------------------------------------- lifecycle
    def put(self, mid: str, model: torch.nn.Module, meta: dict) -> dict:
        self.models[mid] = {"model": model.to(self.device).eval(), "meta": meta}
        return {"ok": True, "id": mid, **meta}

    def _infer_load(self, p):
        if p.get("handle"):   # a published model already loaded through the adapters (vision_load {id, spec})
            from . import ops as model_ops, adapters as A
            h = model_ops.HANDLES[p["handle"]]
            io = h.spec.get("io", {}) if hasattr(h, "spec") else {}
            meta = {"task": p.get("task", "segment"), "num_classes": p.get("num_classes"), "kind": p.get("kind", "3d"),
                    "encoder": {"img_size": p.get("img_size") or [], "in_chans": p.get("in_chans", 1)}, "adapter": p["handle"], "io": io}
            wrapper = _AdapterModule(lambda x: torch.as_tensor(A.infer(h, x)))
            return self.put(p["id"], wrapper, meta)
        # export reads are confined to this store's output dir ∪ MOREGPU_OUTPUT_DIR ∪ MOREGPU_MODEL_ROOTS (relative →
        # this store's output dir)
        path = paths.confine(p["export"], [self.out_root, *paths.export_read_roots()])
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
        try:
            path = paths.confine(name, [self.out_root])
        except paths.ConfinementError:
            raise PermissionError(f"output {name!r} escapes MOREGPU_OUTPUT_DIR") from None
        os.makedirs(os.path.dirname(path) or self.out_root, exist_ok=True)
        return path

    # -------------------------------------------------------------- tile sharding (one volume across workers)
    def _load_volume(self, p):
        from ..data.refs import Ref
        vol = np.asarray(self.plane.read(Ref.from_json(p["ref"])), dtype=np.float32)
        norm = p.get("normalize")
        if norm:
            vol = (vol - float(norm.get("mean", 0.0))) / float(norm.get("std", 1.0))
        return vol

    @torch.no_grad()
    def _predict_part(self, p):
        """Part k of n of one volume. 3D: this worker's share of the sliding-window tiles → weighted-sum accumulators
        over the z-slab they touch (f32). 2.5D: a contiguous slice range → uint8 labels. Merged by vision_merge_write."""
        t0 = time.perf_counter()
        m = self._get(p["id"]); meta = m["meta"]
        k, n = (int(v) for v in p["part"])
        vol = self._load_volume(p)
        t1 = time.perf_counter()
        if meta["kind"] == "3d":
            roi = tuple(meta["encoder"]["img_size"])
            x = torch.from_numpy(vol)[None, None].to(self.device)
            flips = tta_flips(5, p.get("tta", "none"))
            fl_parts, n_units = [], 0
            for f in flips:                 # each flip is its own tiling in flipped coordinates; merged per flip later
                r = sliding_window_part(torch.flip(x, f) if f else x, roi, int(p.get("sw_batch", 4)), m["model"],
                                        float(p.get("overlap", 0.5)), p.get("blend", "gaussian"), part=(k, n))
                q = {"flip": f, "z0": r["z0"], "n_units": r["n_tiles"]}
                if r["sum"] is not None:
                    q.update(sum_shape=list(r["sum"].shape), sum=_b64(r["sum"].cpu().numpy().astype("<f4")),
                             cnt=_b64(r["cnt"].cpu().numpy().astype("<f4")))
                fl_parts.append(q); n_units += r["n_tiles"]
            out = {"ok": True, "kind": "3d", "k": k, "n": n, "n_units": n_units, "vol_shape": list(vol.shape), "roi": list(roi),
                   "flips": fl_parts}
        else:
            Z = vol.shape[0]; z0, z1 = (Z * k) // n, (Z * (k + 1)) // n
            labels = PH.canonical_labels(self._predict_2p5d(m, vol, range(z0, z1), p.get("tta", "none"), int(p.get("sw_batch", 8))))
            out = {"ok": True, "kind": "2p5d", "k": k, "n": n, "n_units": z1 - z0, "z0": z0, "vol_shape": list(vol.shape),
                   "labels_shape": list(labels.shape), "labels_dtype": PH.label_dtype(labels), "labels": _b64(labels)}
        out["timings"] = {"data_s": t1 - t0, "compute_s": time.perf_counter() - t1}
        return out

    def _merge_write(self, p):
        from ..data.refs import Ref
        m = self._get(p["id"]); parts = sorted(p["parts"], key=lambda q: q["k"])
        shape = tuple(parts[0]["vol_shape"])
        if parts[0]["kind"] == "3d":
            roi = tuple(parts[0]["roi"])
            n_fl = len(parts[0]["flips"])
            probs = None
            for fi in range(n_fl):
                f = parts[0]["flips"][fi]["flip"]
                ps = [{"z0": q["flips"][fi]["z0"], "n_tiles": q["flips"][fi]["n_units"],
                       "sum": torch.from_numpy(_unb64(q["flips"][fi]["sum"], "<f4", q["flips"][fi]["sum_shape"]).copy()) if q["flips"][fi].get("sum") else None,
                       "cnt": torch.from_numpy(_unb64(q["flips"][fi]["cnt"], "<f4", [1, 1] + q["flips"][fi]["sum_shape"][2:]).copy()) if q["flips"][fi].get("cnt") else None}
                      for q in parts]
                lg = merge_parts(ps, (1, 1) + shape, roi)
                pr = torch.softmax(lg, 1)
                pr = torch.flip(pr, f) if f else pr
                probs = pr if probs is None else probs + pr
            pred = PH.canonical_labels((probs / n_fl)[0].argmax(0))
        else:
            dts = {"uint8": np.uint8, "uint16": "<u2"}
            pred = PH.canonical_labels(np.concatenate([_unb64(q["labels"], dts[q.get("labels_dtype", "uint8")], q["labels_shape"])
                                                       for q in parts if q["n_units"]], axis=0))
        out_path = self._out_path(p["out"] + (".npy" if not p["out"].endswith(".npy") else ""))
        np.save(out_path, pred)
        res = {"ok": True, "path": out_path, "shape": list(pred.shape), "n_parts": len(parts),
               "pred_sha256": PH.pred_sha256(pred), "pred_dtype": PH.label_dtype(pred)}
        if p.get("mask"):
            gt = torch.as_tensor(np.asarray(self.plane.read(Ref.from_json(p["mask"]))).astype(np.int64))
            d = L.dice_per_class(torch.from_numpy(pred.astype(np.int64))[None, None], gt[None, None], m["meta"]["num_classes"])[0]
            res["dice"] = {str(i + 1): (None if torch.isnan(d[i]) else float(d[i])) for i in range(d.shape[0])}
        return res

    def _predict_2p5d(self, m, vol: np.ndarray, zs, tta: str, sw_batch: int) -> np.ndarray:
        meta, model = m["meta"], m["model"]
        size = tuple(meta["encoder"]["img_size"])
        c = int(meta["encoder"]["in_chans"]); Z, H, W = vol.shape; half = c // 2
        v = torch.from_numpy(vol); zs = list(zs); preds = []
        for i in range(0, len(zs), sw_batch):
            chunk = zs[i:i + sw_batch]
            win = torch.stack([v[[min(Z - 1, max(0, z + kk)) for kk in range(-half, c - half)]] for z in chunk])
            win = F.interpolate(win, size=size, mode="bilinear", align_corners=False).to(self.device)
            lg = sw_predict(win, model, roi=None, tta=tta, average="probs")
            lg = F.interpolate(lg.float(), size=(H, W), mode="bilinear", align_corners=False)
            preds.append(lg.argmax(1).cpu())
        return (torch.cat(preds) if preds else torch.zeros((0, H, W), dtype=torch.long)).numpy()

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
            logits = sw_predict(x, model, roi=size or None, overlap=float(p.get("overlap", 0.5)), sw_batch=sw_batch,
                                mode=p.get("blend", "gaussian"), tta=tta, average="probs")[0]
            pred = logits.argmax(0)
        else:
            pred = torch.from_numpy(self._predict_2p5d(m, vol, range(vol.shape[0]), tta, sw_batch))
        t2 = time.perf_counter()
        pred_np = PH.canonical_labels(pred)          # uint8, or uint16 above 255 classes (never wrapped)
        np.save(out_path, pred_np)
        res = {"ok": True, "path": out_path, "shape": list(pred_np.shape),
               "pred_sha256": PH.pred_sha256(pred_np), "pred_dtype": PH.label_dtype(pred_np),
               "timings": {"data_s": t1 - t0, "compute_s": t2 - t1, "write_s": time.perf_counter() - t2}}
        if p.get("mask"):
            gt = torch.as_tensor(np.asarray(self.plane.read(Ref.from_json(p["mask"]))).astype(np.int64))
            d = L.dice_per_class(torch.from_numpy(pred_np.astype(np.int64))[None, None], gt[None, None], meta["num_classes"])[0]
            res["dice"] = {str(i + 1): (None if torch.isnan(d[i]) else float(d[i])) for i in range(d.shape[0])}
        return res
