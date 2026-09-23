"""Vision fine-tuning tasks (ADR-0112): `segment` (2D / 2.5D / 3D, Dice+CE) and `classify` (CE) on a ViT encoder that
is random, or loaded from a JEPA export (a confined export dir, or a `pushed://<id>` safetensors blob — vision/weights.py). Modes: full fine-tune, frozen encoder + head, or LoRA on attention linears.
DiLoCo-compatible: state_for_sync returns exactly the trainable tensors."""
from __future__ import annotations

import hashlib
import json
import os

import torch
import torch.nn.functional as F

from ... import paths
from ...models.vit import vit_config
from ...vision import losses as L
from ...vision import weights as W
from ...vision.models import build
from ..synthetic import SyntheticSeg, SyntheticVolumes
from ..task import StepReport, TaskContext, Timer, TrainTask
from .llm_lora import LoRAWrap


def resize_mask(mask: torch.Tensor, size) -> torch.Tensor:
    """Label-map resize geometrically aligned with the image path (bilinear, align_corners=False): 'nearest-exact'
    samples pixel centres; plain 'nearest' shifts labels by up to half an input pixel."""
    return F.interpolate(mask[None, None].float(), size=tuple(size), mode="nearest-exact")[0, 0].long()


def _sync(dev: str):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


class _SegDataPlane:
    """Manifest refs carry the image window; ref.meta['mask'] = {uri, slice?} gives the label map
    (2p5d: the centre slice of the window; 3d: the full volume)."""
    def __init__(self, plane, spec: dict, kind: str):
        from ...data.refs import Ref
        self.Ref, self.plane, self.kind = Ref, plane, kind
        self.manifest = plane.open_manifest(spec["manifest"], spec.get("sha256"))
        self.spec = {"kind": kind, **spec.get("spec", {})}

    def __len__(self):
        return len(self.manifest)

    def batch(self, idx):
        return self.plane.load_batch(self.manifest, [int(i) % len(self.manifest) for i in idx], self.spec)

    def label(self, i):
        return int(self.manifest[int(i) % len(self.manifest)].meta.get("label", -1))

    def masks(self, idx):
        out = []
        size = self.spec["size"]
        for i in idx:
            ref = self.manifest[int(i) % len(self.manifest)]
            m = ref.meta.get("mask")
            if m is None:
                raise ValueError(f"manifest entry {i} has no meta.mask")
            arr = torch.as_tensor(self.plane.read(self.Ref.from_json(m)).astype("int64"))
            if self.kind == "2p5d" and arr.dim() == 3:
                arr = arr[arr.shape[0] // 2]
            arr = resize_mask(arr, size)
            out.append(arr)
        return torch.stack(out)


class _VisionBase(TrainTask):
    name = "vision"
    task = "segment"

    def init(self, cfg: dict, ctx: TaskContext) -> dict:
        self.setup(ctx)
        self.cfg = cfg
        self.kind = cfg.get("kind", "2p5d")
        self.keep_inner_state = bool(cfg.get("keep_inner_state", True))
        self.num_classes = int(cfg["num_classes"])
        if cfg.get("synthetic"):
            syn = {"kind": self.kind, **cfg["synthetic"]}
            self.data = SyntheticSeg(**syn) if self.task == "segment" else SyntheticVolumes(**syn)
            size, chans = tuple(syn["size"]), int(syn.get("channels", 1))
        elif cfg.get("data"):
            if ctx.data is None:
                raise RuntimeError("this worker has no data plane configured (MOREGPU_DATA_ROOTS)")
            self.data = _SegDataPlane(ctx.data, cfg["data"], self.kind)
            sp = cfg["data"]["spec"]; size, chans = tuple(sp["size"]), int(sp.get("channels", 1))
        else:
            raise ValueError(f"{self.task} needs cfg.synthetic or cfg.data")
        in_chans = 1 if self.kind == "3d" else chans
        enc = cfg.get("encoder", {"init": "random"})
        init_sd, self.encoder_info = None, {"source": "random"}
        if enc.get("init") == "export":
            # an export dir (confined to MOREGPU_OUTPUT_DIR ∪ MOREGPU_MODEL_ROOTS) or pushed://<id> (BlobStore); either
            # way size-capped, sha256-checked and safetensors-only before any tensor is read (vision/weights.py)
            init_sd, vc, self.encoder_info = W.load_encoder(enc, W.blobs_of(ctx.data))
        else:
            vc = vit_config(enc.get("model", "tiny"), img_size=size, patch=enc.get("patch", 16), in_chans=in_chans)
            vc = {**vc, "img_size": list(size), "patch": list(vc["patch"]) if isinstance(vc["patch"], (list, tuple)) else vc["patch"]}
        self.vit_cfg = vc
        self.dec_ch = tuple(cfg.get("decoder", {}).get("channels", (64, 32, 16)))
        torch.manual_seed(ctx.seed)
        self.model = build(self.task, vc, self.num_classes, self.dec_ch).to(ctx.device)
        self.encoder = self.model.encoder
        if init_sd is not None:
            self.encoder.load_state_dict(init_sd, strict=True)
        self.mode = cfg.get("mode", "full")
        if self.mode == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        elif self.mode == "lora":
            r, alpha = int(cfg.get("lora_rank", 8)), float(cfg.get("lora_alpha", 16))
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for blk in self.encoder.blocks:
                for name in ("qkv", "proj"):
                    lin = getattr(blk.attn, name)
                    setattr(blk.attn, name, LoRAWrap(lin, lin.in_features, lin.out_features, r, alpha).to(ctx.device))
        elif self.mode != "full":
            raise ValueError(f"mode must be full | frozen | lora, got {self.mode!r}")
        self.encoder.pos_embed.requires_grad_(False)       # the fixed sin-cos pos-embed never trains
        self.opt_kind = cfg.get("optimizer", "adamw")
        self.wd = float(cfg.get("weight_decay", 1e-4))
        self.clip = cfg.get("clip_grad")
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"ok": True, "trainable_params": n, "mode": self.mode, "encoder": self.encoder_info}

    def _in(self, b):
        if self.kind == "3d" and b.dim() == 4:
            b = b[:, None]
        return b.to(self.ctx.device, non_blocking=True)

    def _trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _loss(self, x, refs):
        raise NotImplementedError

    def inner_steps(self, batch_refs: list, steps: int, lr: float) -> StepReport:
        if not batch_refs:
            raise ValueError("no samples for this worker")
        opt = self.inner_optimizer(self._trainable(), lr, self.opt_kind, self.wd)
        self.model.train()
        tm, losses = Timer(), []
        per = max(1, len(batch_refs) // steps)
        for s in range(steps):
            refs = batch_refs[s * per:(s + 1) * per] if s < steps - 1 else batch_refs[s * per:]
            refs = refs or batch_refs[-per:]
            with tm.span("data_s"):
                x, y = self._in(self.data.batch(refs)), self._target(refs)
            with tm.span("compute_s"):
                opt.zero_grad(set_to_none=True)
                with self.amp.autocast():
                    out = self.model(x)
                loss = self._criterion(out, y)
                self.amp.backward_step(loss, opt, clip=self.clip, params=self._trainable())
                _sync(self.ctx.device)                # GPU time belongs to compute, not to whatever reads it next
            losses.append(float(loss.detach()))
            self.step += 1
        return StepReport(losses, len(batch_refs), tm.t, {"amp": self.amp.mode})

    def state_for_sync(self):
        return {k: p.detach().clone() for k, p in self.model.named_parameters() if p.requires_grad}

    def load_sync_state(self, tensors):
        with torch.no_grad():
            for k, p in self.model.named_parameters():
                if p.requires_grad and k in tensors:
                    p.copy_(tensors[k].reshape(p.shape).to(p.device, p.dtype))

    def _merged_state(self):
        """State dict with LoRA merged into the base linears (so exports load into the plain architecture)."""
        sd = {}
        for k, v in self.model.state_dict().items():
            if ".base." in k or k.endswith((".A", ".B")):
                continue
            sd[k] = v
        for name, mod in self.model.named_modules():
            if isinstance(mod, LoRAWrap):
                w = mod.base.weight + (mod.B @ mod.A) * mod.scale
                sd[f"{name}.weight"], sd[f"{name}.bias"] = w.detach(), mod.base.bias.detach()
        return {k: v.detach().contiguous().cpu() for k, v in sd.items()}

    def export(self, fmt: str, path: str) -> dict:
        from .jepa import _sha
        from ...vision.models import load_exported
        path = paths.export_dir(path)                     # confined to MOREGPU_OUTPUT_DIR
        x = self._in(self.data.batch([0, 1]))
        if fmt == "safetensors":
            from safetensors.torch import save_file
            w = os.path.join(path, "model.safetensors")
            save_file(self._merged_state(), w)
            c = os.path.join(path, "model_config.json")
            json.dump({"task": self.task, "encoder": self.vit_cfg, "num_classes": self.num_classes,
                       "decoder_channels": list(self.dec_ch), "kind": self.kind}, open(c, "w"))
            return {"task": self.task, "format": fmt, "weights": w, "config": c, "sha256": _sha(w)}
        tmp = os.path.join(path, "_st")
        self.export("safetensors", tmp)
        m = load_exported(tmp).to(self.ctx.device)
        if fmt == "torch_export":
            p = os.path.join(path, "model.pt2")
            torch.export.save(torch.export.export(m, (x,)), p)
            return {"task": self.task, "format": fmt, "path": p, "sha256": _sha(p)}
        if fmt == "onnx":
            p = os.path.join(path, "model.onnx")
            torch.onnx.export(m, (x,), p, input_names=["x"], output_names=["y"], opset_version=17,
                              dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, dynamo=False)
            parity = None
            try:
                import onnxruntime as ort
                s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
                with torch.no_grad():
                    parity = float(abs(s.run(None, {"x": x.cpu().numpy()})[0] - m(x).cpu().numpy()).max())
            except ImportError:
                pass
            return {"task": self.task, "format": fmt, "path": p, "sha256": _sha(p), "parity_max_abs": parity}
        raise ValueError(f"unknown export format {fmt!r}")

    def describe(self) -> dict:
        return {**super().describe(), "kind": self.kind, "mode": self.mode, "num_classes": self.num_classes}


class SegmentTask(_VisionBase):
    name, task = "segment", "segment"

    def _target(self, refs):
        return self.data.masks(refs).to(self.ctx.device)

    def _criterion(self, out, y):
        return L.dice_ce(out.float(), y[:, None])

    @torch.no_grad()
    def evaluate(self, refs: list, kind: str) -> dict:
        refs = [int(r) for r in refs]
        if kind not in ("dice", "loss"):
            raise ValueError(f"segment evaluation kinds: dice | loss (got {kind!r})")
        self.model.eval()
        ds, losses = [], []
        for i in range(0, len(refs), 8):
            r = refs[i:i + 8]
            x, y = self._in(self.data.batch(r)), self._target(r)
            out = self.model(x)
            losses.append(float(self._criterion(out, y)))
            ds.append(L.dice_per_class(out.argmax(1, keepdim=True), y[:, None], self.num_classes))
        self.model.train()
        d = torch.cat(ds)
        per = {str(c + 1): L.nanmean(d[:, c]) for c in range(d.shape[1])}
        return {"dice_mean": L.nanmean(d), "dice_per_class": per, "loss": sum(losses) / len(losses), "n": len(refs)}


class ClassifyTask(_VisionBase):
    name, task = "classify", "classify"

    def _target(self, refs):
        return torch.tensor([self.data.label(r) for r in refs], device=self.ctx.device)

    def _criterion(self, out, y):
        return F.cross_entropy(out.float(), y)

    @torch.no_grad()
    def evaluate(self, refs: list, kind: str) -> dict:
        if kind not in ("accuracy", "loss"):
            raise ValueError(f"classify evaluation kinds: accuracy | loss (got {kind!r})")
        self.model.eval()
        x, y = self._in(self.data.batch([int(r) for r in refs])), self._target(refs)
        out = self.model(x)
        self.model.train()
        return {"accuracy": float((out.argmax(1) == y).float().mean()), "loss": float(self._criterion(out, y)), "n": len(refs)}
