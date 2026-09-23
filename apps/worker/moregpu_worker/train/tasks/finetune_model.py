"""finetune_model — fine-tune ANY published native vision model (loaded exactly as published through the model
adapters: torchvision / timm / MONAI / HF / allowlisted plugin, state_dict or safetensors) as a DiLoCo-compatible
TrainTask. Objectives: classify (CE), segment (Dice+CE, 2D/2.5D/3D), regress (MSE). Trainable subsets: all, head
(parameter-name prefixes), or LoRA on named Linear layers (+ head). Exports reload through the same adapters."""
from __future__ import annotations

import hashlib
import json
import os

import torch
import torch.nn.functional as F

from ...vision import adapters as A
from ...vision import losses as L
from ..synthetic import SyntheticSeg, SyntheticVolumes
from ..task import StepReport, TaskContext, Timer, TrainTask
from .llm_lora import LoRAWrap
from .vision import _SegDataPlane

OBJECTIVES = ("classify", "segment", "regress")


class FinetuneModelTask(TrainTask):
    name = "finetune_model"

    def init(self, cfg: dict, ctx: TaskContext) -> dict:
        self.setup(ctx)
        self.cfg = cfg
        self.objective = cfg.get("objective", "classify")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        self.keep_inner_state = bool(cfg.get("keep_inner_state", False))
        self.num_classes = int(cfg.get("num_classes", 2))
        self.label_map = {int(k): int(v) for k, v in (cfg.get("label_map") or {}).items()}
        syn = cfg.get("synthetic")
        if syn:
            self.kind = syn.get("kind", "2d")
            self.data = SyntheticSeg(**syn) if self.objective == "segment" else SyntheticVolumes(**syn)
        elif cfg.get("data"):
            if ctx.data is None:
                raise RuntimeError("this worker has no data plane configured (MOREGPU_DATA_ROOTS)")
            self.kind = cfg["data"].get("spec", {}).get("kind", "2d")
            self.data = _SegDataPlane(ctx.data, cfg["data"], self.kind)
        else:
            raise ValueError("finetune_model needs cfg.synthetic or cfg.data")
        self.spec = cfg["spec"]
        h = A.load(self.spec)
        self.model = A.train_handle(h).to(ctx.device)          # raises NotNative for torch.export/TorchScript/ONNX
        self.model.train()
        mode = cfg.get("trainable", "all")
        heads = tuple(cfg.get("head_prefixes") or ())
        if mode in ("head", "lora"):
            for n, p in self.model.named_parameters():
                p.requires_grad_(any(n.startswith(h_) for h_ in heads))
        if mode == "lora":
            r, alpha = int(cfg.get("lora_rank", 8)), float(cfg.get("lora_alpha", 16))
            targets = tuple(cfg.get("lora_targets") or ("qkv", "proj"))
            hits = [(n, m) for n, m in self.model.named_modules()
                    if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in targets and not any(n.startswith(h_) for h_ in heads)]
            for n, m in hits:
                parent = self.model.get_submodule(n.rsplit(".", 1)[0]) if "." in n else self.model
                setattr(parent, n.split(".")[-1], LoRAWrap(m, m.in_features, m.out_features, r, alpha).to(ctx.device))
        elif mode not in ("all", "head"):
            raise ValueError("trainable must be all | head | lora")
        self.mode = mode
        self.opt_kind, self.wd, self.clip = cfg.get("optimizer", "adamw"), float(cfg.get("weight_decay", 1e-4)), cfg.get("clip_grad")
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"ok": True, "trainable_params": n, "arch": self.spec.get("arch"), "mode": mode}

    # -------------------------------------------------------------- data
    def _x(self, refs):
        x = self.data.batch(refs)
        if self.kind == "3d" and x.dim() == 4:
            x = x[:, None]
        return x.to(self.ctx.device)

    def _y(self, refs):
        if self.objective == "segment":
            y = self.data.masks(refs).clone()
            for a, b in self.label_map.items():
                y[y == a] = b
            return y.to(self.ctx.device)
        return torch.tensor([self.data.label(r) for r in refs], device=self.ctx.device)

    def _crit(self, out, y):
        if self.objective == "segment":
            return L.dice_ce(out.float(), y[:, None])
        if self.objective == "classify":
            return F.cross_entropy(out.float(), y)
        return F.mse_loss(out.float().squeeze(-1), y.float())

    def _trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    # -------------------------------------------------------------- TrainTask
    def inner_steps(self, batch_refs, steps, lr):
        if not batch_refs:
            raise ValueError("no samples for this worker")
        opt = self.inner_optimizer(self._trainable(), lr, self.opt_kind, self.wd)
        self.model.train()
        tm, losses = Timer(), []
        per = max(1, len(batch_refs) // steps)
        for s in range(steps):
            refs = (batch_refs[s * per:(s + 1) * per] if s < steps - 1 else batch_refs[s * per:]) or batch_refs[-per:]
            with tm.span("data_s"):
                x, y = self._x(refs), self._y(refs)
            with tm.span("compute_s"):
                opt.zero_grad(set_to_none=True)
                with self.amp.autocast():
                    out = self.model(x)
                loss = self._crit(out, y)
                self.amp.backward_step(loss, opt, clip=self.clip, params=self._trainable())
            losses.append(float(loss.detach())); self.step += 1
        return StepReport(losses, len(batch_refs), tm.t, {"amp": self.amp.mode})

    def state_for_sync(self):
        return {k: p.detach().clone() for k, p in self.model.named_parameters() if p.requires_grad}

    def load_sync_state(self, tensors):
        with torch.no_grad():
            for k, p in self.model.named_parameters():
                if p.requires_grad and k in tensors:
                    p.copy_(tensors[k].reshape(p.shape).to(p.device, p.dtype))

    @torch.no_grad()
    def evaluate(self, refs, kind):
        refs = [int(r) for r in refs]
        self.model.eval()
        try:
            if kind == "loss":
                return {"loss": float(self._crit(self.model(self._x(refs)), self._y(refs)))}
            if kind == "accuracy" and self.objective == "classify":
                out = self.model(self._x(refs))
                return {"accuracy": float((out.argmax(1) == self._y(refs)).float().mean()), "n": len(refs)}
            if kind == "dice" and self.objective == "segment":
                ds = [L.dice_per_class(self.model(self._x(refs[i:i + 4])).argmax(1, keepdim=True), self._y(refs[i:i + 4])[:, None],
                                       self.num_classes) for i in range(0, len(refs), 4)]
                d = torch.cat(ds)
                return {"dice_mean": L.nanmean(d), "dice_per_class": {str(c + 1): L.nanmean(d[:, c]) for c in range(d.shape[1])}}
            raise ValueError(f"evaluation {kind!r} not available for objective {self.objective!r}")
        finally:
            self.model.train()

    def _merged_state(self):
        sd = {k: v for k, v in self.model.state_dict().items() if ".base." not in k and not k.endswith((".A", ".B"))}
        for name, mod in self.model.named_modules():
            if isinstance(mod, LoRAWrap):
                sd[f"{name}.weight"] = (mod.base.weight + (mod.B @ mod.A) * mod.scale).detach()
                if mod.base.bias is not None:
                    sd[f"{name}.bias"] = mod.base.bias.detach()
        return {k: v.detach().contiguous().cpu() for k, v in sd.items()}

    def export(self, fmt: str, path: str) -> dict:
        os.makedirs(path, exist_ok=True)
        if fmt != "safetensors":
            raise ValueError("finetune_model exports safetensors (+ a model spec); lower it with /vision/lower for ONNX/WGSL")
        from safetensors.torch import save_file
        w = os.path.join(path, "model.safetensors")
        save_file(self._merged_state(), w)
        sha = hashlib.sha256(open(w, "rb").read()).hexdigest()
        spec = {k: v for k, v in self.spec.items() if k not in ("source", "sha256", "format")}
        spec.update(format="safetensors", source="file://" + os.path.abspath(w), sha256=sha)
        json.dump(spec, open(os.path.join(path, "model_spec.json"), "w"), indent=2)
        return {"format": fmt, "weights": w, "sha256": sha, "spec": spec, "spec_path": os.path.join(path, "model_spec.json")}

    def describe(self):
        return {**super().describe(), "objective": self.objective, "mode": self.mode, "arch": self.spec.get("arch")}
