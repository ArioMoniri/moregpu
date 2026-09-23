"""JEPA self-supervised pretraining tasks (ADR-0109): ijepa_2d, jepa_2p5d (adjacent slices as channels), jepa_3d.

Context encoder + narrow predictor are trained; the target encoder is an EMA of the context encoder that is updated ONLY
in `after_outer_step`, from the freshly synced global, identically on every worker. The per-outer-step momentum is the
product of the per-step I-JEPA schedule over the steps that round covered, so N=1, H=1, η=1, μ=0 DiLoCo is exactly
per-step I-JEPA (tests/py/test_jepa_task.py). The target is never transmitted; its hash is reported for cross-checks.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os

import torch
import torch.nn.functional as F

from ... import paths
from ...models.vit import Predictor, VisionTransformer, vit_config
from .. import monitors as MON
from ..masking import MultiBlockMasker, gather_tokens
from ..synthetic import SyntheticVolumes
from ..task import StepReport, TaskContext, Timer, TrainTask


def ema_update(target: dict, online: dict, m: float) -> None:
    with torch.no_grad():
        for k, t in target.items():
            t.mul_(m).add_(online[k].to(t.dtype), alpha=1.0 - m)


class EmaSchedule:
    """Linear per-step momentum m0 → m1 over total_steps (I-JEPA); flat at m1 afterwards."""
    def __init__(self, m0: float, m1: float, total_steps: int):
        self.m0, self.m1, self.total = float(m0), float(m1), max(1, int(total_steps))

    def at(self, step: int) -> float:
        return self.m0 + (self.m1 - self.m0) * min(step / self.total, 1.0)

    def at_progress(self, p: float) -> float:
        return self.m0 + (self.m1 - self.m0) * min(max(p, 0.0), 1.0)

    def momentum_between(self, s0: int, s1: int) -> float:
        m = 1.0
        for s in range(s0, s1):
            m *= self.at(s)
        return m


class DataPlaneSource:
    """Adapter over moregpu_worker.data.DataPlane: manifest indices → batches (+ optional labels in ref.meta)."""
    def __init__(self, plane, spec: dict, kind: str = "2p5d"):
        self.plane = plane
        self.manifest = plane.open_manifest(spec["manifest"], spec.get("sha256"))
        self.spec = {"kind": kind, **spec.get("spec", {})}

    def __len__(self):
        return len(self.manifest)

    def batch(self, idx):
        return self.plane.load_batch(self.manifest, [int(i) % len(self.manifest) for i in idx], self.spec)

    def label(self, i):
        return int(self.manifest[int(i) % len(self.manifest)].meta.get("label", -1))


class _JepaBase(TrainTask):
    name = "jepa"
    kind = "2p5d"

    def init(self, cfg: dict, ctx: TaskContext) -> dict:
        self.setup(ctx)
        self.cfg = cfg
        self.keep_inner_state = bool(cfg.get("keep_inner_state", True))
        if cfg.get("synthetic"):
            syn = {**cfg["synthetic"]}; syn.setdefault("kind", self.kind)
            self.data = SyntheticVolumes(**syn)
            size, chans = tuple(syn["size"]), int(syn.get("channels", 1 if self.kind == "3d" else 3))
        elif cfg.get("data"):
            if ctx.data is None:
                raise RuntimeError("this worker has no data plane configured (MOREGPU_DATA_ROOTS)")
            self.data = DataPlaneSource(ctx.data, cfg["data"], self.kind)
            sp = cfg["data"].get("spec", {})
            size, chans = tuple(sp["size"]), int(sp.get("channels", 1))
        else:
            raise ValueError("JEPA task needs cfg.synthetic or cfg.data")
        in_chans = 1 if self.kind == "3d" else chans
        vc = vit_config(cfg.get("model", "tiny"), img_size=size, patch=cfg.get("patch", 16), in_chans=in_chans)
        self.vit_cfg = {**vc, "patch": list(vc["patch"]) if isinstance(vc["patch"], (list, tuple)) else vc["patch"],
                        "img_size": list(size)}
        gc = bool(cfg.get("grad_checkpointing", False))
        torch.manual_seed(ctx.seed)
        self.encoder = VisionTransformer(**vc, grad_checkpointing=gc).to(ctx.device)
        self.target = VisionTransformer(**vc).to(ctx.device)
        self.target.load_state_dict(self.encoder.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(self.encoder.embed_dim, int(cfg.get("pred_dim", 192 if vc["embed_dim"] >= 192 else 16)),
                                   int(cfg.get("pred_depth", 6)), int(cfg.get("pred_heads", 6 if vc["embed_dim"] >= 192 else 2)),
                                   self.encoder.grid, grad_checkpointing=gc).to(ctx.device)
        self.masker = MultiBlockMasker(self.encoder.grid, n_targets=int(cfg.get("n_targets", 4)),
                                       target_scale=tuple(cfg.get("target_scale", (0.15, 0.2))),
                                       target_aspect=tuple(cfg.get("target_aspect", (0.75, 1.5))),
                                       context_scale=tuple(cfg.get("context_scale", (0.85, 1.0))),
                                       min_keep=int(cfg.get("min_keep", 4)))
        ema = cfg.get("ema", [0.996, 1.0])
        self.ema = EmaSchedule(ema[0], ema[1], int(cfg.get("total_steps", 10_000)))
        self._ema_step = 0
        self.loss_kind = cfg.get("loss", "smooth_l1")
        self.clip = cfg.get("clip_grad")
        self.wd = float(cfg.get("weight_decay", 0.04))
        self.opt_kind = cfg.get("optimizer", "adamw")
        self.crop = cfg.get("crop_scale")               # e.g. [0.3, 1.0] random-resized-crop; None = off
        self.flip = bool(cfg.get("hflip", False))
        self.monitor_every = int(cfg.get("monitor_every", 1))
        self.std_min, self.rank_min = float(cfg.get("collapse_std_min", 1e-3)), float(cfg.get("collapse_rank_min", 2.0))
        self.probe_idx = list(range(min(int(cfg.get("probe_batch", 256)), len(self.data))))
        n = sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.predictor.parameters())
        self.info = {"encoder": self.vit_cfg, "params": n, "tokens": self.encoder.num_patches}
        return {"ok": True, "params": n, "tokens": self.encoder.num_patches}

    # --------------------------------------------------------------- training
    def _trainable(self):
        return [p for p in list(self.encoder.parameters()) + list(self.predictor.parameters()) if p.requires_grad]

    def _gen(self, refs) -> torch.Generator:
        """Mask/augmentation RNG derived from (seed, the batch's global sample indices) only — never from a worker-local
        step counter — so the same batch gets the same masks in every arm and on every worker."""
        h = hashlib.blake2b(digest_size=8)
        h.update(str(self.ctx.seed).encode()); h.update(",".join(str(int(r)) for r in refs).encode())
        return torch.Generator().manual_seed(int.from_bytes(h.digest(), "little") & 0x7FFFFFFFFFFFFFFF)

    def _augment(self, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        if not self.crop and not self.flip:
            return x
        out = []
        spatial = x.shape[2:]
        for i in range(x.shape[0]):
            xi = x[i:i + 1]
            if self.crop:
                s = self.crop[0] + float(torch.rand(1, generator=g)) * (self.crop[1] - self.crop[0])
                sz = [max(1, int(round(d * s ** (1 / len(spatial))))) for d in spatial]
                st = [int(torch.randint(0, d - z + 1, (1,), generator=g)) for d, z in zip(spatial, sz)]
                xi = xi[(slice(None), slice(None)) + tuple(slice(a, a + z) for a, z in zip(st, sz))]
                mode = "trilinear" if len(spatial) == 3 else "bilinear"
                xi = F.interpolate(xi, size=tuple(spatial), mode=mode, align_corners=False)
            if self.flip and float(torch.rand(1, generator=g)) < 0.5:
                xi = torch.flip(xi, dims=[-1])
            out.append(xi)
        return torch.cat(out)

    def _loss(self, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        ctx_idx, tgt_idx = self.masker(x.shape[0], g)
        ctx_idx, tgt_idx = ctx_idx.to(x.device), [t.to(x.device) for t in tgt_idx]
        with torch.no_grad():
            h = self.target(x)
            h = F.layer_norm(h, (h.shape[-1],))
            h = torch.cat([gather_tokens(h, t) for t in tgt_idx], dim=0)
        z = self.encoder(x, keep=ctx_idx)
        p = self.predictor(z, ctx_idx, tgt_idx)
        if self.loss_kind == "l2":
            return F.mse_loss(p.float(), h.float())
        return F.smooth_l1_loss(p.float(), h.float())

    def _to_input(self, b: torch.Tensor) -> torch.Tensor:
        if self.kind == "3d" and b.dim() == 4:
            b = b[:, None]
        return b.to(self.ctx.device, non_blocking=True)

    def inner_steps(self, batch_refs: list, steps: int, lr: float) -> StepReport:
        if not batch_refs:
            raise ValueError("no samples for this worker")
        opt = self.inner_optimizer(self._trainable(), lr, self.opt_kind, self.wd)
        self.encoder.train(); self.predictor.train()
        tm, losses = Timer(), []
        per = max(1, len(batch_refs) // steps)
        for s in range(steps):
            refs = batch_refs[s * per:(s + 1) * per] if s < steps - 1 else batch_refs[s * per:]
            refs = refs or batch_refs[-per:]
            with tm.span("data_s"):
                g = self._gen(refs)
                x = self._augment(self._to_input(self.data.batch(refs)), g)
            with tm.span("compute_s"):
                opt.zero_grad(set_to_none=True)
                with self.amp.autocast():
                    loss = self._loss(x, g)
                self.amp.backward_step(loss, opt, clip=self.clip, params=self._trainable())
                if self.ctx.device.startswith("cuda"):
                    torch.cuda.synchronize()
            losses.append(float(loss.detach()))
            self.step += 1
        return StepReport(losses, len(batch_refs), tm.t, {"amp": self.amp.mode, **self.amp.describe()})

    # --------------------------------------------------------------- sync + EMA
    def state_for_sync(self):
        out = {}
        for pre, mod in (("encoder.", self.encoder), ("predictor.", self.predictor)):
            for k, p in mod.named_parameters():
                if p.requires_grad:
                    out[pre + k] = p.detach().clone()
        return out

    def load_sync_state(self, tensors):
        mods = {"encoder.": self.encoder, "predictor.": self.predictor}
        with torch.no_grad():
            for pre, mod in mods.items():
                for k, p in mod.named_parameters():
                    if p.requires_grad and pre + k in tensors:
                        p.copy_(tensors[pre + k].reshape(p.shape).to(p.device, p.dtype))

    def target_checksum(self) -> list[float]:
        """[sum, sum of squares] of the target weights in float64 — lets the coordinator accept ulp-level differences
        between devices (CPU/CUDA/MPS) that change the exact hash."""
        s1 = s2 = 0.0
        for v in self.target.state_dict().values():
            d = v.detach().double()
            s1 += float(d.sum()); s2 += float((d * d).sum())
        return [s1, s2]

    def target_hash(self) -> str:
        h = hashlib.sha256()
        for k, v in sorted(self.target.state_dict().items()):
            h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()

    def after_outer_step(self, round: int, info: dict | None = None) -> dict:
        """EMA target update from the freshly synced global. With coordinator `info` = {progress, h} the momentum is
        at_progress(progress)^h — identical on every worker regardless of local step counts (proportional allocation,
        partial failures). Without info (legacy / ad-hoc use) it falls back to this worker's own step counter."""
        if info and info.get("progress") is not None:
            m = self.ema.at_progress(float(info["progress"])) ** float(info.get("h", 1.0))
            self._global_steps = getattr(self, "_global_steps", 0.0) + float(info.get("h", 1.0))
        else:
            m = self.ema.momentum_between(self._ema_step, self.step)
        self._ema_step = self.step
        ema_update(dict(self.target.named_parameters()), dict(self.encoder.named_parameters()), m)
        out = {"target_sha256": self.target_hash(), "target_checksum": self.target_checksum(), "ema_momentum": m}
        if self.monitor_every > 0 and round % self.monitor_every == 0:
            mon = MON.embedding_monitors(self._features(self.probe_idx))
            out["monitors"] = mon
            out["alarms"] = MON.alarms(mon, self.std_min, self.rank_min)
        return out

    def extra_state(self):
        st = super().extra_state()
        st.update({"target." + k: v.detach().clone() for k, v in self.target.state_dict().items()})
        st["_meta.ema_step"] = torch.tensor([float(self._ema_step)])
        st["_meta.global_steps"] = torch.tensor([float(getattr(self, "_global_steps", 0.0))])
        return st

    def load_extra_state(self, tensors):
        with torch.no_grad():
            for k, v in self.target.state_dict().items():
                if "target." + k in tensors:
                    v.copy_(tensors["target." + k].reshape(v.shape).to(v.device, v.dtype))
        super().load_extra_state(tensors)
        if "_meta.ema_step" in tensors:
            self._ema_step = int(tensors["_meta.ema_step"].item())
            self._global_steps = float(tensors["_meta.global_steps"].item())

    # --------------------------------------------------------------- evaluation
    @torch.no_grad()
    def _features(self, refs, bs: int = 32) -> torch.Tensor:
        self.encoder.eval()
        out = []
        for i in range(0, len(refs), bs):
            out.append(self.encoder(self._to_input(self.data.batch(refs[i:i + bs]))).mean(dim=1).float().cpu())
        self.encoder.train()
        return torch.cat(out) if out else torch.zeros(0, self.encoder.embed_dim)

    def evaluate(self, refs: list, kind: str) -> dict:
        refs = [int(r) for r in refs]
        if kind == "loss":
            with torch.no_grad():
                x = self._to_input(self.data.batch(refs))
                return {"loss": float(self._loss(x, torch.Generator().manual_seed(12345)))}
        if kind == "features":
            f = self._features(refs)
            return {"shape": list(f.shape), "data": base64.b64encode(f.numpy().astype("<f4").tobytes()).decode()}
        if kind == "monitors":
            return MON.embedding_monitors(self._features(refs))
        if kind in ("knn", "linear_probe"):
            f = F.normalize(self._features(refs), dim=1)
            y = torch.tensor([self.data.label(r) for r in refs])
            if kind == "knn":
                return {"knn_acc": knn_accuracy(f, y, k=int(self.cfg.get("knn_k", 5))), "n": len(refs)}
            return {"linear_probe_acc": linear_probe_accuracy(f, y), "n": len(refs)}
        raise ValueError(f"unknown evaluation kind {kind!r} (loss|features|monitors|knn|linear_probe)")

    # --------------------------------------------------------------- export
    def export(self, fmt: str, path: str, which: str = "target") -> dict:
        """which='target' (default; the EMA encoder, as I-JEPA evaluates downstream) or 'context'."""
        if which not in ("target", "context"):
            raise ValueError("which must be 'target' or 'context'")
        path = paths.export_dir(path)                     # confined to MOREGPU_OUTPUT_DIR
        enc = (self.target if which == "target" else self.encoder).eval()
        example = self._to_input(self.data.batch([0, 1]))
        if fmt == "safetensors":
            from safetensors.torch import save_file
            w = os.path.join(path, "encoder.safetensors")
            # the ViT config also rides in the safetensors metadata, so the file alone (e.g. pushed:// to another
            # worker) is enough to initialise a segment/classify encoder
            save_file({k: v.detach().contiguous().cpu() for k, v in enc.state_dict().items()}, w,
                      metadata={"moregpu.encoder_config": json.dumps(self.vit_cfg)})
            c = os.path.join(path, "encoder_config.json")
            json.dump({k: v for k, v in self.vit_cfg.items()}, open(c, "w"))
            out = {"format": fmt, "weights": w, "config": c, "sha256": _sha(w), "which": which}
        elif fmt == "torch_export":
            p = os.path.join(path, "encoder.pt2")
            prog = torch.export.export(enc, (example,))
            torch.export.save(prog, p)
            out = {"format": fmt, "path": p, "sha256": _sha(p), "which": which}
        elif fmt == "onnx":
            p = os.path.join(path, "encoder.onnx")
            torch.onnx.export(enc, (example,), p, input_names=["x"], output_names=["tokens"], opset_version=17,
                              dynamic_axes={"x": {0: "batch"}, "tokens": {0: "batch"}}, dynamo=False)
            parity = None
            try:
                import onnxruntime as ort
                s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
                with torch.no_grad():
                    ref = enc(example).cpu().numpy()
                got = s.run(None, {"x": example.cpu().numpy()})[0]
                parity = float(abs(got - ref).max())
            except ImportError:
                pass
            out = {"format": fmt, "path": p, "sha256": _sha(p), "parity_max_abs": parity, "which": which}
        else:
            raise ValueError(f"unknown export format {fmt!r} (safetensors|torch_export|onnx)")
        self.encoder.train(); self.target.eval()
        return out

    def describe(self) -> dict:
        return {**super().describe(), "kind": self.kind, "encoder": self.vit_cfg, "ema_step": self._ema_step}


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def knn_accuracy(f: torch.Tensor, y: torch.Tensor, k: int = 5) -> float:
    """Leave-one-out cosine k-NN majority vote (f rows L2-normalised)."""
    n = f.shape[0]
    if n < 2:
        return float("nan")
    k = max(1, min(k, n - 1))
    sim = f @ f.T
    sim.fill_diagonal_(-float("inf"))
    nn_idx = sim.topk(k, dim=1).indices
    votes = y[nn_idx]
    pred = torch.mode(votes, dim=1).values
    return float((pred == y).float().mean())


def linear_probe_accuracy(f: torch.Tensor, y: torch.Tensor, epochs: int = 200) -> float:
    """Deterministic 2-fold (even/odd index) multinomial logistic regression on standardised features."""
    n = f.shape[0]
    if n < 4:
        return float("nan")
    classes = int(y.max()) + 1
    accs = []
    for fold in (0, 1):
        tr = torch.arange(n) % 2 != fold
        te = ~tr
        mu, sd = f[tr].mean(0), f[tr].std(0).clamp_min(1e-6)
        xtr, xte = (f[tr] - mu) / sd, (f[te] - mu) / sd
        torch.manual_seed(0)
        lin = torch.nn.Linear(f.shape[1], classes)
        opt = torch.optim.LBFGS(lin.parameters(), max_iter=epochs, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(lin(xtr), y[tr]) + 1e-3 * lin.weight.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            accs.append(float((lin(xte).argmax(1) == y[te]).float().mean()))
    return sum(accs) / len(accs)


class IJepa2DTask(_JepaBase):
    name, kind = "ijepa_2d", "2d"


class Jepa2p5DTask(_JepaBase):
    name, kind = "jepa_2p5d", "2p5d"


class Jepa3DTask(_JepaBase):
    name, kind = "jepa_3d", "3d"
