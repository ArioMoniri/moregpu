"""Common adapter interface (ADR-0113): load(spec, fetch) -> Handle; infer; train_handle; describe; unload."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..errors import IntegrityError, NotLoaded, NotNative
from ..fetch import sha256_file

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
Fetch = Callable[..., Path]


@dataclass
class Handle:
    spec: dict
    adapter: str
    sha256: str
    model: Any = None
    native: bool = False
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    info: dict = field(default_factory=dict)    # JSON-able facts reported by describe()
    extra: dict = field(default_factory=dict)   # non-JSON objects (ExportedProgram, artefact path, …)


def resolve_device(placement: dict | None) -> str:
    want = (placement or {}).get("device", "auto")
    if want != "auto":
        return want
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # pragma: no cover
        return "mps"
    return "cpu"


def as_tensor(out: Any) -> torch.Tensor:
    """Model outputs → one tensor: HF ModelOutput.logits, first element of tuples/lists/dicts, numpy arrays."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        return as_tensor(next(iter(out.values())))
    if isinstance(out, list | tuple):
        return as_tensor(out[0])
    return torch.as_tensor(np.asarray(out))


def to_tensor(batch) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(batch)) if isinstance(batch, np.ndarray) else batch


def fetch_verified(spec: dict, fetch: Fetch) -> tuple[Path, str]:
    """Fetch the artefact and verify its sha256 BEFORE anything opens it."""
    path = Path(fetch(spec["source"], spec.get("sha256")))
    actual = sha256_file(path)
    if spec.get("sha256") and actual != spec["sha256"]:
        raise IntegrityError(f"sha256 mismatch for {spec['source']}: the spec pins {spec['sha256']}, the artefact is "
                             f"{actual}; refusing to load")
    return path, actual


def module_sha256(module: torch.nn.Module) -> str:
    """Content hash of a module's class + weights (key, dtype, shape, bytes) — used as the lowering cache identity. The
    class is included so two weight-less modules (or two architectures with identical tensors) never collide."""
    h = hashlib.sha256()
    h.update(f"{type(module).__module__}.{type(module).__qualname__}|".encode())
    for k, v in sorted(module.state_dict().items()):
        t = v.detach().cpu().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run_inference(predict: Callable[[torch.Tensor], Any], x: torch.Tensor, inference: dict) -> torch.Tensor:
    """Apply the spec's inference mode (full | sliding_window) and TTA (none | flip) around a raw predictor."""
    def base(t):
        return as_tensor(predict(t))

    def flip_tta(t):
        preds = [base(t)] + [base(t.flip(d)).flip(d) for d in range(2, t.ndim)]
        return torch.stack(preds).mean(0)

    fn = flip_tta if inference.get("tta") == "flip" else base

    if inference.get("mode") == "sliding_window":
        from monai.inferers import sliding_window_inference
        sw = inference["sliding_window"]
        return sliding_window_inference(x, tuple(sw["roi"]), sw["sw_batch"], fn, overlap=sw["overlap"],
                                        mode=sw["blend"])
    return fn(x)


class Adapter:
    name = ""
    native = False

    def load(self, spec: dict, fetch: Fetch) -> Handle:  # pragma: no cover - interface
        raise NotImplementedError

    def forward(self, handle: Handle, x: torch.Tensor):  # pragma: no cover - interface
        raise NotImplementedError

    def _describe(self, handle: Handle) -> dict:
        return {}

    def _check(self, handle: Handle):
        if handle.model is None:
            raise NotLoaded(f"model {handle.sha256[:12]} was unloaded")

    def infer(self, handle: Handle, batch) -> torch.Tensor:
        self._check(handle)
        x = to_tensor(batch)
        with torch.inference_mode():
            return run_inference(lambda t: self.forward(handle, t), x, handle.spec.get("inference", {}))

    def train_handle(self, handle: Handle) -> torch.nn.Module:
        self._check(handle)
        if not handle.native:
            raise NotNative(f"{self.name} handles are inference-only; training needs a state_dict/safetensors/plugin "
                            "model built from a named architecture")
        return handle.model

    def describe(self, handle: Handle) -> dict:
        s = handle.spec
        d = {"adapter": handle.adapter, "format": s.get("format"), "source": s.get("source"), "native": handle.native,
             "sha256": handle.sha256, "device": handle.device, "dtype": str(handle.dtype).removeprefix("torch."),
             "loaded": handle.model is not None, "inference": s.get("inference"), "io": s.get("io"),
             "licence": s.get("licence"), "citation": s.get("citation")}
        d.update(handle.info)
        if handle.model is not None:
            d.update(self._describe(handle))
        return d

    def unload(self, handle: Handle) -> None:
        handle.model = None
        handle.extra.clear()
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.empty_cache()
