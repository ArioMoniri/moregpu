"""Model adapters (ADR-0113). One interface over every accepted format:

    handle = load(spec, fetch)            # validate spec → fetch → verify sha256 → open with the format's adapter
    infer(handle, batch) -> torch.Tensor  # batch: torch.Tensor | np.ndarray; honours spec.inference (sliding window/TTA)
    train_handle(handle) -> nn.Module     # native (state_dict/safetensors/plugin) handles only
    describe(handle) -> dict              # JSON-able
    unload(handle)

Precedence when a model is published in several forms: state_dict/safetensors + named arch (trainable, lowerable) >
allowlisted plugin > torch.export > TorchScript > ONNX. Pickled full models are refused (RefusedFormat).
"""
from __future__ import annotations

import torch

from .. import spec as S
from ..errors import (IntegrityError, KeyMismatch, NotLoaded, NotNative, Refused, RefusedFormat, RefusedPlugin,
                      RefusedSource, UnknownArch)
from ..fetch import default_fetch
from .base import Adapter, Handle, as_tensor, module_sha256, resolve_device
from .export import TorchExportAdapter, TorchScriptAdapter
from .onnx import OnnxAdapter
from .statedict import StateDictAdapter

ADAPTERS: dict[str, Adapter] = {
    "state_dict": StateDictAdapter("state_dict"),
    "safetensors": StateDictAdapter("safetensors"),
    "plugin": StateDictAdapter("plugin"),
    "torch_export": TorchExportAdapter(),
    "torchscript": TorchScriptAdapter(),
    "onnx": OnnxAdapter(),
}

__all__ = ["ADAPTERS", "Adapter", "Handle", "load", "infer", "train_handle", "describe", "unload", "from_module",
           "as_tensor", "resolve_device", "Refused", "RefusedFormat", "RefusedSource", "RefusedPlugin",
           "IntegrityError", "KeyMismatch", "UnknownArch", "NotNative", "NotLoaded"]


def load(spec: dict, fetch=None) -> Handle:
    spec = S.validate(spec)
    return ADAPTERS[spec["format"]].load(spec, fetch or default_fetch)


def infer(handle: Handle, batch) -> torch.Tensor:
    return ADAPTERS[handle.adapter].infer(handle, batch)


def train_handle(handle: Handle) -> torch.nn.Module:
    return ADAPTERS[handle.adapter].train_handle(handle)


def describe(handle: Handle) -> dict:
    return ADAPTERS[handle.adapter].describe(handle)


def unload(handle: Handle) -> None:
    ADAPTERS[handle.adapter].unload(handle)


def from_module(module: torch.nn.Module, name: str | None = None, spec: dict | None = None) -> Handle:
    """Wrap an in-memory native module (e.g. a just-trained model) as a handle; identity = hash of its weights."""
    s = {**S.DEFAULTS, "format": "state_dict", **(spec or {})}
    s["inference"] = {**S.INFERENCE_DEFAULTS, **s.get("inference", {})}
    p = next(module.parameters(), None)
    return Handle(spec=s, adapter="state_dict", sha256=module_sha256(module), model=module.eval(), native=True,
                  device=str(p.device) if p is not None else "cpu", dtype=p.dtype if p is not None else torch.float32,
                  info={"arch": name or type(module).__name__, "unwrap": [],
                        "params": sum(q.numel() for q in module.parameters())})
