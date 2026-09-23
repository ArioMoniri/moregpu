"""Worker RPC ops for published models (ADR-0113/0114). `worker_torch.py` routes `vision_*` ops here:

    if op in vision.ops.OPS: return vision.ops.handle(op, payload)

    vision_models_describe {}                       → formats, registries, plugins, lowering targets, loaded ids
    vision_load     {id, spec}                      → load (validate → fetch → sha256 → adapter); replaces same id
    vision_describe {id}
    vision_lower    {id, target, example_shape?, include_bytes?} → lowered-artefact report (+ base64 payload)
    vision_unload   {id}

HANDLES / LOWERED are module-level so inference ops added later can share the loaded models.
Errors raise (KeyError for unknown op/id, SpecError/Refused* for bad specs/artefacts); the caller maps them to RPC errors.
"""
from __future__ import annotations

import base64
import importlib.util

import torch

from . import adapters as A
from . import lowering as L
from .adapters import plugins
from .adapters.onnx import available_providers

HANDLES: dict[str, A.Handle] = {}
LOWERED: dict[tuple[str, str], L.LoweredArtifact] = {}
_REGISTRY_MODULES = {"torchvision": "torchvision", "timm": "timm", "monai": "monai", "hf": "transformers"}


def _get(mid: str) -> A.Handle:
    if mid not in HANDLES:
        raise KeyError(f"model {mid!r} is not loaded")
    return HANDLES[mid]


def _id(p: dict) -> str:
    mid = p.get("id")
    if not isinstance(mid, str) or not mid:
        raise ValueError("payload needs a non-empty string 'id'")
    return mid


def lowered(mid: str, target: str) -> L.LoweredArtifact:
    return LOWERED[(mid, target)]


def models_describe(p: dict) -> dict:
    found, refused = plugins.discover()
    regs = {r: importlib.util.find_spec(m) is not None for r, m in _REGISTRY_MODULES.items()}
    regs["plugin"] = True
    return {"formats": list(A.ADAPTERS), "registries": regs,
            "plugins": {"available": sorted(found), "refused": refused},
            "lowering": {"targets": list(L.TARGETS), "wgsl_ops": sorted(L.WGSL_OPS), "version": L.LOWERING_VERSION},
            "onnx_providers": available_providers(), "loaded": sorted(HANDLES)}


def load(p: dict) -> dict:
    mid = _id(p)
    h = A.load(p.get("spec"))
    unload({"id": mid})
    HANDLES[mid] = h
    return {"id": mid, **A.describe(h)}


def describe(p: dict) -> dict:
    mid = _id(p)
    return {"id": mid, **A.describe(_get(mid))}


def lower(p: dict) -> dict:
    mid = _id(p)
    h, target = _get(mid), p.get("target", "wgsl")
    example = None
    if p.get("example_shape"):
        example = torch.randn(list(p["example_shape"]), generator=torch.Generator().manual_seed(0))
    art = L.lower(h, target, example=example)
    LOWERED[(mid, target)] = art
    out = {"id": mid, **art.describe()}
    if p.get("include_bytes"):
        if art.kind == "opgraph":
            out["graph_json"] = art.graph_json()
            out["weights_b64"] = base64.b64encode(art.weights_bytes()).decode()
        elif art.kind == "onnx":
            out["onnx_b64"] = base64.b64encode(art.onnx_bytes).decode()
    return out


def unload(p: dict) -> dict:
    mid = _id(p)
    h = HANDLES.pop(mid, None)
    for k in [k for k in LOWERED if k[0] == mid]:
        del LOWERED[k]
    if h is not None:
        A.unload(h)
    return {"id": mid, "unloaded": h is not None}


OPS = {"vision_models_describe": models_describe, "vision_load": load, "vision_describe": describe,
       "vision_lower": lower, "vision_unload": unload}


def handle(op: str, payload: dict | None = None) -> dict:
    fn = OPS.get(op)
    if fn is None:
        raise KeyError(f"unknown vision op {op!r}; known: {sorted(OPS)}")
    return fn(payload or {})
