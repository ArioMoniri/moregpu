"""The WGSL executor's op-graph contract (ADR-0114, M6), shared by the lowering and the reference interpreter.

The single source of truth is `apps/worker/vision_ops.json` (pinned to the executor's SUPPORTED_OPS by a vitest):

    {"version": 1, "inputs": [{"name", "shape"}], "nodes": [{"op", "inputs", "attrs", "output"}], "outputs": [name],
     "weights": "model.safetensors"}

* op     — 'aten.<name>.<overload>'; the executor matches on the base name 'aten.<name>'.
* inputs — the Tensor arguments in ATen-schema order (graph inputs, earlier outputs or safetensors keys); an absent
           optional tensor is null, trailing nulls dropped, a Tensor[] argument (aten.cat) flattened in place.
* attrs  — every non-tensor argument keyed by its ATen schema argument name, defaults filled; a Python number in a
           Tensor slot (x * 0.5) goes to attrs under that argument's name. Non-finite floats are spelled
           "Infinity" / "-Infinity" / "NaN" (strict JSON; parsed by both Python float() and JS Number()).
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
# repo layout (apps/worker/vision_ops.json) first, then a copy packaged into the wheel next to this module
_CANDIDATES = (_HERE.parents[1] / "vision_ops.json", _HERE / "vision_ops.json")


def contract_path() -> Path:
    env = os.environ.get("MOREGPU_VISION_OPS")
    for p in ((Path(env),) if env else ()) + _CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError("vision_ops.json (the WGSL executor contract) not found; set MOREGPU_VISION_OPS")


@lru_cache(maxsize=1)
def contract() -> dict:
    return json.loads(contract_path().read_text())


def executor_ops() -> frozenset[str]:
    return frozenset(contract()["ops"])


def base_op(op: str) -> str:
    """'aten.add.Tensor' → 'aten.add'; 'aten.relu' stays."""
    p = op.split(".")
    return ".".join(p[:2]) if len(p) > 2 else op


def aten_overload(op: str):
    """'aten.conv2d.default' | 'aten.cat' → the torch OpOverload (overload 'default' when omitted)."""
    parts = op.split(".")
    if len(parts) < 2 or parts[0] != "aten":
        raise KeyError(f"op {op!r} is not an ATen op")
    packet = getattr(torch.ops.aten, parts[1], None)
    if packet is None:
        raise KeyError(f"unknown ATen op {op!r}")
    ovl = parts[2] if len(parts) > 2 else "default"
    fn = getattr(packet, ovl, None)
    if fn is None:
        raise KeyError(f"unknown overload {ovl!r} of {op!r}")
    return fn


def aten_schema(op: str):
    return aten_overload(op)._schema


# ── argument typing ──
def is_tensor_type(t) -> bool:
    if isinstance(t, torch.TensorType):
        return True
    return isinstance(t, torch.OptionalType) and isinstance(t.getElementType(), torch.TensorType)


def is_tensor_list_type(t) -> bool:
    return isinstance(t, torch.ListType) and is_tensor_type(t.getElementType())


def _is_number_type(t) -> bool:
    if isinstance(t, torch.OptionalType):
        t = t.getElementType()
    if isinstance(t, torch.ListType):
        t = t.getElementType()
    return isinstance(t, (torch.FloatType, torch.NumberType, torch.IntType)) or str(t) in ("Scalar", "float", "int", "SymInt")


def _is_dtype_type(t) -> bool:
    if isinstance(t, torch.OptionalType):
        t = t.getElementType()
    return str(t) == "ScalarType" or "ScalarType" in type(t).__name__


# ── JSON-safe values ──
NONFINITE = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan, "inf": math.inf, "-inf": -math.inf, "nan": math.nan}


def to_json(v: Any) -> Any:
    """A non-tensor ATen argument → strict-JSON data (the executor's attr encoding)."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        if math.isfinite(v):
            return v
        return "NaN" if v != v else ("Infinity" if v > 0 else "-Infinity")
    if isinstance(v, (list, tuple)):
        return [to_json(x) for x in v]
    if isinstance(v, torch.dtype):
        return str(v).replace("torch.", "")
    if isinstance(v, (torch.device, torch.layout, torch.memory_format)):
        return None
    if isinstance(v, torch.SymInt):
        return int(v)
    raise TypeError(f"cannot encode op-graph attribute of type {type(v).__name__}")


def from_json(v: Any, ty) -> Any:
    """Inverse of to_json, driven by the schema argument type."""
    if isinstance(v, list):
        return [from_json(x, ty) for x in v]
    if isinstance(v, str):
        if v in NONFINITE and _is_number_type(ty):
            return NONFINITE[v]
        if _is_dtype_type(ty):
            return getattr(torch, v)
    return v
