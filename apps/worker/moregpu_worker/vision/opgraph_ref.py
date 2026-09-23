"""Reference interpreter for the MoreGPU op-graph (ADR-0114) — the SAME schema the WGSL executor runs.

Graph JSON (version 1, pinned by apps/worker/vision_ops.json; see contract.py and docs/WEBGPU_VISION.md):

    {"version": 1, "inputs": [{"name", "shape"}], "nodes": [{"op", "inputs", "attrs", "output"}],
     "outputs": [name, ...], "weights": "model.safetensors"}

Each node is executed by calling the ATen overload itself (torch.ops.aten.<name>.<overload>) with its arguments rebuilt
from the ATen schema: Tensor arguments come from `inputs` in schema order (a Tensor[] argument takes the flattened run,
a missing optional is None, a number given in `attrs` fills a Tensor slot), every other argument from `attrs` by
schema name (schema default when absent). Only ops in the executor's table are accepted, so this is the parity oracle
for what a WebGPU worker will compute. In-place variants (relu_, leaky_relu_) run as their functional twins, as the
executor does, so a weight is never mutated. Tuple-returning ops (native_layer_norm, …) yield output 0 only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .contract import aten_overload, base_op, executor_ops, from_json, is_tensor_list_type, is_tensor_type


def _resolve(op: str):
    if base_op(op) not in executor_ops():
        raise KeyError(f"op-graph op {op!r} is not in the WGSL executor's table (apps/worker/vision_ops.json)")
    base = base_op(op)
    if base.endswith("_") and not base.endswith("__"):  # in-place → functional twin (the executor is out-of-place)
        parts = op.split(".")
        try:
            return aten_overload(".".join([parts[0], parts[1][:-1], *parts[2:]]))
        except KeyError:
            pass
    return aten_overload(op)


def build_args(schema, inputs: list, attrs: dict, value) -> tuple[list, dict]:
    """(positional, kwargs) for an ATen call from the executor's {inputs, attrs} encoding."""
    args = schema.arguments
    n_tensor_after = [0] * len(args)
    for i in range(len(args) - 2, -1, -1):
        nxt = args[i + 1]
        n_tensor_after[i] = n_tensor_after[i + 1] + (1 if is_tensor_type(nxt.type) and nxt.name not in attrs else 0)
    pos, kw, k = [], {}, 0
    for i, a in enumerate(args):
        if is_tensor_list_type(a.type):
            take = max(0, len(inputs) - k - n_tensor_after[i])
            v = [value(x) for x in inputs[k:k + take]]
            k += take
        elif is_tensor_type(a.type):
            if a.name in attrs:        # a Python number in a Tensor slot (x * 0.5)
                v = from_json(attrs[a.name], torch.FloatType.get())
            elif k < len(inputs):
                v = None if inputs[k] is None else value(inputs[k])
                k += 1
            else:
                v = None
        elif a.name in attrs:
            v = from_json(attrs[a.name], a.type)
        elif a.has_default_value():
            v = a.default_value
        else:
            v = None
        if a.kwarg_only:
            kw[a.name] = v
        else:
            pos.append(v)
    return pos, kw


def run(graph: dict, weights: dict, *inputs):
    """Execute an op-graph. Inputs may be torch tensors or numpy arrays; returns a tensor (or tuple)."""
    if graph.get("version") != 1:
        raise ValueError(f"op-graph version must be 1 (got {graph.get('version')})")
    env: dict[str, Any] = {}
    for spec, x in zip(graph["inputs"], inputs):
        env[spec["name"]] = torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) else x

    def value(name: str):
        if name in env:
            return env[name]
        if name in weights:
            w = weights[name]
            return w.float() if w.is_floating_point() and w.dtype != torch.float32 else w
        raise KeyError(f"unknown value {name!r} (not a graph input, earlier node output or weight)")

    with torch.no_grad():
        for n in graph["nodes"]:
            fn = _resolve(n["op"])
            pos, kw = build_args(fn._schema, list(n.get("inputs") or []), dict(n.get("attrs") or {}), value)
            out = fn(*pos, **kw)
            env[n["output"]] = out[0] if isinstance(out, (tuple, list)) else out
        outs = [value(o) for o in graph["outputs"]]
    return outs[0] if len(outs) == 1 else tuple(outs)
