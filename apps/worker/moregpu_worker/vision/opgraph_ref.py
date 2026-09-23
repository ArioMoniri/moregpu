"""Reference interpreter for the MoreGPU op-graph (ADR-0114).

The op-graph is what the WGSL executor will run (M6). This interpreter defines each op's semantics with plain torch
functional calls, one entry per op, taking the op's FULL positional aten argument list (defaults already filled in by
the lowering pass). It is the parity oracle for lowered artefacts and the golden source for future WGSL kernels.

Graph JSON (version 1):
    {"version": 1, "lowering_version": "...", "inputs": [placeholder names], "params": {placeholder: weight key},
     "nodes": [{"name", "op", "args": [encoded args]}], "outputs": [encoded args]}
Encoded args: {"ref": node} | {"float": "inf"|"-inf"|"nan"} | {"dtype"|"device"|"memory_format"|"layout": str} |
lists | JSON scalars.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _conv(x, w, b, stride, padding, dilation, transposed, output_padding, groups):
    n = w.ndim - 2
    if transposed:
        fn = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[n - 1]
        return fn(x, w, b, stride, padding, output_padding, groups, dilation)
    return (F.conv1d, F.conv2d, F.conv3d)[n - 1](x, w, b, stride, padding, dilation, groups)


def _interp(mode, align=False):
    def vec(x, size, *rest):
        # nearest: (x, output_size, scale_factors); linear modes: (x, output_size, align_corners, scale_factors)
        ac, scales = (rest[0], rest[1]) if align else (None, rest[0])
        return F.interpolate(x, size=size, scale_factor=scales, mode=mode, align_corners=ac)
    return vec


def _nearest_default(x, size, *scales):
    return F.interpolate(x, size=size, mode="nearest")


def _pool(fn):
    def pool(x, kernel, stride=(), padding=0, *rest):
        return fn(x, kernel, list(stride) or None, padding, *rest)
    return pool


def _bn_no_training(x, w, b, rm, rv, momentum, eps):
    e = x.new_empty(0)
    return F.batch_norm(x, rm, rv, w, b, False, momentum, eps), e, e


OPS: dict[str, Any] = {
    # convolutions
    "aten.convolution.default": _conv,
    "aten.conv1d.default": F.conv1d,
    "aten.conv2d.default": F.conv2d,
    "aten.conv3d.default": F.conv3d,
    "aten.conv2d.padding": F.conv2d,
    "aten.conv3d.padding": F.conv3d,
    "aten.conv_transpose2d.input": F.conv_transpose2d,
    "aten.conv_transpose3d.input": F.conv_transpose3d,
    # activations / elementwise
    "aten.relu.default": torch.relu,
    "aten.leaky_relu.default": lambda x, slope=0.01: F.leaky_relu(x, slope),
    "aten.prelu.default": F.prelu,
    "aten.hardtanh.default": lambda x, lo=-1.0, hi=1.0: F.hardtanh(x, lo, hi),
    "aten.silu.default": F.silu,
    "aten.sigmoid.default": torch.sigmoid,
    "aten.tanh.default": torch.tanh,
    "aten.gelu.default": lambda x, approximate="none": F.gelu(x, approximate=approximate),
    "aten.abs.default": torch.abs,
    "aten.add.Tensor": lambda a, b, alpha=1: torch.add(a, b, alpha=alpha),
    "aten.sub.Tensor": lambda a, b, alpha=1: torch.sub(a, b, alpha=alpha),
    "aten.mul.Tensor": torch.mul,
    "aten.div.Tensor": torch.div,
    # matmul
    "aten.linear.default": F.linear,
    "aten.addmm.default": lambda b, m1, m2, beta=1, alpha=1: torch.addmm(b, m1, m2, beta=beta, alpha=alpha),
    "aten.mm.default": torch.mm,
    # normalisation (inference semantics)
    "aten.layer_norm.default": lambda x, shape, w=None, b=None, eps=1e-5, *_: F.layer_norm(x, shape, w, b, eps),
    "aten.group_norm.default": lambda x, g, w=None, b=None, eps=1e-5, *_: F.group_norm(x, g, w, b, eps),
    "aten.native_group_norm.default": lambda x, w, b, n, c, hw, g, eps: (F.group_norm(x, g, w, b, eps), None, None),
    "aten.instance_norm.default": lambda x, w, b, rm, rv, use_stats, mom, eps, *_: F.instance_norm(
        x, rm, rv, w, b, use_stats, mom, eps),
    "aten.batch_norm.default": lambda x, w, b, rm, rv, training, mom, eps, *_: F.batch_norm(
        x, rm, rv, w, b, False, mom, eps),
    "aten._native_batch_norm_legit_no_training.default": _bn_no_training,
    # pooling / resampling
    "aten.max_pool2d.default": _pool(F.max_pool2d),
    "aten.max_pool3d.default": _pool(F.max_pool3d),
    "aten.avg_pool2d.default": _pool(F.avg_pool2d),
    "aten.avg_pool3d.default": _pool(F.avg_pool3d),
    "aten.adaptive_avg_pool2d.default": F.adaptive_avg_pool2d,
    "aten.adaptive_avg_pool3d.default": F.adaptive_avg_pool3d,
    "aten.mean.dim": lambda x, dims, keepdim=False, dtype=None: torch.mean(x, dims, keepdim, dtype=dtype),
    "aten.upsample_nearest2d.vec": _interp("nearest"),
    "aten.upsample_nearest3d.vec": _interp("nearest"),
    "aten.upsample_nearest2d.default": _nearest_default,
    "aten.upsample_bilinear2d.vec": _interp("bilinear", align=True),
    "aten.upsample_trilinear3d.vec": _interp("trilinear", align=True),
    "aten.pad.default": lambda x, pad, mode="constant", value=None: F.pad(x, pad, mode, value),
    "aten.constant_pad_nd.default": lambda x, pad, value=0: F.pad(x, pad, "constant", value),
    # softmax / reductions
    "aten._softmax.default": lambda x, dim, half_to_float=False: torch.softmax(x, dim),
    "aten.softmax.int": lambda x, dim, dtype=None: torch.softmax(x, dim, dtype=dtype),
    "aten.argmax.default": lambda x, dim=None, keepdim=False: torch.argmax(x, dim, keepdim),
    # layout
    "aten.cat.default": lambda ts, dim=0: torch.cat(ts, dim),
    "aten.permute.default": lambda x, dims: x.permute(dims),
    "aten.view.default": lambda x, shape: x.reshape(shape),
    "aten.reshape.default": lambda x, shape: x.reshape(shape),
    "aten.flatten.using_ints": lambda x, start=0, end=-1: torch.flatten(x, start, end),
    "aten.transpose.int": lambda x, a, b: x.transpose(a, b),
    "aten.unsqueeze.default": torch.unsqueeze,
    "aten.squeeze.dim": torch.squeeze,
    "aten.clone.default": lambda x, memory_format=None: x.clone(),
    "aten.contiguous.default": lambda x, memory_format=None: x.contiguous(),
    "aten.dropout.default": lambda x, p=0.5, train=False: x,  # inference-only executor
    "getitem": lambda x, i: x[i],
}


def decode_arg(v: Any, env: dict) -> Any:
    if isinstance(v, list):
        return [decode_arg(a, env) for a in v]
    if isinstance(v, dict):
        if "ref" in v:
            return env[v["ref"]]
        if "float" in v:
            return float(v["float"])
        if "device" in v:
            return torch.device(v["device"])
        for k in ("dtype", "memory_format", "layout"):
            if k in v:
                return getattr(torch, v[k])
    return v


def run(graph: dict, weights: dict, *inputs):
    """Execute an op-graph. Inputs may be torch tensors or numpy arrays; returns a tensor (or tuple)."""
    env: dict[str, Any] = {}
    for name, x in zip(graph["inputs"], inputs):
        env[name] = torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) else x
    for node, key in graph["params"].items():
        env[node] = weights[key]
    with torch.no_grad():
        for n in graph["nodes"]:
            fn = OPS.get(n["op"])
            if fn is None:
                raise KeyError(f"op-graph op {n['op']!r} has no reference implementation")
            env[n["name"]] = fn(*[decode_arg(a, env) for a in n["args"]])
        outs = [decode_arg(o, env) for o in graph["outputs"]]
    return outs[0] if len(outs) == 1 else tuple(outs)
