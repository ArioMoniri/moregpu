"""Automatic lowering hooks (ADR-0114, M4): native model → artefact a non-torch worker can run.

    lower(handle, target) -> LoweredArtifact
      1. torch.export the model with an example input (decompositions: none — keep aten-level ops).
      2. target "wgsl": if EVERY op is in WGSL_OPS → op-graph JSON + safetensors weights (kind "opgraph").
      3. otherwise (or target "onnx-web"): in-memory ONNX for onnxruntime-web's WebGPU EP (kind "onnx").
      4. otherwise: kind "native" — not servable off-torch; `unsupported_ops` says why.
    Every op-graph/ONNX artefact must pass a parity probe against the native forward (fp32 max|Δ| ≤ 1e-4) or it is
    refused (`servable=False`, `run()` raises ParityRefused). Results are cached by
    sha256(model sha256 | LOWERING_VERSION | target | example shape), in memory and optionally on disk (re-probed on
    load, so a tampered cache cannot be served).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import operator
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import opgraph_ref
from .adapters import ADAPTERS, Handle, as_tensor

LOWERING_VERSION = "1"
TARGETS = ("wgsl", "onnx-web")
PARITY_TOL = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 5e-2}
ONNX_OPSET = 17

# The WGSL executor's op table (what M6 kernels must implement; opgraph_ref defines the semantics).
WGSL_OPS = frozenset({
    "aten.convolution.default", "aten.conv1d.default", "aten.conv2d.default", "aten.conv3d.default",
    "aten.conv2d.padding", "aten.conv3d.padding", "aten.conv_transpose2d.input", "aten.conv_transpose3d.input",
    "aten.relu.default", "aten.leaky_relu.default", "aten.prelu.default", "aten.hardtanh.default", "aten.silu.default",
    "aten.sigmoid.default", "aten.tanh.default", "aten.gelu.default", "aten.abs.default",
    "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor",
    "aten.linear.default", "aten.addmm.default", "aten.mm.default",
    "aten.layer_norm.default", "aten.group_norm.default", "aten.native_group_norm.default",
    "aten.instance_norm.default", "aten.batch_norm.default", "aten._native_batch_norm_legit_no_training.default",
    "aten.max_pool2d.default", "aten.max_pool3d.default", "aten.avg_pool2d.default", "aten.avg_pool3d.default",
    "aten.adaptive_avg_pool2d.default", "aten.adaptive_avg_pool3d.default", "aten.mean.dim",
    "aten.upsample_nearest2d.vec", "aten.upsample_nearest3d.vec", "aten.upsample_nearest2d.default",
    "aten.upsample_bilinear2d.vec", "aten.upsample_trilinear3d.vec", "aten.pad.default",
    "aten.constant_pad_nd.default",
    "aten._softmax.default", "aten.softmax.int", "aten.argmax.default",
    "aten.cat.default", "aten.permute.default", "aten.view.default", "aten.reshape.default", "aten.flatten.using_ints",
    "aten.transpose.int", "aten.unsqueeze.default", "aten.squeeze.dim", "aten.clone.default",
    "aten.contiguous.default", "aten.dropout.default", "getitem",
})


class ParityRefused(RuntimeError):
    """The lowered artefact is not servable (parity probe failed, or nothing could be lowered)."""


_CACHE: dict[str, "LoweredArtifact"] = {}


def clear_cache() -> None:
    _CACHE.clear()


def cache_key(model_sha256: str, target: str, shape) -> str:
    return hashlib.sha256(f"{model_sha256}|{LOWERING_VERSION}|{target}|{','.join(map(str, shape))}".encode()).hexdigest()


@dataclass
class LoweredArtifact:
    target: str
    kind: str                                  # "opgraph" | "onnx" | "native"
    cache_key: str
    model_sha256: str
    example_shape: list
    graph: dict | None = None
    weights: dict | None = None
    onnx_bytes: bytes | None = None
    unsupported_ops: list = field(default_factory=list)
    ops: list = field(default_factory=list)
    parity: dict = field(default_factory=dict)
    servable: bool = False
    reason: str = ""
    cached: bool = False
    _sess: tuple | None = field(default=None, repr=False)

    def weights_bytes(self) -> bytes:
        from safetensors.torch import save
        return save({k: v.detach().cpu().contiguous() for k, v in (self.weights or {}).items()})

    def graph_json(self) -> str:
        return json.dumps(self.graph, sort_keys=True)

    def payload(self) -> list[bytes]:
        if self.kind == "opgraph":
            return [self.graph_json().encode(), self.weights_bytes()]
        if self.kind == "onnx":
            return [self.onnx_bytes]
        return []

    def _execute(self, x):
        if self.kind == "opgraph":
            return as_tensor(opgraph_ref.run(self.graph, self.weights, x))
        if self.kind == "onnx":
            if self._sess is None or self._sess[0] is not self.onnx_bytes:
                from .adapters.onnx import session
                self._sess = (self.onnx_bytes, session(self.onnx_bytes, ["CPUExecutionProvider"]))
            s = self._sess[1]
            arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            return torch.from_numpy(s.run(None, {s.get_inputs()[0].name: arr.astype(np.float32)})[0])
        raise ParityRefused(f"nothing to run: {self.reason}")

    def run(self, x):
        if not self.servable:
            raise ParityRefused(f"lowered artefact ({self.kind}/{self.target}) refused: {self.reason}")
        return self._execute(x)

    def describe(self) -> dict:
        blobs = self.payload()
        h = hashlib.sha256()
        for b in blobs:
            h.update(hashlib.sha256(b).digest())
        return {"target": self.target, "kind": self.kind, "servable": self.servable, "reason": self.reason,
                "unsupported_ops": self.unsupported_ops, "ops": self.ops, "parity": self.parity,
                "cache_key": self.cache_key, "model_sha256": self.model_sha256, "lowering_version": LOWERING_VERSION,
                "example_shape": self.example_shape, "artifact_sha256": h.hexdigest(),
                "bytes": sum(len(b) for b in blobs), "cached": self.cached}


# ------------------------------------------------------------------ arg encoding (shared with opgraph_ref.decode_arg)
def encode_arg(v: Any) -> Any:
    if isinstance(v, torch.fx.Node):
        return {"ref": v.name}
    if v is None or isinstance(v, bool | int | str):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else {"float": str(v)}
    if isinstance(v, list | tuple):
        return [encode_arg(a) for a in v]
    if isinstance(v, torch.dtype):
        return {"dtype": str(v).removeprefix("torch.")}
    if isinstance(v, torch.device):
        return {"device": str(v)}
    if isinstance(v, torch.memory_format):
        return {"memory_format": str(v).removeprefix("torch.")}
    if isinstance(v, torch.layout):
        return {"layout": str(v).removeprefix("torch.")}
    raise TypeError(f"cannot encode op-graph argument of type {type(v).__name__}")


def _op_name(target) -> str:
    return "getitem" if target is operator.getitem else str(target)


def _full_args(node: torch.fx.Node) -> list:
    schema = getattr(node.target, "_schema", None)
    args = list(node.args)
    if schema is None:
        return args
    for a in schema.arguments[len(args):]:
        if a.name in node.kwargs:
            args.append(node.kwargs[a.name])
        elif a.has_default_value():
            args.append(a.default_value)
        else:  # pragma: no cover - aten schemas always provide defaults for omitted args
            raise TypeError(f"{node.target}: no value for argument {a.name}")
    return args


def _export(module, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.export.export(module, (x,)).run_decompositions({})


def _graph_ops(ep) -> list[str]:
    return sorted({_op_name(n.target) for n in ep.graph.nodes if n.op == "call_function"})


def _emit_opgraph(ep) -> tuple[dict, dict]:
    from torch.export.graph_signature import InputKind, OutputKind
    params, inputs, weights = {}, [], {}
    for spec in ep.graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            inputs.append(spec.arg.name)
        else:  # PARAMETER / BUFFER / CONSTANT_TENSOR (custom objects never reach here: export would need them)
            t = ep.state_dict[spec.target] if spec.target in ep.state_dict else ep.constants[spec.target]
            params[spec.arg.name] = spec.target
            weights[spec.target] = t.detach().clone().contiguous()
    nodes = [{"name": n.name, "op": _op_name(n.target), "args": encode_arg(_full_args(n))}
             for n in ep.graph.nodes if n.op == "call_function"]
    out_node = next(n for n in ep.graph.nodes if n.op == "output")
    user = [o for o, s in zip(out_node.args[0], ep.graph_signature.output_specs) if s.kind == OutputKind.USER_OUTPUT]
    graph = {"version": 1, "lowering_version": LOWERING_VERSION, "inputs": inputs, "params": params, "nodes": nodes,
             "outputs": encode_arg(user)}
    return graph, weights


def _export_onnx(module, x) -> bytes:
    buf = io.BytesIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        torch.onnx.export(module, (x,), buf, dynamo=False, opset_version=ONNX_OPSET, input_names=["input"],
                          output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    return buf.getvalue()


def _example(handle: Handle, example) -> torch.Tensor:
    if example is not None:
        return torch.from_numpy(np.ascontiguousarray(example)) if isinstance(example, np.ndarray) else example
    inputs = (handle.spec.get("io") or {}).get("inputs") or []
    shape = inputs[0].get("shape") if inputs else None
    if not shape:
        raise ValueError("lowering needs an example input: pass example= or declare spec.io.inputs[0].shape")
    g = torch.Generator().manual_seed(0)
    return torch.randn([d or 1 for d in shape], generator=g)


def _native_forward(handle: Handle, x):
    return as_tensor(ADAPTERS[handle.adapter].forward(handle, x))


def probe(art: LoweredArtifact, handle: Handle, example=None) -> dict:
    """Mandatory parity probe: lowered artefact vs the native forward on the same input."""
    x = _example(handle, example)
    tol = PARITY_TOL.get(handle.dtype, 1e-4)
    with torch.inference_mode():
        ref = _native_forward(handle, x).float().cpu()
        try:
            got = as_tensor(art._execute(x)).float().cpu()
            err = None if got.shape == ref.shape else f"shape {tuple(got.shape)} != native {tuple(ref.shape)}"
            max_abs = (got - ref).abs().max().item() if err is None else math.inf
        except Exception as e:
            err, max_abs = f"{type(e).__name__}: {e}", math.inf
    ok = err is None and max_abs <= tol
    art.parity = {"max_abs": max_abs, "tol": tol, "ok": ok, "example_shape": list(x.shape)}
    art.servable = ok
    if not ok:
        art.reason = f"parity probe failed: {err or f'max|Δ|={max_abs:.3g} > {tol}'}; refusing to serve"
    elif art.reason.startswith("parity probe failed"):
        art.reason = ""
    return art.parity


def _lower(handle: Handle, target: str, x: torch.Tensor, key: str) -> LoweredArtifact:
    art = LoweredArtifact(target=target, kind="native", cache_key=key, model_sha256=handle.sha256,
                          example_shape=list(x.shape))
    if handle.adapter == "onnx":  # already ONNX: serve the published bytes as-is (still parity-probed)
        art.kind, art.onnx_bytes = "onnx", Path(handle.extra["path"]).read_bytes()
        art.reason = "published ONNX served via onnxruntime-web"
        return art
    module = handle.model
    ep = None
    try:
        ep = _export(module, x)
        art.ops = _graph_ops(ep)
        art.unsupported_ops = [op for op in art.ops if op not in WGSL_OPS]
    except Exception as e:
        art.reason = f"torch.export failed ({type(e).__name__}: {str(e).splitlines()[0][:200]}); "
    if target == "wgsl" and ep is not None and not art.unsupported_ops:
        art.kind = "opgraph"
        art.graph, art.weights = _emit_opgraph(ep)
        return art
    try:
        art.onnx_bytes, art.kind = _export_onnx(module, x), "onnx"
        if target == "wgsl":
            art.reason += "ops outside the WGSL table; lowered to ONNX for onnxruntime-web"
    except Exception as e:
        art.reason += f"ONNX export failed ({type(e).__name__}: {str(e).splitlines()[0][:200]}); native-only"
    return art


def _disk_path(cache_dir, key: str) -> Path:
    return Path(cache_dir) / key


def _save_disk(art: LoweredArtifact, cache_dir) -> None:
    d = _disk_path(cache_dir, art.cache_key)
    d.mkdir(parents=True, exist_ok=True)
    if art.kind == "opgraph":
        (d / "graph.json").write_text(art.graph_json())
        (d / "weights.safetensors").write_bytes(art.weights_bytes())
    elif art.kind == "onnx":
        (d / "model.onnx").write_bytes(art.onnx_bytes)
    meta = {k: getattr(art, k) for k in ("target", "kind", "model_sha256", "example_shape", "unsupported_ops", "ops",
                                          "reason")}
    tmp = d / f".meta.{os.getpid()}"
    tmp.write_text(json.dumps(meta))
    tmp.replace(d / "meta.json")


def _load_disk(cache_dir, key: str) -> LoweredArtifact | None:
    d = _disk_path(cache_dir, key)
    if not (d / "meta.json").is_file():
        return None
    meta = json.loads((d / "meta.json").read_text())
    art = LoweredArtifact(cache_key=key, cached=True, **meta)
    if art.kind == "opgraph":
        from safetensors.torch import load
        art.graph = json.loads((d / "graph.json").read_text())
        art.weights = load((d / "weights.safetensors").read_bytes())
    elif art.kind == "onnx":
        art.onnx_bytes = (d / "model.onnx").read_bytes()
    return art


def lower(handle: Handle, target: str = "wgsl", example=None, cache: bool = True, cache_dir=None) -> LoweredArtifact:
    if target not in TARGETS:
        raise ValueError(f"unknown lowering target {target!r}; expected one of {TARGETS}")
    x = _example(handle, example)
    key = cache_key(handle.sha256, target, tuple(x.shape))
    if cache and key in _CACHE:
        return _CACHE[key]
    art = _load_disk(cache_dir, key) if cache_dir else None
    fresh = art is None
    if fresh:
        art = _lower(handle, target, x, key)
    if art.kind != "native":
        probe(art, handle, x)
    if cache:
        _CACHE[key] = art
    if cache_dir and fresh:
        _save_disk(art, cache_dir)
    return art
