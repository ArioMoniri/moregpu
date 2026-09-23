"""Automatic lowering hooks (ADR-0114, M4/M6): native model → artefact a non-torch worker can run.

    lower(handle, target) -> LoweredArtifact
      1. torch.export.export the model with an example input, in the DEFAULT (training-IR) dialect — no core-ATen
         decomposition (it would rewrite trilinear/nearest-3d upsampling, replicate pad, instance norm and SDPA into
         index/where graphs the executor does not take; docs/WEBGPU_VISION.md).
      2. target "wgsl": if every op maps onto the WGSL executor's table (apps/worker/vision_ops.json) → an op-graph in
         EXACTLY the executor's schema + safetensors weights (kind "opgraph"):
           {version:1, inputs:[{name,shape}], nodes:[{op, inputs, attrs, output}], outputs:[name], weights}
         with tensor args in ATen-schema order, attrs keyed by ATen schema argument names (defaults filled),
         getitem(multi-output op, 0) mapped onto the node's output, unbind/split/chunk rewritten into select/slice
         nodes, and in-place ops functionalised (add_ → add) when the mutated value is never read again.
      3. otherwise (or target "onnx-web"): in-memory ONNX for onnxruntime-web's WebGPU EP (kind "onnx").
      4. otherwise: kind "native" — not servable off-torch; `unsupported_ops` says why.
    Every op-graph/ONNX artefact must pass a parity probe against the native forward (fp32 max|Δ| ≤ 1e-4) or it is
    refused (`servable=False`, `run()` raises ParityRefused). The op-graph probe runs opgraph_ref, which executes the
    same schema by calling each ATen overload. Results are cached by
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

import numpy as np
import torch

from . import opgraph_ref
from .adapters import ADAPTERS, Handle, as_tensor
from .contract import aten_schema, base_op, executor_ops, is_tensor_list_type, is_tensor_type, to_json
from .errors import brief

__all__ = ["lower", "probe", "LoweredArtifact", "ParityRefused", "WGSL_OPS", "TARGETS", "base_op", "aten_schema"]

LOWERING_VERSION = "2"   # 2: the executor's schema (vision_ops.json), default export dialect
TARGETS = ("wgsl", "onnx-web")
PARITY_TOL = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 5e-2}
ONNX_OPSET = 17
WEIGHTS_FILE = "model.safetensors"

# The WGSL executor's op table, read from its machine-readable contract apps/worker/vision_ops.json (base names, e.g.
# "aten.conv2d"; overloads match on the base). opgraph_ref executes exactly these ops.
WGSL_OPS = executor_ops()
# tuple-of-views ops the lowering rewrites into select/slice nodes instead of emitting
_SPLIT_OPS = frozenset({"aten.unbind", "aten.split", "aten.split_with_sizes", "aten.chunk"})


class ParityRefused(RuntimeError):
    """The lowered artefact is not servable (parity probe failed, or nothing could be lowered)."""


class Unlowerable(Exception):
    def __init__(self, ops: list[str]):
        super().__init__(", ".join(ops))
        self.ops = ops


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
        return json.dumps(self.graph, sort_keys=True, allow_nan=False)

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
        io_ = None
        if self.kind == "opgraph" and self.graph:
            io_ = {"inputs": self.graph["inputs"], "outputs": self.graph["outputs"]}
        return {"target": self.target, "kind": self.kind, "servable": self.servable, "reason": self.reason,
                "unsupported_ops": self.unsupported_ops, "ops": self.ops, "parity": self.parity,
                "cache_key": self.cache_key, "model_sha256": self.model_sha256, "lowering_version": LOWERING_VERSION,
                "example_shape": self.example_shape, "artifact_sha256": h.hexdigest(), "io": io_,
                "bytes": sum(len(b) for b in blobs), "cached": self.cached}


# ------------------------------------------------------------------ torch.export → executor op-graph
def _op_name(target) -> str:
    return "getitem" if target is operator.getitem else str(target)


def _export(module, x):
    """torch.export.export in its default dialect — NO run_decompositions (see the module docstring)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.export.export(module, (x,))


def _graph_ops(ep) -> list[str]:
    return sorted({_op_name(n.target) for n in ep.graph.nodes if n.op == "call_function"})


def _is_inplace(schema) -> bool:
    a = schema.arguments[0] if schema.arguments else None
    return bool(a is not None and a.alias_info is not None and a.alias_info.is_write)


_VIEW_OPS = frozenset({"aten.view", "aten.reshape", "aten._unsafe_view", "aten.flatten", "aten.unsqueeze", "aten.squeeze",
                       "aten.alias", "aten.detach", "aten.permute", "aten.transpose", "aten.t", "aten.expand",
                       "aten.select", "aten.slice", "aten.narrow", "aten.as_strided", "aten.contiguous",
                       "aten.view_as", "aten.dropout", "aten.unbind", "aten.split", "aten.split_with_sizes", "aten.chunk"})


def _is_view(n) -> bool:
    return n.op == "call_function" and (n.target is operator.getitem or base_op(_op_name(n.target)) in _VIEW_OPS)


def _read_after_mutation(n, order) -> bool:
    """In-place node n mutates args[0]. Renaming it to its functional twin is only exact when nothing that shares
    args[0]'s storage (its view base, sibling views, views of it) is read after n; later reads of the mutated value
    itself already go through n's output in torch.export's graph."""
    root = n.args[0]
    while _is_view(root) and isinstance(root.args[0], torch.fx.Node):
        root = root.args[0]
    if root.op == "placeholder":
        return True                        # mutates a graph input / weight: the effect escapes the graph
    seen, stack = set(), [root]
    while stack:
        a = stack.pop()
        if a in seen:
            continue
        seen.add(a)
        for u in a.users:
            if u is n:
                continue
            if _is_view(u):
                stack.append(u)
            if order[u] > order[n]:
                return True
    return False


def _functional(op: str) -> str:
    parts = op.split(".")
    return ".".join([parts[0], parts[1][:-1], *parts[2:]])


def _emit_opgraph(ep) -> tuple[dict, dict]:
    """ExportedProgram → (graph in the executor's schema, weights). Raises Unlowerable(ops) when anything is outside
    the executor's table (after the select/slice and in-place rewrites)."""
    from torch.export.graph_signature import InputKind, OutputKind
    sig = ep.graph_signature
    p2w = {s.arg.name: s.target for s in sig.input_specs if s.kind != InputKind.USER_INPUT}
    order = {n: i for i, n in enumerate(ep.graph.nodes)}
    names: dict[str, str] = {}             # fx node name → op-graph value name
    weights: dict[str, torch.Tensor] = {}
    inputs, nodes, bad = [], [], []
    for n in ep.graph.nodes:
        if n.op != "placeholder":
            continue
        if n.name in p2w:
            t = p2w[n.name]
            names[n.name] = t
        else:
            names[n.name] = n.name
            inputs.append({"name": n.name, "shape": [int(d) for d in n.meta["val"].shape]})

    def ref(v) -> str:
        nm = names[v.name]
        if v.op == "placeholder" and v.name in p2w and nm not in weights:
            t = ep.state_dict[nm] if nm in ep.state_dict else ep.constants[nm]
            t = t.detach()
            if t.is_floating_point() and t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                t = t.float()
            weights[nm] = t.clone().contiguous()
        return nm

    split_src: set[str] = set()
    for n in ep.graph.nodes:
        if n.op != "call_function":
            continue
        if n.target is operator.getitem:
            src, idx = n.args
            if src.name in split_src:
                continue                   # emitted as select/slice by the split rewrite below
            if idx == 0:
                names[n.name] = names[src.name]
                continue
            if n.users:
                bad.append(f"getitem[{idx}] of {_op_name(src.target)} (only output 0 of multi-output ops is supported)")
            names[n.name] = n.name
            continue
        op = _op_name(n.target)
        base = base_op(op)
        names[n.name] = n.name             # (also for a refused node, so the scan continues and reports everything)
        schema = getattr(n.target, "_schema", None)
        if schema is None:
            bad.append(op)
            continue
        full = list(n.args) + [n.kwargs.get(a.name, a.default_value if a.has_default_value() else None)
                               for a in schema.arguments[len(n.args):]]
        if base in _SPLIT_OPS:            # tuple of views → one select/slice node per used getitem
            split_src.add(n.name)
            x = full[0]
            shape = list(n.args[0].meta["val"].shape)
            outs = n.meta["val"]
            if base == "aten.unbind":
                dim = int(full[1] if len(full) > 1 else 0) % len(shape)
            else:
                dim = int(full[2] if len(full) > 2 and full[2] is not None else 0) % len(shape)
            starts, s0 = [], 0
            for o in outs:
                starts.append(s0)
                s0 += int(o.shape[dim]) if base != "aten.unbind" else 1
            for u in sorted(n.users, key=lambda u: order[u]):
                if u.target is not operator.getitem:
                    bad.append(f"{op} used other than by getitem")
                    continue
                i = int(u.args[1])
                if base == "aten.unbind":
                    nodes.append({"op": "aten.select.int", "inputs": [ref(x)], "attrs": {"dim": dim, "index": i}, "output": u.name})
                else:
                    nodes.append({"op": "aten.slice.Tensor", "inputs": [ref(x)], "output": u.name,
                                  "attrs": {"dim": dim, "start": starts[i], "end": starts[i] + int(outs[i].shape[dim]), "step": 1}})
                names[u.name] = u.name
            continue
        if _is_inplace(schema):
            if _read_after_mutation(n, order):
                bad.append(f"{op} (in-place mutation of a value that is read again)")
                continue
            if base not in WGSL_OPS:
                op, base = _functional(op), base[:-1]
                schema = aten_schema(op)
        if base not in WGSL_OPS:
            bad.append(op)
            continue
        ins: list = []
        attrs: dict = {}
        for a, v in zip(schema.arguments, full):
            if is_tensor_list_type(a.type):
                ins.extend(ref(t) for t in v)
            elif is_tensor_type(a.type):
                if isinstance(v, torch.fx.Node):
                    ins.append(ref(v))
                elif v is None:
                    ins.append(None)
                else:                      # a Python number in a Tensor slot (x * 0.5)
                    attrs[a.name] = to_json(v)
            elif isinstance(v, torch.fx.Node):
                bad.append(f"{op} (dynamic non-tensor argument {a.name})")
                break
            else:
                attrs[a.name] = to_json(v)
        while ins and ins[-1] is None:
            ins.pop()
        names[n.name] = n.name
        nodes.append({"op": op, "inputs": ins, "attrs": attrs, "output": n.name})
    if bad:
        raise Unlowerable(sorted(set(bad)))
    out_node = next(n for n in ep.graph.nodes if n.op == "output")
    user = [o for o, s in zip(out_node.args[0], sig.output_specs) if s.kind == OutputKind.USER_OUTPUT]
    outs = [ref(o) if o.op == "placeholder" else names[o.name] for o in user]
    # friendly output names: "output" (single) or "output_<i>"; only for node outputs that appear once
    produced = {nd["output"] for nd in nodes}
    ren = {}
    for i, o in enumerate(outs):
        if o in produced and outs.count(o) == 1:
            ren[o] = "output" if len(outs) == 1 else f"output_{i}"
    taken = produced | {i["name"] for i in inputs} | set(weights)
    ren = {k: v for k, v in ren.items() if v not in taken or v == k}
    for nd in nodes:
        nd["inputs"] = [ren.get(v, v) if v is not None else None for v in nd["inputs"]]
        nd["output"] = ren.get(nd["output"], nd["output"])
    outs = [ren.get(o, o) for o in outs]
    graph = {"version": 1, "lowering_version": LOWERING_VERSION, "inputs": inputs, "nodes": nodes, "outputs": outs,
             "weights": WEIGHTS_FILE}
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
    except Exception as e:
        art.reason = f"torch.export failed ({type(e).__name__}: {brief(e)}); "
    if ep is not None:
        try:
            graph, weights = _emit_opgraph(ep)
            art.ops = sorted({n["op"] for n in graph["nodes"]})
        except Unlowerable as u:
            art.unsupported_ops = u.ops
            graph = weights = None
        if target == "wgsl" and graph is not None:
            art.kind, art.graph, art.weights = "opgraph", graph, weights
            return art
    try:
        art.onnx_bytes, art.kind = _export_onnx(module, x), "onnx"
        if target == "wgsl":
            art.reason += "ops outside the WGSL table; lowered to ONNX for onnxruntime-web"
    except Exception as e:
        art.reason += f"ONNX export failed ({type(e).__name__}: {brief(e)}); native-only"
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
