#!/usr/bin/env python3
"""PyTorch goldens for the WebGPU vision executor (ADR-0114, milestone M6).

Writes to tests/goldens/wgsl/:

* kernels.json: one small case per kernel or op. Each case is a tiny op-graph (exported with the minimal
  torch.export → op-graph exporter below), its tensors (base64 little-endian f32), its inputs, and the
  PyTorch eager outputs.
* <model>.graph.json + <model>.safetensors + <model>.io.json: whole-model parity cases for a tiny MONAI BasicUNet
  (3D), a small 2D seg net (BN + bilinear + cat) and a ViT-Tiny-width encoder (2 blocks). The op-graph
  follows the documented schema (docs/WEBGPU_VISION.md).
* sliding_window.json: monai.inferers.sliding_window_inference outputs, with the UNet op-graph as the predictor.

The exporter here is intentionally small. It exists so these goldens do not depend on the sibling lowering
(apps/worker/moregpu_worker/vision/lowering.py), but it emits the SAME schema, which is pinned in
apps/worker/vision_ops.json:

* op: the ATen overload name (e.g. "aten.conv3d.default").
* inputs: the tensor arguments in ATen-schema order. An absent optional tensor is null, and a Tensor[] argument
  (aten.cat) is flattened in place.
* attrs: every non-tensor argument, keyed by its ATen schema name, with defaults filled in. A Python number
  passed in a Tensor slot (e.g. x * 0.5) goes to attrs under that argument's name.

Deterministic: fixed seeds, CPU, float32. Run:  python3 tests/goldens/make_wgsl_goldens.py
"""
from __future__ import annotations

import base64
import json
import operator
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wgsl")


# ───────────────────────────── helpers ─────────────────────────────
def b64(t: torch.Tensor) -> str:
    a = t.detach().to(torch.float32).contiguous().cpu().numpy().astype("<f4")
    return base64.b64encode(a.tobytes()).decode("ascii")


def tjson(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "b64": b64(t)}


def _jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return str(v)
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, torch.dtype):
        return str(v).replace("torch.", "")
    if isinstance(v, (torch.device, torch.layout, torch.memory_format)):
        return None
    if isinstance(v, torch.SymInt):
        return int(v)
    raise TypeError(f"unsupported attr value {v!r} ({type(v)})")


# ─────────────────────── torch.export → op-graph ───────────────────────
def export_opgraph(model: nn.Module, example: tuple, input_names: list[str], output_names: list[str] | None = None,
                   core_aten: bool = False):
    """Minimal exporter: torch.export (training IR; core_aten=True also runs the default core-ATen decompositions)
    → MoreGPU op-graph v1 + weights dict."""
    model = model.eval()
    ep = torch.export.export(model, example)
    if core_aten:
        ep = ep.run_decompositions()
    sig = ep.graph_signature
    p2w = {}
    p2w.update(sig.inputs_to_parameters)
    p2w.update(sig.inputs_to_buffers)
    p2w.update(getattr(sig, "inputs_to_lifted_tensor_constants", {}) or {})
    state = dict(ep.state_dict)
    state.update(getattr(ep, "constants", {}) or {})
    weights: dict[str, torch.Tensor] = {}
    names: dict[str, str] = {}  # fx node name → op-graph value name
    inputs = []
    user_i = 0
    for n in ep.graph.nodes:
        if n.op != "placeholder":
            continue
        if n.name in p2w:
            wname = p2w[n.name]
            names[n.name] = wname
            weights[wname] = state[wname].detach().to(torch.float32)
        else:
            nm = input_names[user_i]
            user_i += 1
            names[n.name] = nm
            inputs.append({"name": nm, "shape": list(n.meta["val"].shape)})
    nodes = []
    for n in ep.graph.nodes:
        if n.op != "call_function":
            continue
        if n.target is operator.getitem:
            src, idx = n.args
            if idx != 0:
                # extra outputs of native_* ops (mean/rstd/indices) are unsupported unless unused
                if len(n.users) == 0:
                    continue
                raise RuntimeError(f"getitem[{idx}] of {src} is used — multi-output ops are unsupported")
            names[n.name] = names[src.name]
            continue
        schema = n.target._schema
        ins: list = []
        attrs: dict = {}
        for i, arg in enumerate(schema.arguments):
            if i < len(n.args):
                v = n.args[i]
            elif arg.name in n.kwargs:
                v = n.kwargs[arg.name]
            elif arg.has_default_value():
                v = arg.default_value
            else:
                v = None
            ty = str(arg.type)
            is_list = ty in ("Tensor[]", "List[Tensor]", "List[Optional[Tensor]]")
            if is_list or ty in ("Tensor", "Optional[Tensor]", "Tensor?"):
                if is_list:
                    ins.extend(names[x.name] for x in v)
                elif isinstance(v, torch.fx.Node):
                    ins.append(names[v.name])
                elif v is None:
                    ins.append(None)
                else:  # a python number in a Tensor slot (x * 0.5)
                    attrs[arg.name] = _jsonable(v)
            elif isinstance(v, torch.fx.Node):
                raise RuntimeError(f"dynamic non-tensor arg {arg.name} in {n.target}")
            else:
                attrs[arg.name] = _jsonable(v)
        # drop trailing null optional tensors so e.g. conv without bias is [x, w]
        while ins and ins[-1] is None:
            ins.pop()
        names[n.name] = n.name
        nodes.append({"op": str(n.target), "inputs": ins, "attrs": attrs, "output": n.name})
    out_node = [n for n in ep.graph.nodes if n.op == "output"][0]
    outs = [names[a.name] for a in out_node.args[0]]
    if output_names:
        # rename final values to friendly names (rewrite the producing node + any later consumers)
        ren = dict(zip(outs, output_names))
        for nd in nodes:
            nd["inputs"] = [ren.get(x, x) if x is not None else None for x in nd["inputs"]]
            nd["output"] = ren.get(nd["output"], nd["output"])
        outs = [ren.get(o, o) for o in outs]
    graph = {"version": 1, "inputs": inputs, "nodes": nodes, "outputs": outs}
    return graph, weights


def run_eager(model, example):
    with torch.no_grad():
        y = model.eval()(*example)
    if not isinstance(y, (tuple, list)):
        y = (y,)
    return [t.to(torch.float32) for t in y]


# ─────────────────────────── kernel cases ───────────────────────────
class Fn(nn.Module):
    """Wrap a function of (inputs..., params...) as a module so torch.export lifts params as weights."""

    def __init__(self, fn, **params):
        super().__init__()
        self.fn = fn
        for k, v in params.items():
            self.register_parameter(k, nn.Parameter(v, requires_grad=False))

    def forward(self, *xs):
        return self.fn(self, *xs)


def R(*shape, scale=1.0):
    return torch.randn(*shape) * scale


def kernel_cases():
    torch.manual_seed(1234)
    C = []

    def add(name, module, xs, tol=1e-5):
        C.append((name, module, xs, tol))

    # ── convolution (implicit GEMM) ──
    add("conv2d_3x3_pad1", nn.Conv2d(3, 5, 3, padding=1), [R(2, 3, 7, 6)])
    add("conv2d_stride2_dil2_groups2_nobias", nn.Conv2d(4, 6, 3, stride=2, padding=2, dilation=2, groups=2, bias=False), [R(1, 4, 11, 9)])
    add("conv2d_1x1", nn.Conv2d(8, 3, 1), [R(1, 8, 5, 5)])
    add("conv2d_asym_pad", nn.Conv2d(2, 4, (3, 2), stride=(1, 2), padding=(1, 0)), [R(1, 2, 6, 7)])
    add("conv2d_depthwise", nn.Conv2d(4, 4, 3, padding=1, groups=4), [R(1, 4, 6, 6)])
    add("conv3d_3x3x3_pad1", nn.Conv3d(2, 4, 3, padding=1), [R(1, 2, 5, 6, 7)])
    add("conv3d_stride_dil_groups", nn.Conv3d(4, 6, 3, stride=(2, 1, 2), padding=(1, 2, 1), dilation=(1, 2, 1), groups=2), [R(2, 4, 7, 8, 6)])
    add("patch_embed_conv2d_k4s4", nn.Conv2d(3, 16, 4, stride=4), [R(1, 3, 16, 16)])
    add("conv_transpose2d_k3s2p1op1", nn.ConvTranspose2d(4, 3, 3, stride=2, padding=1, output_padding=1), [R(1, 4, 5, 4)])
    add("conv_transpose2d_groups_dil", nn.ConvTranspose2d(4, 6, 3, stride=2, padding=2, dilation=2, groups=2), [R(1, 4, 5, 5)])
    add("conv_transpose3d_k2s2", nn.ConvTranspose3d(4, 2, 2, stride=2), [R(1, 4, 3, 4, 3)])
    add("conv_transpose3d_k3s2p1op1", nn.ConvTranspose3d(2, 3, 3, stride=2, padding=1, output_padding=1, bias=False), [R(2, 2, 3, 3, 4)])
    # ── norms ──
    bn2 = nn.BatchNorm2d(5)
    bn2.running_mean.copy_(R(5)); bn2.running_var.copy_(torch.rand(5) + 0.5)
    bn2.weight.data.copy_(R(5)); bn2.bias.data.copy_(R(5))
    add("batch_norm2d_eval", bn2, [R(2, 5, 4, 6)])
    bn3 = nn.BatchNorm3d(3, eps=1e-3)
    bn3.running_mean.copy_(R(3)); bn3.running_var.copy_(torch.rand(3) + 0.5)
    add("batch_norm3d_eval", bn3, [R(1, 3, 4, 5, 3)])
    in3 = nn.InstanceNorm3d(4, affine=True)
    in3.weight.data.copy_(R(4)); in3.bias.data.copy_(R(4))
    add("instance_norm3d_affine", in3, [R(2, 4, 5, 6, 7, scale=3.0) + 1.5])
    add("instance_norm2d_noaffine", nn.InstanceNorm2d(3), [R(1, 3, 9, 8)])
    gn = nn.GroupNorm(2, 6)
    gn.weight.data.copy_(R(6)); gn.bias.data.copy_(R(6))
    add("group_norm3d_g2", gn, [R(2, 6, 3, 4, 5)])
    add("group_norm2d_g4_noaffine", nn.GroupNorm(4, 8, affine=False), [R(1, 8, 5, 5)])
    ln = nn.LayerNorm(24, eps=1e-6)
    ln.weight.data.copy_(R(24)); ln.bias.data.copy_(R(24))
    add("layer_norm_lastdim", ln, [R(2, 5, 24, scale=2.0)])
    ln2 = nn.LayerNorm([4, 6])
    ln2.weight.data.copy_(R(4, 6)); ln2.bias.data.copy_(R(4, 6))
    add("layer_norm_last2", ln2, [R(3, 4, 6)])
    add("layer_norm_wide_row", nn.LayerNorm(700), [R(3, 700)])
    # ── pooling ──
    add("max_pool2d_k3s2p1", nn.MaxPool2d(3, 2, 1), [R(1, 3, 9, 8)])
    add("max_pool2d_dil2", nn.MaxPool2d(2, 1, 0, dilation=2), [R(1, 2, 6, 6)])
    add("max_pool3d_k2", nn.MaxPool3d(2), [R(1, 3, 6, 4, 8)])
    add("avg_pool2d_k3s2p1_incl", nn.AvgPool2d(3, 2, 1), [R(1, 3, 7, 8)])
    add("avg_pool2d_k3s2p1_excl", nn.AvgPool2d(3, 2, 1, count_include_pad=False), [R(1, 3, 7, 8)])
    add("avg_pool3d_k2", nn.AvgPool3d(2), [R(1, 2, 4, 6, 4)])
    add("avg_pool3d_k3s2p1", nn.AvgPool3d(3, 2, 1), [R(1, 2, 5, 5, 6)])
    # ── upsample ──
    add("upsample_nearest2d_x2", nn.Upsample(scale_factor=2, mode="nearest"), [R(1, 3, 4, 5)])
    add("upsample_nearest3d_size", nn.Upsample(size=(5, 7, 6), mode="nearest"), [R(1, 2, 3, 4, 4)])
    add("upsample_bilinear2d_x2", nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), [R(1, 3, 4, 5)])
    add("upsample_bilinear2d_size_ac", nn.Upsample(size=(7, 9), mode="bilinear", align_corners=True), [R(1, 2, 4, 5)])
    add("upsample_bilinear2d_down", nn.Upsample(size=(3, 4), mode="bilinear", align_corners=False), [R(1, 2, 7, 9)])
    add("upsample_trilinear3d_x2", nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False), [R(1, 2, 3, 4, 3)])
    add("upsample_trilinear3d_size", nn.Upsample(size=(5, 7, 4), mode="trilinear", align_corners=False), [R(1, 2, 3, 4, 3)])
    # ── concat / data movement ──
    add("cat_channels_2", Fn(lambda s, a, b: torch.cat([a, b], dim=1)), [R(1, 3, 4, 4, 2), R(1, 5, 4, 4, 2)])
    add("cat_channels_3_batch2", Fn(lambda s, a, b, c: torch.cat([a, b, c], dim=1)), [R(2, 2, 3, 3), R(2, 1, 3, 3), R(2, 3, 3, 3)])
    add("cat_lastdim", Fn(lambda s, a, b: torch.cat([a, b], dim=-1)), [R(2, 3, 4), R(2, 3, 2)])
    add("permute_5d", Fn(lambda s, a: a.permute(0, 2, 3, 4, 1).contiguous()), [R(1, 3, 2, 4, 5)])
    add("transpose_view_flatten", Fn(lambda s, a: a.flatten(2).transpose(1, 2).reshape(2, -1)), [R(1, 4, 3, 3)])
    add("select_slice_mean", Fn(lambda s, a: (a[:, 0], a[:, 1:].mean(dim=1))), [R(2, 5, 6)])
    add("expand_cat_add", Fn(lambda s, a: torch.cat([s.cls.expand(a.shape[0], -1, -1), a], dim=1) + s.pos, cls=R(1, 1, 6), pos=R(1, 4, 6)), [R(2, 3, 6)])
    add("pad_constant_2d", Fn(lambda s, a: F.pad(a, (1, 2, 0, 1), value=0.5)), [R(1, 2, 3, 4)])
    add("pad_replicate_3d", Fn(lambda s, a: F.pad(a, (1, 0, 2, 1, 0, 1), mode="replicate")), [R(1, 2, 3, 3, 4)])
    # ── elementwise ──
    add("add_broadcast", Fn(lambda s, a, b: a + b), [R(2, 3, 4, 5), R(3, 1, 1)])
    add("sub_mul_div", Fn(lambda s, a, b: (a - b) * a / (b.abs() + 1.0)), [R(2, 3, 4), R(2, 3, 4)])
    add("add_scalar_mul_scalar", Fn(lambda s, a: (a + 0.25) * 3.0), [R(3, 7)])
    add("add_alpha", Fn(lambda s, a, b: torch.add(a, b, alpha=0.5)), [R(4, 5), R(5)])
    add("relu", nn.ReLU(), [R(3, 5, 7)])
    add("leaky_relu_0p2", nn.LeakyReLU(0.2), [R(3, 5, 7)])
    add("gelu_erf", nn.GELU(), [R(4, 64, scale=3.0)])
    add("gelu_tanh", nn.GELU(approximate="tanh"), [R(4, 64, scale=3.0)])
    add("sigmoid", nn.Sigmoid(), [R(3, 40, scale=4.0)])
    add("tanh", nn.Tanh(), [R(3, 40, scale=4.0)])
    add("silu", nn.SiLU(), [R(3, 40, scale=4.0)])
    add("prelu_channels", nn.PReLU(4, init=0.1), [R(2, 4, 3, 3)])
    # ── softmax / argmax ──
    add("softmax_channels_3d", nn.Softmax(dim=1), [R(2, 4, 3, 4, 5, scale=3.0)])
    add("softmax_lastdim", nn.Softmax(dim=-1), [R(3, 5, 17, scale=3.0)])
    add("argmax_channels", Fn(lambda s, a: torch.argmax(a, dim=1)), [R(2, 5, 3, 4)])
    add("argmax_channels_keepdim", Fn(lambda s, a: torch.argmax(a, dim=1, keepdim=True)), [R(1, 3, 2, 3, 4)])
    # ── linear / matmul / attention ──
    add("linear_3d_input", nn.Linear(24, 10), [R(2, 5, 24)])
    add("linear_nobias", nn.Linear(33, 7, bias=False), [R(3, 33)])
    add("addmm", Fn(lambda s, a: torch.addmm(s.b, a, s.w), w=R(20, 9), b=R(9)), [R(6, 20)])
    add("matmul_batched_bcast", Fn(lambda s, a, b: torch.matmul(a, b)), [R(2, 3, 5, 7), R(3, 7, 4)])
    add("bmm", Fn(lambda s, a, b: torch.bmm(a, b)), [R(3, 17, 18), R(3, 18, 19)])
    add("sdpa_vit", Fn(lambda s, q, k, v: F.scaled_dot_product_attention(q, k, v)), [R(1, 3, 5, 64), R(1, 3, 5, 64), R(1, 3, 5, 64)])
    add("sdpa_scale_long", Fn(lambda s, q, k, v: F.scaled_dot_product_attention(q, k, v, scale=0.3)), [R(2, 2, 37, 16), R(2, 2, 41, 16), R(2, 2, 41, 8)])
    return C


def supported_ops() -> set[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "..", "apps", "worker", "vision_ops.json")) as f:
        return set(json.load(f)["ops"])


def base_op(op: str) -> str:
    parts = op.split(".")
    return ".".join(parts[:2])


def build_kernel_goldens():
    """Every case in training IR; plus a '@core' variant (core-ATen decomposed) when the decomposed graph only uses
    ops listed in apps/worker/vision_ops.json — this pins the aten.convolution / _softmax / native_* forms too."""
    supported = supported_ops()
    cases = []
    for name, module, xs, tol in kernel_cases():
        in_names = [f"x{i}" for i in range(len(xs))]
        outs = run_eager(module, tuple(xs))
        variants = [(name, export_opgraph(module, tuple(xs), in_names))]
        try:
            g2 = export_opgraph(module, tuple(xs), in_names, core_aten=True)
            if all(base_op(n["op"]) in supported for n in g2[0]["nodes"]) and \
                    {n["op"] for n in g2[0]["nodes"]} != {n["op"] for n in variants[0][1][0]["nodes"]}:
                variants.append((name + "@core", g2))
        except Exception:  # noqa: BLE001 — a decomposition we cannot represent (e.g. aten.index with None) → skip
            pass
        for vname, (graph, weights) in variants:
            cases.append(case_json(vname, graph, weights, in_names, xs, outs, tol))
    return cases


def case_json(name, graph, weights, in_names, xs, outs, tol):
    if True:
        return ({
            "name": name,
            "graph": graph,
            "tensors": {k: tjson(v) for k, v in weights.items()},
            "inputs": {nm: tjson(x) for nm, x in zip(in_names, xs)},
            "expected": {o: tjson(t) for o, t in zip(graph["outputs"], outs)},
            "tol": tol,
            "ops": sorted({n["op"] for n in graph["nodes"]}),
        })


# ─────────────────────────── model cases ───────────────────────────
class UNetSeg(nn.Module):
    """Tiny MONAI BasicUNet (3D) + channel softmax + argmax segmentation head."""

    def __init__(self):
        super().__init__()
        from monai.networks.nets import BasicUNet

        self.net = BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(4, 4, 8, 8, 16, 4))

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs, torch.argmax(probs, dim=1, keepdim=True)


class Seg2D(nn.Module):
    """Small 2D encoder-decoder: conv-BN-ReLU, maxpool, GroupNorm, bilinear upsample, skip cat, sigmoid head."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.b1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.g2 = nn.GroupNorm(4, 16)
        self.c3 = nn.Conv2d(24, 8, 3, padding=1)
        self.head = nn.Conv2d(8, 2, 1)
        for bn in [self.b1]:
            bn.running_mean.copy_(torch.randn(8) * 0.1)
            bn.running_var.copy_(torch.rand(8) + 0.5)

    def forward(self, x):
        e1 = F.relu(self.b1(self.c1(x)))
        e2 = F.gelu(self.g2(self.c2(F.max_pool2d(e1, 2))))
        u = F.interpolate(e2, scale_factor=2, mode="bilinear", align_corners=False)
        d = F.leaky_relu(self.c3(torch.cat([e1, u], dim=1)), 0.1)
        return torch.sigmoid(self.head(d))


class Block(nn.Module):
    def __init__(self, d, h, mlp):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, eps=1e-6)
        self.fc1 = nn.Linear(d, mlp)
        self.fc2 = nn.Linear(mlp, d)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(self.n1(x)).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class ViTTiny(nn.Module):
    """ViT-Tiny width (dim 192, 3 heads, MLP 768), depth 2, 32×32 input / 16×16 patches. Returns class logits and
    the mean-pooled patch features (the JEPA feature-extraction path)."""

    def __init__(self, d=192, depth=2, h=3, mlp=768, img=32, p=16, nc=10):
        super().__init__()
        self.pe = nn.Conv2d(3, d, p, p)
        n = (img // p) ** 2
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, h, mlp) for _ in range(depth)])
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.head = nn.Linear(d, nc)

    def forward(self, x):
        x = self.pe(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return self.head(x[:, 0]), x[:, 1:].mean(dim=1)


def save_model(name, model, example, out_names):
    from safetensors.torch import save_file

    graph, weights = export_opgraph(model, example, ["x"], out_names)
    graph["weights"] = f"{name}.safetensors"
    outs = run_eager(model, example)
    save_file({k: v.contiguous() for k, v in weights.items()}, os.path.join(OUT, f"{name}.safetensors"))
    with open(os.path.join(OUT, f"{name}.graph.json"), "w") as f:
        json.dump(graph, f, indent=1)
    io = {"inputs": {"x": tjson(example[0])}, "expected": {o: tjson(t) for o, t in zip(out_names, outs)}}
    with open(os.path.join(OUT, f"{name}.io.json"), "w") as f:
        json.dump(io, f)
    print(f"  {name}: {len(graph['nodes'])} nodes, {sum(v.numel() for v in weights.values())} params, ops={sorted({n['op'] for n in graph['nodes']})}")
    return model


def build_sliding_window(unet: UNetSeg):
    from monai.inferers import sliding_window_inference

    cases = []
    torch.manual_seed(7)
    for name, shape, roi, overlap, mode in [
        ("gaussian_overlap0p25", (1, 1, 40, 36, 44), (32, 32, 32), 0.25, "gaussian"),
        ("constant_pad_small_dim", (1, 1, 24, 40, 34), (32, 32, 32), 0.5, "constant"),
    ]:
        x = torch.randn(*shape)
        with torch.no_grad():
            y = sliding_window_inference(x, roi, 1, lambda w: unet.net(w), overlap=overlap, mode=mode, sigma_scale=0.125)
        cases.append({"name": name, "roi": list(roi), "overlap": overlap, "mode": mode, "sigma_scale": 0.125,
                      "input": tjson(x), "expected": tjson(y.as_tensor() if hasattr(y, "as_tensor") else y)})
    return cases


def main():
    os.makedirs(OUT, exist_ok=True)
    print("kernel cases…")
    kc = build_kernel_goldens()
    with open(os.path.join(OUT, "kernels.json"), "w") as f:
        json.dump({"version": 1, "torch": torch.__version__, "cases": kc}, f)
    print(f"  {len(kc)} cases")
    print("models…")
    torch.manual_seed(42)
    unet = UNetSeg().eval()
    # non-trivial instance-norm affine params (default init is 1/0)
    with torch.no_grad():
        for m in unet.modules():
            if isinstance(m, nn.InstanceNorm3d) and m.affine:
                m.weight.add_(torch.randn_like(m.weight) * 0.2)
                m.bias.add_(torch.randn_like(m.bias) * 0.2)
    save_model("unet3d_tiny", unet, (torch.randn(1, 1, 32, 32, 32),), ["logits", "probs", "labels"])
    torch.manual_seed(43)
    save_model("seg2d_tiny", Seg2D().eval(), (torch.randn(1, 3, 16, 20),), ["mask"])
    torch.manual_seed(44)
    save_model("vit_tiny", ViTTiny().eval(), (torch.randn(1, 3, 32, 32),), ["logits", "features"])
    print("sliding window…")
    sw = build_sliding_window(unet)
    with open(os.path.join(OUT, "sliding_window.json"), "w") as f:
        json.dump({"version": 1, "model": "unet3d_tiny", "output": "logits", "cases": sw}, f)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
