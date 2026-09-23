"""Automatic lowering hooks (ADR-0114): op-graph for WGSL, ONNX for onnxruntime-web, native-only fallback, parity probe."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load as st_load

from moregpu_worker.vision import adapters as A
from moregpu_worker.vision import lowering as L
from moregpu_worker.vision import opgraph_ref as OG

from _vision_models import CumsumNet, FftNet, InplaceReuseNet, TinyNet, tiny_basic_unet, tiny_monai_unet, tiny_vit

VISION_OPS_JSON = Path(__file__).resolve().parents[2] / "apps" / "worker" / "vision_ops.json"
CONTRACT = json.loads(VISION_OPS_JSON.read_text())["ops"]


@pytest.fixture(autouse=True)
def _fresh_cache():
    L.clear_cache()
    yield
    L.clear_cache()


def _x2d(b=1, c=3, n=8):
    torch.manual_seed(3)
    return torch.randn(b, c, n, n)


def test_op_table_is_the_executors_contract():
    """WGSL_OPS is read from apps/worker/vision_ops.json (the executor's machine-readable contract), by base name."""
    contract = json.loads(VISION_OPS_JSON.read_text())
    assert L.WGSL_OPS == frozenset(contract["ops"])
    for op in ("aten.convolution", "aten.conv2d", "aten.conv3d", "aten.relu", "aten.add", "aten.mul", "aten.linear",
               "aten.addmm", "aten.layer_norm", "aten.instance_norm", "aten.batch_norm", "aten.max_pool2d",
               "aten.upsample_trilinear3d", "aten.cat", "aten._softmax", "aten.gelu", "aten.permute", "aten.view",
               "aten.argmax", "aten.scaled_dot_product_attention", "aten.adaptive_avg_pool2d"):
        assert op in L.WGSL_OPS, op
    assert L.base_op("aten.add.Tensor") == "aten.add" and L.base_op("aten.relu") == "aten.relu"
    assert L.LOWERING_VERSION


def test_tiny_cnn_lowers_to_opgraph_with_parity():
    m = TinyNet().eval()
    h = A.from_module(m)
    x = _x2d()
    art = L.lower(h, "wgsl", example=x)
    assert art.kind == "opgraph" and art.servable and art.unsupported_ops == []
    assert art.parity["max_abs"] <= 1e-4 and art.parity["ok"]
    g = json.loads(json.dumps(art.graph))
    w = st_load(art.weights_bytes())
    out = OG.run(g, w, x)
    with torch.no_grad():
        assert (out - m(x)).abs().max().item() <= 1e-4
    assert torch.allclose(art.run(x), out)
    d = art.describe()
    json.dumps(d)
    assert d["kind"] == "opgraph" and d["target"] == "wgsl" and len(d["artifact_sha256"]) == 64
    assert {L.base_op(o) for o in d["ops"]} <= set(L.WGSL_OPS)
    assert_executor_schema(art.graph, w)


def test_opgraph_handles_other_batch_sizes_via_numpy():
    m = TinyNet().eval()
    art = L.lower(A.from_module(m), "wgsl", example=_x2d())
    x = _x2d(b=1).numpy()
    out = OG.run(art.graph, art.weights, x)
    with torch.no_grad():
        assert (out - m(torch.from_numpy(x))).abs().max().item() <= 1e-4


def test_monai_unets_lower_to_opgraph():
    for m, n in ((tiny_monai_unet(), 16), (tiny_basic_unet(), 32)):
        torch.manual_seed(0)
        x = torch.randn(1, 1, n, n, n)
        art = L.lower(A.from_module(m), "wgsl", example=x)
        assert art.kind == "opgraph", art.unsupported_ops
        assert art.parity["max_abs"] <= 1e-4
        with torch.no_grad():
            ref = m(x)
        assert (art.run(x).argmax(1) == ref.argmax(1)).float().mean() >= 0.999


def test_resnet18_lowers_to_opgraph():
    tv = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    m = tv.models.resnet18(weights=None, num_classes=10).eval()
    art = L.lower(A.from_module(m), "wgsl", example=torch.randn(1, 3, 32, 32))
    assert art.kind == "opgraph" and art.servable, art.unsupported_ops


def test_timm_vit_now_lowers_to_opgraph():
    """SDPA / unbind (→ select) / layer_norm are in the executor's table, so a timm ViT no longer needs ONNX."""
    pytest.importorskip("timm")
    m = tiny_vit()
    x = torch.randn(1, 3, 32, 32)
    art = L.lower(A.from_module(m), "wgsl", example=x)
    assert art.kind == "opgraph" and art.servable, (art.unsupported_ops, art.reason)
    assert art.parity["max_abs"] <= 1e-4
    assert_executor_schema(art.graph, art.weights)


def test_op_outside_the_table_falls_back_to_onnx_and_reports_it():
    m = CumsumNet().eval()
    x = torch.randn(1, 2, 6, 6)
    art = L.lower(A.from_module(m), "wgsl", example=x)
    assert art.kind == "onnx" and art.servable
    assert "aten.cumsum.default" in art.unsupported_ops
    assert art.parity["max_abs"] <= 1e-4
    import onnxruntime as ort
    s = ort.InferenceSession(art.onnx_bytes, providers=["CPUExecutionProvider"])
    out = s.run(None, {s.get_inputs()[0].name: x.numpy()})[0]
    with torch.no_grad():
        assert np.abs(out - m(x).numpy()).max() <= 1e-4


def test_onnx_web_target():
    m = TinyNet().eval()
    art = L.lower(A.from_module(m), "onnx-web", example=_x2d())
    assert art.kind == "onnx" and art.servable and art.target == "onnx-web"
    assert art.describe()["bytes"] == len(art.onnx_bytes)


def test_fft_model_is_native_only_with_unsupported_ops():
    m = FftNet().eval()
    art = L.lower(A.from_module(m), "wgsl", example=torch.randn(1, 1, 8, 8))
    assert art.kind == "native" and not art.servable
    assert any("fft" in op for op in art.unsupported_ops), art.unsupported_ops
    assert "onnx" in art.reason.lower()
    with pytest.raises(L.ParityRefused):
        art.run(torch.randn(1, 1, 8, 8))


def test_corrupted_lowered_weight_fails_parity_and_refuses_serving():
    m = TinyNet().eval()
    h = A.from_module(m)
    x = _x2d()
    art = L.lower(h, "wgsl", example=x)
    assert art.servable
    k = next(k for k, v in art.weights.items() if v.ndim == 4)
    art.weights[k] = art.weights[k] + 0.5
    rep = L.probe(art, h, example=x)
    assert not rep["ok"] and rep["max_abs"] > 1e-4
    assert not art.servable and "parity" in art.reason
    with pytest.raises(L.ParityRefused):
        art.run(x)


def test_lower_refuses_when_emitted_artifact_disagrees(monkeypatch):
    orig = L._emit_opgraph

    def corrupt(ep):
        g, w = orig(ep)
        k = next(iter(w))
        w[k] = w[k] * 3 + 1
        return g, w

    monkeypatch.setattr(L, "_emit_opgraph", corrupt)
    art = L.lower(A.from_module(TinyNet().eval()), "wgsl", example=_x2d())
    assert art.kind == "opgraph" and not art.servable and not art.parity["ok"]


def test_corrupted_onnx_fails_parity():
    m = TinyNet().eval()
    h = A.from_module(m)
    art = L.lower(h, "onnx-web", example=_x2d())
    import onnx
    from onnx import numpy_helper
    mp = onnx.load_from_string(art.onnx_bytes)
    init = mp.graph.initializer[0]
    arr = numpy_helper.to_array(init).copy() + 1.0
    init.CopyFrom(numpy_helper.from_array(arr, init.name))
    art.onnx_bytes = mp.SerializeToString()
    assert not L.probe(art, h, example=_x2d())["ok"]


def test_cache_is_content_addressed(monkeypatch):
    m = TinyNet().eval()
    h = A.from_module(m)
    a1 = L.lower(h, "wgsl", example=_x2d())
    monkeypatch.setattr(L.torch.export, "export", lambda *a, **k: pytest.fail("cache miss"))
    assert L.lower(A.from_module(m), "wgsl", example=_x2d()) is a1
    assert a1.cache_key == L.cache_key(h.sha256, "wgsl", (1, 3, 8, 8))
    assert L.cache_key(h.sha256, "onnx-web", (1, 3, 8, 8)) != a1.cache_key
    assert L.cache_key(h.sha256, "wgsl", (2, 3, 8, 8)) != a1.cache_key


def test_disk_cache_roundtrip(tmp_path):
    m = TinyNet().eval()
    h = A.from_module(m)
    x = _x2d()
    art = L.lower(h, "wgsl", example=x, cache_dir=tmp_path)
    L.clear_cache()
    again = L.lower(h, "wgsl", example=x, cache_dir=tmp_path)
    assert again is not art and again.cached and again.kind == "opgraph" and again.servable
    assert torch.allclose(again.run(x), art.run(x))
    o = L.lower(h, "onnx-web", example=x, cache_dir=tmp_path)
    L.clear_cache()
    o2 = L.lower(h, "onnx-web", example=x, cache_dir=tmp_path)
    assert o2.cached and o2.onnx_bytes == o.onnx_bytes and o2.servable


def test_example_from_spec_io_and_bad_target():
    m = TinyNet().eval()
    h = A.from_module(m, spec={"io": {"inputs": [{"name": "x", "shape": [None, 3, 8, 8], "dtype": "float32"}]}})
    assert L.lower(h, "wgsl").servable
    with pytest.raises(ValueError):
        L.lower(h, "metal")
    with pytest.raises(ValueError, match="example"):
        L.lower(A.from_module(m), "wgsl")


def test_lowering_non_native_handles(tmp_path):
    from moregpu_worker.vision import fetch as FE
    from _vision_models import spec_for
    m = TinyNet().eval()
    x = _x2d()
    p = tmp_path / "t.onnx"
    torch.onnx.export(m, (x,), str(p), dynamo=False, opset_version=17)
    fetch = FE.make_fetch(roots=[tmp_path])
    h = A.load(spec_for(p, "onnx"), fetch)
    art = L.lower(h, "wgsl", example=x)
    assert art.kind == "onnx" and art.servable and art.onnx_bytes == p.read_bytes()
    p2 = tmp_path / "t.pt2"
    torch.export.save(torch.export.export(m, (x,)), str(p2))
    art2 = L.lower(A.load(spec_for(p2, "torch_export"), fetch), "wgsl", example=x)
    assert art2.kind == "opgraph" and art2.servable


# ───────────────────────── executor schema (apps/worker/vision_ops.json, docs/WEBGPU_VISION.md) ─────────────────────────
def assert_executor_schema(graph: dict, weights: dict) -> None:
    """The lowered graph is EXACTLY the WGSL executor's schema: {version, inputs:[{name,shape}], nodes:[{op, inputs,
    attrs, output}], outputs:[name], weights}, ops in the table, attrs keyed by ATen schema argument names, no getitem."""
    g = json.loads(json.dumps(graph, allow_nan=False))  # JSON round-trip: plain, strict-JSON data only
    assert g["version"] == 1 and g["weights"] == "model.safetensors"
    assert all(set(i) == {"name", "shape"} and all(isinstance(d, int) for d in i["shape"]) for i in g["inputs"])
    assert all(isinstance(o, str) for o in g["outputs"])
    defined = {i["name"] for i in g["inputs"]} | set(weights)
    for n in g["nodes"]:
        assert set(n) == {"op", "inputs", "attrs", "output"}, n
        assert n["op"].startswith("aten.") and L.base_op(n["op"]) in CONTRACT, n["op"]
        for v in n["inputs"]:
            assert v is None or v in defined, (n, v)
        names = {a.name for a in L.aten_schema(n["op"]).arguments}
        assert set(n["attrs"]) <= names, (n["op"], set(n["attrs"]) - names)
        assert n["output"] not in defined, f"{n['output']} defined twice"
        defined.add(n["output"])
    for o in g["outputs"]:
        assert o in defined


def _segvit_micro():
    from moregpu_worker.models.vit import vit_config
    from moregpu_worker.vision import models as VM
    torch.manual_seed(5)
    return VM.build("segment", vit_config("micro", (32, 32), 8, 3), 3).eval()


def _vit_tiny_width():
    from moregpu_worker.models.vit import VisionTransformer
    torch.manual_seed(6)
    return VisionTransformer(img_size=(32, 32), patch=16, in_chans=3, embed_dim=192, depth=2, heads=3).eval()


@pytest.mark.parametrize("name,make,shape", [
    ("basic_unet3d", tiny_basic_unet, (1, 1, 32, 32, 32)),
    ("segvit_micro", _segvit_micro, (1, 3, 32, 32)),
    ("vit_tiny_width", _vit_tiny_width, (1, 3, 32, 32)),
])
def test_models_lower_to_the_executor_schema(name, make, shape):
    m = make()
    torch.manual_seed(1)
    x = torch.randn(*shape)
    art = L.lower(A.from_module(m), "wgsl", example=x)
    assert art.kind == "opgraph" and art.servable, (art.unsupported_ops, art.reason)
    assert art.parity["max_abs"] <= 1e-4 and art.parity["tol"] == 1e-4
    w = st_load(art.weights_bytes())
    assert_executor_schema(art.graph, w)
    # the reference interpreter runs the SAME schema (from JSON + safetensors bytes only)
    with torch.no_grad():
        ref = m(x)
    out = OG.run(json.loads(art.graph_json()), w, x)
    assert (out - ref).abs().max().item() <= 1e-5 * max(1.0, ref.abs().max().item())


def test_default_dialect_no_core_aten_decomposition(monkeypatch):
    """torch.export.export default (training-IR) dialect: high-level ops survive (instance_norm, conv3d, SDPA,
    upsample_bilinear2d) and run_decompositions is never called."""
    from torch.export import ExportedProgram
    monkeypatch.setattr(ExportedProgram, "run_decompositions", lambda *a, **k: pytest.fail("decomposition ran"))
    art = L.lower(A.from_module(tiny_basic_unet()), "wgsl", example=torch.randn(1, 1, 32, 32, 32))
    ops = {n["op"] for n in art.graph["nodes"]}
    assert "aten.instance_norm.default" in ops and "aten.conv3d.default" in ops
    art2 = L.lower(A.from_module(_segvit_micro()), "wgsl", example=torch.randn(1, 3, 32, 32))
    ops2 = {n["op"] for n in art2.graph["nodes"]}
    assert "aten.scaled_dot_product_attention.default" in ops2 and "aten.upsample_bilinear2d.vec" in ops2


def test_unbind_getitem_becomes_select_nodes():
    art = L.lower(A.from_module(_segvit_micro()), "wgsl", example=torch.randn(1, 3, 32, 32))
    ops = [n["op"] for n in art.graph["nodes"]]
    assert not any("unbind" in o or "getitem" in o for o in ops)
    sel = [n for n in art.graph["nodes"] if n["op"] == "aten.select.int"]
    assert len(sel) >= 3 and {n["attrs"]["index"] for n in sel} == {0, 1, 2} and all(n["attrs"]["dim"] == 0 for n in sel)


def test_multi_output_getitem0_maps_to_the_node_output():
    class LN(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.native_layer_norm.default(x, [6], None, None, 1e-5)[0] * 2.0
    art = L.lower(A.from_module(LN().eval()), "wgsl", example=torch.randn(2, 6))
    assert art.kind == "opgraph" and art.servable, art.reason
    n0, n1 = art.graph["nodes"]
    assert n0["op"] == "aten.native_layer_norm.default" and n1["inputs"] == [n0["output"]] and n1["attrs"]["other"] == 2.0

    class Bad(torch.nn.Module):
        def forward(self, x):
            y, mean, rstd = torch.ops.aten.native_layer_norm.default(x, [6], None, None, 1e-5)
            return y + mean
    art2 = L.lower(A.from_module(Bad().eval()), "wgsl", example=torch.randn(2, 6))
    assert art2.kind != "opgraph" and any("getitem" in u for u in art2.unsupported_ops), art2.unsupported_ops


def test_attrs_match_the_reference_exporter_on_the_golden_vit():
    """The lowering emits the same nodes (op, attrs, input structure) as the executor goldens' reference exporter."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "goldens"))
    import make_wgsl_goldens as MG
    torch.manual_seed(44)
    m = MG.ViTTiny().eval()
    x = torch.randn(1, 3, 32, 32)
    ref_graph, _ = MG.export_opgraph(m, (x,), ["x"])
    art = L.lower(A.from_module(m), "wgsl", example=x)
    assert art.servable
    got = art.graph["nodes"]
    assert [n["op"] for n in got] == [n["op"] for n in ref_graph["nodes"]]
    assert [n["attrs"] for n in got] == [n["attrs"] for n in ref_graph["nodes"]]
    assert [[v is None for v in n["inputs"]] for n in got] == [[v is None for v in n["inputs"]] for n in ref_graph["nodes"]]


def test_inplace_ops_are_functionalised_or_refused():
    tv = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    m = tv.models.resnet18(weights=None, num_classes=10).eval()
    art = L.lower(A.from_module(m), "wgsl", example=torch.randn(1, 3, 32, 32))
    assert art.kind == "opgraph" and art.servable
    ops = {n["op"] for n in art.graph["nodes"]}
    assert "aten.add.Tensor" in ops and "aten.add_.Tensor" not in ops           # add_ → add (not in the table)
    assert "aten.adaptive_avg_pool2d.default" in ops
    # a value that is mutated in place and READ AGAIN afterwards cannot be functionalised safely
    art2 = L.lower(A.from_module(InplaceReuseNet().eval()), "wgsl", example=torch.randn(1, 4))
    assert art2.kind != "opgraph" and any("in-place" in u for u in art2.unsupported_ops), art2.unsupported_ops


def test_non_finite_attrs_use_json_safe_spellings():
    class P(torch.nn.Module):
        def forward(self, x):  # -inf padding then a max pool: the output is finite
            return torch.nn.functional.max_pool2d(torch.nn.functional.pad(x, (1, 1), value=float("-inf")), (1, 3), 1)
    art = L.lower(A.from_module(P().eval()), "wgsl", example=torch.randn(1, 2, 2, 3))
    assert art.servable, art.reason
    pad = next(n for n in art.graph["nodes"] if L.base_op(n["op"]) in ("aten.pad", "aten.constant_pad_nd"))
    assert pad["attrs"]["value"] == "-Infinity"
    json.loads(json.dumps(art.graph, allow_nan=False))
    padded = OG.run({**art.graph, "outputs": [pad["output"]]}, {}, torch.zeros(1, 2, 2, 3))
    assert torch.isneginf(padded[..., 0]).all() and torch.isneginf(padded[..., -1]).all()


def test_opgraph_reference_executes_a_hand_written_executor_graph():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 6, 6)
    w = {"c.weight": torch.randn(4, 2, 3, 3), "c.bias": torch.randn(4)}
    w0 = {k: v.clone() for k, v in w.items()}
    g = {"version": 1, "inputs": [{"name": "x", "shape": [1, 2, 6, 6]}], "weights": "model.safetensors", "nodes": [
        {"op": "aten.conv2d.default", "inputs": ["x", "c.weight", "c.bias"],
         "attrs": {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1}, "output": "c"},
        {"op": "aten.mul.Tensor", "inputs": ["c"], "attrs": {"other": 0.5}, "output": "m"},
        {"op": "aten.cat", "inputs": ["c", "m"], "attrs": {"dim": 1}, "output": "cat"},
        {"op": "aten.scaled_dot_product_attention.default", "inputs": ["cat", "cat", "cat"],
         "attrs": {"dropout_p": 0.0, "is_causal": False, "scale": 0.25, "enable_gqa": False}, "output": "a"},
        {"op": "aten.relu_.default", "inputs": ["a"], "attrs": {}, "output": "y"},
        {"op": "aten.adaptive_avg_pool2d.default", "inputs": ["y"], "attrs": {"output_size": [3, 2]}, "output": "p"},
    ], "outputs": ["p"]}
    F = torch.nn.functional
    c = F.conv2d(x, w["c.weight"], w["c.bias"], padding=1)
    cat = torch.cat([c, c * 0.5], 1)
    ref = F.adaptive_avg_pool2d(F.relu(F.scaled_dot_product_attention(cat, cat, cat, scale=0.25)), (3, 2))
    assert torch.allclose(OG.run(g, w, x), ref, atol=1e-6)
    assert all(torch.equal(w[k], w0[k]) for k in w)


def test_opgraph_reference_rejects_ops_outside_the_table():
    g = {"version": 1, "inputs": [{"name": "x", "shape": [1]}], "nodes": [
        {"op": "aten.fft_rfft2.default", "inputs": ["x"], "attrs": {}, "output": "y"}], "outputs": ["y"]}
    with pytest.raises(KeyError, match="fft"):
        OG.run(g, {}, torch.zeros(1))
    with pytest.raises(ValueError, match="version"):
        OG.run({**g, "version": 2}, {}, torch.zeros(1))


