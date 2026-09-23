"""Automatic lowering hooks (ADR-0114): op-graph for WGSL, ONNX for onnxruntime-web, native-only fallback, parity probe."""
import json

import numpy as np
import pytest
import torch
from safetensors.torch import load as st_load

from moregpu_worker.vision import adapters as A
from moregpu_worker.vision import lowering as L
from moregpu_worker.vision import opgraph_ref as OG

from _vision_models import FftNet, TinyNet, tiny_basic_unet, tiny_monai_unet, tiny_vit


@pytest.fixture(autouse=True)
def _fresh_cache():
    L.clear_cache()
    yield
    L.clear_cache()


def _x2d(b=1, c=3, n=8):
    torch.manual_seed(3)
    return torch.randn(b, c, n, n)


def test_op_table_covers_the_adr_list():
    for op in ("aten.convolution.default", "aten.conv2d.default", "aten.conv3d.default", "aten.relu.default",
               "aten.add.Tensor", "aten.mul.Tensor", "aten.linear.default", "aten.addmm.default", "aten.mm.default",
               "aten.layer_norm.default", "aten.native_group_norm.default", "aten.instance_norm.default",
               "aten.batch_norm.default", "aten.max_pool2d.default", "aten.avg_pool2d.default",
               "aten.upsample_nearest2d.vec", "aten.upsample_trilinear3d.vec", "aten.cat.default",
               "aten._softmax.default", "aten.sigmoid.default", "aten.gelu.default", "aten.permute.default",
               "aten.view.default", "aten.reshape.default", "aten.argmax.default"):
        assert op in L.WGSL_OPS, op
    assert set(L.WGSL_OPS) <= set(OG.OPS), "the reference interpreter must implement every declared op"
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
    assert set(d["ops"]) <= set(L.WGSL_OPS)


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


def test_vit_falls_back_to_onnx_and_reports_unsupported_ops():
    pytest.importorskip("timm")
    m = tiny_vit()
    x = torch.randn(1, 3, 32, 32)
    art = L.lower(A.from_module(m), "wgsl", example=x)
    assert art.kind == "onnx" and art.servable
    assert "aten.scaled_dot_product_attention.default" in art.unsupported_ops
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


def test_opgraph_reference_rejects_unknown_ops():
    g = {"version": 1, "inputs": ["x"], "params": {}, "nodes": [{"name": "y", "op": "aten.fft_rfft2.default",
                                                                   "args": [{"ref": "x"}]}], "outputs": [{"ref": "y"}]}
    with pytest.raises(KeyError, match="fft"):
        OG.run(g, {}, torch.zeros(1))


def test_opgraph_reference_ops_match_torch():
    """Spot-check reference semantics for ops not exercised by the model tests."""
    t = torch.randn(2, 4, 6, 6)
    R = OG.OPS
    assert torch.allclose(R["aten.avg_pool2d.default"](t, [2, 2]), torch.nn.functional.avg_pool2d(t, 2))
    assert torch.equal(R["aten.argmax.default"](t, 1), t.argmax(1))
    assert torch.allclose(R["aten.addmm.default"](torch.ones(3), torch.eye(3), torch.eye(3)), torch.eye(3) + 1)
    assert torch.allclose(R["aten.mm.default"](torch.eye(2), torch.ones(2, 2)), torch.ones(2, 2))
    assert torch.allclose(R["aten.upsample_bilinear2d.vec"](t, None, False, [2.0, 2.0]),
                          torch.nn.functional.interpolate(t, scale_factor=2.0, mode="bilinear"))
    t3 = torch.randn(1, 2, 4, 4, 4)
    assert torch.allclose(R["aten.upsample_trilinear3d.vec"](t3, [8, 8, 8], False, None),
                          torch.nn.functional.interpolate(t3, size=(8, 8, 8), mode="trilinear"))
    w = torch.randn(4, 2, 3, 3)
    assert torch.allclose(R["aten.convolution.default"](t, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2),
                          torch.nn.functional.conv2d(t, w, None, 1, 1, 1, 2))
    wt = torch.randn(4, 3, 2, 2)
    assert torch.allclose(R["aten.convolution.default"](t, wt, None, [2, 2], [0, 0], [1, 1], True, [0, 0], 1),
                          torch.nn.functional.conv_transpose2d(t, wt, stride=2))
    out, mean, rstd = R["aten.native_group_norm.default"](t, None, None, 2, 4, 36, 2, 1e-5)
    assert torch.allclose(out, torch.nn.functional.group_norm(t, 2), atol=1e-5)
    assert torch.allclose(R["aten.batch_norm.default"](t, None, None, torch.zeros(4), torch.ones(4), False, 0.1,
                                                       1e-5, True), t / (1 + 1e-5) ** 0.5, atol=1e-5)
    assert torch.allclose(R["aten._softmax.default"](t, 1, False), t.softmax(1))
    assert torch.allclose(R["aten.softmax.int"](t, 1), t.softmax(1))
    assert R["aten.reshape.default"](t, [2, -1]).shape == (2, 144)
    assert R["aten.flatten.using_ints"](t, 1, -1).shape == (2, 144)
    assert R["aten.transpose.int"](t, 1, 2).shape == (2, 6, 4, 6)
    assert torch.equal(R["aten.sub.Tensor"](t, t, 1), torch.zeros_like(t))
    assert torch.allclose(R["aten.div.Tensor"](t, 2.0), t / 2)
    assert torch.equal(R["aten.dropout.default"](t, 0.5, False), t)
    assert torch.allclose(R["aten.mean.dim"](t, [2, 3], True), t.mean((2, 3), keepdim=True))
    assert torch.allclose(R["aten.tanh.default"](t), t.tanh())
    assert R["aten.constant_pad_nd.default"](t, [1, 1], 0.0).shape == (2, 4, 6, 8)
    assert R["aten.upsample_nearest2d.default"](t, [12, 12]).shape == (2, 4, 12, 12)
    assert R["aten.avg_pool3d.default"](t3, [2, 2, 2]).shape == (1, 2, 2, 2, 2)
    assert R["aten.adaptive_avg_pool3d.default"](t3, [1, 1, 1]).shape == (1, 2, 1, 1, 1)
    assert R["aten.upsample_nearest3d.vec"](t3, None, [2.0, 2.0, 2.0]).shape == (1, 2, 8, 8, 8)
    assert R["aten.conv1d.default"](torch.randn(1, 2, 5), torch.randn(3, 2, 1)).shape == (1, 3, 5)
    assert R["aten.conv_transpose2d.input"](t, wt, None, [2, 2]).shape == (2, 3, 12, 12)
    assert R["aten.hardtanh.default"](t, 0.0, 6.0).max() <= 6
    assert R["aten.silu.default"](t).shape == t.shape
    assert R["aten.clone.default"](t) is not t
    assert R["aten.contiguous.default"](t).is_contiguous()
    assert R["aten.unsqueeze.default"](t, 0).shape == (1, 2, 4, 6, 6)
    assert R["aten.squeeze.dim"](t[:1], 0).shape == (4, 6, 6)


def test_decode_encode_special_values():
    for v in (float("inf"), float("-inf"), torch.float16, torch.device("cpu"), [1, (2, 3)], None, "s",
              torch.contiguous_format, torch.strided):
        enc = L.encode_arg(v)
        json.dumps(enc)
        dec = OG.decode_arg(enc, {})
        if isinstance(v, tuple | list):
            assert dec == [1, [2, 3]]
        else:
            assert dec == v
    nan = OG.decode_arg(L.encode_arg(float("nan")), {})
    assert nan != nan
    with pytest.raises(TypeError):
        L.encode_arg(object())
