"""Model adapters (ADR-0113): state_dict/safetensors + registries, plugins, torch.export/TorchScript, ONNX, refusals."""
import io
import json
import os
import pickle
import zipfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from moregpu_worker.vision import adapters as A
from moregpu_worker.vision import fetch as FE
from moregpu_worker.vision.adapters import onnx as AO
from moregpu_worker.vision.adapters import plugins as P
from moregpu_worker.vision.adapters import statedict as SD

from _vision_models import (TinyNet, save_state_dict, sha256_file, spec_for, tiny_basic_unet, tiny_monai_unet,
                            tiny_vit)

UNET_ARCH = {"registry": "monai", "name": "UNet", "kwargs": {"spatial_dims": 3, "in_channels": 1, "out_channels": 2,
                                                            "channels": [4, 8, 16], "strides": [2, 2],
                                                            "num_res_units": 1}}


@pytest.fixture
def fetch(tmp_path):
    return FE.make_fetch(roots=[tmp_path])


def x3d(n=16, b=1):
    torch.manual_seed(1)
    return torch.randn(b, 1, n, n, n)


# ------------------------------------------------------------------ state_dict + registries
def test_monai_unet_state_dict_roundtrip(tmp_path, fetch):
    m = tiny_monai_unet()
    p = tmp_path / "unet.pt"
    save_state_dict(m, p)
    h = A.load(spec_for(p, "state_dict", UNET_ARCH), fetch)
    x = x3d()
    with torch.no_grad():
        ref = m(x)
    out = A.infer(h, x)
    assert torch.allclose(out, ref, atol=1e-6)
    d = A.describe(h)
    assert d["adapter"] == "state_dict" and d["native"] and d["sha256"] == sha256_file(p)
    assert d["arch"] == "monai:UNet" and d["params"] > 0 and d["unwrap"] == []
    assert isinstance(A.train_handle(h), nn.Module)
    json.dumps(d)
    A.unload(h)
    assert h.model is None


def test_numpy_batch_is_accepted(tmp_path, fetch):
    m = tiny_monai_unet()
    p = tmp_path / "unet.pt"
    save_state_dict(m, p)
    h = A.load(spec_for(p, "state_dict", UNET_ARCH), fetch)
    out = A.infer(h, x3d().numpy())
    assert isinstance(out, torch.Tensor) and out.shape == (1, 2, 16, 16, 16)


def test_basic_unet_safetensors(tmp_path, fetch):
    from safetensors.torch import save_file
    m = tiny_basic_unet()
    p = tmp_path / "bunet.safetensors"
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(p))
    arch = {"registry": "monai", "name": "BasicUNet",
            "kwargs": {"spatial_dims": 3, "in_channels": 1, "out_channels": 2, "features": [4, 4, 8, 8, 16, 4]}}
    h = A.load(spec_for(p, "safetensors", arch), fetch)
    x = x3d(32)
    with torch.no_grad():
        assert torch.allclose(A.infer(h, x), m(x), atol=1e-6)


def test_torchvision_resnet18(tmp_path, fetch):
    tv = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    m = tv.models.resnet18(weights=None, num_classes=5).eval()
    p = tmp_path / "r18.pth"
    save_state_dict(m, p)
    h = A.load(spec_for(p, "state_dict", {"registry": "torchvision", "name": "resnet18",
                                          "kwargs": {"num_classes": 5}}), fetch)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(A.infer(h, x), m(x), atol=1e-5)


def test_timm_vit_tiny(tmp_path, fetch):
    pytest.importorskip("timm")
    m = tiny_vit()
    p = tmp_path / "vit.pt"
    save_state_dict(m, p)
    h = A.load(spec_for(p, "state_dict", {"registry": "timm", "name": "vit_tiny_patch16_224",
                                          "kwargs": {"img_size": 32, "num_classes": 10}}), fetch)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(A.infer(h, x), m(x), atol=1e-5)


def test_hf_vision_model_from_config_dict(tmp_path, fetch):
    tr = pytest.importorskip("transformers")
    cfg = {"model_type": "vit", "image_size": 32, "patch_size": 16, "num_channels": 3, "hidden_size": 32,
           "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 64, "num_labels": 3}
    torch.manual_seed(0)
    c = tr.AutoConfig.for_model(**cfg)
    m = tr.AutoModelForImageClassification.from_config(c).eval()
    from safetensors.torch import save_file
    p = tmp_path / "vit_hf.safetensors"
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(p))
    h = A.load(spec_for(p, "safetensors", {"registry": "hf", "name": "AutoModelForImageClassification",
                                           "kwargs": {"config": cfg}}), fetch)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = A.infer(h, x)
        assert torch.allclose(out, m(pixel_values=x).logits, atol=1e-5)


def test_hf_refuses_non_auto_class(tmp_path, fetch):
    p = tmp_path / "w.safetensors"
    from safetensors.torch import save_file
    save_file({"a": torch.zeros(1)}, str(p))
    with pytest.raises(A.RefusedFormat, match="Auto"):
        A.load(spec_for(p, "safetensors", {"registry": "hf", "name": "ViTModel", "kwargs": {"config": {}}}), fetch)


def test_unknown_arch_name(tmp_path, fetch):
    m = tiny_monai_unet()
    p = tmp_path / "unet.pt"
    save_state_dict(m, p)
    with pytest.raises(A.UnknownArch):
        A.load(spec_for(p, "state_dict", {"registry": "monai", "name": "NoSuchNet"}), fetch)
    with pytest.raises(A.UnknownArch):
        A.load(spec_for(p, "state_dict", {"registry": "monai", "name": "_private"}), fetch)


@pytest.mark.parametrize("wrap, expect", [
    (lambda sd: {f"module.{k}": v for k, v in sd.items()}, ["prefix:module."]),
    (lambda sd: {"state_dict": sd, "epoch": 3}, ["key:state_dict"]),
    (lambda sd: {"model": {"state_dict": {f"module.{k}": v for k, v in sd.items()}}},
     ["key:model", "key:state_dict", "prefix:module."]),
    (lambda sd: {f"_orig_mod.{k}": v for k, v in sd.items()}, ["prefix:_orig_mod."]),
])
def test_common_containers_are_unwrapped_and_reported(tmp_path, fetch, wrap, expect):
    m = tiny_monai_unet()
    p = tmp_path / "unet.pt"
    save_state_dict(m, p, wrap=wrap)
    h = A.load(spec_for(p, "state_dict", UNET_ARCH), fetch)
    assert A.describe(h)["unwrap"] == expect
    x = x3d()
    with torch.no_grad():
        assert torch.allclose(A.infer(h, x), m(x), atol=1e-6)


def test_strict_load_reports_a_key_diff(tmp_path, fetch):
    m = tiny_monai_unet()
    sd = m.state_dict()
    k_missing = next(iter(sd))
    sd.pop(k_missing)
    sd["extra.weight"] = torch.zeros(3)
    k_shape = [k for k, v in sd.items() if v.ndim == 5][0]
    sd[k_shape] = torch.zeros(1, 1, 1, 1, 1)
    p = tmp_path / "bad.pt"
    torch.save(sd, p)
    with pytest.raises(A.KeyMismatch) as e:
        A.load(spec_for(p, "state_dict", UNET_ARCH), fetch)
    err = e.value
    assert err.missing == [k_missing] and err.unexpected == ["extra.weight"]
    assert [s[0] for s in err.shape_mismatch] == [k_shape]
    msg = str(err)
    assert "strict" in msg and k_missing in msg and "extra.weight" in msg and k_shape in msg and "(1, 1, 1, 1, 1)" in msg


def test_float16_dtype(tmp_path, fetch):
    m = tiny_monai_unet()
    p = tmp_path / "unet.pt"
    save_state_dict(m, p)
    h = A.load(spec_for(p, "state_dict", UNET_ARCH, dtype="bfloat16"), fetch)
    assert next(h.model.parameters()).dtype == torch.bfloat16
    assert A.infer(h, x3d()).dtype == torch.bfloat16


# ------------------------------------------------------------------ refusals
class _Evil:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def test_malicious_pickle_in_torch_zip_is_refused_and_not_executed(tmp_path, fetch):
    marker = tmp_path / "PWNED"
    p = tmp_path / "evil.pt"
    torch.save({"w": torch.zeros(2), "x": _Evil(marker)}, p)
    with pytest.raises(A.RefusedFormat) as e:
        A.load(spec_for(p, "state_dict", UNET_ARCH), fetch)
    assert not marker.exists()
    for word in ("state_dict", "safetensors", "torch.export", "ONNX", "plugin"):
        assert word in str(e.value)


def test_malicious_raw_pickle_is_refused_and_not_executed(tmp_path, fetch):
    marker = tmp_path / "PWNED2"
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(_Evil(marker)))
    with pytest.raises(A.RefusedFormat):
        A.load(spec_for(p, "state_dict", UNET_ARCH), fetch)
    assert not marker.exists()


def test_pickled_full_model_is_refused(tmp_path, fetch):
    p = tmp_path / "full.pt"
    torch.save(tiny_monai_unet(), p)
    with pytest.raises(A.RefusedFormat, match="export a state_dict"):
        A.load(spec_for(p, "state_dict", UNET_ARCH), fetch)


@pytest.mark.parametrize("obj", [[torch.zeros(1)], {"a": 1, "b": "x"}, {"a": {"b": torch.zeros(1)}}])
def test_non_tensor_dicts_are_refused(tmp_path, fetch, obj):
    p = tmp_path / "odd.pt"
    torch.save(obj, p)
    with pytest.raises(A.RefusedFormat, match="dict of tensors"):
        A.load(spec_for(p, "state_dict", UNET_ARCH), fetch)


def test_garbage_safetensors_refused(tmp_path, fetch):
    p = tmp_path / "junk.safetensors"
    p.write_bytes(b"not a safetensors file at all")
    with pytest.raises(A.RefusedFormat):
        A.load(spec_for(p, "safetensors", UNET_ARCH), fetch)


# ------------------------------------------------------------------ sha256 + fetch
def test_sha256_mismatch_refused_before_load(tmp_path, fetch, monkeypatch):
    m = tiny_monai_unet()
    p = tmp_path / "unet.pt"
    save_state_dict(m, p)
    s = spec_for(p, "state_dict", UNET_ARCH, sha256="0" * 64)
    called = []
    monkeypatch.setattr(SD, "read_state_dict", lambda *a, **k: called.append(1))
    with pytest.raises(A.IntegrityError, match="sha256"):
        A.load(s, fetch)
    assert not called


def test_file_source_outside_allowed_roots_refused(tmp_path):
    inside, outside = tmp_path / "in", tmp_path / "out"
    inside.mkdir(), outside.mkdir()
    p = outside / "w.pt"
    p.write_bytes(b"x")
    f = FE.make_fetch(roots=[inside])
    with pytest.raises(A.RefusedSource, match="allowed"):
        f(p.as_uri())
    (inside / "link.pt").symlink_to(p)
    with pytest.raises(A.RefusedSource):
        f((inside / "link.pt").as_uri())
    with pytest.raises(A.RefusedSource):
        FE.make_fetch(roots=[])(p.as_uri())


def test_default_fetch_uses_env_roots(tmp_path, monkeypatch):
    p = tmp_path / "w.pt"
    p.write_bytes(b"x")
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(tmp_path))
    assert FE.default_fetch(p.as_uri()) == p.resolve()
    monkeypatch.delenv("MOREGPU_MODEL_ROOTS")
    with pytest.raises(A.RefusedSource):
        FE.default_fetch(p.as_uri())


def test_pushed_source(tmp_path):
    (tmp_path / "blob-1").write_bytes(b"data")
    f = FE.make_fetch(pushed_dir=tmp_path)
    assert f("pushed://blob-1").read_bytes() == b"data"
    for bad in ("pushed://../etc/passwd", "pushed://a/b", "pushed://missing"):
        with pytest.raises(A.RefusedSource):
            f(bad)
    with pytest.raises(A.RefusedSource):
        FE.make_fetch()("pushed://blob-1")


def test_https_source_is_content_addressed(tmp_path, monkeypatch):
    body = b"weights-bytes"
    import hashlib
    sha = hashlib.sha256(body).hexdigest()
    calls = []

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(url, timeout=None):
        calls.append(url)
        return Resp(body)

    monkeypatch.setattr(FE.urllib.request, "urlopen", urlopen)
    f = FE.make_fetch(cache_dir=tmp_path)
    p = f("https://example.org/w.pt", sha)
    assert p.read_bytes() == body and p.name == sha
    f("https://example.org/w.pt", sha)
    assert len(calls) == 1  # cache hit, no second download
    with pytest.raises(A.RefusedSource, match="sha256"):
        f("https://example.org/w.pt")
    with pytest.raises(A.IntegrityError):
        f("https://example.org/other.pt", "1" * 64)
    assert not (tmp_path / ("1" * 64)).exists()


def test_hf_source_uses_hub_download(tmp_path, monkeypatch):
    got = {}
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"x")

    def fake(repo_id, filename, revision=None):
        got.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(target)

    monkeypatch.setattr(FE, "_hf_hub_download", fake)
    f = FE.make_fetch()
    assert f("hf://org/repo@v1.0/sub/model.safetensors") == target
    assert got == {"repo_id": "org/repo", "filename": "sub/model.safetensors", "revision": "v1.0"}
    f("hf://org/repo/model.safetensors")
    assert got["revision"] is None
    with pytest.raises(A.RefusedSource):
        f("hf://org")


def test_unknown_scheme_refused():
    with pytest.raises(A.RefusedSource):
        FE.make_fetch()("ftp://x/y")


# ------------------------------------------------------------------ plugins
class _Dist:
    def __init__(self, name, version, sha):
        self.metadata = {"Name": name}
        self.version = version
        self._sha = sha

    def read_text(self, f):
        if f == "direct_url.json" and self._sha:
            return json.dumps({"archive_info": {"hash": f"sha256={self._sha}"}})
        return None


class _EP:
    def __init__(self, name, dist):
        self.name, self.dist, self.value = name, dist, "tiny:build"

    def load(self):
        return lambda **kw: TinyNet(**kw)


def _plugin_env(tmp_path, monkeypatch, eps, allow):
    path = tmp_path / "models_allow.json"
    path.write_text(json.dumps(allow))
    monkeypatch.setenv("MOREGPU_MODEL_PLUGIN_ALLOWLIST", str(path))
    monkeypatch.setattr(P, "_entry_points", lambda: eps)
    return path


PIN = [{"dist": "tiny-models", "version": "1.2", "wheel_sha256": "ab" * 32}]


def test_pinned_plugin_loads_and_accepts_weights(tmp_path, fetch, monkeypatch):
    _plugin_env(tmp_path, monkeypatch, [_EP("tiny", _Dist("tiny-models", "1.2", "ab" * 32))], PIN)
    torch.manual_seed(0)
    h = A.load({"format": "plugin", "arch": {"registry": "plugin", "name": "tiny", "kwargs": {"classes": 3}}}, fetch)
    assert A.describe(h)["plugin"] == {"dist": "tiny-models", "version": "1.2", "wheel_sha256": "ab" * 32}
    assert A.infer(h, torch.randn(1, 3, 8, 8)).shape == (1, 3)
    ref = TinyNet(classes=3).eval()
    p = tmp_path / "tiny.pt"
    save_state_dict(ref, p)
    h2 = A.load(spec_for(p, "state_dict", {"registry": "plugin", "name": "tiny", "kwargs": {"classes": 3}}), fetch)
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        assert torch.allclose(A.infer(h2, x), ref(x), atol=1e-6)
    assert "tiny" in P.discover()[0]


@pytest.mark.parametrize("dist, why", [
    (_Dist("tiny-models", "1.3", "ab" * 32), "pinned version"),
    (_Dist("tiny-models", "1.2", "cd" * 32), "sha256"),
    (_Dist("tiny-models", "1.2", None), "sha256"),
    (_Dist("other-models", "1.2", "ab" * 32), "allowlist"),
])
def test_unpinned_wrong_version_or_hash_mismatched_plugins_refused(tmp_path, fetch, monkeypatch, dist, why):
    _plugin_env(tmp_path, monkeypatch, [_EP("tiny", dist)], PIN)
    with pytest.raises(A.RefusedPlugin, match=why):
        A.load({"format": "plugin", "arch": {"registry": "plugin", "name": "tiny"}}, fetch)


def test_train_task_allowlist_does_not_admit_model_plugins(tmp_path, fetch, monkeypatch):
    allow = tmp_path / "train_allow.json"
    allow.write_text(json.dumps(PIN))
    monkeypatch.setenv("MOREGPU_PLUGIN_ALLOWLIST", str(allow))
    monkeypatch.delenv("MOREGPU_MODEL_PLUGIN_ALLOWLIST", raising=False)
    monkeypatch.setattr(P, "_entry_points", lambda: [_EP("tiny", _Dist("tiny-models", "1.2", "ab" * 32))])
    with pytest.raises(A.RefusedPlugin):
        A.load({"format": "plugin", "arch": {"registry": "plugin", "name": "tiny"}}, fetch)


def test_unknown_plugin_name(tmp_path, fetch, monkeypatch):
    _plugin_env(tmp_path, monkeypatch, [], PIN)
    with pytest.raises(A.UnknownArch):
        A.load({"format": "plugin", "arch": {"registry": "plugin", "name": "nope"}}, fetch)


def test_real_entry_point_group_is_queried():
    assert P.GROUP == "moregpu.models"
    assert isinstance(list(P._entry_points()), list)


# ------------------------------------------------------------------ torch.export / TorchScript
def test_torch_export_roundtrip(tmp_path, fetch):
    m = tiny_monai_unet()
    x = x3d()
    ep = torch.export.export(m, (x,))
    p = tmp_path / "unet.pt2"
    torch.export.save(ep, str(p))
    h = A.load(spec_for(p, "torch_export"), fetch)
    with torch.no_grad():
        assert torch.allclose(A.infer(h, x), m(x), atol=1e-6)
    d = A.describe(h)
    assert d["adapter"] == "torch_export" and not d["native"]
    with pytest.raises(A.NotNative):
        A.train_handle(h)
    A.unload(h)


def _repack(src, dst, replace):
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w") as zo:
        for n in zi.namelist():
            data = zi.read(n)
            for suffix, fn in replace.items():
                if n.endswith(suffix):
                    data = fn(data)
            zo.writestr(n, data)


def test_torch_export_with_malicious_sample_inputs_is_refused(tmp_path, fetch):
    marker = tmp_path / "PWNED3"
    m = TinyNet().eval()
    ep = torch.export.export(m, (torch.randn(1, 3, 8, 8),))
    good = tmp_path / "good.pt2"
    torch.export.save(ep, str(good))
    evil_buf = io.BytesIO()
    torch.save((_Evil(marker),), evil_buf)
    bad = tmp_path / "bad.pt2"
    _repack(good, bad, {"sample_inputs/model.pt": lambda _: evil_buf.getvalue()})
    with pytest.raises(A.RefusedFormat):
        A.load(spec_for(bad, "torch_export"), fetch)
    assert not marker.exists()


def test_torch_export_with_pickled_weights_is_refused(tmp_path, fetch):
    m = TinyNet().eval()
    good = tmp_path / "good.pt2"
    torch.export.save(torch.export.export(m, (torch.randn(1, 3, 8, 8),)), str(good))

    def flip(data):
        cfg = json.loads(data)
        for meta in cfg["config"].values():
            meta["use_pickle"] = True
        return json.dumps(cfg).encode()

    bad = tmp_path / "bad.pt2"
    _repack(good, bad, {"model_weights_config.json": flip})
    with pytest.raises(A.RefusedFormat, match="pickle"):
        A.load(spec_for(bad, "torch_export"), fetch)
    bad2 = tmp_path / "bad2.pt2"
    _repack(good, bad2, {"model_constants_config.json": lambda _: json.dumps(
        {"config": {"c": {"path_name": "opaque_obj_0", "use_pickle": False}}}).encode()})
    with pytest.raises(A.RefusedFormat, match="constant"):
        A.load(spec_for(bad2, "torch_export"), fetch)


def test_torch_export_rejects_non_archive(tmp_path, fetch):
    p = tmp_path / "x.pt2"
    p.write_bytes(b"nope")
    with pytest.raises(A.RefusedFormat):
        A.load(spec_for(p, "torch_export"), fetch)


def test_torchscript_roundtrip(tmp_path, fetch):
    m = tiny_monai_unet()
    x = x3d()
    p = tmp_path / "unet.ts"
    torch.jit.save(torch.jit.trace(m, x), str(p))
    h = A.load(spec_for(p, "torchscript"), fetch)
    with torch.no_grad():
        assert torch.allclose(A.infer(h, x), m(x), atol=1e-6)
    assert A.describe(h)["adapter"] == "torchscript"
    with pytest.raises(A.NotNative):
        A.train_handle(h)


def test_torchscript_rejects_pickle(tmp_path, fetch):
    marker = tmp_path / "PWNED4"
    p = tmp_path / "evil.ts"
    p.write_bytes(pickle.dumps(_Evil(marker)))
    with pytest.raises(A.RefusedFormat):
        A.load(spec_for(p, "torchscript"), fetch)
    assert not marker.exists()


# ------------------------------------------------------------------ ONNX
def _onnx_file(m, x, path):
    torch.onnx.export(m, (x,), str(path), dynamo=False, opset_version=17, input_names=["input"],
                      output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})


def test_onnx_parity_resnet18(tmp_path, fetch):
    tv = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    m = tv.models.resnet18(weights=None, num_classes=10).eval()
    x = torch.randn(4, 3, 32, 32)
    p = tmp_path / "r18.onnx"
    _onnx_file(m, x[:1], p)
    h = A.load(spec_for(p, "onnx"), fetch)
    out = A.infer(h, x)
    with torch.no_grad():
        ref = m(x)
    assert (out - ref).abs().max().item() <= 1e-4
    assert (out.argmax(1) == ref.argmax(1)).float().mean().item() >= 0.999


def test_onnx_parity_monai_unet_argmax_agreement(tmp_path, fetch):
    m = tiny_monai_unet()
    x = x3d(16, b=2)
    p = tmp_path / "unet.onnx"
    _onnx_file(m, x[:1], p)
    h = A.load(spec_for(p, "onnx"), fetch)
    out = A.infer(h, x.numpy())
    with torch.no_grad():
        ref = m(x)
    assert (out - ref).abs().max().item() <= 1e-4
    assert (out.argmax(1) == ref.argmax(1)).float().mean().item() >= 0.999
    d = A.describe(h)
    assert d["adapter"] == "onnx" and d["providers"][-1] == "CPUExecutionProvider" and d["inputs"][0]["name"] == "input"
    with pytest.raises(A.NotNative):
        A.train_handle(h)
    A.unload(h)


def test_onnx_parity_timm_vit(tmp_path, fetch):
    pytest.importorskip("timm")
    m = tiny_vit()
    x = torch.randn(2, 3, 32, 32)
    p = tmp_path / "vit.onnx"
    _onnx_file(m, x[:1], p)
    h = A.load(spec_for(p, "onnx"), fetch)
    with torch.no_grad():
        assert (A.infer(h, x) - m(x)).abs().max().item() <= 1e-4


def test_onnx_provider_preference():
    assert AO.pick_providers(["CPUExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider"]) == [
        "CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert AO.pick_providers(["AzureExecutionProvider", "CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert AO.pick_providers([]) == ["CPUExecutionProvider"]


def test_onnx_garbage_refused(tmp_path, fetch):
    p = tmp_path / "x.onnx"
    p.write_bytes(b"garbage")
    with pytest.raises(A.RefusedFormat):
        A.load(spec_for(p, "onnx"), fetch)


# ------------------------------------------------------------------ inference modes
def test_sliding_window_matches_monai(tmp_path, fetch):
    from monai.inferers import sliding_window_inference
    m = tiny_monai_unet()
    p = tmp_path / "unet.pt"
    save_state_dict(m, p)
    inf = {"mode": "sliding_window", "sliding_window": {"roi": [16, 16, 16], "overlap": 0.25, "blend": "gaussian",
                                                        "sw_batch": 2}}
    h = A.load(spec_for(p, "state_dict", UNET_ARCH, inference=inf), fetch)
    x = x3d(24)
    with torch.no_grad():
        ref = sliding_window_inference(x, (16, 16, 16), 2, m, overlap=0.25, mode="gaussian")
    assert torch.allclose(A.infer(h, x), ref, atol=1e-5)


def test_flip_tta_averages_flips(tmp_path, fetch):
    m = tiny_monai_unet()
    p = tmp_path / "unet.pt"
    save_state_dict(m, p)
    h = A.load(spec_for(p, "state_dict", UNET_ARCH, inference={"mode": "full", "tta": "flip"}), fetch)
    x = x3d()
    with torch.no_grad():
        preds = [m(x)] + [m(x.flip(d)).flip(d) for d in (2, 3, 4)]
    assert torch.allclose(A.infer(h, x), torch.stack(preds).mean(0), atol=1e-5)


def test_from_module_wraps_in_memory_models():
    m = TinyNet().eval()
    h = A.from_module(m, name="tiny")
    assert A.describe(h)["native"] and len(h.sha256) == 64
    assert A.from_module(m).sha256 == h.sha256  # content hash of the weights
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        assert torch.allclose(A.infer(h, x), m(x))


def test_describe_after_unload_and_format_table():
    assert set(A.ADAPTERS) == {"state_dict", "safetensors", "plugin", "torch_export", "torchscript", "onnx"}
    h = A.from_module(TinyNet())
    A.unload(h)
    assert A.describe(h)["loaded"] is False
    with pytest.raises(A.NotLoaded):
        A.infer(h, torch.zeros(1, 3, 8, 8))


def test_invalid_spec_rejected_before_fetch():
    from moregpu_worker.vision.spec import SpecError
    with pytest.raises(SpecError):
        A.load({"format": "pickle", "source": "file:///x"}, lambda s, sha=None: pytest.fail("fetched"))


def test_placement_device_resolution():
    assert A.resolve_device({"device": "cpu"}) == "cpu"
    assert A.resolve_device({}) in ("cpu", "cuda", "mps")
    assert A.resolve_device(None) in ("cpu", "cuda", "mps")


def test_output_unwrapping():
    class O:
        logits = torch.ones(1)
    assert A.as_tensor(O()) is O.logits
    t = torch.zeros(1)
    assert A.as_tensor((t, 1)) is t and A.as_tensor({"out": t}) is t and A.as_tensor([t]) is t
    assert A.as_tensor(np.zeros(2)).shape == (2,)
