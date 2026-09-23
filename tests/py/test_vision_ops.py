"""Worker RPC surface for model adapters: moregpu_worker.vision.ops.handle(op, payload)."""
import base64
import json

import pytest

from moregpu_worker.vision import lowering as L
from moregpu_worker.vision import ops
from moregpu_worker.vision.spec import SpecError

from _vision_models import save_state_dict, spec_for, tiny_monai_unet

ARCH = {"registry": "monai", "name": "UNet", "kwargs": {"spatial_dims": 3, "in_channels": 1, "out_channels": 2,
                                                       "channels": [4, 8, 16], "strides": [2, 2], "num_res_units": 1}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    ops.HANDLES.clear()
    L.clear_cache()
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(tmp_path))
    monkeypatch.delenv("MOREGPU_MODEL_PLUGIN_ALLOWLIST", raising=False)
    yield
    ops.HANDLES.clear()


def test_models_describe_lists_capabilities():
    d = ops.handle("vision_models_describe", {})
    json.dumps(d)
    assert set(d["formats"]) == {"state_dict", "safetensors", "plugin", "torch_export", "torchscript", "onnx"}
    assert d["registries"]["monai"] is True and "torchvision" in d["registries"]
    assert d["lowering"]["targets"] == ["wgsl", "onnx-web"] and "aten.conv2d.default" in d["lowering"]["wgsl_ops"]
    assert d["plugins"] == {"available": [], "refused": {}}
    assert d["loaded"] == []
    assert "CPUExecutionProvider" in d["onnx_providers"]


def test_load_describe_lower_unload_cycle(tmp_path):
    m = tiny_monai_unet()
    p = tmp_path / "unet.pt"
    save_state_dict(m, p)
    spec = spec_for(p, "state_dict", ARCH, io={"inputs": [{"name": "x", "shape": [None, 1, 16, 16, 16]}]})
    r = ops.handle("vision_load", {"id": "seg", "spec": spec})
    assert r["id"] == "seg" and r["adapter"] == "state_dict"
    assert "seg" in ops.HANDLES
    assert ops.handle("vision_describe", {"id": "seg"})["arch"] == "monai:UNet"
    assert ops.handle("vision_models_describe", {})["loaded"] == ["seg"]
    lo = ops.handle("vision_lower", {"id": "seg", "target": "wgsl"})
    assert lo["kind"] == "opgraph" and lo["servable"] and "graph" not in lo
    full = ops.handle("vision_lower", {"id": "seg", "target": "wgsl", "include_bytes": True})
    assert json.loads(full["graph_json"])["version"] == 1 and base64.b64decode(full["weights_b64"])
    assert ops.lowered("seg", "wgsl").servable
    on = ops.handle("vision_lower", {"id": "seg", "target": "onnx-web", "include_bytes": True,
                                     "example_shape": [1, 1, 16, 16, 16]})
    assert on["kind"] == "onnx" and base64.b64decode(on["onnx_b64"])
    assert ops.handle("vision_unload", {"id": "seg"}) == {"id": "seg", "unloaded": True}
    assert "seg" not in ops.HANDLES
    with pytest.raises(KeyError):
        ops.lowered("seg", "wgsl")


def test_errors():
    with pytest.raises(KeyError, match="unknown vision op"):
        ops.handle("vision_nope", {})
    with pytest.raises(KeyError, match="not loaded"):
        ops.handle("vision_describe", {"id": "missing"})
    with pytest.raises(SpecError):
        ops.handle("vision_load", {"id": "x", "spec": {"format": "pickle"}})
    with pytest.raises(ValueError, match="id"):
        ops.handle("vision_load", {"spec": {}})
    assert ops.handle("vision_unload", {"id": "never"}) == {"id": "never", "unloaded": False}


def test_reload_same_id_replaces(tmp_path):
    m = tiny_monai_unet()
    p = tmp_path / "unet.pt"
    save_state_dict(m, p)
    s = spec_for(p, "state_dict", ARCH)
    ops.handle("vision_load", {"id": "a", "spec": s})
    first = ops.HANDLES["a"]
    ops.handle("vision_load", {"id": "a", "spec": s})
    assert ops.HANDLES["a"] is not first and first.model is None


def test_ops_are_listed():
    assert set(ops.OPS) == {"vision_models_describe", "vision_load", "vision_describe", "vision_lower",
                            "vision_unload"}
