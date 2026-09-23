import base64, json, os

import numpy as np
import pytest
import torch

from moregpu_worker.data.plane import DataPlane
from moregpu_worker.data.refs import DataPolicy
from moregpu_worker.train import registry as R
from moregpu_worker.train.task import TaskContext
from moregpu_worker.vision.infer import InferenceStore

SEG = {"kind": "2p5d", "num_classes": 3, "encoder": {"init": "random", "model": "micro", "patch": 8},
       "decoder": {"channels": [16, 8]}, "synthetic": {"kind": "2p5d", "n": 8, "size": [32, 32], "channels": 3, "seed": 0}}


@pytest.fixture
def exported(tmp_path):
    t = R.create("segment"); t.init(SEG, TaskContext(device="cpu", amp="fp32"))
    for _ in range(20):
        t.inner_steps(list(range(8)), 1, 3e-3)
    t.export("safetensors", str(tmp_path / "model"))
    return t, str(tmp_path / "model")


def _plane(root):
    return DataPlane(DataPolicy(roots=[str(root)]))


def test_load_infer_and_describe(exported, tmp_path):
    t, path = exported
    st = InferenceStore(device="cpu", plane=_plane(tmp_path), out_root=str(tmp_path / "out"))
    d = st.handle("vision_infer_load", {"id": "m", "export": path})
    assert d["ok"] and d["task"] == "segment" and d["num_classes"] == 3
    x = t.data.batch([0]).numpy().astype("<f4")
    r = st.handle("vision_infer", {"id": "m", "shape": list(x.shape), "data": base64.b64encode(x.tobytes()).decode()})
    y = np.frombuffer(base64.b64decode(r["data"]), dtype="<f4").reshape(r["shape"])
    with torch.no_grad():
        ref = t.model.eval()(torch.from_numpy(x)).numpy()
    assert np.abs(y - ref).max() < 1e-4
    assert st.handle("vision_infer_describe", {"id": "m"})["kind"] == "2p5d"
    st.handle("vision_infer_unload", {"id": "m"})
    with pytest.raises(KeyError):
        st.handle("vision_infer", {"id": "m", "shape": [1], "data": ""})


def test_predict_volume_2p5d_writes_mask_and_scores(exported, tmp_path):
    t, path = exported
    root = tmp_path / "data"; root.mkdir()
    # build a volume from synthetic slabs: Z=6, each z uses sample (z % 8)'s centre slice
    imgs = t.data.batch(list(range(6)))           # (6, 3, 32, 32)
    vol = imgs[:, 1].numpy().astype("float32")    # (Z, H, W)
    gt = t.data.masks(list(range(6))).numpy().astype("uint8")
    np.save(root / "v.npy", vol); np.save(root / "v_gt.npy", gt)
    st = InferenceStore(device="cpu", plane=_plane(root), out_root=str(tmp_path / "out"))
    st.handle("vision_infer_load", {"id": "m", "export": path})
    r = st.handle("vision_predict", {"id": "m", "ref": {"uri": "file://v.npy"}, "mask": {"uri": "file://v_gt.npy"},
                                     "out": "case_v", "tta": "flip", "sw_batch": 4})
    assert r["ok"] and r["shape"] == [6, 32, 32]
    pred = np.load(r["path"])
    assert pred.shape == (6, 32, 32) and pred.dtype == np.uint8
    assert set(r["dice"]) == {"1", "2"} and 0 <= r["dice"]["1"] <= 1
    assert r["timings"]["compute_s"] > 0
    with pytest.raises(PermissionError):
        st.handle("vision_predict", {"id": "m", "ref": {"uri": "file://v.npy"}, "out": "../escape"})


def test_predict_volume_3d_sliding_window(tmp_path):
    cfg = {**SEG, "kind": "3d", "synthetic": {"kind": "3d", "n": 2, "size": [16, 16, 16], "channels": 1, "seed": 0},
           "encoder": {"init": "random", "model": "micro", "patch": [4, 8, 8]}}
    t = R.create("segment"); t.init(cfg, TaskContext(device="cpu", amp="fp32"))
    t.export("safetensors", str(tmp_path / "m3"))
    root = tmp_path / "d"; root.mkdir()
    np.save(root / "big.npy", np.random.default_rng(0).standard_normal((20, 24, 18)).astype("float32"))
    st = InferenceStore(device="cpu", plane=_plane(root), out_root=str(tmp_path / "o"))
    st.handle("vision_infer_load", {"id": "m3", "export": str(tmp_path / "m3")})
    r = st.handle("vision_predict", {"id": "m3", "ref": {"uri": "file://big.npy"}, "out": "big", "overlap": 0.5})
    assert r["shape"] == [20, 24, 18]


def test_encoder_export_features(tmp_path):
    j = R.create("jepa_2p5d")
    j.init({"model": "micro", "patch": 8, "pred_dim": 16, "pred_depth": 1, "pred_heads": 2, "synthetic": SEG["synthetic"]},
           TaskContext(device="cpu", amp="fp32"))
    j.export("safetensors", str(tmp_path / "enc"))
    st = InferenceStore(device="cpu", plane=None, out_root=str(tmp_path))
    d = st.handle("vision_infer_load", {"id": "e", "export": str(tmp_path / "enc")})
    assert d["task"] == "encoder"
    x = j.data.batch([0]).numpy().astype("<f4")
    r = st.handle("vision_infer", {"id": "e", "shape": list(x.shape), "data": base64.b64encode(x.tobytes()).decode(), "pool": "mean"})
    assert r["shape"] == [1, 32]


def test_published_model_via_adapter_predicts_volume(tmp_path, monkeypatch):
    """A third-party published model (MONAI UNet state_dict, loaded by the M4 adapters) runs the same whole-volume
    prediction path as MoreGPU's own exports."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from _vision_models import save_state_dict, spec_for, tiny_monai_unet
    from moregpu_worker.vision import ops as MO
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(tmp_path))
    m = tiny_monai_unet(); p = tmp_path / "unet.pt"; save_state_dict(m, p)
    arch = {"registry": "monai", "name": "UNet", "kwargs": {"spatial_dims": 3, "in_channels": 1, "out_channels": 2,
            "channels": [4, 8, 16], "strides": [2, 2], "num_res_units": 1}}
    MO.handle("vision_load", {"id": "pub", "spec": spec_for(p, "state_dict", arch)})
    root = tmp_path / "d"; root.mkdir()
    vol = np.random.default_rng(0).standard_normal((16, 16, 16)).astype("float32"); np.save(root / "v.npy", vol)
    st = InferenceStore(device="cpu", plane=_plane(root), out_root=str(tmp_path / "o"))
    st.handle("vision_infer_load", {"id": "pub", "handle": "pub", "task": "segment", "num_classes": 2, "kind": "3d"})
    r = st.handle("vision_predict", {"id": "pub", "ref": {"uri": "file://v.npy"}, "out": "pub_v"})
    with torch.no_grad():
        ref = m.eval()(torch.from_numpy(vol)[None, None]).argmax(1)[0].numpy()
    assert (np.load(r["path"]) == ref).mean() > 0.999
    MO.handle("vision_unload", {"id": "pub"})
