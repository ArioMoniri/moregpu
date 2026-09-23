"""pred_sha256 (moregpu.pred/1): a deterministic hash of a predicted label volume, identical in Python (worker) and
TypeScript (coordinator) — tests/goldens/pred_sha256.json is checked by both sides."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from moregpu_worker.data.plane import DataPlane
from moregpu_worker.data.refs import DataPolicy
from moregpu_worker.train import registry as R
from moregpu_worker.train.task import TaskContext
from moregpu_worker.vision import pred_hash as P
from moregpu_worker.vision.infer import InferenceStore

G = json.loads((Path(__file__).parents[1] / "goldens" / "pred_sha256.json").read_text())


@pytest.mark.parametrize("case", G["cases"], ids=[c["name"] for c in G["cases"]])
def test_golden_cases(case):
    a = np.asarray(case["labels"], dtype=np.int64).reshape(tuple(case["shape"]))
    assert P.label_dtype(a) == case["dtype"]
    assert P.preimage(a).hex() == case["preimage_hex"]
    assert P.pred_sha256(a) == case["sha256"]
    # the input integer dtype and memory layout do not matter, only the values and the shape
    for dt in (np.uint8, np.int32, np.uint16, np.int64):
        if a.size and a.max() > np.iinfo(dt).max:
            continue
        assert P.pred_sha256(a.astype(dt)) == case["sha256"]
    assert P.pred_sha256(torch.from_numpy(a)) == case["sha256"]


def test_golden_logits_argmax():
    lg = G["logits"]
    x = np.asarray([np.nan if v is None else v for v in lg["logits"]], dtype=np.float32).reshape(lg["shape"])
    assert bytes.fromhex(lg["logits_f32le_hex"]) == x.astype("<f4").tobytes()
    labels = P.labels_from_logits(x)
    assert labels.shape == tuple(lg["labels_shape"]) and labels.ravel().tolist() == lg["labels"]
    assert P.pred_sha256(labels) == lg["sha256"]
    assert P.pred_sha256(P.labels_from_logits(torch.from_numpy(x))) == lg["sha256"]


def test_non_c_contiguous_input_hashes_in_c_order():
    a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4) % 5
    f = np.asfortranarray(a)
    assert P.pred_sha256(f) == P.pred_sha256(a)
    assert P.pred_sha256(a.transpose(1, 0, 2)) != P.pred_sha256(a)          # a different volume


def test_refusals():
    with pytest.raises(ValueError, match="65535"):
        P.pred_sha256(np.array([70000]))
    with pytest.raises(ValueError, match="negative"):
        P.pred_sha256(np.array([-1, 2]))
    with pytest.raises(ValueError, match="integer"):
        P.pred_sha256(np.array([0.5]))
    with pytest.raises(ValueError, match="class axis"):
        P.labels_from_logits(np.zeros((3,), np.float32))


def test_canonical_labels():
    a = np.array([[1, 300]], dtype=np.int64)
    c = P.canonical_labels(a)
    assert c.dtype == np.dtype("<u2") and c.flags.c_contiguous and c.tolist() == [[1, 300]]
    assert P.canonical_labels(np.array([3], np.int64)).dtype == np.uint8


# ---------------------------------------------------------------- worker results carry pred_sha256
SEG = {"kind": "2p5d", "num_classes": 3, "encoder": {"init": "random", "model": "micro", "patch": 8},
       "decoder": {"channels": [16, 8]}, "synthetic": {"kind": "2p5d", "n": 8, "size": [32, 32], "channels": 3, "seed": 0}}


def _stores(tmp_path, n, cfg=SEG):
    t = R.create("segment"); t.init(cfg, TaskContext(device="cpu", amp="fp32")); t.export("safetensors", str(tmp_path / "m"))
    root = tmp_path / "d"; root.mkdir(exist_ok=True)
    out = []
    for _ in range(n):
        s = InferenceStore(device="cpu", plane=DataPlane(DataPolicy(roots=[str(root)])), out_root=str(tmp_path / "o"))
        s.handle("vision_infer_load", {"id": "m", "export": str(tmp_path / "m")})
        out.append(s)
    return out, root


@pytest.mark.parametrize("kind", ["2p5d", "3d"])
def test_vision_predict_and_merge_write_report_pred_sha256(tmp_path, kind):
    cfg = SEG if kind == "2p5d" else {**SEG, "kind": "3d", "encoder": {"init": "random", "model": "micro", "patch": [4, 8, 8]},
                                      "synthetic": {"kind": "3d", "n": 2, "size": [16, 16, 16], "channels": 1, "seed": 0}}
    stores, root = _stores(tmp_path, 2, cfg)
    vol = np.random.default_rng(0).standard_normal((9, 20, 24) if kind == "2p5d" else (20, 18, 20)).astype("float32")
    np.save(root / "v.npy", vol)
    r = stores[0].handle("vision_predict", {"id": "m", "ref": {"uri": "file://v.npy"}, "out": "one"})
    saved = np.load(r["path"])
    assert r["pred_sha256"] == P.pred_sha256(saved) and r["pred_dtype"] == P.label_dtype(saved)
    assert len(r["pred_sha256"]) == 64
    # a second worker holding the same model computes the same hash for the same prediction
    r2 = stores[1].handle("vision_predict", {"id": "m", "ref": {"uri": "file://v.npy"}, "out": "two"})
    assert r2["pred_sha256"] == r["pred_sha256"]
    parts = [s.handle("vision_predict_part", {"id": "m", "ref": {"uri": "file://v.npy"}, "part": [k, 2]})
             for k, s in enumerate(stores)]
    w = stores[0].handle("vision_merge_write", {"id": "m", "parts": parts, "out": "tiled"})
    assert w["pred_sha256"] == P.pred_sha256(np.load(w["path"]))


def test_vision_predict_many_classes_is_not_truncated(tmp_path):
    """> 256 classes: the label map is written (and hashed) as uint16 instead of wrapping to uint8."""
    stores, root = _stores(tmp_path, 1, {**SEG, "num_classes": 300})
    np.save(root / "v.npy", np.random.default_rng(1).standard_normal((3, 16, 16)).astype("float32"))
    st = stores[0]
    m = st.models["m"]["model"]
    with torch.no_grad():                        # force a class > 255 everywhere
        m.decoder.head.bias.zero_(); m.decoder.head.bias[299] = 1e4
    r = st.handle("vision_predict", {"id": "m", "ref": {"uri": "file://v.npy"}, "out": "big"})
    saved = np.load(r["path"])
    assert saved.dtype == np.uint16 and int(saved.max()) == 299 and r["pred_dtype"] == "uint16"
    assert r["pred_sha256"] == P.pred_sha256(saved)
