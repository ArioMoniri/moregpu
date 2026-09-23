import math

import pytest
import torch

from moregpu_worker.train import registry as R
from moregpu_worker.train.local import run_diloco, run_plain
from moregpu_worker.train.synthetic import SyntheticSeg
from moregpu_worker.train.task import TaskContext

SEG_SYN = {"kind": "2p5d", "n": 16, "size": [32, 32], "channels": 3, "seed": 0}
SEG = {"kind": "2p5d", "num_classes": 3, "encoder": {"init": "random", "model": "micro", "patch": 8},
       "decoder": {"channels": [16, 8]}, "synthetic": SEG_SYN, "mode": "full", "optimizer": "adamw", "weight_decay": 0.0}


def ctx():
    return TaskContext(device="cpu", amp="fp32", seed=0)


def test_synthetic_seg_has_three_classes_and_aligned_masks():
    s = SyntheticSeg(**SEG_SYN)
    x, y = s.batch([0, 1]), s.masks([0, 1])
    assert x.shape == (2, 3, 32, 32) and y.shape == (2, 32, 32)
    assert set(torch.unique(y).tolist()) <= {0, 1, 2} and (y == 1).any() and (y == 2).any()
    s3 = SyntheticSeg(kind="3d", n=2, size=[16, 16, 16], channels=1, seed=0)
    assert s3.batch([0]).shape == (1, 1, 16, 16, 16) and s3.masks([0]).shape == (1, 16, 16, 16)


def test_segment_overfits_tiny_batch():
    t = R.create("segment"); t.init(SEG, ctx())
    d0 = t.evaluate([0, 1, 2, 3], "dice")["dice_mean"]
    losses = []
    for _ in range(30):
        losses += t.inner_steps([0, 1, 2, 3], 1, 3e-3).losses
    d1 = t.evaluate([0, 1, 2, 3], "dice")
    assert losses[-1] < 0.6 * losses[0]
    assert d1["dice_mean"] > d0 and d1["dice_mean"] > 0.5, d1
    assert set(d1["dice_per_class"]) == {"1", "2"}


def test_frozen_mode_keeps_encoder_and_syncs_only_head():
    t = R.create("segment"); t.init({**SEG, "mode": "frozen"}, ctx())
    before = {k: v.clone() for k, v in t.encoder.state_dict().items()}
    t.inner_steps([0, 1], 2, 1e-2)
    assert all(torch.equal(before[k], v) for k, v in t.encoder.state_dict().items())
    keys = t.state_for_sync().keys()
    assert all(k.startswith("decoder.") for k in keys)


def test_lora_mode_trains_adapters_and_decoder_only():
    t = R.create("segment"); t.init({**SEG, "mode": "lora", "lora_rank": 4}, ctx())
    keys = list(t.state_for_sync())
    assert any(k.endswith(".A") for k in keys) and any(k.startswith("decoder.") for k in keys)
    assert not any(k.endswith("qkv.base.weight") for k in keys)
    base = t.encoder.blocks[0].attn.qkv.base.weight.clone()
    t.inner_steps([0, 1], 2, 1e-2)
    assert torch.equal(base, t.encoder.blocks[0].attn.qkv.base.weight)


def test_encoder_from_jepa_export(tmp_path):
    j = R.create("jepa_2p5d")
    j.init({"model": "micro", "patch": 8, "pred_dim": 16, "pred_depth": 1, "pred_heads": 2, "synthetic": SEG_SYN}, ctx())
    ex = j.export("safetensors", str(tmp_path))
    t = R.create("segment"); t.init({**SEG, "encoder": {"init": "export", "path": str(tmp_path)}}, ctx())
    for k, v in j.encoder.state_dict().items():
        assert torch.equal(v, t.encoder.state_dict()[k])
    with pytest.raises(FileNotFoundError):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": str(tmp_path / "nope")}}, ctx())
    assert ex["sha256"]


def test_segment_3d_and_2d_variants():
    for kind, syn, patch in (("3d", {"kind": "3d", "n": 4, "size": [16, 16, 16], "channels": 1, "seed": 0}, [4, 8, 8]),
                             ("2d", {"kind": "2d", "n": 4, "size": [32, 32], "channels": 1, "seed": 0}, 8)):
        t = R.create("segment")
        t.init({**SEG, "kind": kind, "synthetic": syn, "encoder": {"init": "random", "model": "micro", "patch": patch}}, ctx())
        rep = t.inner_steps([0, 1], 1, 1e-3)
        assert math.isfinite(rep.losses[0])
        assert "dice_mean" in t.evaluate([0, 1], "dice")


def test_segment_diloco_two_workers_runs_and_n1_equivalence():
    cfg = {**SEG, "keep_inner_state": True}
    plain = run_plain("segment", cfg, steps=4, batch=2, lr=1e-3, seed=0, manifest_len=16)
    dl = run_diloco("segment", cfg, n_workers=1, rounds=4, inner_steps=1, batch=2, lr=1e-3, outer_lr=1.0,
                    outer_momentum=0.0, seed=0, manifest_len=16, keep_inner_state=True)
    for k in plain["state"]:
        assert torch.allclose(plain["state"][k], dl["state"][k], atol=1e-6), k
    r = run_diloco("segment", SEG, n_workers=2, rounds=2, inner_steps=2, batch=2, lr=1e-3, outer_lr=0.7,
                   outer_momentum=0.9, seed=0, manifest_len=16)
    assert r["samples_seen"] == 16


def test_segment_export_parity(tmp_path):
    t = R.create("segment"); t.init(SEG, ctx())
    x = t.data.batch([0, 1])
    with torch.no_grad():
        ref = t.model.eval()(x)
    st = t.export("safetensors", str(tmp_path / "a"))
    from moregpu_worker.vision.models import load_exported
    m = load_exported(str(tmp_path / "a"))
    with torch.no_grad():
        assert torch.allclose(m(x), ref, atol=1e-5)
    te = t.export("torch_export", str(tmp_path / "b"))
    with torch.no_grad():
        assert torch.allclose(torch.export.load(te["path"]).module()(x), ref, atol=1e-4)
    pytest.importorskip("onnxruntime")
    ox = t.export("onnx", str(tmp_path / "c"))
    assert ox["parity_max_abs"] < 1e-4
    assert st["task"] == "segment"


def test_classify_learns_and_evaluates():
    cfg = {"kind": "2p5d", "num_classes": 3, "encoder": {"init": "random", "model": "micro", "patch": 8},
           "synthetic": {"kind": "2p5d", "n": 24, "size": [32, 32], "channels": 3, "seed": 0, "classes": 3},
           "mode": "full", "optimizer": "adamw", "weight_decay": 0.0}
    t = R.create("classify"); t.init(cfg, ctx())
    refs = list(range(12))
    losses = []
    for _ in range(40):
        losses += t.inner_steps(refs, 1, 3e-3).losses
    assert losses[-1] < 0.5 * losses[0]
    acc = t.evaluate(refs, "accuracy")["accuracy"]
    assert acc > 0.8
    with pytest.raises(ValueError):
        t.evaluate(refs, "dice")
