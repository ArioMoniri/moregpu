import math

import pytest
import torch

from moregpu_worker.train import registry as R, monitors as MON
from moregpu_worker.train.local import run_diloco, run_plain
from moregpu_worker.train.synthetic import SyntheticVolumes
from moregpu_worker.train.task import TaskContext
from moregpu_worker.train.tasks import jepa as J

SYN = {"kind": "2p5d", "n": 48, "size": [32, 32], "channels": 3, "seed": 0, "classes": 3}
CFG = {"model": "micro", "patch": 8, "pred_dim": 16, "pred_depth": 1, "pred_heads": 2, "synthetic": SYN,
       "ema": [0.9, 1.0], "total_steps": 40, "n_targets": 2, "optimizer": "adamw", "weight_decay": 0.0,
       "probe_batch": 16}


def ctx(seed=0):
    return TaskContext(device="cpu", amp="fp32", seed=seed)


def test_synthetic_volumes_are_structured_deterministic_and_labelled():
    s = SyntheticVolumes(**SYN)
    a, b = s.batch([0, 1]), s.batch([0, 1])
    assert a.shape == (2, 3, 32, 32) and torch.equal(a, b)
    assert s.label(0) in range(3) and len(s) == 48
    assert a.std() > 0.1


def test_ema_update_math():
    tgt = {"w": torch.tensor([1.0, 1.0])}; onl = {"w": torch.tensor([3.0, -1.0])}
    J.ema_update(tgt, onl, 0.75)
    assert torch.allclose(tgt["w"], torch.tensor([1.5, 0.5]))


def test_ema_schedule_product_over_round():
    s = J.EmaSchedule(0.9, 1.0, total_steps=10)
    assert math.isclose(s.momentum_between(0, 1), 0.9)
    assert math.isclose(s.momentum_between(0, 3), s.at(0) * s.at(1) * s.at(2))
    assert s.at(10) == 1.0 and s.at(99) == 1.0


def test_jepa_2p5d_learns_on_synthetic():
    r = run_plain("jepa_2p5d", CFG, steps=40, batch=8, lr=2e-3, seed=0, manifest_len=48)
    first, last = sum(r["losses"][:5]) / 5, sum(r["losses"][-5:]) / 5
    assert last < first * 0.8, (first, last)


def test_diloco_n1_h1_equals_plain_ijepa_with_per_step_ema():
    cfg = {**CFG, "keep_inner_state": True}
    plain = run_plain("jepa_2p5d", cfg, steps=6, batch=4, lr=1e-3, seed=0, manifest_len=48)
    dl = run_diloco("jepa_2p5d", cfg, n_workers=1, rounds=6, inner_steps=1, batch=4, lr=1e-3, outer_lr=1.0,
                    outer_momentum=0.0, seed=0, manifest_len=48, keep_inner_state=True)
    for k in plain["state"]:
        assert torch.allclose(plain["state"][k], dl["state"][k], atol=1e-6), k
    tp, td = plain["task"].target.state_dict(), dl["workers"][0].target.state_dict()
    for k in tp:
        assert torch.allclose(tp[k], td[k], atol=1e-6), k


def test_two_worker_diloco_targets_stay_identical():
    r = run_diloco("jepa_2p5d", CFG, n_workers=2, rounds=3, inner_steps=2, batch=4, lr=1e-3, outer_lr=0.7,
                   outer_momentum=0.9, seed=0, manifest_len=48)
    h = [w.after_outer_step(99)["target_sha256"] for w in r["workers"]]
    assert h[0] == h[1]


def test_sync_state_excludes_target_and_includes_predictor():
    t = R.create("jepa_2p5d"); t.init(CFG, ctx())
    keys = t.state_for_sync().keys()
    assert any(k.startswith("encoder.") for k in keys) and any(k.startswith("predictor.") for k in keys)
    assert not any(k.startswith("target.") for k in keys)
    assert not any(k.endswith("pos_embed") for k in keys)          # frozen buffers-as-params don't sync


def test_monitors_and_forced_collapse_alarm():
    good = torch.randn(64, 16)
    m = MON.embedding_monitors(good)
    assert m["std_mean"] > 0.5 and m["rankme"] > 10
    collapsed = torch.ones(64, 16) + 1e-7 * torch.randn(64, 16)
    m2 = MON.embedding_monitors(collapsed)
    assert m2["rankme"] < 2
    al = MON.alarms(m2, std_min=1e-3, rank_min=2.0)
    assert any("collapse" in a for a in al)
    t = R.create("jepa_2p5d"); t.init(CFG, ctx())
    with torch.no_grad():
        for p in t.encoder.parameters():
            p.zero_()                                              # forced collapse: constant embeddings
    out = t.after_outer_step(1)
    assert any("collapse" in a for a in out["alarms"])


def test_evaluate_loss_knn_linear_probe_features():
    t = R.create("jepa_2p5d"); t.init(CFG, ctx())
    refs = list(range(24))
    assert "loss" in t.evaluate(refs, "loss")
    k = t.evaluate(refs, "knn")
    assert 0.0 <= k["knn_acc"] <= 1.0
    lp = t.evaluate(refs, "linear_probe")
    assert 0.0 <= lp["linear_probe_acc"] <= 1.0
    f = t.evaluate(refs[:4], "features")
    assert f["shape"] == [4, t.encoder.embed_dim]
    with pytest.raises(ValueError):
        t.evaluate(refs, "nope")


@pytest.mark.parametrize("task,syn", [("ijepa_2d", {"kind": "2d", "n": 16, "size": [32, 32], "channels": 1, "seed": 0}),
                                      ("jepa_3d", {"kind": "3d", "n": 8, "size": [16, 16, 16], "channels": 1, "seed": 0})])
def test_other_jepa_variants_step(task, syn):
    cfg = {**CFG, "synthetic": syn, "patch": 8 if task == "ijepa_2d" else [4, 8, 8]}
    t = R.create(task); t.init(cfg, ctx())
    rep = t.inner_steps([0, 1, 2, 3], 2, 1e-3)
    assert len(rep.losses) == 2 and all(math.isfinite(x) for x in rep.losses)
    assert rep.timings["data_s"] >= 0 and rep.samples == 4


def test_export_parity_safetensors_torch_export_onnx(tmp_path):
    t = R.create("jepa_2p5d"); t.init(CFG, ctx())
    x = t.data.batch([0, 1])
    with torch.no_grad():
        ref = t.encoder.eval()(x)
    st = t.export("safetensors", str(tmp_path / "enc"))
    from safetensors.torch import load_file
    from moregpu_worker.models.vit import VisionTransformer
    import json
    cfg = json.load(open(st["config"]))
    m = VisionTransformer(**cfg); m.load_state_dict(load_file(st["weights"]), strict=True); m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), ref, atol=1e-6)
    te = t.export("torch_export", str(tmp_path / "enc2"))
    prog = torch.export.load(te["path"])
    with torch.no_grad():
        assert torch.allclose(prog.module()(x), ref, atol=1e-5)
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    ox = t.export("onnx", str(tmp_path / "enc3"))
    sess = ort.InferenceSession(ox["path"], providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    assert abs(out - ref.numpy()).max() < 1e-4
    assert ox["parity_max_abs"] < 1e-4
    with pytest.raises(ValueError):
        t.export("pickle", str(tmp_path / "x"))


def test_amp_bf16_cpu_runs():
    t = R.create("jepa_2p5d")
    t.init(CFG, TaskContext(device="cpu", amp="bf16", seed=0))
    rep = t.inner_steps([0, 1, 2, 3], 1, 1e-3)
    assert math.isfinite(rep.losses[0]) and rep.metrics["amp"] == "bf16"


def test_augmentation_and_l2_loss_and_describe():
    t = R.create("jepa_2p5d")
    t.init({**CFG, "crop_scale": [0.5, 1.0], "hflip": True, "loss": "l2", "clip_grad": 1.0, "grad_checkpointing": True}, ctx())
    x = t.data.batch([0, 1])
    g = torch.Generator().manual_seed(0)
    y = t._augment(x, g)
    assert y.shape == x.shape and not torch.equal(x, y)
    rep = t.inner_steps([0, 1, 2, 3], 2, 1e-3)
    assert all(math.isfinite(v) for v in rep.losses)
    d = t.describe()
    assert d["kind"] == "2p5d" and d["encoder"]["embed_dim"] == 32


def test_data_plane_source_and_missing_data():
    class FakeRef:
        def __init__(self, i): self.meta = {"label": i % 2}
    class FakeManifest(list):
        pass
    class FakePlane:
        def open_manifest(self, uri, sha=None): return FakeManifest([FakeRef(i) for i in range(10)])
        def load_batch(self, man, idx, spec): return torch.randn(len(idx), 3, 32, 32)
    t = R.create("jepa_2p5d")
    t.init({**CFG, "synthetic": None, "data": {"manifest": "file://x", "spec": {"size": [32, 32], "channels": 3}}},
           TaskContext(device="cpu", amp="fp32", data=FakePlane()))
    assert len(t.data) == 10 and t.data.label(3) == 1
    assert len(t.inner_steps([0, 1], 1, 1e-3).losses) == 1
    with pytest.raises(RuntimeError):
        R.create("jepa_2p5d").init({**CFG, "synthetic": None, "data": {"manifest": "x", "spec": {"size": [8, 8]}}}, ctx())
    with pytest.raises(ValueError):
        R.create("jepa_2p5d").init({**CFG, "synthetic": None}, ctx())


def test_knn_and_probe_edge_cases():
    assert math.isnan(J.knn_accuracy(torch.randn(1, 4), torch.tensor([0])))
    assert math.isnan(J.linear_probe_accuracy(torch.randn(3, 4), torch.tensor([0, 1, 0])))
    f = torch.nn.functional.normalize(torch.cat([torch.randn(10, 4) + 5, torch.randn(10, 4) - 5]), dim=1)
    y = torch.tensor([0] * 10 + [1] * 10)
    assert J.knn_accuracy(f, y) > 0.9 and J.linear_probe_accuracy(f, y) > 0.9
