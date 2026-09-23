"""Regression tests for the ML-engineer review findings (P0/P1)."""
import math

import pytest
import torch

from moregpu_worker.train import registry as R
from moregpu_worker.train.local import run_diloco, run_plain
from moregpu_worker.train.masking import MultiBlockMasker
from moregpu_worker.train.runner import TaskRunner
from moregpu_worker.train.sessions import SessionStore
from moregpu_worker.train.task import TaskContext

SYN = {"kind": "2p5d", "n": 48, "size": [32, 32], "channels": 3, "seed": 0, "classes": 3}
JEPA = {"model": "micro", "patch": 8, "pred_dim": 16, "pred_depth": 1, "pred_heads": 2, "synthetic": SYN,
        "ema": [0.9, 1.0], "n_targets": 2, "weight_decay": 0.04, "probe_batch": 8}


def test_p0_1_proportional_allocation_keeps_ema_targets_identical():
    r = run_diloco("jepa_2p5d", JEPA, n_workers=2, rounds=3, inner_steps=3, batch=4, lr=1e-3, outer_lr=0.7,
                   outer_momentum=0.9, seed=0, manifest_len=48, alloc_mode="proportional", speeds=[1.0, 1.5],
                   target_samples=72)
    steps = [w.step for w in r["workers"]]
    assert steps[0] != steps[1]                       # workers really did different numbers of local steps
    assert r["workers"][0].target_hash() == r["workers"][1].target_hash()


def test_p0_2_resume_restores_target_and_counters_via_extra_state():
    a = TaskRunner(SessionStore(2), "cpu")
    a.handle("task_init", {"session": "s", "task": "jepa_2p5d", "cfg": JEPA, "amp": "fp32", "seed": 0})
    a.handle("task_inner", {"session": "s", "refs": list(range(8)), "steps": 2, "lr": 1e-3})
    a.handle("task_after_outer", {"session": "s", "round": 1, "progress": 0.1, "h": 2})
    t = a.sessions.get("s")
    before = (t.target_hash(), t.step)
    import base64
    from moregpu_worker.train import tensorwire as tw
    out = a.handle("task_state_get", {"session": "s", "which": "extra"})
    blob = base64.b64decode(out["chunk0"])
    for k in range(1, out["nchunks"]):
        blob += base64.b64decode(a.handle("task_state_chunk", {"session": "s", "k": k})["data"])
    b = TaskRunner(SessionStore(2), "cpu")
    b.handle("task_init", {"session": "s", "task": "jepa_2p5d", "cfg": JEPA, "amp": "fp32", "seed": 0})
    assert b.sessions.get("s").target_hash() != before[0]
    b.handle("task_state_put", {"session": "s", "which": "extra", "header": out["header"], "k": 0, "n": 1,
                                "data": base64.b64encode(blob).decode()})
    t2 = b.sessions.get("s")
    assert (t2.target_hash(), t2.step) == before


def test_p1_1_inner_optimizer_state_kept_by_default_h1_equals_plain():
    cfg = {"n": 64, "dim": 5, "batch": 4, "optimizer": "adamw"}
    plain = run_plain("toy_linear", cfg, steps=10, batch=4, lr=0.05, seed=0, manifest_len=64)
    dl = run_diloco("toy_linear", cfg, n_workers=1, rounds=10, inner_steps=1, batch=4, lr=0.05, outer_lr=1.0,
                    outer_momentum=0.0, seed=0, manifest_len=64)
    for k in plain["state"]:
        assert torch.allclose(plain["state"][k], dl["state"][k], atol=1e-6)


def test_p1_2_ema_momentum_from_global_progress():
    t = R.create("jepa_2p5d"); t.init({**JEPA, "ema": [0.9, 1.0]}, TaskContext(amp="fp32"))
    out = t.after_outer_step(1, {"progress": 0.5, "h": 3})
    assert math.isclose(out["ema_momentum"], 0.95 ** 3, rel_tol=1e-9)


def test_p1_4_export_target_encoder_by_default(tmp_path):
    t = R.create("jepa_2p5d"); t.init(JEPA, TaskContext(amp="fp32"))
    t.inner_steps(list(range(8)), 2, 1e-2); t.after_outer_step(1, {"progress": 0.2, "h": 2})
    from safetensors.torch import load_file
    ex = t.export("safetensors", str(tmp_path / "a"))
    w = load_file(ex["weights"])
    assert ex["which"] == "target"
    assert all(torch.equal(w[k], v) for k, v in t.target.state_dict().items())
    ex2 = t.export("safetensors", str(tmp_path / "b"), which="context")
    w2 = load_file(ex2["weights"])
    assert all(torch.equal(w2[k], v) for k, v in t.encoder.state_dict().items())


def test_p1_5_no_weight_decay_on_bias_norm_and_1d_params():
    t = R.create("jepa_2p5d"); t.init(JEPA, TaskContext(amp="fp32"))
    opt = t.make_optimizer(t._trainable(), 1e-3, "adamw", 0.04)
    groups = {g["weight_decay"]: g["params"] for g in opt.param_groups}
    assert all(p.ndim >= 2 for p in groups[0.04]) and all(p.ndim <= 1 for p in groups[0.0])


def test_p1_6_mask_blocks_have_equal_size_before_truncation():
    mk = MultiBlockMasker((14, 14), n_targets=4)
    g = torch.Generator().manual_seed(0)
    shapes = set()
    for _ in range(8):
        _, tg = mk.sample_one(g, *mk.sample_shapes(g))
        shapes |= {len(x) for x in tg}
    ctx, tgts = mk(32, torch.Generator().manual_seed(1))
    full = mk.last_target_sizes
    assert len(set(full)) == 1 and tgts[0].shape[1] == full[0]     # nothing truncated from targets


def test_p1_7_mask_resize_is_aligned_with_image_resize():
    import torch.nn.functional as F
    from moregpu_worker.train.tasks.vision import resize_mask
    m = torch.zeros(512, 512); m[200:260, 300:340] = 1
    img = F.interpolate(m[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    lab = resize_mask(m.long(), (224, 224)).float()
    com = lambda a: torch.stack([(a.sum(1) * torch.arange(a.shape[0])).sum(), (a.sum(0) * torch.arange(a.shape[1])).sum()]) / a.sum()
    assert (com(img) - com(lab)).abs().max() <= 0.3


def test_p1_8_finetune_model_syncs_batchnorm_buffers(tmp_path, monkeypatch):
    tv = pytest.importorskip("torchvision")
    from _vision_models import save_state_dict, spec_for
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(tmp_path))
    m = tv.models.resnet18(weights=None, num_classes=3); p = tmp_path / "r.pt"; save_state_dict(m, p)
    spec = spec_for(p, "state_dict", {"registry": "torchvision", "name": "resnet18", "kwargs": {"num_classes": 3}})
    cfg = {"spec": spec, "objective": "classify", "num_classes": 3,
           "synthetic": {"kind": "2d", "n": 16, "size": [32, 32], "channels": 3, "seed": 0, "classes": 3}}
    r = run_diloco("finetune_model", cfg, n_workers=2, rounds=1, inner_steps=1, batch=4, lr=1e-3, outer_lr=1.0,
                   outer_momentum=0.0, seed=0, manifest_len=16)
    a, b = (dict(w.model.named_buffers()) for w in r["workers"])
    assert torch.equal(a["bn1.running_mean"], b["bn1.running_mean"])
    assert "buffer:bn1.running_mean" in r["workers"][0].state_for_sync()
    assert not any(k.endswith("num_batches_tracked") for k in r["workers"][0].state_for_sync())


def test_p2_label_map_is_a_lut_not_sequential():
    from moregpu_worker.train.tasks.finetune_model import apply_label_map
    y = torch.tensor([0, 1, 2]); assert apply_label_map(y, {1: 2, 2: 1}).tolist() == [0, 2, 1]


def test_inner_optimizer_state_round_trips_through_extra_state():
    cfg = {"n": 64, "dim": 5, "batch": 4, "optimizer": "adamw"}
    a = R.create("toy_linear"); a.init(cfg, TaskContext(amp="fp32"))
    a.inner_steps(list(range(8)), 2, 0.05)
    ex = a.extra_state()
    assert any(k.startswith("opt.") for k in ex)
    b = R.create("toy_linear"); b.init(cfg, TaskContext(amp="fp32"))
    b.load_sync_state(a.state_for_sync()); b.load_extra_state(ex)
    a.inner_steps(list(range(8, 16)), 2, 0.05); b.inner_steps(list(range(8, 16)), 2, 0.05)
    for k, v in a.state_for_sync().items():
        assert torch.equal(v, b.state_for_sync()[k])
