"""Generic fine-tuning of ANY published native vision model (torchvision / timm / MONAI / HF / plugin) as a TrainTask,
DiLoCo-compatible, loaded exactly as published through the M4 adapters."""
import math

import pytest
import torch

from moregpu_worker.train import registry as R
from moregpu_worker.train.local import run_diloco, run_plain
from moregpu_worker.train.task import TaskContext
from _vision_models import save_state_dict, spec_for, tiny_monai_unet


def ctx():
    return TaskContext(device="cpu", amp="fp32", seed=0)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(tmp_path))
    return tmp_path


def _resnet_spec(root):
    tv = pytest.importorskip("torchvision")
    m = tv.models.resnet18(weights=None, num_classes=3)
    p = root / "r18.pt"; save_state_dict(m, p)
    return spec_for(p, "state_dict", {"registry": "torchvision", "name": "resnet18", "kwargs": {"num_classes": 3}}), m


def test_resnet18_classification_learns_head_only_and_full(roots):
    spec, _ = _resnet_spec(roots)
    syn = {"kind": "2d", "n": 24, "size": [32, 32], "channels": 3, "seed": 0, "classes": 3}
    t = R.create("finetune_model")
    info = t.init({"spec": spec, "objective": "classify", "num_classes": 3, "synthetic": syn, "trainable": "head",
                   "head_prefixes": ["fc."], "weight_decay": 0.0}, ctx())
    assert set(t.state_for_sync()) == {"fc.weight", "fc.bias"} and info["trainable_params"] == 512 * 3 + 3
    t2 = R.create("finetune_model")
    t2.init({"spec": spec, "objective": "classify", "num_classes": 3, "synthetic": syn, "trainable": "all", "weight_decay": 0.0}, ctx())
    losses = []
    for _ in range(25):
        losses += t2.inner_steps(list(range(12)), 1, 1e-3).losses
    assert losses[-1] < 0.5 * losses[0]
    assert 0 <= t2.evaluate(list(range(12)), "accuracy")["accuracy"] <= 1


def test_monai_unet_3d_segmentation_finetune_and_export_reloads(roots, tmp_path):
    m = tiny_monai_unet(); p = roots / "unet.pt"; save_state_dict(m, p)
    arch = {"registry": "monai", "name": "UNet", "kwargs": {"spatial_dims": 3, "in_channels": 1, "out_channels": 2,
            "channels": [4, 8, 16], "strides": [2, 2], "num_res_units": 1}}
    spec = spec_for(p, "state_dict", arch)
    syn = {"kind": "3d", "n": 4, "size": [16, 16, 16], "channels": 1, "seed": 0}
    t = R.create("finetune_model")
    t.init({"spec": spec, "objective": "segment", "num_classes": 2, "synthetic": syn, "label_map": {"2": 1}, "weight_decay": 0.0}, ctx())
    losses = []
    for _ in range(15):
        losses += t.inner_steps([0, 1], 1, 3e-3).losses
    assert losses[-1] < losses[0]
    d = t.evaluate([0, 1], "dice")
    assert "1" in d["dice_per_class"]
    ex = t.export("safetensors", str(tmp_path / "ft"))
    from moregpu_worker.vision import adapters as A
    h = A.load(ex["spec"])                                  # the fine-tuned model reloads exactly as a published model
    x = t.data.batch([0])[:, None] if t.data.batch([0]).dim() == 4 else t.data.batch([0])
    with torch.no_grad():
        assert torch.allclose(A.infer(h, x), t.model.eval()(x), atol=1e-5)


def test_timm_vit_lora_trains_adapters_only(roots):
    timm = pytest.importorskip("timm")
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=32, num_classes=3)
    p = roots / "vit.pt"; save_state_dict(m, p)
    spec = spec_for(p, "state_dict", {"registry": "timm", "name": "vit_tiny_patch16_224", "kwargs": {"img_size": 32, "num_classes": 3}})
    syn = {"kind": "2d", "n": 8, "size": [32, 32], "channels": 3, "seed": 0, "classes": 3}
    t = R.create("finetune_model")
    t.init({"spec": spec, "objective": "classify", "num_classes": 3, "synthetic": syn, "trainable": "lora", "lora_rank": 4,
            "lora_targets": ["qkv", "proj"], "head_prefixes": ["head."]}, ctx())
    keys = list(t.state_for_sync())
    assert any(k.endswith(".A") for k in keys) and any(k.startswith("head.") for k in keys)
    assert not any(k.endswith("qkv.base.weight") for k in keys)
    assert math.isfinite(t.inner_steps([0, 1], 1, 1e-3).losses[0])


def test_finetune_model_diloco_equivalence_and_refuses_non_native(roots):
    spec, _ = _resnet_spec(roots)
    cfg = {"spec": spec, "objective": "classify", "num_classes": 3, "trainable": "head", "head_prefixes": ["fc."],
           "synthetic": {"kind": "2d", "n": 16, "size": [32, 32], "channels": 3, "seed": 0, "classes": 3}, "keep_inner_state": True}
    plain = run_plain("finetune_model", cfg, steps=3, batch=4, lr=1e-3, seed=0, manifest_len=16)
    dl = run_diloco("finetune_model", cfg, n_workers=1, rounds=3, inner_steps=1, batch=4, lr=1e-3, outer_lr=1.0,
                    outer_momentum=0.0, seed=0, manifest_len=16, keep_inner_state=True)
    for k in plain["state"]:
        assert torch.allclose(plain["state"][k], dl["state"][k], atol=1e-6)
    with pytest.raises(ValueError):
        R.create("finetune_model").init({**cfg, "objective": "nope"}, ctx())
