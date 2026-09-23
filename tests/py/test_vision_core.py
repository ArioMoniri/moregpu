from pathlib import Path

import numpy as np
import pytest
import torch

from moregpu_worker.vision import losses as L, sliding as SW

G = np.load(Path(__file__).parents[1] / "goldens" / "vision_goldens.npz")


@pytest.mark.parametrize("nd", ["2d", "3d"])
def test_dice_ce_matches_monai(nd):
    got = L.dice_ce(torch.from_numpy(G[f"dicece_{nd}_logits"]), torch.from_numpy(G[f"dicece_{nd}_tgt"]))
    assert abs(float(got) - float(G[f"dicece_{nd}_loss"])) < 1e-5


def test_dice_metric_matches_monai():
    got = L.dice_per_class(torch.from_numpy(G["dice_pred"]), torch.from_numpy(G["dice_gt"]), 3, include_background=False)
    assert np.allclose(got.numpy(), G["dice_val"], atol=1e-6)


def test_dice_metric_empty_conventions():
    z = torch.zeros(1, 1, 4, 4, dtype=torch.long)
    d = L.dice_per_class(z, z, 2, include_background=False)
    assert torch.isnan(d).all()                     # MONAI convention: empty GT+pred → nan (ignored in means)
    assert L.nanmean(torch.tensor([[float("nan"), 0.5]])) == pytest.approx(0.5)


def _conv(nd):
    c = (torch.nn.Conv3d(1, 2, 3, padding=1) if nd == 3 else torch.nn.Conv2d(1, 2, 3, padding=1))
    with torch.no_grad():
        c.weight.copy_(torch.from_numpy(G[f"conv{nd}_w"])); c.bias.copy_(torch.from_numpy(G[f"conv{nd}_b"]))
    return c


@pytest.mark.parametrize("mode", ["gaussian", "constant"])
def test_sliding_window_matches_monai(mode):
    with torch.no_grad():
        y3 = SW.sliding_window(torch.from_numpy(G["sw3_x"]), (8, 8, 8), 3, _conv(3), overlap=0.5, mode=mode)
        y2 = SW.sliding_window(torch.from_numpy(G["sw2_x"]), (16, 16), 2, _conv(2), overlap=0.25, mode=mode)
    assert np.abs(y3.numpy() - G[f"sw3_{mode}"]).max() < 1e-5
    assert np.abs(y2.numpy() - G[f"sw2_{mode}"]).max() < 1e-5


def test_flip_tta_is_average_of_flipped_predictions():
    conv = _conv(2)
    x = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        y = SW.predict(x, conv, roi=None, tta="flip")
        ref = (conv(x) + torch.flip(conv(torch.flip(x, [-1])), [-1]) + torch.flip(conv(torch.flip(x, [-2])), [-2])) / 3
    assert torch.allclose(y, ref, atol=1e-6)
    with torch.no_grad():
        assert torch.allclose(SW.predict(x, conv, roi=(8, 8), overlap=0.5, mode="constant"), SW.sliding_window(x, (8, 8), 4, conv, 0.5), atol=1e-6)
