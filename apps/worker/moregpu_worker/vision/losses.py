"""Segmentation losses/metrics with MONAI-equivalent semantics (goldens: tests/goldens/vision_goldens.npz).

dice_ce = MONAI DiceCELoss(to_onehot_y=True, softmax=True): soft Dice over all classes incl. background, batch-wise
per-sample mean, smooth_nr = smooth_dr = 1e-5, plus mean cross-entropy (weights 1:1)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _onehot(t: torch.Tensor, c: int) -> torch.Tensor:
    t = t[:, 0] if t.dim() >= 3 and t.shape[1] == 1 else t
    return F.one_hot(t.long(), c).movedim(-1, 1).float()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    p = torch.softmax(logits.float(), dim=1)
    y = _onehot(target, logits.shape[1])
    dims = tuple(range(2, p.dim()))
    inter = (p * y).sum(dims)
    den = p.sum(dims) + y.sum(dims)
    return (1.0 - (2.0 * inter + smooth) / (den + smooth)).mean()


def dice_ce(logits: torch.Tensor, target: torch.Tensor, w_dice: float = 1.0, w_ce: float = 1.0) -> torch.Tensor:
    t = target[:, 0] if target.dim() == logits.dim() else target
    return w_dice * dice_loss(logits, target) + w_ce * F.cross_entropy(logits.float(), t.long())


def dice_per_class(pred: torch.Tensor, gt: torch.Tensor, num_classes: int, include_background: bool = False) -> torch.Tensor:
    """Hard Dice per (sample, class) from label maps; nan where both are empty (MONAI DiceMetric convention)."""
    p, g = _onehot(pred, num_classes), _onehot(gt, num_classes)
    if not include_background:
        p, g = p[:, 1:], g[:, 1:]
    dims = tuple(range(2, p.dim()))
    inter = (p * g).sum(dims); den = p.sum(dims) + g.sum(dims)
    d = 2 * inter / den
    return torch.where(g.sum(dims) > 0, d, torch.where(den > 0, torch.zeros_like(d), torch.full_like(d, float("nan"))))


def nanmean(x: torch.Tensor, dim=None):
    m = ~torch.isnan(x)
    s = torch.where(m, x, torch.zeros_like(x)).sum() if dim is None else torch.where(m, x, torch.zeros_like(x)).sum(dim)
    n = m.sum() if dim is None else m.sum(dim)
    return float(s / n) if dim is None else s / n
