"""Sliding-window inference equal to MONAI `sliding_window_inference` (constant/gaussian blend, sigma_scale 0.125,
padding='constant' 0) plus flip test-time augmentation. Goldens: tests/goldens/vision_goldens.npz (max|Δ| < 1e-5)."""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F


def _scan_starts(size: int, roi: int, overlap: float) -> list[int]:
    if roi >= size:
        return [0]
    interval = max(int(roi * (1 - overlap)), 1)
    n = math.ceil((size - roi) / interval) + 1
    starts = []
    for i in range(n):
        s = i * interval
        starts.append(size - roi if s + roi > size else s)
    return sorted(set(starts))


def _importance(roi, mode: str, sigma_scale: float = 0.125, device="cpu") -> torch.Tensor:
    if mode == "constant":
        return torch.ones(roi, device=device)
    # MONAI compute_importance_map: separable gaussian (unnormalised, centre = 1), clamped at max(min, 1e-3)
    m = None
    for i, r in enumerate(roi):
        sigma = r * sigma_scale
        x = torch.arange(-(r - 1) / 2.0, (r - 1) / 2.0 + 1, dtype=torch.float, device=device)
        x = torch.exp(x ** 2 / (-2 * sigma ** 2))
        m = x if m is None else m.unsqueeze(-1) * x[(None,) * i]
    return torch.clamp(m, min=max(float(m.min()), 1e-3))


def sliding_window(x: torch.Tensor, roi, sw_batch: int, predictor: Callable, overlap: float = 0.25,
                   mode: str = "constant", sigma_scale: float = 0.125) -> torch.Tensor:
    nd = x.dim() - 2
    roi = tuple(int(r) for r in roi)
    spatial = tuple(x.shape[2:])
    # pad up to roi where the image is smaller (MONAI pads symmetrically with constant 0)
    pad = []
    for s, r in zip(reversed(spatial), reversed(roi)):
        d = max(r - s, 0)
        pad += [d // 2, d - d // 2]
    xp = F.pad(x, pad) if any(pad) else x
    ps = tuple(xp.shape[2:])
    starts = [_scan_starts(s, r, overlap) for s, r in zip(ps, roi)]
    grid = torch.cartesian_prod(*[torch.tensor(s) for s in starts]).reshape(-1, nd).tolist()
    imp = _importance(roi, mode, sigma_scale, x.device)
    out = cnt = None
    B = xp.shape[0]
    windows = [(b, tuple(s)) for b in range(B) for s in grid]
    for i in range(0, len(windows), sw_batch):
        chunk = windows[i:i + sw_batch]
        inp = torch.cat([xp[b:b + 1][(slice(None), slice(None)) + tuple(slice(a, a + r) for a, r in zip(s, roi))]
                         for b, s in chunk])
        pred = predictor(inp)
        if out is None:
            out = torch.zeros((B, pred.shape[1]) + ps, dtype=torch.float32, device=x.device)
            cnt = torch.zeros((B, 1) + ps, dtype=torch.float32, device=x.device)
        for j, (b, s) in enumerate(chunk):
            sl = (slice(b, b + 1), slice(None)) + tuple(slice(a, a + r) for a, r in zip(s, roi))
            out[sl] += pred[j:j + 1].float() * imp
            cnt[sl] += imp
    out = out / cnt
    if any(pad):
        crop = []
        for k in range(nd):
            lo = pad[2 * (nd - 1 - k)]
            crop.append(slice(lo, lo + spatial[k]))
        out = out[(slice(None), slice(None)) + tuple(crop)]
    return out


def predict(x: torch.Tensor, model: Callable, roi=None, overlap: float = 0.5, sw_batch: int = 4,
            mode: str = "gaussian", tta: str = "none") -> torch.Tensor:
    run = (lambda t: sliding_window(t, roi, sw_batch, model, overlap, mode)) if roi else model
    y = run(x)
    if tta == "flip":
        n = 1
        for d in range(2, x.dim()):
            y = y + torch.flip(run(torch.flip(x, [-(x.dim() - d)])), [-(x.dim() - d)])
            n += 1
        y = y / n
    elif tta not in ("none", None):
        raise ValueError(f"unknown tta {tta!r}")
    return y
