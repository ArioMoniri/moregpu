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


# ---- tile sharding: one volume's sliding-window tiles split across workers (ADR-0112 amendment) -------------------
def _pad_and_grid(shape, roi, overlap):
    nd = len(shape) - 2
    spatial = tuple(shape[2:])
    pad = []
    for s, r in zip(reversed(spatial), reversed(roi)):
        d = max(r - s, 0)
        pad += [d // 2, d - d // 2]
    ps = tuple(max(s, r) for s, r in zip(spatial, roi))
    starts = [_scan_starts(s, r, overlap) for s, r in zip(ps, roi)]
    grid = torch.cartesian_prod(*[torch.tensor(s) for s in starts]).reshape(-1, nd).tolist()
    return pad, ps, grid


def count_tiles(shape, roi, overlap: float) -> int:
    return len(_pad_and_grid(shape, tuple(roi), overlap)[2])


def sliding_window_part(x: torch.Tensor, roi, sw_batch: int, predictor: Callable, overlap: float, mode: str,
                        part=(0, 1), sigma_scale: float = 0.125) -> dict:
    """Run only this part's tiles (contiguous block k of n in scan order, grouped along the first spatial axis) and
    return the weighted-sum and weight accumulators over the axis-0 slab they touch (padded coordinates)."""
    roi = tuple(int(r) for r in roi)
    pad, ps, grid = _pad_and_grid(tuple(x.shape), roi, overlap)
    k, n = part
    lo, hi = (len(grid) * k) // n, (len(grid) * (k + 1)) // n
    mine = grid[lo:hi]
    xp = F.pad(x, pad) if any(pad) else x
    if not mine:
        return {"z0": 0, "sum": None, "cnt": None, "n_tiles": 0, "padded_shape": list(ps)}
    z0 = min(t[0] for t in mine); z1 = max(t[0] for t in mine) + roi[0]
    imp = _importance(roi, mode, sigma_scale, x.device)
    out = cnt = None
    B = xp.shape[0]
    windows = [(b, tuple(s)) for b in range(B) for s in mine]
    for i in range(0, len(windows), sw_batch):
        chunk = windows[i:i + sw_batch]
        inp = torch.cat([xp[b:b + 1][(slice(None), slice(None)) + tuple(slice(a, a + r) for a, r in zip(s, roi))] for b, s in chunk])
        pred = predictor(inp)
        if out is None:
            out = torch.zeros((B, pred.shape[1], z1 - z0) + ps[1:], dtype=torch.float32, device=x.device)
            cnt = torch.zeros((B, 1, z1 - z0) + ps[1:], dtype=torch.float32, device=x.device)
        for j, (b, s) in enumerate(chunk):
            sl = (slice(b, b + 1), slice(None), slice(s[0] - z0, s[0] - z0 + roi[0])) + tuple(slice(a, a + r) for a, r in zip(s[1:], roi[1:]))
            out[sl] += pred[j:j + 1].float() * imp
            cnt[sl] += imp
    return {"z0": z0, "sum": out, "cnt": cnt, "n_tiles": len(mine), "padded_shape": list(ps)}


def merge_parts(parts: list[dict], shape, roi) -> torch.Tensor:
    """Sum the parts' accumulators, normalise, crop the padding — equals single-node sliding_window."""
    roi = tuple(int(r) for r in roi)
    pad, ps, _ = _pad_and_grid(tuple(shape), roi, 0.0)
    live = [p for p in parts if p["sum"] is not None]
    B, C = live[0]["sum"].shape[:2]
    out = torch.zeros((B, C) + ps, dtype=torch.float32)
    cnt = torch.zeros((B, 1) + ps, dtype=torch.float32)
    for p in live:
        z0, zl = p["z0"], p["sum"].shape[2]
        out[:, :, z0:z0 + zl] += p["sum"].cpu()
        cnt[:, :, z0:z0 + zl] += p["cnt"].cpu()
    out = out / cnt
    nd = len(shape) - 2
    if any(pad):
        crop = []
        for k in range(nd):
            lo = pad[2 * (nd - 1 - k)]
            crop.append(slice(lo, lo + shape[2 + k]))
        out = out[(slice(None), slice(None)) + tuple(crop)]
    return out


def tta_flips(x_dim: int, tta: str) -> list[list[int]]:
    """Flip sets for test-time augmentation: identity plus one flip per spatial axis (MONAI/nnU-Net style)."""
    if tta in ("none", None):
        return [[]]
    if tta != "flip":
        raise ValueError(f"unknown tta {tta!r}")
    return [[]] + [[-(x_dim - d)] for d in range(2, x_dim)]


def predict(x: torch.Tensor, model: Callable, roi=None, overlap: float = 0.5, sw_batch: int = 4,
            mode: str = "gaussian", tta: str = "none", average: str = "logits") -> torch.Tensor:
    """average='probs' softmaxes each augmented prediction before averaging (MONAI / nnU-Net practice for
    segmentation); 'logits' averages raw outputs (for regression / feature outputs)."""
    run = (lambda t: sliding_window(t, roi, sw_batch, model, overlap, mode)) if roi else model
    post = (lambda t: torch.softmax(t.float(), 1)) if average == "probs" else (lambda t: t)
    flips = tta_flips(x.dim(), tta)
    y = None
    for f in flips:
        out = post(run(torch.flip(x, f) if f else x))
        out = torch.flip(out, f) if f else out
        y = out if y is None else y + out
    return y / len(flips)
