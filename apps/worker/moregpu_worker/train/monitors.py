"""Representation-collapse monitors (ADR-0109): per-dimension std and RankMe effective rank of embeddings.

``rankme`` is RankMe as published (raw, uncentred embeddings); it is reported for curves. The rank ALARM uses
``rankme_centered`` (the same entropy-rank of the mean-centred embeddings): a large mean vector shared by all inputs
dominates the uncentred spectrum (mean-pooled ViT features of CT slices read ~1.1 at random init), which is not
collapse. Complete collapse (constant embeddings) is caught by the per-dimension std; dimensional collapse (spread
along one direction) by the centred rank.
"""
from __future__ import annotations

import torch


def embedding_monitors(z: torch.Tensor) -> dict:
    """z: (N, D) embeddings (e.g. mean-pooled encoder tokens on a fixed probe batch)."""
    z = z.detach().float()
    std = z.std(dim=0)
    zc = z - z.mean(dim=0, keepdim=True)
    return {"std_mean": float(std.mean()), "std_min": float(std.min()), "rankme": _rankme(z),
            "rankme_centered": _rankme(zc), "centered_norm": float(zc.norm(dim=1).mean()), "dim": int(z.shape[1]),
            "n": int(z.shape[0])}


def _rankme(z: torch.Tensor) -> float:
    s = torch.linalg.svdvals(z.double())
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


def alarms(m: dict, std_min: float = 1e-3, rank_min: float = 2.0) -> list[str]:
    out = []
    if m["std_mean"] < std_min:
        out.append(f"representation collapse: mean per-dim std {m['std_mean']:.2e} < {std_min:g}")
    rank = m.get("rankme_centered", m["rankme"])
    if rank < rank_min:
        out.append(f"representation collapse: centred RankMe {rank:.2f} < {rank_min:g}")
    return out
