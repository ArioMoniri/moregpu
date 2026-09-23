"""Representation-collapse monitors (ADR-0109): per-dimension std and RankMe effective rank of embeddings."""
from __future__ import annotations

import torch


def embedding_monitors(z: torch.Tensor) -> dict:
    """z: (N, D) embeddings (e.g. mean-pooled encoder tokens on a fixed probe batch)."""
    z = z.detach().float()
    std = z.std(dim=0)
    zc = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)                     # RankMe uses the raw (uncentred) embeddings
    p = s / s.sum().clamp_min(1e-12)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return {"std_mean": float(std.mean()), "std_min": float(std.min()), "rankme": float(torch.exp(ent)),
            "centered_norm": float(zc.norm(dim=1).mean()), "dim": int(z.shape[1]), "n": int(z.shape[0])}


def alarms(m: dict, std_min: float = 1e-3, rank_min: float = 2.0) -> list[str]:
    out = []
    if m["std_mean"] < std_min:
        out.append(f"representation collapse: mean per-dim std {m['std_mean']:.2e} < {std_min:g}")
    if m["rankme"] < rank_min:
        out.append(f"representation collapse: RankMe {m['rankme']:.2f} < {rank_min:g}")
    return out
