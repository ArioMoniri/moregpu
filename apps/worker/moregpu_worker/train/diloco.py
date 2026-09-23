"""Reference DiLoCo outer-loop math (ADR-0106). The coordinator's TS implementation
(apps/coordinator/lib/diloco.ts) must produce identical results; tests/py/test_diloco_crosslang.py pins goldens.

Outer step on the pseudo-gradient Δ = global − avg (PyTorch SGD(nesterov=True), no dampening):
    v ← μ·v + Δ ;  global ← global − η·(Δ + μ·v)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

Tensors = dict[str, torch.Tensor]


def weighted_average(items: list[tuple[Tensors, float]]) -> Tensors:
    total = float(sum(w for _, w in items))
    if not items or total <= 0:
        raise ValueError("weighted_average needs at least one item with positive total weight")
    keys = items[0][0].keys()
    out: Tensors = {}
    for k in keys:
        shape = items[0][0][k].shape
        acc = torch.zeros(shape, dtype=torch.float64)
        for t, w in items:
            if t[k].shape != shape:
                raise ValueError(f"{k}: shape {tuple(t[k].shape)} != {tuple(shape)}")
            acc += t[k].to(torch.float64) * (float(w) / total)
        out[k] = acc.to(torch.float32)
    return out


def drop_nonfinite(items: list[tuple[str, Tensors, float]]) -> tuple[list[tuple[str, Tensors, float]], list[str]]:
    """Split (worker, tensors, weight) items into finite ones and the ids of workers with any NaN/Inf."""
    kept = [x for x in items if all(torch.isfinite(v).all() for v in x[1].values())]
    dropped = [x[0] for x in items if x not in kept]
    return kept, dropped


@dataclass
class OuterState:
    global_: Tensors
    momentum: Tensors = field(default_factory=dict)
    round: int = 0

    @classmethod
    def init(cls, global_: Tensors) -> "OuterState":
        g = {k: v.detach().to(torch.float32).clone() for k, v in global_.items()}
        return cls(g, {k: torch.zeros_like(v) for k, v in g.items()}, 0)


def outer_step(st: OuterState, avg: Tensors, lr: float, momentum: float) -> OuterState:
    for k, g in st.global_.items():
        d = g - avg[k].to(torch.float32)
        v = st.momentum[k]
        v.mul_(momentum).add_(d)
        g.sub_(lr * (d + momentum * v))
    st.round += 1
    return st
