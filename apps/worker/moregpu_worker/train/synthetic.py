"""Deterministic synthetic structured images/volumes for tests, examples and CI (no real data).

Each sample is a smooth random field plus 1–3 ellipsoidal "organs" with class-dependent texture, so a JEPA encoder
has real structure to learn and a label (the class of the dominant blob) for probes."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class SyntheticVolumes:
    def __init__(self, kind="2p5d", n=64, size=(64, 64), channels=3, seed=0, classes=3):
        self.kind, self.n, self.size, self.c, self.seed, self.classes = kind, int(n), tuple(size), int(channels), int(seed), int(classes)

    def __len__(self):
        return self.n

    def label(self, i: int) -> int:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(i))
        return int(torch.randint(0, self.classes, (1,), generator=g))

    def _one(self, i: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(i))
        cls = int(torch.randint(0, self.classes, (1,), generator=g))
        nd = 3 if self.kind in ("3d", "2p5d") else 2
        shape = (self.c,) + tuple(self.size) if self.kind == "2p5d" else tuple(self.size) if nd == 3 else tuple(self.size)
        if self.kind == "2p5d":
            spatial = (self.c,) + tuple(self.size)          # adjacent slices are a thin 3D slab
        elif self.kind == "3d":
            spatial = tuple(self.size)
        else:
            spatial = tuple(self.size)
        coarse = torch.randn((1, 1) + tuple(max(2, s // 8) for s in spatial), generator=g)
        mode = "trilinear" if len(spatial) == 3 else "bilinear"
        field = F.interpolate(coarse, size=spatial, mode=mode, align_corners=False)[0, 0] * 0.3
        grids = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in spatial], indexing="ij")
        for b in range(1 + int(torch.randint(0, 3, (1,), generator=g))):
            c = [float(torch.rand(1, generator=g)) * 1.2 - 0.6 for _ in spatial]
            r = [0.2 + 0.3 * float(torch.rand(1, generator=g)) for _ in spatial]
            if self.kind == "2p5d":
                c[0], r[0] = 0.0, 10.0                        # slab: blobs extend through adjacent slices
            inside = sum(((x - ci) / ri) ** 2 for x, ci, ri in zip(grids, c, r)) <= 1
            k = cls if b == 0 else int(torch.randint(0, self.classes, (1,), generator=g))
            freq = 3 + 4 * k
            tex = torch.sin(freq * grids[-1] * 3.14159) * torch.cos(freq * grids[-2] * 3.14159)
            field = field + inside.float() * (0.8 + 0.3 * tex)
        del shape
        if self.kind == "2p5d":
            return field.float()                              # (C, H, W)
        if self.kind == "3d":
            return field.float()[None]                        # (1, D, H, W)
        return field.float()[None].repeat(self.c, 1, 1)       # (C, H, W)

    def batch(self, idx: list[int]) -> torch.Tensor:
        return torch.stack([self._one(i % self.n) for i in idx])
