"""I-JEPA multi-block masking for 2D (h, w) and 3D (d, h, w) token grids (ADR-0109).

Per sample: draw `n_targets` target blocks (scale × aspect), one large context block, remove every target token from the
context. Across the batch, each index set is truncated to the minimum size (as the I-JEPA collator does) so tensors
are rectangular; all target blocks share one length. Fully determined by the torch.Generator passed in."""
from __future__ import annotations

import math

import torch


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class MultiBlockMasker:
    def __init__(self, grid, n_targets=4, target_scale=(0.15, 0.2), target_aspect=(0.75, 1.5),
                 context_scale=(0.85, 1.0), min_keep=4, max_tries=50):
        self.grid = tuple(int(g) for g in grid)
        self.n = math.prod(self.grid)
        self.n_targets, self.ts, self.ta, self.cs = n_targets, target_scale, target_aspect, context_scale
        self.min_keep, self.max_tries = min_keep, max_tries

    def _uniform(self, g, lo, hi):
        return lo + float(torch.rand(1, generator=g)) * (hi - lo)

    def _block_shape(self, g, scale, aspect):
        s = self._uniform(g, *scale)
        n_tok = max(1, int(round(s * self.n)))
        if len(self.grid) == 2:
            H, W = self.grid
            a = math.exp(self._uniform(g, math.log(aspect[0]), math.log(aspect[1])))
            h = max(1, min(H, int(round(math.sqrt(n_tok * a)))))
            w = max(1, min(W, int(round(math.sqrt(n_tok / a)))))
            return (h, w)
        D, H, W = self.grid
        a1 = math.exp(self._uniform(g, math.log(aspect[0]), math.log(aspect[1])))
        a2 = math.exp(self._uniform(g, math.log(aspect[0]), math.log(aspect[1])))
        side = n_tok ** (1 / 3)
        return (max(1, min(D, int(round(side * a1)))), max(1, min(H, int(round(side / a1 * a2 ** 0.5)))),
                max(1, min(W, int(round(side / a2 ** 0.5)))))

    def _place(self, g, shape) -> torch.Tensor:
        m = torch.zeros(self.grid, dtype=torch.bool)
        starts = [int(torch.randint(0, gs - s + 1, (1,), generator=g)) for gs, s in zip(self.grid, shape)]
        m[tuple(slice(st, st + s) for st, s in zip(starts, shape))] = True
        return m.flatten()

    def sample_one(self, g):
        tshape = self._block_shape(g, self.ts, self.ta)
        targets = [self._place(g, tshape) for _ in range(self.n_targets)]
        union = torch.zeros(self.n, dtype=torch.bool)
        for t in targets:
            union |= t
        cshape = self._block_shape(g, self.cs, (1.0, 1.0))
        for _ in range(self.max_tries):
            ctx = self._place(g, cshape) & ~union
            if int(ctx.sum()) >= self.min_keep:
                break
        else:  # fall back to "everything that is not a target"
            ctx = ~union
        return ctx.nonzero().flatten(), [t.nonzero().flatten() for t in targets]

    def __call__(self, batch: int, g: torch.Generator):
        ctxs, tgts = [], []
        for _ in range(batch):
            c, ts = self.sample_one(g)
            ctxs.append(c); tgts.append(ts)
        kc = min(len(c) for c in ctxs)
        kt = min(len(t) for ts in tgts for t in ts)
        ctx = torch.stack([c[:kc] for c in ctxs])
        targets = [torch.stack([tgts[b][m][:kt] for b in range(batch)]) for m in range(self.n_targets)]
        return ctx, targets
