"""Deterministic sample assignment (ADR-0107).

A manifest of N samples + a seed defines one permutation per epoch (SplitMix64-driven Fisher–Yates, identical in the
coordinator's apps/coordinator/lib/sharding.ts). Each round consumes the next Σ allocation indices from that stream
and splits them into contiguous slices in worker order. Only indices travel, never pixels.
"""
from __future__ import annotations

_M = (1 << 64) - 1


def _splitmix(state: int):
    while True:
        state = (state + 0x9E3779B97F4A7C15) & _M
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M
        yield z ^ (z >> 31)


def permutation(n: int, seed: int, epoch: int) -> list[int]:
    rng = _splitmix((seed * 0x100000001B3 + epoch) & _M)
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = next(rng) % (i + 1)
        p[i], p[j] = p[j], p[i]
    return p


class SampleStream:
    def __init__(self, n: int, seed: int, epoch: int = 0, cursor: int = 0):
        if n <= 0:
            raise ValueError("manifest is empty")
        self.n, self.seed, self.epoch, self.cursor = n, seed, epoch, cursor
        self._perm = permutation(n, seed, epoch)

    def take(self, k: int) -> list[int]:
        out: list[int] = []
        while len(out) < k:
            if self.cursor >= self.n:
                self.epoch += 1
                self.cursor = 0
                self._perm = permutation(self.n, self.seed, self.epoch)
            m = min(k - len(out), self.n - self.cursor)
            out += self._perm[self.cursor: self.cursor + m]
            self.cursor += m
        return out

    def state(self) -> dict:
        return {"n": self.n, "seed": self.seed, "epoch": self.epoch, "cursor": self.cursor}

    @classmethod
    def from_state(cls, st: dict) -> "SampleStream":
        return cls(st["n"], st["seed"], st["epoch"], st["cursor"])


def split(indices: list[int], sizes: list[int]) -> list[list[int]]:
    if sum(sizes) != len(indices):
        raise ValueError(f"sizes sum {sum(sizes)} != {len(indices)} indices")
    out, o = [], 0
    for s in sizes:
        out.append(indices[o: o + s])
        o += s
    return out


def allocate(per_worker: int, n: int, speeds: list[float], mode: str = "fixed", remaining: int | None = None) -> list[int]:
    """Samples per worker this round. `fixed`: per_worker each. `proportional`: same total, ∝ measured speed (≥1 each).
    `remaining` truncates the round so the session stops exactly at target_samples (largest-first trim)."""
    total = per_worker * n
    if mode == "fixed":
        alloc = [per_worker] * n
    elif mode == "proportional":
        s = [max(float(x), 1e-9) for x in speeds]
        raw = [total * x / sum(s) for x in s]
        alloc = [max(1, int(r)) for r in raw]
        # distribute the rounding remainder by largest fractional part
        order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        k = 0
        while sum(alloc) < total:
            alloc[order[k % n]] += 1; k += 1
        while sum(alloc) > total:
            i = max(range(n), key=lambda j: alloc[j]); alloc[i] -= 1
    else:
        raise ValueError(f"unknown allocation mode {mode!r}")
    if remaining is not None and remaining < sum(alloc):
        alloc = [int(a * remaining / sum(alloc)) for a in alloc]
        i = 0
        while sum(alloc) < remaining:
            alloc[i % n] += 1; i += 1
    return alloc
