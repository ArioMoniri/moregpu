"""Per-round inner learning-rate schedule, shared by the coordinator (apps/coordinator/lib/schedule.ts) and the
in-process reference (moregpu_worker.train.local). Cross-language golden: tests/goldens/lr_schedule.json.

progress = midpoint of the round: (samples_seen + round_samples/2) / target_samples, or (round + 0.5) / max_rounds.
Schedule `cosine`: linear warmup over the first warmup_frac of progress, then cosine decay from lr to min_lr."""
from __future__ import annotations

import math


def progress(cfg: dict, samples_seen: int, round_samples: int, round_: int) -> float | None:
    if cfg.get("target_samples"):
        return (samples_seen + round_samples / 2) / float(cfg["target_samples"])
    if cfg.get("max_rounds"):
        return (round_ + 0.5) / float(cfg["max_rounds"])
    return None


def lr_at(cfg: dict, p: float | None) -> float:
    base = float(cfg["lr"])
    sch = cfg.get("lr_schedule") or {}
    if sch.get("kind", "constant") == "constant" or p is None:
        return base
    if sch["kind"] != "cosine":
        raise ValueError(f"unknown lr schedule {sch['kind']!r}")
    w, lo = float(sch.get("warmup_frac", 0.0)), float(sch.get("min_lr", 0.0))
    p = min(1.0, max(0.0, p))
    if w > 0 and p < w:
        return base * p / w
    q = (p - w) / max(1e-12, 1.0 - w)
    return lo + (base - lo) * 0.5 * (1.0 + math.cos(math.pi * q))
