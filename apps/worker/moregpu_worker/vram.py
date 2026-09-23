"""``MOREGPU_VRAM_FRACTION``: cap the share of each CUDA device this worker process may allocate.

When the variable is set to a fraction ``0 < f <= 1`` and the worker runs on CUDA, :func:`apply` calls
``torch.cuda.set_per_process_memory_fraction(f, device=i)`` for every visible CUDA device at start-up. The caching
allocator then raises an out-of-memory error instead of growing past ``f × total memory``, so a donor can keep part of
the GPU for its own work. The value is validated even on a CPU/MPS worker: NaN, infinities, non-numbers and anything
outside ``(0, 1]`` are refused with :class:`VramFractionError`, so a typo fails the start-up instead of being ignored.

The applied value is reported as ``vram_fraction`` in :func:`moregpu_worker.telemetry.hw.fingerprint` (and so in every
telemetry record's ``hw``); it is ``None`` when unset or when there is no CUDA device to apply it to.
"""
from __future__ import annotations

import math
import os

ENV = "MOREGPU_VRAM_FRACTION"


class VramFractionError(ValueError):
    """``MOREGPU_VRAM_FRACTION`` is not a number in (0, 1]."""


_REQUESTED: float | None = None
_APPLIED: float | None = None


def parse(raw) -> float | None:
    """``None`` / blank → ``None``; otherwise a float with ``0 < f <= 1`` or :class:`VramFractionError`."""
    if raw is None or not str(raw).strip():
        return None
    try:
        f = float(str(raw).strip())
    except ValueError:
        raise VramFractionError(f"{ENV}={raw!r} is not a number; expected a fraction 0 < f <= 1 (e.g. 0.5)") from None
    if not math.isfinite(f) or not 0.0 < f <= 1.0:
        raise VramFractionError(f"{ENV}={raw!r} is out of range; expected a finite fraction 0 < f <= 1 (e.g. 0.5)")
    return f


def apply(torch_mod=None, env=None, device: str = "cuda") -> float | None:
    """Validate ``MOREGPU_VRAM_FRACTION`` (from ``env``, default ``os.environ``) and, when ``device`` is CUDA and CUDA is
    available, apply it to every visible CUDA device. Returns the applied fraction, or ``None`` if nothing was applied."""
    global _REQUESTED, _APPLIED
    f = parse((os.environ if env is None else env).get(ENV))
    _REQUESTED, _APPLIED = f, None
    if f is None or not str(device).startswith("cuda"):
        return None
    if torch_mod is None:
        import torch as torch_mod
    if not torch_mod.cuda.is_available():
        return None
    for i in range(int(torch_mod.cuda.device_count())):
        torch_mod.cuda.set_per_process_memory_fraction(f, device=i)
    _APPLIED = f
    return f


def applied() -> float | None:
    """The fraction applied at start-up (``None``: unset, or no CUDA device)."""
    return _APPLIED


def status() -> dict:
    return {"requested": _REQUESTED, "applied": _APPLIED}


def reset() -> None:
    """Forget the recorded state (tests)."""
    global _REQUESTED, _APPLIED
    _REQUESTED = _APPLIED = None
