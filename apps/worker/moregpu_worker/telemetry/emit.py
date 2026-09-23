"""Standalone telemetry emitter (ADR-0111) — the same ``moregpu.telemetry/1`` JSONL the coordinator writes, for
non-MoreGPU runners too (a DDP or single-process control), so their numbers land in the same tables.

    from moregpu_worker.telemetry.emit import JsonlEmitter, PhaseTimer
    em = JsonlEmitter("runs/ddp.jsonl", config=cfg)          # or JsonlEmitter.from_env("ddp-rank0")
    pt = PhaseTimer()
    for step, batch in enumerate(loader_iter):
        with pt.phase("data"):    x, y = next(loader_iter)
        with pt.phase("compute"): loss = train_step(x, y)
        with pt.phase("network"): dist.all_reduce(...)      # if comm is not overlapped with compute
        em.emit("external_round", runner="ddp", world_size=W, rank=R, round=step, samples=len(x), **pt.lap())
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .hw import fingerprint
from .schema import PHASES, SCHEMA, SCHEMA_ID, config_hash, validate

_AUTO = object()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def default_git_sha() -> str | None:
    """MOREGPU_GIT_SHA if set, else ``git rev-parse HEAD`` of the cwd, else None."""
    env = os.environ.get("MOREGPU_GIT_SHA")
    if env:
        return env
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        sha = p.stdout.strip()
        return sha if p.returncode == 0 and sha else None
    except (OSError, subprocess.SubprocessError):
        return None


class JsonlEmitter:
    """Append-only JSONL writer. ``emit(kind, **fields)`` fills ``schema``/``kind``/``ts`` and — when the kind has
    those fields and the caller did not pass them — ``git_sha``, ``config_hash`` and ``hw``; validates (``strict``
    raises ``ValueError`` and writes nothing) and appends one line. Thread-safe within a process; one file per process
    (e.g. per DDP rank) avoids interleaving across processes."""

    def __init__(self, path, *, git_sha=_AUTO, config=None, config_hash: str | None = None, hw=_AUTO,
                 strict: bool = True):
        self.path = Path(path)
        self.git_sha = default_git_sha() if git_sha is _AUTO else git_sha
        self.config_hash = config_hash if config_hash is not None else (
            _cfg_hash(config) if config is not None else None)
        self._hw = hw
        self.strict = strict
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, name: str = "run", **kw) -> "JsonlEmitter | None":
        """MOREGPU_TELEMETRY_FILE, else $MOREGPU_TELEMETRY_DIR/<name>.jsonl, else None (telemetry off).
        ``moregpu bench`` sets both for each repeat."""
        f = os.environ.get("MOREGPU_TELEMETRY_FILE")
        if f:
            return cls(f, **kw)
        d = os.environ.get("MOREGPU_TELEMETRY_DIR")
        return cls(Path(d) / f"{name}.jsonl", **kw) if d else None

    @property
    def hw(self):
        if self._hw is _AUTO:
            self._hw = fingerprint()
        return self._hw

    def emit(self, kind: str, **fields) -> dict:
        rec = {"schema": SCHEMA_ID, "kind": kind, "ts": utc_now(), **fields}
        props = SCHEMA["$defs"].get(kind, {}).get("properties", {})
        for key, val in (("git_sha", lambda: self.git_sha), ("config_hash", lambda: self.config_hash),
                         ("hw", lambda: self.hw)):
            if key in props and key not in rec:
                rec[key] = val()
        errs = validate(rec)
        if errs and self.strict:
            raise ValueError(f"invalid {kind} telemetry record: " + "; ".join(errs))
        line = json.dumps(rec) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
        return rec


def _cfg_hash(cfg) -> str:
    return config_hash(cfg)


class PhaseTimer:
    """Accumulates compute/data/serialize/network/wait seconds over a window and returns a breakdown whose five
    phases sum to ``wall_s`` exactly: untracked time becomes ``wait_s`` (wall − others). If explicitly added phases
    exceed the measured wall (e.g. overlapped async comm), wait is 0 and ``wall_s`` is the phase sum — never a
    negative phase. Phases must not nest (double counting)."""

    NAMES = tuple(p[:-2] for p in PHASES)   # compute, data, serialize, network, wait

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self._t0 = clock()
        self._t1 = None
        self._acc = dict.fromkeys(self.NAMES, 0.0)
        self._active: str | None = None

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *exc):
        self._t1 = self.clock()
        return False

    def reset(self):
        self._t0, self._t1 = self.clock(), None
        self._acc = dict.fromkeys(self.NAMES, 0.0)

    def add(self, name: str, seconds: float):
        if name not in self._acc:
            raise ValueError(f"unknown phase {name!r} (one of {self.NAMES})")
        if seconds < 0:
            raise ValueError(f"negative duration for phase {name!r}")
        self._acc[name] += float(seconds)

    @contextmanager
    def phase(self, name: str):
        if name not in self._acc:
            raise ValueError(f"unknown phase {name!r} (one of {self.NAMES})")
        if self._active is not None:
            raise RuntimeError(f"nested phase {name!r} inside {self._active!r} would double count")
        self._active = name
        t = self.clock()
        try:
            yield
        finally:
            self._acc[name] += self.clock() - t
            self._active = None

    def record(self) -> dict:
        end = self._t1 if self._t1 is not None else self.clock()
        wall = end - self._t0
        a = self._acc
        others = a["compute"] + a["data"] + a["serialize"] + a["network"]
        wait = a["wait"] + max(0.0, wall - others - a["wait"])
        # sum in the documented order so sum(phases) == wall_s bit-for-bit
        total = a["compute"] + a["data"] + a["serialize"] + a["network"] + wait
        return {"wall_s": total, "compute_s": a["compute"], "data_s": a["data"], "serialize_s": a["serialize"],
                "network_s": a["network"], "wait_s": wait}

    def lap(self) -> dict:
        rec = self.record()
        self.reset()
        return rec
