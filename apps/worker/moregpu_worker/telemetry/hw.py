"""Hardware/software fingerprint for telemetry records (ADR-0111). No PII: the hostname is only kept as a salted,
truncated SHA-256 (enough to tell machines apart within one run's JSONL, not to recover the name); no user names,
paths, IPs, serial numbers or UUIDs are collected.

The salt is a fixed domain tag plus 16 random bytes drawn once per run (per worker process): ``host_hash`` is stable
within a run, but differs across runs and machines, so it cannot be dictionary-attacked back to a hostname with a
precomputed table, nor used to link the same machine across separate runs / exported JSONL files."""
from __future__ import annotations

import hashlib
import os
import platform
import socket

_SALT = b"moregpu.telemetry/1:host:" + os.urandom(16)   # per-run random salt (see module doc)


def host_hash(hostname: str | None = None) -> str:
    name = socket.gethostname() if hostname is None else hostname
    return hashlib.sha256(_SALT + name.encode()).hexdigest()[:16]


def _import_torch():
    try:
        import torch
        return torch
    except Exception:  # pragma: no cover - torch is a hard dependency of the worker
        return None


def _ram_bytes():
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def fingerprint(torch_mod=None) -> dict:
    t = torch_mod if torch_mod is not None else _import_torch()
    fp = {
        "host_hash": host_hash(),
        "os": platform.system(), "os_release": platform.release(), "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": None, "cuda": None, "device": None, "capability": None, "total_mem_bytes": None,
        "cpu_count": os.cpu_count(), "ram_bytes": _ram_bytes(),
    }
    if t is None:
        return fp
    fp["torch"] = str(getattr(t, "__version__", None))
    fp["cuda"] = getattr(getattr(t, "version", None), "cuda", None)
    try:
        if t.cuda.is_available():
            p = t.cuda.get_device_properties(t.cuda.current_device())
            fp["device"], fp["capability"] = str(p.name), f"{p.major}.{p.minor}"
            fp["total_mem_bytes"] = int(p.total_memory)
        else:
            fp["device"] = "cpu"
    except Exception:
        pass
    return fp
