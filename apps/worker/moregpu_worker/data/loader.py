"""Fast path (ADR-0110): pre-tiled ``.npy`` shards + ``index.json``, memory-mapped reads, a seeded DataLoader.

``write_shards`` turns ``(name, array)`` pairs — each array a stack of samples ``(N, *sample_shape)`` — into one
``.npy`` shard per pair (cast to ``dtype``) and an ``index.json`` carrying every shard's sha256. ``ShardDataset``
memory-maps shards lazily in each DataLoader worker process, so reads touch only the requested samples.
"""
from __future__ import annotations

import bisect
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cache import sha256_file
from .refs import IntegrityError

INDEX_VERSION = 1


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-.")[:60] or "shard"


def write_shards(arrays: Iterable[tuple[str, np.ndarray]], out_dir, dtype: str = "float16") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np_dtype = np.dtype(dtype)
    shards, sample_shape, total = [], None, 0
    for i, (name, arr) in enumerate(arrays):
        arr = np.asarray(arr)
        if arr.ndim < 1:
            raise ValueError(f"shard {name!r}: need an array of samples (N, ...), got a scalar")
        if sample_shape is None:
            sample_shape = list(arr.shape[1:])
        elif list(arr.shape[1:]) != sample_shape:
            raise ValueError(f"shard {name!r}: sample shape {list(arr.shape[1:])} != {sample_shape}")
        fname = f"{i:05d}-{_safe(name)}.npy"
        np.save(out / fname, np.ascontiguousarray(arr, dtype=np_dtype))
        shards.append({"name": str(name), "file": fname, "n": int(arr.shape[0]), "sha256": sha256_file(out / fname)})
        total += int(arr.shape[0])
    if not shards:
        raise ValueError("write_shards: no arrays given")
    index = {"version": INDEX_VERSION, "dtype": np_dtype.name, "sample_shape": sample_shape, "total": total,
             "shards": shards}
    tmp = out / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True))
    os.replace(tmp, out / "index.json")
    return index


class ShardDataset(Dataset):
    """Samples of a shard directory as float32 tensors (``dtype`` to override)."""

    def __init__(self, index, verify: bool = False, dtype: torch.dtype = torch.float32):
        p = Path(index)
        if p.is_dir():
            p = p / "index.json"
        self.root = p.parent
        self.index = json.loads(p.read_text())
        self.dtype = dtype
        self._starts, acc = [], 0
        for sh in self.index["shards"]:
            self._starts.append(acc)
            acc += int(sh["n"])
        self._len = acc
        self._mm: dict[int, np.ndarray] = {}
        if verify:
            for sh in self.index["shards"]:
                got = sha256_file(self.root / sh["file"])
                if got != sh["sha256"]:
                    raise IntegrityError(f"shard {sh['file']}: sha256 {got} != {sh['sha256']}")

    def __len__(self) -> int:
        return self._len

    def __getstate__(self):
        st = dict(self.__dict__)
        st["_mm"] = {}  # memmaps are reopened in each worker process
        return st

    def _shard(self, s: int) -> np.ndarray:
        mm = self._mm.get(s)
        if mm is None:
            mm = self._mm[s] = np.load(self.root / self.index["shards"][s]["file"], mmap_mode="r")
        return mm

    def __getitem__(self, i: int) -> torch.Tensor:
        if i < 0:
            i += self._len
        if not 0 <= i < self._len:
            raise IndexError(i)
        s = bisect.bisect_right(self._starts, i) - 1
        return torch.from_numpy(np.array(self._shard(s)[i - self._starts[s]], dtype=np.float32)).to(self.dtype)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)


def make_loader(dataset, batch_size: int, workers: int = 2, pin_memory: bool | None = None, prefetch_factor: int = 2,
                persistent_workers: bool = True, seed: int = 0, shuffle: bool = True, drop_last: bool = False
                ) -> DataLoader:
    """A DataLoader whose sample order depends only on ``seed`` (seeded generator; per-worker seeds derived from it).
    ``pin_memory`` defaults to ``torch.cuda.is_available()``."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    kw: dict = {}
    if workers > 0:
        kw.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, worker_init_fn=_seed_worker)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, generator=g,
                      pin_memory=torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
                      drop_last=drop_last, **kw)


def _batch_len(b) -> int:
    if isinstance(b, (list, tuple)):
        b = b[0]
    return int(b.shape[0]) if hasattr(b, "shape") else len(b)


def measure_throughput(loader, max_batches: int) -> dict:
    n = batches = 0
    t0 = time.perf_counter()
    for b in loader:
        n += _batch_len(b)
        batches += 1
        if batches >= max_batches:
            break
    dt = max(time.perf_counter() - t0, 1e-9)
    return {"samples_per_s": n / dt, "batches": batches, "samples": n, "seconds": dt}
