"""Content-addressed (sha256) on-disk cache with a byte cap and LRU eviction (ADR-0110).

Entries are files named by their lowercase hex sha256 directly under ``dir``. Recency is the file mtime (touched on
every hit) so the LRU order survives a restart. Every ``put`` hashes the content; a declared sha256 that does not
match raises :class:`~moregpu_worker.data.refs.IntegrityError` and nothing is stored.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

from .refs import IntegrityError, check_sha256

_BUF = 1 << 20


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_BUF)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class ContentCache:
    def __init__(self, dir, cap_bytes: int):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cap_bytes = int(cap_bytes)
        self._lock = threading.RLock()
        self._lru: "OrderedDict[str, int]" = OrderedDict()  # sha -> size, oldest first
        self._hits = self._misses = self._evictions = 0
        found = []
        for p in self.dir.iterdir():
            try:
                check_sha256(p.name)
            except ValueError:
                continue
            if p.is_file():
                st = p.stat()
                found.append((st.st_mtime, p.name, st.st_size))
        for _, sha, size in sorted(found):
            self._lru[sha] = size
        with self._lock:
            self._evict(0)

    # ------------------------------------------------------------------ public
    def get(self, sha256: str) -> Path | None:
        sha = check_sha256(sha256)
        with self._lock:
            p = self.dir / sha
            if sha in self._lru and p.exists():
                self._lru.move_to_end(sha)
                try:
                    os.utime(p)
                except OSError:  # pragma: no cover - read-only cache dir
                    pass
                self._hits += 1
                return p
            self._lru.pop(sha, None)
            self._misses += 1
            return None

    def put(self, data_or_path, sha256: str | None = None, move: bool = False) -> Path:
        """Store bytes or a file. ``move=True`` renames the source file into the cache (it must be on the same
        filesystem, e.g. a temp file created under ``self.dir``); otherwise the file is copied."""
        want = check_sha256(sha256) if sha256 is not None else None
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=self.dir)
        try:
            if isinstance(data_or_path, (bytes, bytearray, memoryview)):
                with os.fdopen(fd, "wb") as f:
                    f.write(data_or_path)
            else:
                os.close(fd)
                src = os.fspath(data_or_path)
                if move:
                    shutil.move(src, tmp)
                else:
                    shutil.copyfile(src, tmp)
            got = sha256_file(tmp)
            if want is not None and got != want:
                raise IntegrityError(f"sha256 mismatch: expected {want}, got {got}")
            size = os.path.getsize(tmp)
            if size > self.cap_bytes:
                raise ValueError(f"entry of {size} bytes exceeds cache cap {self.cap_bytes}")
            with self._lock:
                dst = self.dir / got
                if got in self._lru and dst.exists():
                    os.unlink(tmp)
                    self._lru.move_to_end(got)
                    os.utime(dst)
                    return dst
                self._evict(size)
                os.replace(tmp, dst)
                self._lru[got] = size
                return dst
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def clear(self) -> None:
        with self._lock:
            for sha in list(self._lru):
                self._remove(sha)

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._lru), "bytes": sum(self._lru.values()), "cap_bytes": self.cap_bytes,
                    "hits": self._hits, "misses": self._misses, "evictions": self._evictions}

    # ------------------------------------------------------------------ internals
    def _remove(self, sha: str) -> None:
        self._lru.pop(sha, None)
        try:
            os.unlink(self.dir / sha)
        except FileNotFoundError:  # pragma: no cover - removed behind our back
            pass

    def _evict(self, incoming: int) -> None:
        total = sum(self._lru.values())
        while self._lru and total + incoming > self.cap_bytes:
            sha, size = next(iter(self._lru.items()))
            self._remove(sha)
            total -= size
            self._evictions += 1
