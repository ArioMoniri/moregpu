"""``pushed://<id>`` blobs: data the coordinator streams to this worker in ordered chunks (ADR-0110).

Staging is RAM-first: ``MOREGPU_STAGE_DIR`` (alias: ``MOREGPU_PUSHED_DIR``) if set, else ``/dev/shm`` when it has at
least max(2 GiB, the blob's size) free, else the OS temp dir — which is usually DISK, so a staged blob can touch
persistent storage there. A blob is only readable after ``end`` has verified its declared size and sha256. Staged
files are deleted by ``drop``/``close`` (``close`` also runs at interpreter exit and on a coordinator ``welcome``); a
worker that is killed (SIGKILL / power loss) can leave ``moregpu-blob-*`` files behind in a disk staging dir.

Caps: one blob <= ``MOREGPU_PUSH_MAX_BYTES`` (20 GiB), all staged blobs together <= ``MOREGPU_BLOB_TOTAL_MAX_BYTES``
(40 GiB), and a blob is refused up front when its staging filesystem has less free space than its declared size.

One process-wide store (:func:`default_store`) is shared by the data plane (``pushed://`` data refs) and model
fetches (``pushed://`` model sources), so ``/data/push`` delivers both.
"""
from __future__ import annotations

import atexit
import hashlib
import os
import re
import shutil
import tempfile
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path

from .refs import GiB, IntegrityError, check_sha256

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUFFIX_RE = re.compile(r"^(\.[A-Za-z0-9]{1,16}){0,3}$")   # e.g. .nii.gz, .safetensors


def stage_root(need: int = 0) -> str:
    """Staging directory: ``MOREGPU_STAGE_DIR`` (or its alias ``MOREGPU_PUSHED_DIR``) if writable, else ``/dev/shm``
    with >= max(2 GiB, ``need``) free, else the temp dir."""
    env = os.environ.get("MOREGPU_STAGE_DIR") or os.environ.get("MOREGPU_PUSHED_DIR")
    if env:
        try:
            os.makedirs(env, exist_ok=True)
            if os.access(env, os.W_OK):
                return env
        except OSError:  # pragma: no cover - unwritable override
            pass
    shm = "/dev/shm"
    if os.path.isdir(shm) and os.access(shm, os.W_OK):
        try:
            if shutil.disk_usage(shm).free >= max(2 * GiB, int(need)):
                return shm
        except OSError:  # pragma: no cover
            pass
    return tempfile.gettempdir()  # pragma: no cover - depends on host /dev/shm


@dataclass
class _Blob:
    path: Path
    sha256: str
    size: int
    next_k: int = 0
    written: int = 0
    done: bool = False
    hasher: "hashlib._Hash" = field(default_factory=hashlib.sha256)


class BlobStore:
    def __init__(self, stage_dir=None, max_bytes: int | None = None, total_max_bytes: int | None = None):
        self._stage = str(stage_dir) if stage_dir is not None else None
        self.max_bytes = int(max_bytes if max_bytes is not None
                             else os.environ.get("MOREGPU_PUSH_MAX_BYTES", str(20 * GiB)))
        self.total_max_bytes = int(total_max_bytes if total_max_bytes is not None
                                   else os.environ.get("MOREGPU_BLOB_TOTAL_MAX_BYTES") or 40 * GiB)
        self._blobs: dict[str, _Blob] = {}
        self._lock = threading.RLock()
        ref = weakref.ref(self)
        atexit.register(lambda: (ref() is not None) and ref().close())

    @property
    def stage_dir(self) -> str:
        return self._stage_dir(0)

    def _stage_dir(self, need: int) -> str:
        d = self._stage or stage_root(need)
        os.makedirs(d, exist_ok=True)
        return d

    def staged_bytes(self) -> int:
        """Declared bytes of every blob currently staged (in progress or ended)."""
        with self._lock:
            return sum(b.size for b in self._blobs.values())

    def begin(self, id: str, sha256: str, size: int, suffix: str = "") -> dict:
        """Start (or restart) blob ``id``. ``suffix`` (e.g. ``.nii.gz``) is kept on the staged file for readers."""
        if not isinstance(id, str) or not _ID_RE.match(id):
            raise ValueError(f"bad blob id {id!r}")
        sha = check_sha256(sha256)
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"bad blob size {size!r}")
        if size > self.max_bytes:
            raise ValueError(f"blob of {size} bytes exceeds staging cap {self.max_bytes} (MOREGPU_PUSH_MAX_BYTES)")
        suffix = suffix or ""
        if not _SUFFIX_RE.match(suffix):
            raise ValueError(f"bad blob suffix {suffix!r}")
        with self._lock:
            self.drop(id)
            total = self.staged_bytes() + size
            if total > self.total_max_bytes:
                raise ValueError(f"blob {id!r}: staging {size} more bytes would hold {total} bytes, over the total cap "
                                 f"{self.total_max_bytes} (MOREGPU_BLOB_TOTAL_MAX_BYTES) — drop blobs first")
            root = self._stage_dir(size)
            try:
                free = shutil.disk_usage(root).free
            except OSError:  # pragma: no cover
                free = None
            if free is not None and free < size:
                raise ValueError(f"blob {id!r}: {size} bytes do not fit in staging dir {root} ({free} bytes free)")
            fd, p = tempfile.mkstemp(prefix=f"moregpu-blob-{id}-", suffix=suffix, dir=root)
            os.close(fd)
            self._blobs[id] = _Blob(Path(p), sha, size)
            return {"ok": True, "id": id, "staging": "ram" if os.path.realpath(root) == "/dev/shm" else "disk"}

    def chunk(self, id: str, k: int, data: bytes) -> dict:
        """Append chunk ``k``; chunks must arrive strictly in order 0, 1, 2, ..."""
        with self._lock:
            b = self._get(id)
            if b.done:
                raise ValueError(f"blob {id!r} already ended")
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise ValueError("chunk data must be bytes")
            if k != b.next_k:
                raise ValueError(f"blob {id!r}: expected chunk {b.next_k}, got {k}")
            if b.written + len(data) > b.size:
                self.drop(id)
                raise ValueError(f"blob {id!r}: more bytes than the declared size {b.size} — aborted")
            with open(b.path, "ab") as f:
                f.write(data)
            b.hasher.update(data)
            b.written += len(data)
            b.next_k += 1
            return {"ok": True, "id": id, "k": k, "bytes": b.written}

    def end(self, id: str) -> Path:
        with self._lock:
            b = self._get(id)
            if b.done:
                return b.path
            got = b.hasher.hexdigest()
            if b.written != b.size or got != b.sha256:
                self.drop(id)
                raise IntegrityError(f"blob {id!r}: got {b.written}/{b.size} bytes, sha256 {got} != {b.sha256}")
            b.done = True
            return b.path

    def info(self, id: str) -> dict:
        with self._lock:
            b = self._get(id)
            return {"id": id, "sha256": b.sha256, "size": b.size, "done": b.done}

    def path(self, id: str) -> Path:
        with self._lock:
            b = self._blobs.get(id)
            if b is None or not b.done:
                raise KeyError(f"blob {id!r} not pushed (or not ended)")
            return b.path

    def drop(self, id: str) -> None:
        with self._lock:
            b = self._blobs.pop(id, None)
            if b is not None:
                try:
                    os.unlink(b.path)
                except FileNotFoundError:  # pragma: no cover
                    pass

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._blobs)

    def close(self) -> None:
        with self._lock:
            for id in list(self._blobs):
                self.drop(id)

    def _get(self, id: str) -> _Blob:
        b = self._blobs.get(id)
        if b is None:
            raise KeyError(f"blob {id!r} not begun — call blob_begin first")
        return b


_DEFAULT: BlobStore | None = None
_DEFAULT_LOCK = threading.Lock()


def default_store() -> BlobStore:
    """The process-wide BlobStore (created on first use): the data plane's and ``pushed://`` model sources' store."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = BlobStore()
        return _DEFAULT
