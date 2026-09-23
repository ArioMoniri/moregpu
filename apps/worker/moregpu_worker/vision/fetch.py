"""Artefact fetchers for model specs (ADR-0113). A fetch is `fetch(source, sha256=None) -> Path`.

* `file:///p`   — only under an allowed root (MOREGPU_MODEL_ROOTS, os.pathsep-separated); symlinks are resolved first.
* `pushed://id` — a blob pushed to this worker through the data plane (`/data/push` → `blob_begin/chunk/end`) and held
  by its :class:`~moregpu_worker.data.blobs.BlobStore`: only a fully received blob whose size + sha256 were verified at
  `blob_end` resolves, and its sha256 must equal the spec's. A file that merely exists in a directory never resolves.
* `https://…`   — host must be in MOREGPU_MODEL_HOSTS (comma-separated; falls back to MOREGPU_DATA_HOSTS when unset),
  every redirect hop is re-checked, the body is capped at MOREGPU_MODEL_MAX_BYTES (default 20 GiB) and sha256 is
  mandatory; downloaded into a content-addressed cache (MOREGPU_MODEL_CACHE) named by its sha256. Plain `http://` only
  to an allowlisted loopback host (local mirrors / tests). Same helper as the data plane (moregpu_worker.data.http).
* `hf://org/repo[@rev]/path` — one file via huggingface_hub (never `from_pretrained`, never remote code). Must be
  immutable: either the spec pins a sha256, or `rev` is a full 40-hex commit id (branches / tags are refused).

Every loaded artefact is ALSO hashed and compared by the adapter layer before it is opened (`adapters.base.fetch_verified`).
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.parse
from pathlib import Path

from ..data import http as H
from ..data.refs import GiB, RefDenied
from .errors import IntegrityError, RefusedSource

_PUSHED_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CHUNK = 1 << 20


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_hub_download(repo_id: str, filename: str, revision: str | None = None) -> str:  # pragma: no cover - network
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def _env_paths(name: str) -> list[Path]:
    return [Path(p) for p in os.environ.get(name, "").split(os.pathsep) if p]


def model_hosts() -> tuple[list[str], str]:
    """(allowlisted hosts, the env var they came from): MOREGPU_MODEL_HOSTS, else MOREGPU_DATA_HOSTS."""
    for name in ("MOREGPU_MODEL_HOSTS", "MOREGPU_DATA_HOSTS"):
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return [h.strip().lower() for h in raw.split(",") if h.strip()], name
    return [], "MOREGPU_MODEL_HOSTS"


def model_max_bytes() -> int:
    return int(os.environ.get("MOREGPU_MODEL_MAX_BYTES") or 20 * GiB)


def make_fetch(roots=None, cache_dir=None, blobs=None, hosts=None, max_bytes=None):
    """``blobs``: the BlobStore that holds ``pushed://`` blobs (default: the process-wide data-plane store).
    ``hosts`` / ``max_bytes``: override the env allowlist / cap (read at call time otherwise)."""
    roots = [Path(r).resolve() for r in (roots or [])]
    cache = Path(cache_dir) if cache_dir else Path(os.environ.get(
        "MOREGPU_MODEL_CACHE", Path.home() / ".cache" / "moregpu" / "models"))

    def fetch(source: str, sha256: str | None = None) -> Path:
        scheme, _, rest = source.partition("://")
        if scheme == "file":
            p = Path(urllib.parse.unquote(urllib.parse.urlparse(source).path)).resolve()
            if not any(p.is_relative_to(r) for r in roots):
                raise RefusedSource(f"{p} is not under an allowed model root (MOREGPU_MODEL_ROOTS={roots or 'unset'})")
            return p
        if scheme == "pushed":
            return _pushed(rest, sha256, blobs)
        if scheme in ("https", "http"):
            return _download(source, sha256, cache, hosts, max_bytes)
        if scheme == "hf":
            parts = rest.split("/", 2)
            if len(parts) != 3 or not all(parts):
                raise RefusedSource(f"hf source must be hf://org/repo[@rev]/file, got {source!r}")
            name, _, rev = parts[1].partition("@")
            if not sha256 and not _COMMIT.match(rev):
                raise RefusedSource(f"hf source {source!r} is mutable: pin a sha256 in the spec or use a full 40-hex "
                                    f"commit revision (hf://org/repo@<commit>/file); branches/tags are refused")
            return Path(_hf_hub_download(f"{parts[0]}/{name}", parts[2], revision=rev or None))
        raise RefusedSource(f"unsupported source scheme in {source!r}")

    return fetch


def _pushed(bid: str, sha256: str | None, blobs) -> Path:
    if not _PUSHED_ID.match(bid) or bid in (".", ".."):
        raise RefusedSource(f"pushed blob id {bid!r} is not valid")
    if blobs is None:
        from ..data.blobs import default_store
        blobs = default_store()
    try:
        p = blobs.path(bid)                      # KeyError unless fully received AND verified at blob_end
        have = blobs.info(bid)["sha256"]
    except KeyError:
        raise RefusedSource(f"pushed blob {bid!r} was not pushed to this worker (or is incomplete) — "
                            f"push it with /data/push first") from None
    if sha256 is not None and sha256 != have:
        raise IntegrityError(f"pushed://{bid}: blob sha256 {have} != spec sha256 {sha256}")
    return p


def _download(source: str, sha256: str | None, cache: Path, hosts, max_bytes) -> Path:
    if not sha256:
        raise RefusedSource(f"{source.partition(':')[0]}:// sources need a pinned sha256")
    allowed, env_name = (list(hosts), "hosts") if hosts is not None else model_hosts()
    cap = int(max_bytes) if max_bytes is not None else model_max_bytes()
    try:
        H.host_allowed(source, allowed, env_name)
    except RefDenied as e:
        raise RefusedSource(f"{e} (allowlist model download hosts in MOREGPU_MODEL_HOSTS)") from None
    target = cache / sha256
    if target.is_file() and sha256_file(target) == sha256:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache, prefix=".part-")
    os.close(fd)
    try:
        try:
            H.http_get(source, tmp, allowed, cap, env_name)
        except RefDenied as e:
            raise RefusedSource(f"{e} (MOREGPU_MODEL_MAX_BYTES / MOREGPU_MODEL_HOSTS)") from None
        except FileNotFoundError:
            raise RefusedSource(f"{source}: not found (HTTP 404)") from None
        got = sha256_file(tmp)
        if got != sha256:
            raise IntegrityError(f"sha256 mismatch for {source}: expected {sha256}, got {got}")
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def default_fetch(source: str, sha256: str | None = None) -> Path:
    """Fetch with the worker's environment configuration (read at call time); ``pushed://`` resolves through the
    process-wide data-plane BlobStore."""
    return make_fetch(roots=_env_paths("MOREGPU_MODEL_ROOTS"))(source, sha256)
