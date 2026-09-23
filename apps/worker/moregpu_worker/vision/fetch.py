"""Artefact fetchers for model specs (ADR-0113). A fetch is `fetch(source, sha256=None) -> Path`.

* `file:///p`   — only under an allowed root (MOREGPU_MODEL_ROOTS, os.pathsep-separated); symlinks are resolved first.
* `pushed://id` — a blob the coordinator pushed into MOREGPU_PUSHED_DIR (sha256 mandatory at the spec level).
* `https://…`   — downloaded into a content-addressed cache (MOREGPU_MODEL_CACHE) named by its sha256; sha256 mandatory.
* `hf://org/repo[@rev]/path` — one file via huggingface_hub (never `from_pretrained`, never remote code).

Every loaded artefact is ALSO hashed and compared by the adapter layer before it is opened (`adapters.base.fetch_verified`).
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from .errors import IntegrityError, RefusedSource

_PUSHED_ID = re.compile(r"^[A-Za-z0-9._-]+$")
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


def make_fetch(roots=None, pushed_dir=None, cache_dir=None):
    roots = [Path(r).resolve() for r in (roots or [])]
    pushed = Path(pushed_dir).resolve() if pushed_dir else None
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
            if not pushed or not _PUSHED_ID.match(rest) or rest in (".", ".."):
                raise RefusedSource(f"pushed blob {rest!r} is not resolvable (MOREGPU_PUSHED_DIR set? id valid?)")
            p = pushed / rest
            if not p.is_file():
                raise RefusedSource(f"pushed blob {rest!r} not found")
            return p
        if scheme == "https":
            if not sha256:
                raise RefusedSource("https:// sources need a pinned sha256")
            target = cache / sha256
            if target.is_file() and sha256_file(target) == sha256:
                return target
            cache.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=cache, prefix=".part-")
            try:
                with os.fdopen(fd, "wb") as out, urllib.request.urlopen(source, timeout=60) as resp:
                    for chunk in iter(lambda: resp.read(_CHUNK), b""):
                        out.write(chunk)
                got = sha256_file(tmp)
                if got != sha256:
                    raise IntegrityError(f"sha256 mismatch for {source}: expected {sha256}, got {got}")
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            return target
        if scheme == "hf":
            parts = rest.split("/", 2)
            if len(parts) != 3 or not all(parts):
                raise RefusedSource(f"hf source must be hf://org/repo[@rev]/file, got {source!r}")
            name, _, rev = parts[1].partition("@")
            return Path(_hf_hub_download(f"{parts[0]}/{name}", parts[2], revision=rev or None))
        raise RefusedSource(f"unsupported source scheme in {source!r}")

    return fetch


def default_fetch(source: str, sha256: str | None = None) -> Path:
    """Fetch with the worker's environment configuration (read at call time)."""
    pushed = os.environ.get("MOREGPU_PUSHED_DIR")
    return make_fetch(roots=_env_paths("MOREGPU_MODEL_ROOTS"), pushed_dir=pushed)(source, sha256)
