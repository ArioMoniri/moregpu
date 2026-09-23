"""Data references and the worker-side data policy (ADR-0110).

A :class:`Ref` names one piece of data (``file://``, ``https://``, ``s3://``, ``gs://`` or ``pushed://``) plus an
optional sha256, an optional axis-0 slice window and free-form metadata. :class:`DataPolicy` is what the *donor*
allows: local roots, download hosts and whether public buckets may be read. Anything outside it is :class:`RefDenied`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GiB = 1024 ** 3


class RefDenied(PermissionError):
    """The ref is outside the donor's data policy (scheme, root, host or bucket not allowed)."""


class IntegrityError(ValueError):
    """Content did not match its declared sha256 (or declared size)."""


def check_sha256(value, what: str = "sha256") -> str:
    """Return ``value`` as a normalised lowercase hex sha256 or raise ValueError."""
    if not isinstance(value, str) or not _SHA_RE.match(value.lower()):
        raise ValueError(f"{what} must be 64 hex chars, got {value!r}")
    return value.lower()


@dataclass(frozen=True)
class Ref:
    uri: str
    sha256: str | None = None
    slice: tuple[int, int] | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict) -> "Ref":
        if not isinstance(d, dict) or not isinstance(d.get("uri"), str):
            raise ValueError(f"ref needs a string 'uri': {d!r}")
        sha = d.get("sha256")
        if sha is not None:
            sha = check_sha256(sha)
        sl = d.get("slice")
        if sl is not None:
            if (not isinstance(sl, (list, tuple)) or len(sl) != 2
                    or not all(isinstance(v, int) and not isinstance(v, bool) for v in sl)
                    or sl[0] < 0 or sl[1] < sl[0]):
                raise ValueError(f"slice must be [start, stop] with 0 <= start <= stop, got {sl!r}")
            sl = (int(sl[0]), int(sl[1]))
        meta = d.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError(f"meta must be an object, got {meta!r}")
        return cls(d["uri"], sha, sl, dict(meta))

    def to_json(self) -> dict:
        out: dict = {"uri": self.uri}
        if self.sha256 is not None:
            out["sha256"] = self.sha256
        if self.slice is not None:
            out["slice"] = [int(self.slice[0]), int(self.slice[1])]
        if self.meta:
            out["meta"] = dict(self.meta)
        return out


@dataclass
class DataPolicy:
    roots: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    allow_buckets: bool = False
    max_download_bytes: int = 20 * GiB

    @classmethod
    def from_env(cls) -> "DataPolicy":
        """``MOREGPU_DATA_ROOTS`` (os.pathsep-separated), ``MOREGPU_DATA_HOSTS`` (comma-separated),
        ``MOREGPU_DATA_BUCKETS=1``, optional ``MOREGPU_DATA_MAX_DOWNLOAD_BYTES``."""
        roots = [r for r in os.environ.get("MOREGPU_DATA_ROOTS", "").split(os.pathsep) if r.strip()]
        hosts = [h.strip().lower() for h in os.environ.get("MOREGPU_DATA_HOSTS", "").split(",") if h.strip()]
        buckets = os.environ.get("MOREGPU_DATA_BUCKETS", "").strip().lower() in ("1", "true", "yes", "on")
        cap = int(os.environ.get("MOREGPU_DATA_MAX_DOWNLOAD_BYTES", str(20 * GiB)))
        return cls(roots=roots, hosts=hosts, allow_buckets=buckets, max_download_bytes=cap)
