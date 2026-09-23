"""The worker-side vision data plane (ADR-0110): resolve refs under the donor's policy, read arrays, build batches.

Security properties enforced here (all failures raise :class:`RefDenied` unless noted):

* ``file://`` — the *realpath* (symlinks and ``..`` resolved) must lie inside a policy root; relative URIs resolve
  against the first root. A declared sha256 is verified (:class:`IntegrityError`).
* ``https://`` — host must be in ``policy.hosts`` (redirects are re-checked), sha256 is required and verified before
  the content enters the cache; downloads are capped at ``policy.max_download_bytes``. Plain ``http://`` is accepted
  only for loopback hosts that are allowlisted (local mirrors / tests); integrity still comes from the sha256.
* ``s3://`` / ``gs://`` — only if ``policy.allow_buckets`` and ``s5cmd`` / ``gsutil`` are on PATH; always anonymous
  (``--no-sign-request``; cloud credentials are stripped from the tool's environment); sha256 required and verified.
* ``pushed://<id>`` — only a blob that has been fully pushed and verified by :class:`BlobStore`.
* Any other scheme (including bare paths) is denied.
"""
from __future__ import annotations

import ipaddress
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch

from . import readers
from .blobs import BlobStore
from .cache import ContentCache, sha256_file
from .manifest import Manifest
from .refs import GiB, DataPolicy, IntegrityError, Ref, RefDenied

_BUCKET_RE = re.compile(r"^(s3|gs)://[a-z0-9][a-z0-9._-]{1,221}/[^\x00-\x1f]*$")
_CRED_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
             "AWS_SHARED_CREDENTIALS_FILE", "GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_AUTH_ACCESS_TOKEN",
             "BOTO_CONFIG", "BOTO_PATH")
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "float64": torch.float64}
_BUF = 1 << 20


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class DataPlane:
    def __init__(self, policy: DataPolicy | None = None, cache: ContentCache | None = None,
                 blobs: BlobStore | None = None):
        self.policy = policy if policy is not None else DataPolicy.from_env()
        self._cache = cache
        self.blobs = blobs if blobs is not None else BlobStore()
        self._lock = threading.RLock()
        self._manifests: dict[tuple[str, str | None], Manifest] = {}
        self._verified: dict[tuple[str, int, int], str] = {}
        self._stats = {"bytes_read": 0, "reads": 0, "cache_hits": 0, "cache_misses": 0, "read_seconds": 0.0}

    # ------------------------------------------------------------------ cache (lazy default)
    @property
    def cache(self) -> ContentCache:
        if self._cache is None:
            d = os.environ.get("MOREGPU_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "moregpu", "data")
            self._cache = ContentCache(d, int(os.environ.get("MOREGPU_CACHE_BYTES", str(20 * GiB))))
        return self._cache

    # ------------------------------------------------------------------ resolve
    def resolve(self, ref: Ref) -> Path:
        uri = ref.uri if isinstance(ref.uri, str) else ""
        scheme = uri.split(":", 1)[0].lower() if ":" in uri else ""
        if scheme == "file":
            return self._resolve_file(uri, ref.sha256)
        if scheme in ("https", "http"):
            return self._resolve_http(uri, ref.sha256)
        if scheme in ("s3", "gs"):
            return self._resolve_bucket(uri, scheme, ref.sha256)
        if scheme == "pushed":
            return self._resolve_pushed(uri, ref.sha256)
        raise RefDenied(f"scheme not allowed: {uri[:80]!r}")

    def _resolve_file(self, uri: str, sha: str | None) -> Path:
        rest = urllib.parse.unquote(uri[5:])
        if rest.startswith("//"):
            rest = rest[2:]
        if "\x00" in rest or not rest:
            raise RefDenied(f"bad file uri {uri!r}")
        roots = [os.path.realpath(r) for r in self.policy.roots]
        if not roots:
            raise RefDenied("no data roots configured (MOREGPU_DATA_ROOTS)")
        cand = rest if os.path.isabs(rest) else os.path.join(roots[0], rest)
        real = os.path.realpath(cand)
        if not any(os.path.commonpath([real, r]) == r for r in roots):
            raise RefDenied(f"{uri!r} is outside the data roots")
        if not os.path.exists(real):
            raise FileNotFoundError(real)
        if sha is not None:
            if os.path.isdir(real):
                raise ValueError(f"sha256 cannot be checked for a directory ref {uri!r}")
            st = os.stat(real)
            key = (real, st.st_size, st.st_mtime_ns)
            got = self._verified.get(key)
            if got is None:
                got = self._verified[key] = sha256_file(real)
            if got != sha:
                raise IntegrityError(f"{uri!r}: sha256 {got} != {sha}")
        return Path(real)

    def _host_allowed(self, url: str) -> None:
        p = urllib.parse.urlsplit(url)
        host = (p.hostname or "").lower()
        if p.scheme not in ("https", "http") or p.username or p.password or not host:
            raise RefDenied(f"url not allowed: {url[:120]!r}")
        allowed = {h.strip().lower() for h in self.policy.hosts}
        with_port = f"{host}:{p.port}" if p.port else None
        if host not in allowed and with_port not in allowed:
            raise RefDenied(f"host {host!r} not in MOREGPU_DATA_HOSTS")
        if p.scheme == "http" and not _is_loopback(host):
            raise RefDenied("plain http is only allowed for loopback hosts; use https")

    def _cached(self, sha: str) -> Path | None:
        p = self.cache.get(sha)
        with self._lock:
            self._stats["cache_hits" if p is not None else "cache_misses"] += 1
        return p

    def _fetch(self, uri: str, sha: str, download) -> Path:
        """Cache hit, or ``download(tmp_path)`` into a temp file in the cache dir, cap its size, verify + store it."""
        hit = self._cached(sha)
        if hit is not None:
            return hit
        fd, tmp = tempfile.mkstemp(prefix=".dl-", dir=self.cache.dir)
        os.close(fd)
        try:
            download(tmp)
            if os.path.getsize(tmp) > self.policy.max_download_bytes:
                raise RefDenied(f"{uri}: exceeds download cap {self.policy.max_download_bytes}")
            return self.cache.put(tmp, sha256=sha, move=True)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _resolve_http(self, uri: str, sha: str | None) -> Path:
        self._host_allowed(uri)
        if sha is None:
            raise RefDenied("remote refs need a sha256")
        return self._fetch(uri, sha, lambda tmp: self._http_get(uri, tmp))

    def _http_get(self, uri: str, tmp: str) -> None:
        plane = self

        class _Redirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                plane._host_allowed(newurl)  # every hop must stay on the allowlist
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        handlers: list = [_Redirect()]
        if _is_loopback((urllib.parse.urlsplit(uri).hostname or "").lower()):
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        cap = self.policy.max_download_bytes
        try:
            resp = opener.open(urllib.request.Request(uri, headers={"User-Agent": "moregpu-worker"}), timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(uri) from e
            raise OSError(f"GET {uri}: HTTP {e.code}") from e  # pragma: no cover
        with resp, open(tmp, "wb") as out:
            declared = resp.headers.get("Content-Length")
            if declared is not None and int(declared) > cap:
                raise RefDenied(f"{uri}: {declared} bytes exceeds download cap {cap}")
            n = 0
            while True:
                b = resp.read(_BUF)
                if not b:
                    break
                n += len(b)
                if n > cap:  # pragma: no cover - server sent more than its Content-Length
                    raise RefDenied(f"{uri}: exceeds download cap {cap}")
                out.write(b)

    def _resolve_bucket(self, uri: str, scheme: str, sha: str | None) -> Path:
        if not self.policy.allow_buckets:
            raise RefDenied("bucket refs are disabled (MOREGPU_DATA_BUCKETS=1 to allow public buckets)")
        if sha is None:
            raise RefDenied("bucket refs need a sha256")
        if not _BUCKET_RE.match(uri):
            raise RefDenied(f"bad bucket uri {uri[:120]!r}")
        name = "s5cmd" if scheme == "s3" else "gsutil"
        tool = shutil.which(name)
        if tool is None:
            raise RefDenied(f"{name} is not on PATH")

        def download(tmp: str) -> None:
            env = {k: v for k, v in os.environ.items() if k not in _CRED_ENV}  # anonymous only
            with tempfile.TemporaryDirectory() as empty_cfg:
                if scheme == "s3":
                    cmd = [tool, "--no-sign-request", "cp", uri, tmp]
                else:
                    env["BOTO_CONFIG"] = os.devnull
                    env["CLOUDSDK_CONFIG"] = empty_cfg
                    cmd = [tool, "-q", "cp", uri, tmp]
                r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=6 * 3600)
            if r.returncode != 0:
                raise FileNotFoundError(f"{uri}: {name} exited {r.returncode}: "
                                        f"{r.stderr.decode(errors='replace')[-300:]}")

        return self._fetch(uri, sha, download)

    def _resolve_pushed(self, uri: str, sha: str | None) -> Path:
        bid = uri[len("pushed://"):] if uri.lower().startswith("pushed://") else ""
        try:
            p = self.blobs.path(bid)
        except KeyError as e:
            raise RefDenied(f"no pushed blob {bid!r}") from e
        have = self.blobs.info(bid)["sha256"]
        if sha is not None and sha != have:
            raise IntegrityError(f"pushed://{bid}: sha256 {have} != {sha}")
        return p

    # ------------------------------------------------------------------ read
    def read(self, ref: Ref) -> np.ndarray:
        t0 = time.perf_counter()
        path = self.resolve(ref)
        fmt = ref.meta.get("fmt") if ref.meta else None
        if fmt is None and not ref.uri.lower().startswith("file:"):
            try:  # cached / staged files have no meaningful suffix: take the format from the URI
                fmt = readers._detect(Path(urllib.parse.urlsplit(ref.uri).path or "x"))
            except ValueError:
                fmt = None
        arr = readers.read_array(path, fmt)
        if ref.slice is not None:
            a, b = ref.slice
            if arr.ndim == 0 or b > arr.shape[0]:
                raise ValueError(f"slice {ref.slice} out of range for axis 0 of {arr.shape} ({ref.uri})")
            arr = arr[a:b]
        with self._lock:
            self._stats["reads"] += 1
            self._stats["bytes_read"] += int(arr.nbytes)
            self._stats["read_seconds"] += time.perf_counter() - t0
        return arr

    def open_manifest(self, uri: str, sha256: str | None = None) -> Manifest:
        key = (uri, sha256)
        with self._lock:
            m = self._manifests.get(key)
        if m is not None:
            return m
        path = self.resolve(Ref(uri, sha256=sha256))
        m = Manifest(path.read_bytes())
        with self._lock:
            self._manifests[key] = m
        return m

    # ------------------------------------------------------------------ batches
    def load_batch(self, manifest: Manifest, indices: list[int], spec: dict) -> torch.Tensor:
        kind = spec.get("kind")
        size = [int(s) for s in (spec.get("size") or [])]
        if kind in ("2d", "2p5d"):
            if len(size) != 2:
                raise ValueError(f"{kind} size must be [H, W], got {spec.get('size')!r}")
        elif kind == "3d":
            if len(size) != 3:
                raise ValueError(f"3d size must be [D, H, W], got {spec.get('size')!r}")
        else:
            raise ValueError(f"spec kind must be 2d, 2p5d or 3d, got {kind!r}")
        dtype_name = spec.get("dtype") or "float32"
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported dtype {dtype_name!r}")
        channels = spec.get("channels")
        items = [self._sample(self.read(manifest[i]), kind, size, channels, spec.get("layout")) for i in indices]
        x = torch.stack(items)
        norm = spec.get("normalize")
        if norm:
            shape = (1, -1) + (1,) * (x.ndim - 2)
            mean = torch.as_tensor(norm.get("mean", 0.0), dtype=torch.float32)
            std = torch.as_tensor(norm.get("std", 1.0), dtype=torch.float32)
            if mean.ndim:
                mean = mean.reshape(shape)
            if std.ndim:
                std = std.reshape(shape)
            x = (x - mean) / std
        return x.to(_DTYPES[dtype_name])

    @staticmethod
    def _sample(arr: np.ndarray, kind: str, size: list[int], channels, layout) -> torch.Tensor:
        x = torch.from_numpy(np.array(arr, dtype=np.float32))
        if kind == "3d":
            if x.ndim == 3:
                x = x[None]
            elif x.ndim != 4:
                raise ValueError(f"3d sample must be (D,H,W) or (C,D,H,W), got {tuple(x.shape)}")
            mode = "trilinear"
        else:
            if x.ndim == 2:
                x = x[None]
            elif x.ndim == 3 and kind == "2d":
                hwc = layout == "hwc" or (layout is None and x.shape[-1] <= 4 and x.shape[0] > x.shape[-1])
                if hwc:
                    x = x.permute(2, 0, 1)
            elif x.ndim != 3:
                raise ValueError(f"{kind} sample must be 2-D or 3-D, got {tuple(x.shape)}")
            mode = "bilinear"
        if channels is not None and x.shape[0] != int(channels):
            if x.shape[0] == 1 and kind == "2d":
                x = x.expand(int(channels), *x.shape[1:])
            else:
                raise ValueError(f"{kind} sample has {x.shape[0]} channels/slices, spec wants {channels}")
        if list(x.shape[1:]) != size:
            x = torch.nn.functional.interpolate(x[None], size=tuple(size), mode=mode, align_corners=False)[0]
        return x.contiguous()

    # ------------------------------------------------------------------ introspection
    def capabilities(self) -> dict:
        return {"readers": readers.capabilities(), "n_roots": len(self.policy.roots), "n_hosts": len(self.policy.hosts),
                "buckets": bool(self.policy.allow_buckets), "max_download_bytes": self.policy.max_download_bytes,
                "cache": self.cache.stats()}

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

