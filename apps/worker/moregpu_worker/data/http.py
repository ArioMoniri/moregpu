"""Allowlisted HTTP(S) downloads shared by the data plane (``https://`` refs) and model fetches (``https://`` sources).

* the host (or ``host:port``) must be in the caller's allowlist; userinfo in the URL is refused;
* plain ``http://`` only for loopback hosts (local mirrors / tests) — integrity always comes from a sha256;
* every redirect hop is re-checked against the allowlist;
* the body is capped: a declared ``Content-Length`` above the cap is refused up front, and the stream is cut off as
  soon as it passes the cap.

Violations raise :class:`~moregpu_worker.data.refs.RefDenied`.
"""
from __future__ import annotations

import ipaddress
import urllib.error
import urllib.parse
import urllib.request

from .refs import RefDenied

_BUF = 1 << 20


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def host_allowed(url: str, hosts, env_name: str = "MOREGPU_DATA_HOSTS") -> None:
    """Raise RefDenied unless ``url`` is http(s) to an allowlisted host (plain http: loopback only)."""
    p = urllib.parse.urlsplit(url)
    host = (p.hostname or "").lower()
    if p.scheme not in ("https", "http") or p.username or p.password or not host:
        raise RefDenied(f"url not allowed: {url[:120]!r}")
    allowed = {h.strip().lower() for h in hosts}
    try:
        port = p.port
    except ValueError:
        raise RefDenied(f"bad port in {url[:120]!r}") from None
    with_port = f"{host}:{port}" if port else None
    if host not in allowed and with_port not in allowed:
        raise RefDenied(f"host {host!r} not in {env_name}")
    if p.scheme == "http" and not is_loopback(host):
        raise RefDenied("plain http is only allowed for loopback hosts; use https")


def http_get(url: str, dest: str, hosts, cap: int, env_name: str = "MOREGPU_DATA_HOSTS") -> None:
    """GET ``url`` into the file ``dest`` (allowlist + redirect re-check + size cap). 404 → FileNotFoundError."""
    host_allowed(url, hosts, env_name)

    class _Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            host_allowed(newurl, hosts, env_name)  # every hop must stay on the allowlist
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    handlers: list = [_Redirect()]
    if is_loopback((urllib.parse.urlsplit(url).hostname or "").lower()):
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    try:
        resp = opener.open(urllib.request.Request(url, headers={"User-Agent": "moregpu-worker"}), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(url) from e
        raise OSError(f"GET {url}: HTTP {e.code}") from e  # pragma: no cover
    with resp, open(dest, "wb") as out:
        declared = resp.headers.get("Content-Length")
        if declared is not None and declared.strip().isdigit() and int(declared) > cap:
            raise RefDenied(f"{url}: {declared} bytes exceeds download cap {cap}")
        n = 0
        while True:
            b = resp.read(_BUF)
            if not b:
                break
            n += len(b)
            if n > cap:
                raise RefDenied(f"{url}: exceeds download cap {cap}")
            out.write(b)
