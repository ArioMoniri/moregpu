#!/usr/bin/env python3
"""
tls_install_path.py — the DENO worker install path works over the DEFAULT self-signed wss, and is MITM-safe.

`tls_default.py` proves the PYTHON torch worker pins the coordinator's cert fingerprint on the live socket.
But the flagship path users actually run is `curl scripts/install.sh | sh`, which runs the DENO worker
(apps/worker/worker.ts). Deno's `WebSocket` has no per-connection "trust this self-signed cert" switch, so
the installer instead does a MITM-safe trust-on-fetch (scripts/install.sh):

    fetch the coordinator's public leaf cert from GET /cert.pem  →  verify sha256(DER) == the out-of-band
    MOREGPU_PIN from the join banner  →  hand exactly that cert to Deno via DENO_CERT so the live wss
    handshake is validated against it (and nothing else).

This test proves that whole chain against the REAL server.ts + worker.ts (no re-implemented TLS), CPU-only,
loopback, no model download:

  (1) DEFAULT wss — the coordinator (no MOREGPU_INSECURE) mints a self-signed cert, serves GET /cert.pem, and
      the fetched cert's sha256(DER) — computed INDEPENDENTLY in Python (hashlib) — equals the fingerprint the
      coordinator printed as MOREGPU_PIN (Deno WebCrypto) and returned in the x-moregpu-cert-sha256 header.
  (2) RIGHT cert CONNECTS — the real worker.ts run with DENO_CERT=<pinned cert> completes the wss handshake
      and registers (shows up in the admin /workers list, fetched over https). This is exactly what install.sh
      does after the pin check passes.
  (3) cert trust is LOAD-BEARING — the same worker.ts with NO DENO_CERT is REFUSED by the self-signed cert
      (never registers). Proves (2) isn't succeeding by accident / a disabled check.
  (4) WRONG cert is REFUSED (the MITM defense at the TLS layer) — worker.ts pointed at the real coordinator but
      handed a DIFFERENT coordinator's cert as DENO_CERT never registers: Deno validates the live cert against
      exactly the trusted one, so a substituted cert fails the handshake.
  (5) the installer's pin GATE rejects the wrong cert BEFORE running anything — sha256(DER) of that other cert
      != MOREGPU_PIN, which is the check scripts/install.sh makes (so it aborts before the token is sent).

Run:  python3 tests/security/tls_install_path.py     (exit 0 = the Deno install path works and fails closed)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCS: list[subprocess.Popen] = []
_RESULTS: list[tuple[bool, str]] = []


def check(passed: bool, label: str) -> None:
    _RESULTS.append((passed, label))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


def _deno() -> str:
    d = shutil.which("deno")
    if not d:
        raise RuntimeError("deno not found on PATH — the installer's runtime is required")
    return d


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


_NOVERIFY = ssl.create_default_context()
_NOVERIFY.check_hostname = False
_NOVERIFY.verify_mode = ssl.CERT_NONE


def _get(url: str, token: str | None = None, want_headers: bool = False):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=5, context=_NOVERIFY) as r:
        body = r.read()
        return (body, dict(r.headers)) if want_headers else body


def der_fingerprint_py(cert_pem: str) -> str:
    """sha256 of the DER of the leaf cert — independent of Deno; matches `openssl x509 -fingerprint -sha256`
    and server.ts certFingerprint(). Uses only base64+hashlib so it needs no extra deps."""
    m = re.search(r"-----BEGIN CERTIFICATE-----(.*?)-----END CERTIFICATE-----", cert_pem, re.S)
    if not m:
        raise ValueError("no CERTIFICATE block")
    der = base64.b64decode(re.sub(r"\s+", "", m.group(1)))
    return hashlib.sha256(der).hexdigest()


def start_coord(root: str, tag: str) -> dict:
    """Start a DEFAULT-TLS coordinator (no MOREGPU_INSECURE) on loopback; MOREGPU_HOST=localhost so the minted
    cert's SAN matches wss://localhost. Returns {port, join, admin, pin, cert_pem, cert_file}."""
    port = free_port()
    # Each coordinator gets its OWN config dir: the self-signed cert is persisted beside MOREGPU_CONFIG (for a
    # stable pin across restarts), so a shared dir would make B reuse A's cert. Separate dirs ⇒ distinct certs.
    cdir = os.path.join(root, tag)
    os.makedirs(cdir, exist_ok=True)
    cfg = os.path.join(cdir, "mg.json")
    log = os.path.join(cdir, "coord.log")
    env = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_HOST="localhost")
    env.pop("MOREGPU_INSECURE", None)  # DEFAULT transport = wss:// (the whole point of this test)
    p = subprocess.Popen(
        [_deno(), "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write", "apps/coordinator/server.ts"],
        cwd=REPO, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT,
    )
    PROCS.append(p)
    for _ in range(80):
        try:
            _get(f"https://127.0.0.1:{port}/health")
            break
        except Exception:
            time.sleep(0.5)
    conf = json.load(open(cfg))
    pin = ""
    for _ in range(20):
        txt = open(log).read()
        m = re.search(r"sha256:([0-9a-f]{64})", txt)
        if m:
            pin = m.group(1)
            break
        time.sleep(0.3)
    cert_pem = _get(f"https://127.0.0.1:{port}/cert.pem").decode()
    cert_file = os.path.join(root, f"cert-{tag}.pem")
    with open(cert_file, "w") as f:
        f.write(cert_pem)
    return {"port": port, "join": conf["joinToken"], "admin": conf["adminToken"],
            "pin": pin, "cert_pem": cert_pem, "cert_file": cert_file}


def run_worker(port: int, join: str, name: str, deno_cert: str | None, root: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop("MOREGPU_INSECURE", None)
    if deno_cert:
        env["DENO_CERT"] = deno_cert
    else:
        env.pop("DENO_CERT", None)
    p = subprocess.Popen(
        [_deno(), "run", "--unstable-webgpu", "--allow-net", "--allow-env", "--allow-sys",
         "apps/worker/worker.ts", "--server", f"wss://localhost:{port}/ws", "--token", join, "--name", name, "--cpu"],
        cwd=REPO, env=env, stdout=open(os.path.join(root, f"w-{name}.log"), "w"), stderr=subprocess.STDOUT,
    )
    PROCS.append(p)
    return p


def registered(port: int, admin: str, name: str) -> bool:
    try:
        body = _get(f"https://127.0.0.1:{port}/workers", token=admin)
        d = json.loads(body)
        ids = [w.get("id") for w in (d if isinstance(d, list) else d.get("workers", []))]
        return name in ids
    except Exception:
        return False


def wait_registered(port: int, admin: str, name: str, timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if registered(port, admin, name):
            return True
        time.sleep(0.5)
    return False


def cleanup() -> None:
    for p in PROCS:
        try:
            p.terminate()
        except Exception:
            pass
    for p in PROCS:
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def main() -> int:
    print("=" * 92)
    print("TLS INSTALL PATH — Deno worker.ts joins the default self-signed wss (trust-on-fetch + DENO_CERT), MITM-safe")
    print("=" * 92)
    root = tempfile.mkdtemp(prefix="mg-tls-install-")
    try:
        A = start_coord(root, "A")           # the real coordinator the worker will join
        B = start_coord(root, "B")           # an unrelated coordinator — its cert is the "attacker/wrong" cert
        print(f"\n  coordinator A on :{A['port']}  pin sha256:{A['pin'][:16]}…")

        # (1) DEFAULT wss + /cert.pem fingerprint agreement (Deno WebCrypto pin == Python hashlib(DER)).
        print("\n(1) default wss + /cert.pem fingerprint agreement")
        _, hdrs = _get(f"https://127.0.0.1:{A['port']}/cert.pem", want_headers=True)
        hdr_fp = (hdrs.get("x-moregpu-cert-sha256") or hdrs.get("X-Moregpu-Cert-Sha256") or "").lower()
        py_fp = der_fingerprint_py(A["cert_pem"])
        check(bool(A["pin"]) and py_fp == A["pin"],
              f"printed MOREGPU_PIN == independent Python sha256(DER) of the fetched cert ({py_fp[:16]}…)")
        check(hdr_fp == A["pin"], "x-moregpu-cert-sha256 response header == printed pin")

        # (2) RIGHT cert connects — exactly what install.sh runs after the pin check passes.
        print("\n(2) right cert connects (worker.ts + DENO_CERT=pinned cert)")
        run_worker(A["port"], A["join"], "good", A["cert_file"], root)
        check(wait_registered(A["port"], A["admin"], "good", 25),
              "worker.ts with DENO_CERT=<pinned cert> registers over wss://")

        # (3) cert trust is load-bearing — no DENO_CERT ⇒ self-signed cert is refused, worker never registers.
        print("\n(3) cert trust is load-bearing (no DENO_CERT ⇒ refused)")
        run_worker(A["port"], A["join"], "nocert", None, root)
        time.sleep(9)  # ~4 reconnect attempts; a self-signed cert with no trust anchor never completes the handshake
        check(not registered(A["port"], A["admin"], "nocert"),
              "worker.ts with NO DENO_CERT does NOT register (self-signed cert rejected)")

        # (4) WRONG cert is refused at the TLS layer — B's cert can't authenticate A's live socket.
        print("\n(4) wrong cert refused (MITM: B's cert handed as DENO_CERT for A)")
        run_worker(A["port"], A["join"], "mitm", B["cert_file"], root)
        time.sleep(9)
        check(not registered(A["port"], A["admin"], "mitm"),
              "worker.ts pinned to the WRONG cert does NOT register (Deno rejects the mismatched live cert)")

        # (5) the installer's pin GATE would abort before running: sha256(DER) of B's cert != A's pin.
        print("\n(5) installer pin gate rejects the wrong cert before any token is sent")
        b_fp = der_fingerprint_py(B["cert_pem"])
        check(b_fp != A["pin"],
              "sha256(DER) of the wrong cert != MOREGPU_PIN ⇒ scripts/install.sh aborts (no token leaked)")
    finally:
        cleanup()
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(1 for ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 92)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 92)
    if passed != total:
        for ok, label in _RESULTS:
            if not ok:
                print(f"  FAILED: {label}")
        return 1
    print("THE DENO INSTALL PATH WORKS OVER THE DEFAULT SELF-SIGNED wss AND FAILS CLOSED ON A WRONG CERT. ✔")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        cleanup()
