#!/usr/bin/env python3
"""
tls_default.py — TLS IS THE DEFAULT transport (self-signed on first run + worker cert pinning).

Proves, against the REAL production code (no re-implemented TLS/crypto), that:

  (1) DEFAULT wss:// — started with NO MOREGPU_TLS_CERT/_KEY and NO MOREGPU_INSECURE, the coordinator
      (apps/coordinator/server.ts) MINTS a self-signed cert, PERSISTS it beside MOREGPU_CONFIG
      (moregpu-cert.pem / moregpu-key.pem, key mode 0600), serves https + wss, and PRINTS the cert's
      SHA-256 fingerprint as the worker pin. We cross-check the coordinator's printed fingerprint against
      an INDEPENDENT hash of the DER from the persisted cert file (server.ts certFingerprint() is Deno
      WebCrypto; ours is python hashlib — a genuine cross-runtime agreement, same as openssl -sha256).

  (2) RIGHT pin CONNECTS — the REAL worker (apps/worker/worker_torch.py) with the correct MOREGPU_PIN
      completes the wss handshake, its _verify_pin() accepts the cert, it registers, and it shows up in
      the admin /workers list (fetched over https).

  (3) WRONG pin is REFUSED — the same real worker with a mismatched MOREGPU_PIN (same valid join token)
      NEVER registers (never appears in /workers) and logs the REFUSED / MITM line — the pin check fires
      BEFORE the join token is sent, so a wrong cert can't harvest the token/tenant key.

  (4) ws:// opt-out still works — with MOREGPU_INSECURE=1 the coordinator serves plaintext ws:// (no cert,
      no pin) and a plain ws:// worker registers, exactly as the existing CI does. This is the guard that
      making TLS the default did not break the simple local/CI path.

Runs on CPU, no GPU, no model download, no network beyond loopback. Needs `deno` + `python3`+torch (the
worker imports torch, as every e2e test already requires). The private key lives only in a tempdir that is
deleted at the end — nothing is ever committed.

  python3 tests/security/tls_default.py        # exit 0 = all checks pass
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
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCS: list[subprocess.Popen] = []
_NOVERIFY = ssl.create_default_context()
_NOVERIFY.check_hostname = False
_NOVERIFY.verify_mode = ssl.CERT_NONE  # self-signed coordinator cert; this is a test client, not a worker

_RESULTS: list[tuple[bool, str]] = []


def check(passed: bool, label: str) -> None:
    _RESULTS.append((bool(passed), label))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}", flush=True)


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def api(scheme: str, port: int, path: str, admin: str | None = None, timeout: int = 15):
    """Hit the coordinator's HTTP API (https uses the no-verify context, matching a self-signed cert)."""
    h = {"content-type": "application/json"}
    if admin:
        h["authorization"] = "Bearer " + admin
    req = urllib.request.Request(f"{scheme}://127.0.0.1:{port}{path}", method="GET", headers=h)
    ctx = _NOVERIFY if scheme == "https" else None
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout, context=ctx))
    except urllib.error.HTTPError as e:
        return {"httperror": e.code, "body": e.read().decode()[:200]}


def wait_health(scheme: str, port: int, tries: int = 80) -> bool:
    ctx = _NOVERIFY if scheme == "https" else None
    for _ in range(tries):
        try:
            urllib.request.urlopen(f"{scheme}://127.0.0.1:{port}/health", timeout=2, context=ctx)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def start_coord(root: str, port: int, cfg: str, tag: str, insecure: bool) -> str:
    env = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_HOST="localhost")
    if insecure:
        env["MOREGPU_INSECURE"] = "1"
    else:
        env.pop("MOREGPU_INSECURE", None)
        env.pop("MOREGPU_TLS_CERT", None)  # ensure the DEFAULT self-signed mint path (no operator cert)
        env.pop("MOREGPU_TLS_KEY", None)
    logp = os.path.join(root, f"coord-{tag}.log")
    PROCS.append(subprocess.Popen(
        ["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write",
         "apps/coordinator/server.ts"], cwd=REPO, env=env,
        stdout=open(logp, "w"), stderr=subprocess.STDOUT))
    return logp


def start_worker(root: str, scheme: str, port: int, join: str, name: str, pin: str | None) -> str:
    env = dict(os.environ)
    if pin is not None:
        env["MOREGPU_PIN"] = pin
    else:
        env.pop("MOREGPU_PIN", None)
    env["PYTHONUNBUFFERED"] = "1"  # so the worker's pin-check prints reach the log file promptly (harness reads them)
    logp = os.path.join(root, f"{name}.log")
    PROCS.append(subprocess.Popen(
        ["python3", "-u", "apps/worker/worker_torch.py", "--server", f"{scheme}://127.0.0.1:{port}/ws",
         "--token", join, "--name", name, "--cpu"], cwd=REPO, env=env,
        stdout=open(logp, "w"), stderr=subprocess.STDOUT))
    return logp


def cert_fp_from_file(cert_pem_path: str) -> str:
    """SHA-256 of the leaf cert DER — the same value server.ts certFingerprint() and `openssl -sha256` print."""
    pem = open(cert_pem_path).read()
    body = re.search(r"-----BEGIN CERTIFICATE-----(.*?)-----END CERTIFICATE-----", pem, re.S).group(1)
    der = base64.b64decode("".join(body.split()))
    return hashlib.sha256(der).hexdigest()


def worker_registered(scheme: str, port: int, admin: str, name: str, tries: int, interval: float = 1.0) -> bool:
    for _ in range(tries):
        w = api(scheme, port, "/workers", admin=admin)
        if isinstance(w, list) and any(x.get("id") == name for x in w):
            return True
        time.sleep(interval)
    return False


def log_has(path: str, needle: str, tries: int = 20, interval: float = 0.5) -> bool:
    for _ in range(tries):
        try:
            if needle in open(path, encoding="utf-8", errors="replace").read():
                return True
        except FileNotFoundError:
            pass
        time.sleep(interval)
    return False


def main() -> int:
    print("=" * 92)
    print("TLS DEFAULT — self-signed wss:// on first run + worker cert pinning (real server.ts + worker_torch.py)")
    print("=" * 92)
    root = tempfile.mkdtemp(prefix="moregpu-tls-")
    try:
        # ── (1) default wss:// coordinator: mints + persists a cert, prints its fingerprint ───────────────
        print("\n(1) coordinator comes up on wss:// with a generated cert", flush=True)
        port = free_port(); cfg = os.path.join(root, "mg.json")
        coord_log = start_coord(root, port, cfg, "tls", insecure=False)
        up = wait_health("https", port)
        check(up, "coordinator answers HTTPS /health (serving TLS by default, no cert provided)")
        if not up:
            print("  coord log tail:\n" + "".join(open(coord_log).readlines()[-25:]))
            return 1

        cert_file = os.path.join(root, "moregpu-cert.pem")
        key_file = os.path.join(root, "moregpu-key.pem")
        check(os.path.isfile(cert_file) and os.path.isfile(key_file),
              "self-signed cert + key persisted beside MOREGPU_CONFIG (moregpu-cert.pem / moregpu-key.pem)")
        mode = os.stat(key_file).st_mode & 0o777
        check(mode & 0o077 == 0, f"private key is not group/other-readable (mode {oct(mode)})")
        check("PRIVATE KEY" in open(key_file).read() and "PRIVATE KEY" not in open(cert_file).read(),
              "key file holds the private key; cert file does not (key stays on the admin box, in tempdir)")

        expected_fp = cert_fp_from_file(cert_file)
        banner = open(coord_log).read()
        m = re.search(r"sha256:([0-9a-f]{64})", banner)
        printed_fp = m.group(1) if m else None
        check(printed_fp == expected_fp,
              f"coordinator PRINTED fingerprint matches independent DER hash (sha256:{expected_fp[:16]}…)")
        check("MOREGPU_PIN=" + expected_fp in banner, "wizard join command carries MOREGPU_PIN=<fingerprint>")

        conf = json.load(open(cfg)); JOIN, ADMIN = conf["joinToken"], conf["adminToken"]

        # ── (2) a worker with the RIGHT pin connects ─────────────────────────────────────────────────────
        print("\n(2) a worker with the correct pin connects over wss://", flush=True)
        start_worker(root, "wss", port, JOIN, "w-goodpin", pin=expected_fp)
        ok_good = worker_registered("https", port, ADMIN, "w-goodpin", tries=90)
        check(ok_good, "worker with the correct MOREGPU_PIN registered (visible in admin /workers)")
        check(log_has(os.path.join(root, "w-goodpin.log"), "pinned OK"),
              "worker logged the pin match (coordinator TLS cert pinned OK)")

        # ── (3) a worker with the WRONG pin is refused ───────────────────────────────────────────────────
        print("\n(3) a worker with a wrong pin is refused (same valid join token)", flush=True)
        wrong_fp = ("f" if expected_fp[0] != "f" else "0") + expected_fp[1:]  # flip one nibble → still 64 hex chars
        start_worker(root, "wss", port, JOIN, "w-badpin", pin=wrong_fp)
        refused = log_has(os.path.join(root, "w-badpin.log"), "REFUSED", tries=40)
        check(refused, "worker with a mismatched pin logged REFUSED (does not send the join token)")
        never = not worker_registered("https", port, ADMIN, "w-badpin", tries=8)
        check(never, "worker with a mismatched pin NEVER registered (rejected before the token crossed the wire)")

        # ── (4) ws:// opt-out (MOREGPU_INSECURE=1) still works — the existing CI path ─────────────────────
        print("\n(4) MOREGPU_INSECURE=1 opt-out still serves plaintext ws:// for local/CI", flush=True)
        iport = free_port(); icfg = os.path.join(root, "mg-insecure.json")
        istart_log = start_coord(root, iport, icfg, "insecure", insecure=True)
        up2 = wait_health("http", iport)
        check(up2, "insecure coordinator answers plain HTTP /health (ws:// mode)")
        check("sha256:" not in open(istart_log).read(), "insecure coordinator prints NO cert fingerprint (plaintext, no TLS)")
        iconf = json.load(open(icfg)); IJOIN, IADMIN = iconf["joinToken"], iconf["adminToken"]
        start_worker(root, "ws", iport, IJOIN, "w-plain", pin=None)
        ok_plain = worker_registered("http", iport, IADMIN, "w-plain", tries=90)
        check(ok_plain, "plain ws:// worker (no pin) registered against the insecure coordinator")

        passed = sum(1 for ok, _ in _RESULTS if ok)
        total = len(_RESULTS)
        print("\n" + "=" * 92)
        print(f"RESULT: {passed}/{total} checks passed")
        print("=" * 92)
        if passed != total:
            for ok, label in _RESULTS:
                if not ok:
                    print(f"  FAILED: {label}")
            print("RESULT: FAIL")
            return 1
        print("TLS is the default; the right pin connects, a wrong pin is refused, ws:// opt-out intact. ✔")
        print("RESULT: PASS")
        return 0
    finally:
        for p in PROCS:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(1)
        for p in PROCS:
            try:
                p.kill()
            except Exception:
                pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
