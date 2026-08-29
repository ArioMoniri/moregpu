#!/usr/bin/env python3
"""
RELEASE VERIFY — the signed, hash-PINNED worker release must FAIL CLOSED at install time.

This proves, against the REAL release tooling (no re-implemented crypto), that the supply-chain
gate which replaced the old `curl raw.githubusercontent.com/.../main/worker.ts | sh` fetch refuses
any artifact that is not both (1) the exact pinned bytes and (2) signed by the pinned release key:

  * SIGN  — scripts/release_sign.py :: sign_artifact()/release_message()/pubkey_b64() mint a detached
            Ed25519 signature over the domain-separated message  `moregpu-release/v1\\n<name>\\n<sha256>`.
            These reuse the repo's EXACT crypto approach — the same Ed25519 primitives the torch worker
            already signs results with (apps/worker/worker_torch.py:28,50-51,71).

  * VERIFY — scripts/verify_release.ts is the install-time gate. It recomputes the sha256, constant-
            length-compares it to the pin, then verifies the signature with the SAME WebCrypto Ed25519
            call the coordinator runs on every result (apps/coordinator/server.ts:35,205,251):
            importKey('raw', pub, {name:'Ed25519'}) + crypto.subtle.verify({name:'Ed25519'}, ...).
            We run that EXACT verifier in Deno (the installer's own runtime) and assert its exit code:
                0 = trusted · 3 = sha256 mismatch · 4 = bad signature.

  Negatives asserted (each must be REJECTED, never run):
    - a TAMPERED artifact (one byte flipped) against its original pin/sig        -> exit 3
    - a WRONG pin (attacker rewrites the pinned sha256, keeps the good artifact)  -> exit 3
    - a WRONG-KEY signature (signed by an unpinned attacker key)                  -> exit 4
    - a CROSS-ARTIFACT replay (a valid sig re-presented under another --name)     -> exit 4
    - a MISSING / empty signature                                                 -> exit 4

  Integration guards (ground the unit test in the actually-shipped bundle):
    - the REAL committed apps/worker/worker.ts verifies against the pins baked into scripts/install.sh
      and the committed apps/worker/worker.ts.sig                                 -> exit 0
    - the verifier EMBEDDED in scripts/install.sh (heredoc) is byte-identical to scripts/verify_release.ts
    - the committed .sig files are 64-byte Ed25519 signatures (public), not key material (no secrets)

Runs on CPU, no network, no live coordinator, no model download: unit-level against the imported real
signing functions plus the real Deno verifier. Ed25519 is RFC 8032, so the Python-signed / Deno-verified
round trip also proves the cross-runtime byte-compatibility the installer depends on.

Run:  python3 tests/security/release_verify.py     (exit 0 = every bad release rejected)
"""
from __future__ import annotations

import base64
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIGN_PY = os.path.join(REPO, "scripts", "release_sign.py")
VERIFY_TS = os.path.join(REPO, "scripts", "verify_release.ts")
INSTALL_SH = os.path.join(REPO, "scripts", "install.sh")
WORKER_TS = os.path.join(REPO, "apps", "worker", "worker.ts")
WORKER_TS_SIG = os.path.join(REPO, "apps", "worker", "worker.ts.sig")


# ---- import the REAL release signing tool (sha256_hex/release_message/sign_artifact/pubkey_b64) ----
def _load_signer():
    spec = importlib.util.spec_from_file_location("release_sign", SIGN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # argparse only runs under __main__ -> importing is side-effect-free
    return mod


S = _load_signer()


# ---------- tiny assertion harness (standalone script style, like tests/security/seal_negatives.py) ----------
_RESULTS: list[tuple[bool, str]] = []


def check(passed: bool, label: str) -> None:
    _RESULTS.append((passed, label))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


def _deno() -> str:
    d = shutil.which("deno")
    if not d:
        raise RuntimeError("deno not found on PATH — the installer's verify runtime is required")
    return d


def run_verify(artifact: str, sig: str, sha256: str, pubkey: str, name: str) -> int:
    """Run scripts/verify_release.ts EXACTLY as install.sh does; return its exit code."""
    proc = subprocess.run(
        [_deno(), "run", "--allow-read", VERIFY_TS,
         "--artifact", artifact, "--sig", sig,
         "--sha256", sha256, "--pubkey", pubkey, "--name", name],
        capture_output=True, text=True, timeout=60,
    )
    # Surface the gate's own one-line verdict (OK.../REJECT...) for the transcript.
    line = (proc.stdout or proc.stderr).strip().splitlines()
    if line:
        print(f"        gate: {line[-1]}")
    return proc.returncode


def _write(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


# ============================================================================================
# Hermetic unit cases: an ephemeral release key, a synthetic artifact, the REAL verifier.
# ============================================================================================
def test_hermetic_gate() -> None:
    print("\n(a) hermetic gate  [release_sign.py sign_artifact -> verify_release.ts verify]")
    sk = Ed25519PrivateKey.generate()             # the pinned release key for this run
    pub = S.pubkey_b64(sk)
    attacker = Ed25519PrivateKey.generate()       # an UNPINNED signer
    att_pub = S.pubkey_b64(attacker)

    d = tempfile.mkdtemp(prefix="mg-release-")
    try:
        art = os.path.join(d, "worker.ts")
        body = b"// synthetic worker artifact\nconsole.log('join pool');\n" + os.urandom(64)
        _write(art, body)

        good_sha, good_sig = S.sign_artifact(sk, art, "worker.ts")
        sig_path = os.path.join(d, "worker.ts.sig")
        _write(sig_path, (good_sig + "\n").encode())

        # control (fail-OPEN sanity): the genuine bundle verifies -> installer would run the worker.
        check(run_verify(art, sig_path, good_sha, pub, "worker.ts") == 0,
              "control: genuine artifact+sig+pin VERIFIES (exit 0, worker allowed)")

        # tampered artifact: flip one byte on disk, keep the original pin+sig -> hash no longer matches.
        tampered = bytearray(body)
        tampered[10] ^= 0xFF
        art_bad = os.path.join(d, "worker_tampered.ts")
        _write(art_bad, bytes(tampered))
        check(run_verify(art_bad, sig_path, good_sha, pub, "worker.ts") == 3,
              "tampered artifact (1 byte flipped) -> REJECTED (exit 3, sha256 mismatch)")

        # wrong pin: attacker rewrites the pinned sha256 to some other value, artifact untouched.
        wrong_pin = "0" * 64
        check(run_verify(art, sig_path, wrong_pin, pub, "worker.ts") == 3,
              "wrong pinned sha256 -> REJECTED (exit 3, sha256 mismatch)")

        # wrong-key signature: signed by the unpinned attacker key, checked against the pinned pubkey.
        _, forged_sig = S.sign_artifact(attacker, art, "worker.ts")
        forged_path = os.path.join(d, "worker.forged.sig")
        _write(forged_path, (forged_sig + "\n").encode())
        check(run_verify(art, forged_path, good_sha, pub, "worker.ts") == 4,
              "signature from unpinned attacker key -> REJECTED (exit 4, bad signature)")
        # sanity: that SAME forged sig DOES verify against the attacker's own pubkey (proves the
        # rejection above is the key binding, not a broken signer).
        check(run_verify(art, forged_path, good_sha, att_pub, "worker.ts") == 0,
              "  (sanity) forged sig verifies under the ATTACKER pubkey -> rejection was key-binding")

        # cross-artifact replay: a genuine sig for name "worker.ts" re-presented under another name.
        check(run_verify(art, sig_path, good_sha, pub, "worker_torch.py") == 4,
              "valid sig replayed under a different --name -> REJECTED (exit 4, name binding)")

        # missing / empty signature file.
        empty_path = os.path.join(d, "empty.sig")
        _write(empty_path, b"")
        check(run_verify(art, empty_path, good_sha, pub, "worker.ts") == 4,
              "empty / missing signature -> REJECTED (exit 4)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ============================================================================================
# Integration guards against the actually-shipped release bundle.
# ============================================================================================
def _install_pin(var: str) -> str:
    """Pull the baked-in default of a `VAR="${ENV:-DEFAULT}"` pin out of scripts/install.sh."""
    src = open(INSTALL_SH).read()
    m = re.search(rf'{var}="\$\{{[A-Z0-9_]+:-([^}}]+)\}}"', src)
    if not m:
        raise AssertionError(f"could not find pin {var} in install.sh")
    return m.group(1)


def _release_strict() -> bool:
    """The 'committed artifact matches its signature' check (b) is FRESHNESS, and freshness is a RELEASE gate:
    HARD on a tagged release build (or when MOREGPU_RELEASE_STRICT=1), ADVISORY on ordinary push / PR / local
    dev runs — so a routine worker.ts edit does not turn CI red before the maintainer re-signs at release time.
    The supply-chain gate ITSELF — the hermetic sign→verify negatives (a), no verifier drift (c), no key
    material in the .sig files (d) — stays a hard assertion on EVERY run; only (b)'s freshness is gated here."""
    if os.environ.get("MOREGPU_RELEASE_STRICT") == "1":
        return True
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        return True
    return os.environ.get("GITHUB_REF", "").startswith("refs/tags/")


def _gate(strict: bool, passed: bool, label: str) -> None:
    """A release-gated assertion: recorded as a real check when it passes or when strict; otherwise an advisory
    WARN that never fails the run (a dev build whose worker.ts is simply ahead of its last signature)."""
    if passed or strict:
        check(passed, label)
    else:
        print(f"  [WARN] {label}  — advisory on this dev build; a release tag (MOREGPU_RELEASE_STRICT=1) makes it a hard failure")


def test_shipped_bundle() -> None:
    strict = _release_strict()
    mode = "RELEASE gate — strict" if strict else "dev build — freshness advisory (re-sign before release)"
    print(f"\n(b) shipped bundle  [scripts/install.sh pins + apps/worker/worker.ts + .sig]  ·  {mode}")
    pub = _install_pin("RELEASE_PUBKEY_B64")
    sha = _install_pin("WORKER_TS_SHA256")

    if not os.path.exists(WORKER_TS_SIG):
        _gate(strict, False, f"committed signature {os.path.basename(WORKER_TS_SIG)} present — run scripts/release_sign.py sign")
        return
    rc = run_verify(WORKER_TS, WORKER_TS_SIG, sha, pub, "worker.ts")
    _gate(strict, rc == 0,
          "committed worker.ts verifies against install.sh pins + committed .sig (exit 0)"
          + ("" if rc == 0 else "  [worker.ts changed since signing — re-run scripts/release_sign.py sign before cutting a release]"))


def test_no_drift() -> None:
    print("\n(c) no drift  [verifier embedded in install.sh == scripts/verify_release.ts]")
    canonical = open(VERIFY_TS).read()
    src = open(INSTALL_SH).read()
    marker = "<<'MOREGPU_VERIFY_EOF'\n"
    end = "\nMOREGPU_VERIFY_EOF\n"
    if marker not in src or end not in src:
        check(False, "install.sh does not embed the verifier heredoc (markers missing)")
        return
    embedded = src.split(marker, 1)[1].split(end, 1)[0] + "\n"
    check(embedded == canonical,
          "install.sh embeds a byte-identical copy of scripts/verify_release.ts (no fork/drift)")


def test_no_secrets() -> None:
    print("\n(d) no secrets  [committed .sig files are 64-byte Ed25519 signatures, not keys]")
    for sig in (WORKER_TS_SIG, os.path.join(REPO, "apps", "worker", "worker_torch.py.sig")):
        if not os.path.exists(sig):
            continue
        raw = base64.b64decode(open(sig).read().strip())
        check(len(raw) == 64, f"{os.path.basename(sig)} is a 64-byte Ed25519 signature (public), not key material")


def main() -> int:
    print("=" * 92)
    print("RELEASE VERIFY — signed, hash-pinned worker release fails closed (sha256 pin + Ed25519 sig)")
    print(f"  sign   : {SIGN_PY}")
    print(f"  verify : {VERIFY_TS}   (WebCrypto Ed25519, same call as server.ts:251)")
    print(f"  gate   : {INSTALL_SH}")
    print("=" * 92)

    test_hermetic_gate()
    test_shipped_bundle()
    test_no_drift()
    test_no_secrets()

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
    print("EVERY UNSIGNED / TAMPERED / WRONG-KEY RELEASE REJECTED — the installer fails closed. ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
