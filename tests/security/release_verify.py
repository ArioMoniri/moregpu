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

  Second artefact + torch-worker manifest (throwaway keys only):
    - (e) the REAL install.sh against a file:// release tree: a genuine vision_wgsl.ts is fetched + verified and the
      worker boots with it; a tampered / missing / unsigned / wrong-key vision_wgsl.ts is REMOVED (vision disabled)
      while the install succeeds and the worker still starts; a tampered worker.ts still aborts the install
    - (f) install.sh pins VISION_WGSL_TS_SHA256; the coordinator's built-in worker --reload refreshes vision_wgsl.ts
    - (g) ADR-0103 MANIFEST.sha256 over EVERY file under apps/worker (worker_torch.py, pyproject.toml, moregpu_worker/**,
      ...; not *.sig / the manifest itself): release_sign.py manifest/verify-manifest and verify_release.ts
      --manifest-root reject a tampered (5) / added (5) / missing (5) file, a __pycache__ dir or sourceless .pyc (5),
      a planted root-level numpy.py (5), a rewritten manifest (4), a wrong key (4), and a missing / mismatched
      version header or a rollback below --expect-version / --state (6)
    - (h) `moregpu torch-join` purges bytecode caches and runs verify-manifest before exec'ing worker_torch.py (refuses
      on any failure; warns "unsigned dev tree" when no MANIFEST exists; MOREGPU_VERIFY_MANIFEST=1 makes it mandatory)

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
    for sig in (WORKER_TS_SIG, os.path.join(REPO, "apps", "worker", "worker_torch.py.sig"),
                os.path.join(REPO, "apps", "worker", "vision_wgsl.ts.sig"), os.path.join(REPO, "apps", "worker", "MANIFEST.sha256.sig")):
        if not os.path.exists(sig):
            continue
        raw = base64.b64decode(open(sig).read().strip())
        check(len(raw) == 64, f"{os.path.basename(sig)} is a 64-byte Ed25519 signature (public), not key material")


# ============================================================================================
# (e) vision_wgsl.ts — a SECOND signed artefact. The installer fetches + verifies it; a tampered / missing /
#     unsigned vision module is DROPPED (vision disabled) while the worker itself still installs and starts.
# ============================================================================================
VISION_TS = os.path.join(REPO, "apps", "worker", "vision_wgsl.ts")
VISION_TS_SIG = VISION_TS + ".sig"


def _raw_tree(d: str, sk, *, vision: str = "good", worker: str = "good") -> tuple[str, str, str]:
    """A fake raw.githubusercontent tree (file://) holding the REAL worker.ts + vision_wgsl.ts, signed by `sk`.
    Returns (raw_base_url, worker_sha, vision_sha)."""
    raw = os.path.join(d, "raw")
    wdir = os.path.join(raw, "apps", "worker")
    os.makedirs(wdir, exist_ok=True)
    shutil.copy(WORKER_TS, os.path.join(wdir, "worker.ts"))
    shutil.copy(VISION_TS, os.path.join(wdir, "vision_wgsl.ts"))
    wsha, wsig = S.sign_artifact(sk, os.path.join(wdir, "worker.ts"), "worker.ts")
    vsha, vsig = S.sign_artifact(sk, os.path.join(wdir, "vision_wgsl.ts"), "vision_wgsl.ts")
    _write(os.path.join(wdir, "worker.ts.sig"), (wsig + "\n").encode())
    _write(os.path.join(wdir, "vision_wgsl.ts.sig"), (vsig + "\n").encode())
    if worker == "tampered":
        with open(os.path.join(wdir, "worker.ts"), "ab") as f:
            f.write(b"\n// injected\n")
    if vision == "tampered":
        with open(os.path.join(wdir, "vision_wgsl.ts"), "ab") as f:
            f.write(b"\nexport const PWNED = 1;\n")
    elif vision == "missing":
        os.remove(os.path.join(wdir, "vision_wgsl.ts"))
        os.remove(os.path.join(wdir, "vision_wgsl.ts.sig"))
    elif vision == "nosig":
        os.remove(os.path.join(wdir, "vision_wgsl.ts.sig"))
    elif vision == "wrongkey":
        _, bad = S.sign_artifact(Ed25519PrivateKey.generate(), os.path.join(wdir, "vision_wgsl.ts"), "vision_wgsl.ts")
        _write(os.path.join(wdir, "vision_wgsl.ts.sig"), (bad + "\n").encode())
    return "file://" + raw, wsha, vsha


def run_install(d: str, raw_base: str, pub: str, wsha: str, vsha: str) -> tuple[int, str, str]:
    """Run the REAL scripts/install.sh against the file:// tree, stage + verify only (MOREGPU_INSTALL_ONLY=1)."""
    home = os.path.join(d, "home")
    os.makedirs(home, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MOREGPU_")}
    env.update(HOME=home, MOREGPU_RAW_BASE=raw_base, MOREGPU_RELEASE_PUBKEY=pub, MOREGPU_WORKER_SHA256=wsha,
               MOREGPU_VISION_WGSL_SHA256=vsha, MOREGPU_INSTALL_ONLY="1", MOREGPU_SERVER="ws://127.0.0.1:9/ws")
    p = subprocess.run(["sh", INSTALL_SH], env=env, capture_output=True, text=True, timeout=180)
    return p.returncode, p.stdout + p.stderr, os.path.join(home, ".moregpu")


def boot_worker(mg: str, timeout: float = 90.0) -> str:
    """Start the installed worker exactly as install.sh's RUN_ARGS do (no --allow-read), until it logs its backend
    line, then stop it. Returns its output. A worker that cannot start never prints that line."""
    env = dict(os.environ, MOREGPU_FORCE_CPU="1")
    p = subprocess.Popen([_deno(), "run", "--unstable-webgpu", "--allow-net", "--allow-env", "--allow-sys",
                          os.path.join(mg, "worker.ts"), "--server", "ws://127.0.0.1:9/ws", "--token", "t", "--name", "rv"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    out = []
    import time
    t0 = time.time()
    try:
        while time.time() - t0 < timeout:
            line = p.stdout.readline()
            if not line:
                break
            out.append(line)
            if "backend=" in line:
                break
    finally:
        p.kill()
        p.wait()
    return "".join(out)


def test_vision_second_artifact() -> None:
    print("\n(e) vision_wgsl.ts second signed artefact  [install.sh fetch+verify; tampered/missing → vision off, worker starts]")
    sk = Ed25519PrivateKey.generate()   # throwaway key: NEVER the release key
    pub = S.pubkey_b64(sk)
    for case in ("good", "tampered", "missing", "nosig", "wrongkey"):
        d = tempfile.mkdtemp(prefix=f"mg-inst-{case}-")
        try:
            raw, wsha, vsha = _raw_tree(d, sk, vision=case)
            rc, log, mg = run_install(d, raw, pub, wsha, vsha)
            staged = os.path.exists(os.path.join(mg, "vision_wgsl.ts"))
            if case == "good":
                check(rc == 0 and staged and "vision_wgsl.ts" in log and "verified" in log,
                      "genuine vision_wgsl.ts + sig → installed and VERIFIED next to worker.ts")
            else:
                check(rc == 0 and not staged and "vision disabled" in log,
                      f"{case} vision_wgsl.ts → dropped (vision disabled), install still succeeds (exit {rc})")
            boot = boot_worker(mg) if rc == 0 else ""
            started = "backend=" in boot
            off = "vision module unavailable" in boot
            if case == "good":
                check(started and not off, "  worker starts WITH the verified vision module")
            else:
                check(started and off, f"  worker still STARTS with vision disabled ({case})")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    # the worker itself stays fail-CLOSED
    d = tempfile.mkdtemp(prefix="mg-inst-wbad-")
    try:
        raw, wsha, vsha = _raw_tree(d, sk, worker="tampered")
        rc, log, _ = run_install(d, raw, pub, wsha, vsha)
        check(rc != 0 and "REFUSING" in log, "tampered worker.ts → install REFUSED (fail closed, unchanged)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_vision_shipped_and_reload() -> None:
    strict = _release_strict()
    print("\n(f) vision_wgsl.ts shipped pin + coordinator built-in worker reload")
    src = open(INSTALL_SH).read()
    check('VISION_WGSL_TS_SHA256="${MOREGPU_VISION_WGSL_SHA256:-' in src, "install.sh pins VISION_WGSL_TS_SHA256 (env-overridable)")
    if os.path.exists(VISION_TS_SIG):
        rc = run_verify(VISION_TS, VISION_TS_SIG, _install_pin("VISION_WGSL_TS_SHA256"), _install_pin("RELEASE_PUBKEY_B64"), "vision_wgsl.ts")
        _gate(strict, rc == 0, "committed vision_wgsl.ts verifies against install.sh pins + committed .sig")
    else:
        _gate(strict, False, "committed vision_wgsl.ts.sig present — run scripts/release_sign.py sign apps/worker/vision_wgsl.ts at release")
    server = open(os.path.join(REPO, "apps", "coordinator", "server.ts")).read()
    check(re.search(r"--reload=\$\{[^}]*\}.*vision_wgsl\.ts|visionUrl", server) is not None and "vision_wgsl.ts" in server,
          "coordinator's built-in worker --reload also refreshes vision_wgsl.ts")


# ============================================================================================
# (g) ADR-0103 — signed MANIFEST.sha256 of the torch worker tree (apps/worker/moregpu_worker/** + vision_ops.json)
# ============================================================================================
def run_manifest_verify_py(root: str, pub: str, *extra: str) -> int:
    p = subprocess.run([sys.executable, SIGN_PY, "verify-manifest", "--root", root, "--pubkey", pub, *extra],
                       capture_output=True, text=True, timeout=120)
    line = (p.stdout or p.stderr).strip().splitlines()
    if line:
        print(f"        py gate: {line[-1]}")
    return p.returncode


def run_manifest_verify_ts(root: str, pub: str) -> int:
    p = subprocess.run([_deno(), "run", "--allow-read", VERIFY_TS, "--manifest-root", root,
                        "--artifact", os.path.join(root, "MANIFEST.sha256"), "--sig", os.path.join(root, "MANIFEST.sha256.sig"),
                        "--pubkey", pub, "--name", "MANIFEST.sha256"], capture_output=True, text=True, timeout=120)
    line = (p.stdout or p.stderr).strip().splitlines()
    if line:
        print(f"        ts gate: {line[-1]}")
    return p.returncode


def _both(root: str, pub: str) -> tuple[int, int]:
    return run_manifest_verify_py(root, pub), run_manifest_verify_ts(root, pub)


WORKER_DIR = os.path.join(REPO, "apps", "worker")
_TREE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "MANIFEST.sha256", "MANIFEST.sha256.sig")


def _pyproject_version(root: str) -> str:
    return re.search(r'(?m)^version\s*=\s*"([^"]+)"', open(os.path.join(root, "pyproject.toml")).read()).group(1)


def _sign_manifest(root: str, sk, text: bytes | None = None) -> None:
    """(Re)write MANIFEST.sha256 (+ .sig) under root with a THROWAWAY key — the real tool, or explicit bytes."""
    if text is None:
        text = S.build_manifest(root).encode()
    _write(os.path.join(root, "MANIFEST.sha256"), text)
    _write(os.path.join(root, "MANIFEST.sha256.sig"),
           (base64.b64encode(sk.sign(S.release_message("MANIFEST.sha256", S.sha256_hex(text)))).decode() + "\n").encode())


def test_torch_worker_manifest() -> None:
    print("\n(g) torch worker MANIFEST.sha256  [release_sign.py manifest → verify-manifest (py) + verify_release.ts --manifest-root]")
    sk = Ed25519PrivateKey.generate()   # throwaway key
    pub = S.pubkey_b64(sk)
    d = tempfile.mkdtemp(prefix="mg-manifest-")
    try:
        root = os.path.join(d, "worker")
        shutil.copytree(WORKER_DIR, root, ignore=_TREE_IGNORE)
        key = os.path.join(d, "k.b64")
        _write(key, (base64.b64encode(sk.private_bytes_raw()).decode() + "\n").encode())
        p = subprocess.run([sys.executable, SIGN_PY, "manifest", "--key", key, "--root", root], capture_output=True, text=True, timeout=120)
        man = os.path.join(root, "MANIFEST.sha256")
        check(p.returncode == 0 and os.path.exists(man) and os.path.exists(man + ".sig"), "manifest + detached sig written")
        text = open(man).read()
        head, lines = text.splitlines()[0], text.splitlines()[1:]
        paths = [l.split("  ", 1)[1] for l in lines]
        version = _pyproject_version(root)
        check(head == f"# moregpu-worker-version: {version}",
              f"manifest carries a version header bound to apps/worker/pyproject.toml ({head!r})")
        check(paths == sorted(paths) and all(x in paths for x in ("moregpu_worker/vision/lowering.py", "vision_ops.json",
                                                                  "worker_torch.py", "pyproject.toml", "worker.ts"))
              and not any("__pycache__" in x or x.endswith(".sig") or x.startswith("MANIFEST") for x in paths),
              f"manifest covers EVERY file under apps/worker (worker_torch.py, pyproject.toml, ...) except .sig/MANIFEST ({len(lines)} files)")
        check(_both(root, pub) == (0, 0), "control: genuine tree VERIFIES (py + ts, exit 0)")
        # a planted bytecode cache can shadow a verified .py (timestamp pycs are trusted by the importer): REFUSE it
        cache = os.path.join(root, "moregpu_worker", "__pycache__")
        os.makedirs(cache, exist_ok=True)
        _write(os.path.join(cache, "bench.cpython-311.pyc"), b"\0")
        check(_both(root, pub) == (5, 5), "__pycache__ present → REJECTED (py + ts, exit 5)")
        check(run_manifest_verify_py(root, pub, "--purge-pycache") == 0 and not os.path.exists(cache),
              "  verify-manifest --purge-pycache deletes bytecode caches, then VERIFIES")
        sourceless = os.path.join(root, "moregpu_worker", "vision", "evil.pyc")
        _write(sourceless, b"\0")
        check(_both(root, pub) == (5, 5), "sourceless .pyc in the package → REJECTED (exit 5)")
        os.remove(sourceless)
        # a module planted next to worker_torch.py shadows a dependency (sys.path[0] is the script dir)
        planted = os.path.join(root, "numpy.py")
        _write(planted, b"import os; os.system('id')\n")
        check(_both(root, pub) == (5, 5), "planted root-level numpy.py next to worker_torch.py → REJECTED (exit 5)")
        os.remove(planted)
        victim_root = os.path.join(root, "worker_torch.py")
        orig_root = open(victim_root, "rb").read()
        _write(victim_root, orig_root + b"\n# evil\n")
        check(_both(root, pub) == (5, 5), "tampered worker_torch.py → REJECTED (exit 5)")
        _write(victim_root, orig_root)
        victim = os.path.join(root, "moregpu_worker", "vision", "lowering.py")
        orig = open(victim, "rb").read()
        _write(victim, orig + b"\nimport os; os.system('id')\n")
        check(_both(root, pub) == (5, 5), "tampered package file → REJECTED (exit 5)")
        _write(victim, orig)
        added = os.path.join(root, "moregpu_worker", "vision", "evil.py")
        _write(added, b"print('x')\n")
        check(_both(root, pub) == (5, 5), "ADDED package file → REJECTED (exit 5)")
        os.remove(added)
        os.remove(victim)
        check(_both(root, pub) == (5, 5), "MISSING package file → REJECTED (exit 5)")
        _write(victim, orig)
        check(_both(root, pub) == (0, 0), "  restored tree verifies again")
        # attacker rewrites the manifest to match a tampered file → the manifest signature breaks
        _write(victim, orig + b"\n# evil\n")
        import hashlib
        new = hashlib.sha256(open(victim, "rb").read()).hexdigest()
        _write(man, (head + "\n" + "\n".join(f"{new}  {l.split('  ', 1)[1]}" if l.endswith("  moregpu_worker/vision/lowering.py") else l
                                            for l in lines) + "\n").encode())
        check(_both(root, pub) == (4, 4), "rewritten manifest → REJECTED (exit 4, signature)")
        _write(victim, orig)
        _sign_manifest(root, sk)
        check(run_manifest_verify_py(root, S.pubkey_b64(Ed25519PrivateKey.generate())) == 4, "wrong release key → REJECTED (exit 4)")

        # ---- version binding / rollback (exit 6)
        print("  version binding / rollback:")
        good = open(man, "rb").read()
        body = b"\n".join(good.split(b"\n")[1:])
        _sign_manifest(root, sk, body)                                   # validly signed, but no version header
        check(_both(root, pub) == (6, 6), "manifest WITHOUT a version header → REJECTED (exit 6)")
        _sign_manifest(root, sk, b"# moregpu-worker-version: 99.0.0\n" + body)
        check(_both(root, pub) == (6, 6), "manifest version != pyproject.toml version → REJECTED (exit 6)")
        _sign_manifest(root, sk, good)
        check(run_manifest_verify_py(root, pub, "--expect-version", version) == 0, "  --expect-version <current> → OK")
        check(run_manifest_verify_py(root, pub, "--expect-version", "99.0.0") == 6, "--expect-version mismatch → REJECTED (exit 6)")
        state = os.path.join(d, "state", "torch-worker.version")
        check(run_manifest_verify_py(root, pub, "--state", state) == 0 and open(state).read().strip() == version,
              "  --state records the verified version")
        _write(state, b"99.0.0\n")
        check(run_manifest_verify_py(root, pub, "--state", state) == 6,
              "ROLLBACK below the last verified version (--state) → REJECTED (exit 6)")
        check(S.version_key("0.7.0.dev0") < S.version_key("0.7.0rc1") < S.version_key("0.7.0") < S.version_key("0.7.1")
              < S.version_key("0.10.0"), "  version ordering: dev < rc < final, numeric components")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ============================================================================================
# (h) `moregpu torch-join` runs the manifest gate before exec'ing worker_torch.py
# ============================================================================================
MOREGPU_SH = os.path.join(REPO, "scripts", "moregpu")


def _torch_join_tree(d: str) -> str:
    """A throwaway clone layout: scripts/{moregpu,release_sign.py} + a stub apps/worker tree whose worker prints RAN."""
    os.makedirs(os.path.join(d, "scripts"))
    for f in ("moregpu", "release_sign.py"):
        shutil.copy2(os.path.join(REPO, "scripts", f), os.path.join(d, "scripts", f))
    w = os.path.join(d, "apps", "worker")
    os.makedirs(os.path.join(w, "moregpu_worker"))
    _write(os.path.join(w, "worker_torch.py"), b"import sys\nprint('WORKER-RAN', sys.argv[1:], flush=True)\n")
    _write(os.path.join(w, "pyproject.toml"), b'[project]\nname = "moregpu-worker"\nversion = "0.7.0.dev0"\n')
    _write(os.path.join(w, "moregpu_worker", "__init__.py"), b"")
    return w


def run_torch_join(d: str, env: dict) -> tuple[int, str]:
    e = {k: v for k, v in os.environ.items() if not k.startswith("MOREGPU_")}
    e.update(NO_COLOR="1", MOREGPU_STATE_DIR=os.path.join(d, "state"), **env)
    p = subprocess.run(["bash", os.path.join(d, "scripts", "moregpu"), "torch-join", "--name", "t"], env=e,
                       capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def test_torch_join_gate() -> None:
    print("\n(h) `moregpu torch-join` verifies the signed MANIFEST before exec  [scripts/moregpu]")
    sk = Ed25519PrivateKey.generate()   # throwaway key
    pub = S.pubkey_b64(sk)
    d = tempfile.mkdtemp(prefix="mg-tjoin-")
    try:
        w = _torch_join_tree(d)
        rc, out = run_torch_join(d, {"MOREGPU_RELEASE_PUBKEY": pub})
        check(rc == 0 and "WORKER-RAN" in out and "unsigned dev tree" in out,
              "no MANIFEST → runs with a clear 'unsigned dev tree' warning")
        rc, out = run_torch_join(d, {"MOREGPU_RELEASE_PUBKEY": pub, "MOREGPU_VERIFY_MANIFEST": "1"})
        check(rc != 0 and "WORKER-RAN" not in out, "MOREGPU_VERIFY_MANIFEST=1 without a MANIFEST → REFUSED")
        _sign_manifest(w, sk)
        rc, out = run_torch_join(d, {"MOREGPU_RELEASE_PUBKEY": pub})
        check(rc == 0 and "WORKER-RAN" in out and "--name" in out, "signed tree → verified, then worker_torch.py exec'd with its args")
        cache = os.path.join(w, "moregpu_worker", "__pycache__")
        os.makedirs(cache)
        _write(os.path.join(cache, "x.cpython-311.pyc"), b"\0")
        rc, out = run_torch_join(d, {"MOREGPU_RELEASE_PUBKEY": pub})
        check(rc == 0 and "WORKER-RAN" in out and not os.path.exists(cache), "stale __pycache__ is purged before verification")
        _write(os.path.join(w, "numpy.py"), b"raise SystemExit('pwned')\n")
        rc, out = run_torch_join(d, {"MOREGPU_RELEASE_PUBKEY": pub})
        check(rc != 0 and "WORKER-RAN" not in out, "planted numpy.py next to worker_torch.py → REFUSED (never exec'd)")
        rc, out = run_torch_join(d, {"MOREGPU_RELEASE_PUBKEY": pub, "MOREGPU_VERIFY_MANIFEST": "0"})
        check(rc == 0 and "WORKER-RAN" in out and "NOT verified" in out,
              "  MOREGPU_VERIFY_MANIFEST=0 is an explicit, loudly-warned opt-out")
        os.remove(os.path.join(w, "numpy.py"))
        rc, out = run_torch_join(d, {"MOREGPU_RELEASE_PUBKEY": S.pubkey_b64(Ed25519PrivateKey.generate())})
        check(rc != 0 and "WORKER-RAN" not in out, "manifest signed by another key → REFUSED")
        os.makedirs(os.path.join(d, "state"), exist_ok=True)
        _write(os.path.join(d, "state", "torch-worker.version"), b"0.8.0\n")
        rc, out = run_torch_join(d, {"MOREGPU_RELEASE_PUBKEY": pub})
        check(rc != 0 and "WORKER-RAN" not in out, "rollback below the last verified worker version → REFUSED")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    src = open(MOREGPU_SH).read()
    m = re.search(r'RELEASE_PUBKEY_B64="\$\{MOREGPU_RELEASE_PUBKEY:-([^}]+)\}"', src)
    check(m is not None and m.group(1) == _install_pin("RELEASE_PUBKEY_B64"),
          "scripts/moregpu pins the same release public key as scripts/install.sh (no drift)")


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
    test_vision_second_artifact()
    test_vision_shipped_and_reload()
    test_torch_worker_manifest()
    test_torch_join_gate()

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
