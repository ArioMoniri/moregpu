#!/usr/bin/env python3
"""
SEAL NEGATIVES — the sealed shard/compute dispatch path must FAIL CLOSED.

This proves, against the REAL production code (no re-implemented crypto), that a tampered or forged
frame on the sealed WebSocket protocol is REJECTED rather than silently processed:

  (a) SEAL/UNSEAL (AES-256-GCM) — apps/worker/worker_torch.py :: seal()/unseal()  (lines 63-67).
      These are the exact functions handle_assign()/handle_relay()/handle_cache() call on every
      `assign`/`model`/`shard_forward`/`cache` frame (worker_torch.py:854,867,876). We assert that a
      tampered ciphertext body, a flipped GCM tag, a wrong nonce (IV), and a wrong tenant key each
      raise cryptography's InvalidTag — a GCM authentication failure — and that NO plaintext is
      ever returned. Fail-closed, not fail-open.

  (b) Ed25519 RESULT SIGNING — worker signs, coordinator verifies.
      The worker signs every result over `${shardId}|${iv}|${ct}` with sign_result()/_sk
      (worker_torch.py:68-69, sent at :858). The coordinator VERIFIES with WebCrypto Ed25519
      (apps/coordinator/server.ts:205 importKey + :251 crypto.subtle.verify) and, on failure,
      does `p.reject(new Error('invalid result signature'))` (server.ts:252). We run that EXACT
      coordinator verify in a Deno subprocess (the coordinator's own runtime) fed by REAL worker
      signatures, and assert it returns okSig=false — i.e. the shard is rejected — for:
        - a tampered signature,
        - a forged signature from an unregistered signer (wrong pubkey),
        - a ciphertext mutated after signing (sig no longer matches the blob),
        - a REPLAYED signature: a valid sig captured for shard S0 and re-presented as the result
          for a different in-flight shard S9. Because the signature binds the shardId, the rebind
          fails verification — that is the replay defense at the signature layer. (The identical-
          frame replay is additionally defeated one level up by the coordinator's one-shot
          pending-map + `p.workerId !== id` guard at server.ts:246, out of scope for this crypto test.)

Runs on CPU, no GPU, no network, no live coordinator: unit-level against the imported real functions
plus a tiny Deno verify shim that copies server.ts's verify call verbatim. Ed25519 is RFC 8032, so
the Python-signed / Deno-verified round trip also proves the cross-runtime byte-compatibility the
protocol depends on.

Run:  python3 tests/security/seal_negatives.py      (exit 0 = all negatives rejected)
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKER_PY = os.path.join(REPO, "apps", "worker", "worker_torch.py")


# ---- import the REAL worker crypto (seal/unseal/sign_result/_sk/PUBKEY_B64/f32_to_b64) ----
def _load_worker():
    spec = importlib.util.spec_from_file_location("worker_torch", WORKER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module-level run() only fires under __main__, so importing is side-effect-free
    return mod


W = _load_worker()

# The exact WebCrypto-compatible verify the coordinator runs, in the coordinator's own runtime (Deno).
# b64d + importKey('raw', ..., {name:'Ed25519'}) + verify(...) are copied verbatim from server.ts:35,205,251.
DENO_VERIFY_SHIM = r"""
function b64d(s) {
  const F = Uint8Array.fromBase64;                       // server.ts:35-38
  if (typeof F === "function") return F(s);
  const bin = atob(s); const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}
const { cases } = JSON.parse(await new Response(Deno.stdin.readable).text());
const out = [];
for (const c of cases) {
  // server.ts:205 — import the worker's raw Ed25519 public key for verification
  const pubkey = await crypto.subtle.importKey("raw", b64d(c.pubkey), { name: "Ed25519" }, false, ["verify"]);
  // server.ts:251 — verify the result signature over `${shardId}|${iv}|${ct}`
  const okSig = await crypto.subtle.verify(
    { name: "Ed25519" }, pubkey, b64d(c.sig),
    new TextEncoder().encode(`${c.shardId}|${c.iv}|${c.ct}`),
  );
  out.push({ name: c.name, okSig });                     // server.ts:252 — if(!okSig) reject('invalid result signature')
}
console.log(JSON.stringify(out));
"""


# ---------- tiny assertion harness (standalone script style, like tests/e2e/*.py) ----------
_RESULTS: list[tuple[bool, str]] = []


def check(passed: bool, label: str) -> None:
    _RESULTS.append((passed, label))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


def flip_b64_byte(b64: str, idx: int) -> str:
    """Return b64 with one raw byte flipped (0-based idx; negative indexes from the end)."""
    raw = bytearray(base64.b64decode(b64))
    raw[idx] ^= 0xFF
    return base64.b64encode(bytes(raw)).decode()


# ============================================================================================
# (a) AES-256-GCM seal/unseal — the sealed compute-dispatch payload must fail closed on tamper
# ============================================================================================
def test_gcm_fail_closed() -> None:
    print("\n(a) AES-256-GCM seal/unseal  [worker_torch.py:63-67 · used by handle_assign :854]")
    import torch

    key = os.urandom(32)  # 256-bit tenant key, as the coordinator's welcome{tenantKeyB64} delivers

    # A REALISTIC sealed result payload: exactly what handle_assign seals at worker_torch.py:856,
    # {"out": f32_to_b64(tensor)}, using the worker's real wire<->tensor encoder.
    out_tensor = torch.tensor([1.5, -2.25, 3.0, 0.0, 42.0], dtype=torch.float32)
    plaintext = json.dumps({"out": W.f32_to_b64(out_tensor)}).encode()
    blob = W.seal(key, plaintext)  # {"iv": b64, "ct": ciphertext||16-byte GCM tag, b64}

    # Control (fail-OPEN sanity): the untouched blob unseals back to the exact bytes.
    check(W.unseal(key, blob) == plaintext, "control: untouched blob round-trips to original plaintext")

    def rejects(bad_blob: dict, bad_key: bytes, label: str) -> None:
        leaked = None
        try:
            leaked = W.unseal(bad_key, bad_blob)
        except InvalidTag:
            check(True, f"{label} -> REJECTED (InvalidTag / GCM auth failure)")
            return
        except Exception as e:  # any other rejection is still fail-closed, but flag the type
            check(True, f"{label} -> REJECTED ({type(e).__name__})")
            return
        check(False, f"{label} -> SILENTLY ACCEPTED, leaked {len(leaked)} bytes")

    # Tampered ciphertext body (flip a byte before the 16-byte tag).
    rejects({"iv": blob["iv"], "ct": flip_b64_byte(blob["ct"], 0)}, key, "tampered ciphertext body")
    # Flipped GCM tag (last byte of ct == last tag byte).
    rejects({"iv": blob["iv"], "ct": flip_b64_byte(blob["ct"], -1)}, key, "flipped GCM auth tag")
    # Wrong nonce / IV (flip a byte of the 12-byte IV).
    rejects({"iv": flip_b64_byte(blob["iv"], 0), "ct": blob["ct"]}, key, "wrong nonce (mutated IV)")
    # Wrong tenant key (cross-tenant / stolen-blob attempt).
    rejects(dict(blob), os.urandom(32), "wrong tenant key")
    # Truncated ciphertext (drops part of the tag -> auth cannot succeed).
    trunc = base64.b64encode(base64.b64decode(blob["ct"])[:-1]).decode()
    rejects({"iv": blob["iv"], "ct": trunc}, key, "truncated ciphertext/tag")


# ============================================================================================
# (b) Ed25519 result signing — the coordinator's verify must reject bad / replayed signatures
# ============================================================================================
def run_coordinator_verify(cases: list[dict]) -> dict[str, bool]:
    """Run server.ts's verify call verbatim in Deno; return {caseName: okSig}."""
    deno = shutil.which("deno")
    if not deno:
        raise RuntimeError("deno not found on PATH — the coordinator's verify runtime is required")
    with tempfile.NamedTemporaryFile("w", suffix=".ts", delete=False) as f:
        f.write(DENO_VERIFY_SHIM)
        shim = f.name
    try:
        proc = subprocess.run(
            [deno, "run", shim],  # crypto.subtle + stdin need no --allow flags
            input=json.dumps({"cases": cases}),
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"deno verify shim failed (rc={proc.returncode}):\n{proc.stderr}")
        return {r["name"]: r["okSig"] for r in json.loads(proc.stdout.strip().splitlines()[-1])}
    finally:
        os.unlink(shim)


def test_ed25519_verify_rejects() -> None:
    print("\n(b) Ed25519 result signing  [worker sign_result :68-69 -> coordinator verify server.ts:251]")
    key = os.urandom(32)

    # A genuine signed result for shard S0, exactly as handle_assign builds it (worker_torch.py:857-859).
    shard0 = "job-1:shard-0"
    blob0 = W.seal(key, json.dumps({"out": "AAAAAA=="}).encode())
    sig0 = W.sign_result(shard0, blob0)  # real _sk.sign over `${shard0}|${iv}|${ct}`
    pub = W.PUBKEY_B64  # the raw Ed25519 pubkey the worker registered (server.ts:205 imports this)

    # An UNREGISTERED signer (attacker key not in the worker registry) signs the same message.
    attacker = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    forged_sig = base64.b64encode(
        attacker.sign(f"{shard0}|{blob0['iv']}|{blob0['ct']}".encode())
    ).decode()

    # A second live shard S9 the replayed frame will be aimed at.
    shard9 = "job-1:shard-9"

    cases = [
        # control (fail-OPEN sanity): the genuine signature verifies -> shard accepted.
        {"name": "valid_signature", "shardId": shard0, "iv": blob0["iv"], "ct": blob0["ct"], "sig": sig0, "pubkey": pub},
        # tampered signature bytes.
        {"name": "tampered_signature", "shardId": shard0, "iv": blob0["iv"], "ct": blob0["ct"], "sig": flip_b64_byte(sig0, 10), "pubkey": pub},
        # forged signature from an unregistered signer, checked against the worker's registered pubkey.
        {"name": "forged_wrong_signer", "shardId": shard0, "iv": blob0["iv"], "ct": blob0["ct"], "sig": forged_sig, "pubkey": pub},
        # ciphertext mutated AFTER signing: the sig no longer covers the delivered blob.
        {"name": "ciphertext_mutated_after_sign", "shardId": shard0, "iv": blob0["iv"], "ct": flip_b64_byte(blob0["ct"], 0), "sig": sig0, "pubkey": pub},
        # REPLAY: valid sig captured for S0, re-presented as the result for a different shard S9.
        {"name": "replayed_cross_shard", "shardId": shard9, "iv": blob0["iv"], "ct": blob0["ct"], "sig": sig0, "pubkey": pub},
    ]

    res = run_coordinator_verify(cases)

    check(res.get("valid_signature") is True, "control: genuine signature VERIFIES (shard accepted)")
    check(res.get("tampered_signature") is False, "tampered signature -> REJECTED (invalid result signature)")
    check(res.get("forged_wrong_signer") is False, "forged sig from unregistered signer -> REJECTED")
    check(res.get("ciphertext_mutated_after_sign") is False, "ciphertext mutated after signing -> REJECTED")
    check(res.get("replayed_cross_shard") is False, "REPLAYED sig re-aimed at another shard -> REJECTED")


def main() -> int:
    print("=" * 92)
    print("SEAL NEGATIVES — sealed shard/compute dispatch path fails closed (GCM auth + Ed25519 verify)")
    print(f"  worker      : {WORKER_PY}")
    print(f"  coordinator : {os.path.join(REPO, 'apps', 'coordinator', 'server.ts')}  (verify @ :205,:251,:252)")
    print("=" * 92)

    test_gcm_fail_closed()
    test_ed25519_verify_rejects()

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
    print("ALL NEGATIVES REJECTED — the sealed protocol fails closed. ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
