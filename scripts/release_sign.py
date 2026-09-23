#!/usr/bin/env python3
"""
release_sign.py — sign a MoreGPU worker artifact for a hash-PINNED, signed release.

This is the maintainer side of the supply-chain gate that replaces the old
`curl raw.githubusercontent.com/.../main/... | sh` fetch that ran WHATEVER `main` happened
to contain at that instant (no integrity check, no authenticity check). Instead, each release
artifact (apps/worker/worker.ts, apps/worker/worker_torch.py, ...) is:

  1. hashed  — sha256 of the exact bytes (the pin baked into scripts/install.sh), and
  2. signed  — a detached Ed25519 signature over a domain-separated message that binds the
               artifact's ROLE (basename) to that hash, so a signature minted for one artifact
               can never be replayed for another.

It REUSES the repo's existing crypto approach verbatim — the same primitives the torch worker
already signs results with (apps/worker/worker_torch.py:28,50-51,71):
    Ed25519PrivateKey / Encoding.Raw+PublicFormat.Raw / base64.
The install-time verifier (scripts/verify_release.ts) checks these with the SAME WebCrypto
Ed25519 verify the coordinator uses on every result (apps/coordinator/server.ts:205,251).

The signed MESSAGE (must match verify_release.ts exactly):

    moregpu-release/v1\n<basename>\n<sha256-hex>

KEY MANAGEMENT — read before you run:
  * The PRIVATE key never lives in the repo. `keygen` writes it (0600) to a path you choose;
    keep it in a secret store / hardware token. Only the PUBLIC key + sha256 + .sig ship.
  * `sign` prints the exact pins (RELEASE_PUBKEY_B64, <ARTIFACT>_SHA256) to paste into
    scripts/install.sh, and writes a detached `<artifact>.sig` next to each artifact.

Usage:
    # one-time: mint a release keypair (PRIVATE key stays OUT of the repo)
    python3 scripts/release_sign.py keygen --out /secure/moregpu_release_key.b64

    # each release: sign the artifacts the installer pins (worker.ts AND the lazily imported vision_wgsl.ts — the
    # installer verifies both; a vision module that fails verification is dropped, the worker still runs)
    python3 scripts/release_sign.py sign --key /secure/moregpu_release_key.b64 \\
        apps/worker/worker.ts apps/worker/vision_wgsl.ts apps/worker/worker_torch.py

    # each release (ADR-0103): a signed MANIFEST.sha256 of the torch worker tree (moregpu_worker/** + vision_ops.json)
    python3 scripts/release_sign.py manifest --key /secure/moregpu_release_key.b64 --root apps/worker
    # ...and its verify step (also: deno run --allow-read scripts/verify_release.ts --manifest-root apps/worker ...)
    python3 scripts/release_sign.py verify-manifest --root apps/worker --pubkey <RELEASE_PUBKEY_B64>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import stat
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

# Domain tag for the signed message. Bump the version if the signing scheme ever changes.
MSG_DOMAIN = "moregpu-release/v1"


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode()


def b64d(s: str) -> bytes:
    return base64.b64decode(s)


def sha256_hex(data: bytes) -> str:
    """Lowercase hex sha256 of the artifact bytes — the pin baked into install.sh."""
    return hashlib.sha256(data).hexdigest()


def release_message(name: str, sha256hex: str) -> bytes:
    """The exact bytes that get signed / verified (kept in lockstep with verify_release.ts).

    Domain-separated and binds the artifact ROLE (basename) to its content hash, so a valid
    signature for `worker.ts` can never be presented as the signature for a different file.
    """
    return f"{MSG_DOMAIN}\n{name}\n{sha256hex}".encode()


def pubkey_b64(sk: Ed25519PrivateKey) -> str:
    """Raw 32-byte Ed25519 public key, base64 — the same encoding worker_torch.py:51 registers."""
    return b64e(sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def sign_artifact(sk: Ed25519PrivateKey, path: str, name: str | None = None) -> tuple[str, str]:
    """Return (sha256_hex, signature_b64) for one artifact.

    signature = Ed25519(sk, release_message(basename, sha256_hex)).
    """
    with open(path, "rb") as f:
        data = f.read()
    sha = sha256_hex(data)
    name = name if name is not None else os.path.basename(path)
    sig = b64e(sk.sign(release_message(name, sha)))
    return sha, sig


# ---------------------------------------------------------------- ADR-0103: signed MANIFEST.sha256 (torch worker tree)
MANIFEST_NAME = "MANIFEST.sha256"
MANIFEST_DIRS = ("moregpu_worker",)       # covered recursively
MANIFEST_FILES = ("vision_ops.json",)     # covered top-level files (the WGSL executor contract the lowering reads)


def _ignored(rel: str) -> bool:
    return "__pycache__" in rel.split("/") or rel.endswith(".pyc")


def manifest_files(root: str) -> list[str]:
    out = []
    for d in MANIFEST_DIRS:
        for dp, dns, fns in os.walk(os.path.join(root, d)):
            dns[:] = sorted(x for x in dns if x != "__pycache__")
            for fn in fns:
                rel = os.path.relpath(os.path.join(dp, fn), root).replace(os.sep, "/")
                if not _ignored(rel):
                    out.append(rel)
    out += [f for f in MANIFEST_FILES if os.path.isfile(os.path.join(root, f))]
    return sorted(out)


def build_manifest(root: str) -> str:
    """`<sha256>  <path>` lines (sha256sum format), sorted by path, relative to root."""
    lines = []
    for rel in manifest_files(root):
        with open(os.path.join(root, rel), "rb") as f:
            lines.append(f"{sha256_hex(f.read())}  {rel}")
    return "\n".join(lines) + "\n"


def verify_manifest(root: str, manifest: bytes, sig_b64: str, pubkey: str) -> tuple[int, list[str]]:
    """(0, []) when the manifest's signature verifies AND every covered file matches; (4, why) bad signature;
    (5, problems) tampered / missing / unlisted files."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        Ed25519PublicKey.from_public_bytes(b64d(pubkey)).verify(b64d(sig_b64), release_message(MANIFEST_NAME, sha256_hex(manifest)))
    except (InvalidSignature, ValueError) as e:
        return 4, [f"signature does not verify against the release key ({type(e).__name__})"]
    listed, bad = {}, []
    for line in manifest.decode().splitlines():
        if not line:
            continue
        h, sep, rel = line.partition("  ")
        if not sep or len(h) != 64 or rel.startswith("/") or ".." in rel.split("/"):
            bad.append(f"malformed line: {line[:80]}")
            continue
        listed[rel] = h
    for rel, want in listed.items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            bad.append(f"missing {rel}")
            continue
        with open(p, "rb") as f:
            if sha256_hex(f.read()) != want:
                bad.append(f"tampered {rel}")
    for top in sorted({r.split("/", 1)[0] for r in listed if "/" in r}):
        for dp, _dns, fns in os.walk(os.path.join(root, top)):
            for fn in fns:
                rel = os.path.relpath(os.path.join(dp, fn), root).replace(os.sep, "/")
                if not _ignored(rel) and rel not in listed:
                    bad.append(f"unlisted {rel}")
    return (5, bad) if bad else (0, [])


def load_private_key(path: str) -> Ed25519PrivateKey:
    """Load a raw (base64) 32-byte Ed25519 seed written by `keygen`."""
    with open(path) as f:
        raw = b64d(f.read().strip())
    return Ed25519PrivateKey.from_private_bytes(raw)


def _cmd_keygen(args: argparse.Namespace) -> int:
    sk = Ed25519PrivateKey.generate()
    seed = sk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    if os.path.exists(args.out) and not args.force:
        print(f"refusing to overwrite existing key at {args.out} (pass --force)", file=sys.stderr)
        return 2
    # Write 0600 so the private seed is not world-readable.
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as f:
        f.write(b64e(seed) + "\n")
    pub = pubkey_b64(sk)
    print(f"wrote PRIVATE key (raw seed, base64) -> {args.out}  [mode 0600 — keep OUT of the repo]")
    print("")
    print("Paste this PUBLIC pin into scripts/install.sh:")
    print(f'    RELEASE_PUBKEY_B64="{pub}"')
    return 0


def _cmd_sign(args: argparse.Namespace) -> int:
    sk = load_private_key(args.key)
    pub = pubkey_b64(sk)
    print(f"# signer public key (Ed25519, raw, base64): {pub}")
    print(f'RELEASE_PUBKEY_B64="{pub}"')
    for path in args.artifacts:
        name = os.path.basename(path)
        sha, sig = sign_artifact(sk, path, name)
        sig_path = path + ".sig"
        with open(sig_path, "w") as f:
            f.write(sig + "\n")
        pin_var = name.replace(".", "_").replace("-", "_").upper() + "_SHA256"
        print(f"# {name}: sha256={sha}  sig-> {sig_path}")
        print(f'{pin_var}="{sha}"')
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    sk = load_private_key(args.key)
    text = build_manifest(args.root).encode()
    out = args.out or os.path.join(args.root, MANIFEST_NAME)
    with open(out, "wb") as f:
        f.write(text)
    sig = b64e(sk.sign(release_message(MANIFEST_NAME, sha256_hex(text))))
    with open(out + ".sig", "w") as f:
        f.write(sig + "\n")
    print(f'RELEASE_PUBKEY_B64="{pubkey_b64(sk)}"')
    n = len(text.splitlines())
    print(f"# {MANIFEST_NAME}: {n} files, sha256={sha256_hex(text)}  sig-> {out}.sig")
    return 0


def _cmd_verify_manifest(args: argparse.Namespace) -> int:
    man = args.manifest or os.path.join(args.root, MANIFEST_NAME)
    sig = args.sig or man + ".sig"
    try:
        with open(man, "rb") as f:
            data = f.read()
        with open(sig) as f:
            sig_b64 = f.read().strip()
    except OSError as e:
        print(f"[verify] REJECT {MANIFEST_NAME}: {e}", file=sys.stderr)
        return 4
    code, problems = verify_manifest(args.root, data, sig_b64, args.pubkey)
    if code:
        print(f"[verify] REJECT {MANIFEST_NAME}: " + "; ".join(problems[:20]), file=sys.stderr)
        return code
    print(f"[verify] OK {MANIFEST_NAME}: signed by the release key · every covered file matches")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sign MoreGPU worker artifacts for a pinned release.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    kg = sub.add_parser("keygen", help="generate a release signing keypair (private key stays out of the repo)")
    kg.add_argument("--out", required=True, help="path to write the PRIVATE key (raw seed, base64, mode 0600)")
    kg.add_argument("--force", action="store_true", help="overwrite an existing key file")
    kg.set_defaults(fn=_cmd_keygen)

    sg = sub.add_parser("sign", help="sign one or more artifacts; writes <artifact>.sig, prints the pins")
    sg.add_argument("--key", required=True, help="path to the release PRIVATE key from `keygen`")
    sg.add_argument("artifacts", nargs="+", help="artifact files to sign (e.g. apps/worker/worker.ts)")
    sg.set_defaults(fn=_cmd_sign)

    mf = sub.add_parser("manifest", help="write + sign MANIFEST.sha256 of the torch worker tree (ADR-0103)")
    mf.add_argument("--key", required=True, help="path to the release PRIVATE key from `keygen`")
    mf.add_argument("--root", default="apps/worker", help="worker root holding moregpu_worker/ (default apps/worker)")
    mf.add_argument("--out", help="manifest path (default <root>/MANIFEST.sha256; the .sig goes next to it)")
    mf.set_defaults(fn=_cmd_manifest)

    vm = sub.add_parser("verify-manifest", help="verify MANIFEST.sha256's signature + every covered file (exit 0/4/5)")
    vm.add_argument("--root", default="apps/worker")
    vm.add_argument("--pubkey", required=True, help="release PUBLIC key (raw, base64)")
    vm.add_argument("--manifest")
    vm.add_argument("--sig")
    vm.set_defaults(fn=_cmd_verify_manifest)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
