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

    # each release: sign the artifacts the installer pins
    python3 scripts/release_sign.py sign --key /secure/moregpu_release_key.b64 \\
        apps/worker/worker.ts apps/worker/worker_torch.py
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

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
