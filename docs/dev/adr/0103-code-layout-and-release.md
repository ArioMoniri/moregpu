# ADR-0103 — Code layout and signed release

**Status:** Accepted (2026-09-23) · **Milestone:** M1

## Context
`worker_torch.py` (1.9k lines) and `server.ts` (3.2k lines) are single-file monoliths; releases sign those single files
(`*.sig`, `tests/security/release_verify.py`); the Dockerfile copies only `server.ts`.

## Decision
- **Worker:** new importable package `apps/worker/moregpu_worker/` with sub-packages `train/` (framework, tasks,
  diloco client), `vision/` (adapters, inference), `data/` (refs, readers, cache, loader), `telemetry/`.
  `worker_torch.py` stays the entry point and dispatches new ops into the package; existing code is moved only when a
  test covers it.
- **Packaging:** `apps/worker/pyproject.toml` (`moregpu-worker`) with extras `train`, `vision` (nibabel, pydicom,
  SimpleITK, pillow, tifffile), `onnx`, `monai`, `timm`; installer honours `MOREGPU_EXTRAS=vision,...`.
- **Coordinator:** new logic in `apps/coordinator/lib/*.ts` (pure functions: averaging, outer step, sharding plan,
  telemetry writer) imported by `server.ts`; route glue stays in `server.ts`. Dockerfile copies `apps/coordinator/`.
- **Release signing:** extend `release_sign.py` to sign a `MANIFEST.sha256` (sorted path → sha256) for the worker tree;
  the verifier checks the manifest signature then every file hash. Single-file `.sig` kept for backward compatibility for
  one minor version. `release_verify.py` gets negative tests for a tampered/added/missing package file.

## Consequences
One-file curl-install of the torch worker becomes "fetch tarball + verify manifest". Needs security-engineer sign-off.
