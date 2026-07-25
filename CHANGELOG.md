# Changelog

All notable changes to MoreGPU are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-25

Initial release: a native GPU compute pool with a networked coordinator and
cross-platform workers.

### Added

- **Tested core libraries** — a TypeScript monorepo (npm workspaces) of covered
  libraries: crypto, protocol, scheduler, integrity, transport, runtime, and the
  GPU layer.
- **Real GPU execution** — WGSL kernels run on the physical GPU via WebGPU
  (Metal / Vulkan / D3D12), with a CPU reference fallback so CPU-only machines
  contribute compute too.
- **Native compute pool** — jobs are row-sharded across workers, results are
  pooled and verified. Built-in task types: matmul and vector_add, extensible by
  adding a WGSL kernel plus a CPU reference.
- **Networked coordinator + worker** — an admin runs the coordinator/admin
  server; worker machines join the pool over an outbound WebSocket connection.
  Dashboard on `http://HOST:8787`; `wss` supported via TLS cert/key env vars.
- **First-run token wizard** — the coordinator's first run generates that pool's
  own admin token, worker join token, and encryption key, so every pool is
  isolated and nobody shares another pool's tokens.
- **Sealed jobs** — each work unit is AES-GCM sealed; only ciphertext travels on
  the wire between coordinator and workers.
- **CPU duty-cycle throttle** — CPU workers contribute with a configurable duty
  cycle (`MOREGPU_THROTTLE`, `MOREGPU_DUTY`) so the interactive user is not
  disturbed and power draw stays low.
- **One-liner installers + services** — cross-OS worker install via
  `scripts/install.sh` (Linux/macOS) and `scripts/install.ps1` (Windows). The
  installer self-heals, and `MOREGPU_SERVICE=1` installs a reboot-surviving,
  self-healing service (systemd / launchd / Windows scheduled task).

[Unreleased]: https://github.com/ArioMoniri/moregpu/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ArioMoniri/moregpu/releases/tag/v0.1.0
