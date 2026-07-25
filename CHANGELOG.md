# Changelog

All notable changes to MoreGPU are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Contribution scheduling** — a worker sets `MOREGPU_SCHEDULE` (`always` ·
  `idle-only` · `HH:MM-HH:MM` active window, may wrap midnight) to control *when*
  its machine is lent. Outside the window it takes no new work; in-flight shards
  always finish.
- **Remote fleet control** — `POST /workers/:id/control` lets an admin pause,
  resume, cap the duty ceiling, reschedule, relabel, or remove any worker; the
  coordinator pushes a control frame to that worker. Exposed in the dashboard
  (per-row controls, search, "pause/resume all") and the CLI (`moregpu pause /
  resume / set / schedule / rm / workers`).
- **High-worker-count admin UI** — the fleet table now searches, sorts by
  contribution, and caps the rendered rows (with a "showing N of M" count) so the
  dashboard stays responsive with many machines. No cap on how many can join.
- **Linux hardware isolation** — [`scripts/isolate-linux.sh`](scripts/isolate-linux.sh)
  (and `moregpu isolate`) pin a worker to a bounded cgroups-v2 scope (CPU quota,
  cpuset, memory cap, idle I/O), degrading gracefully when systemd/cgroups v2 are
  absent.
- **CLI & admin-UI ASCII banners** and richer `--help`.
- **`examples/verify_workloads.py`** — replays real GPU-user workloads (Linear,
  MLP, LayerNorm, softmax head, single-head attention) against a live pool.

### Changed

- Documented the honest execution split (matmul on the GPU; memory-bound
  elementwise/row-wise kernels on the worker CPU) and the real cryptography model
  (AES-256-GCM sealing, Ed25519 result signatures, single-trust-domain limits) in
  `SECURITY.md`, plus a hardware-grounded confidential-computing/TEE roadmap.
- Client SDKs distributed as GitHub Release artifacts (wheel + npm tarball);
  publishing to PyPI/npm/Homebrew documented in `CONTRIBUTING.md`.

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
