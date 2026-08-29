# `packages/*` — reference library, NOT the running system

> [!IMPORTANT]
> The **production** MoreGPU pool is three self-contained files, and they import **nothing** from `packages/*`:
> - `apps/coordinator/server.ts` — the Deno coordinator (HTTP API, sharding, MoE, training, sealed protocol)
> - `apps/worker/worker.ts` — the Deno / WebGPU worker (WGSL runtime, tokenizer, kernels)
> - `apps/worker/worker_torch.py` — the opt-in native torch worker
>
> The CLI (`scripts/moregpu`) launches those files directly (`deno run … server.ts` / `worker.ts`). Grep the apps for `@moregpu/` — there are no imports.

## What this directory is

`packages/*` is an **experimental / reference** decomposition of the same ideas into a typed library layer — `crypto` (sealing/HKDF), `gpu` (WGSL kernels + a CPU reference), `transport`, `scheduler`, `protocol`, `integrity` (quorum/canary/reputation), `runtime` (governor), `coordinator` (orchestrator), and `client`. Each has its own unit tests, and `npm test` runs them in CI.

## Why it is not the production path (and the caveats that follow)

- **It is not wired in.** The live coordinator/worker re-implement throttling, sharding, and integrity inline; the packages are referenced only by their own tests. Fixing a bug here has **no effect** on a deployed pool.
- **The wire protocol differs.** `packages/protocol` defines an `Envelope { v, id, type, payload }` with a `MessageType` enum; the live wire is a flat `{ t: 'register' | 'welcome' | 'assign' | 'result' | … }`. A worker built on `@moregpu/transport` + `@moregpu/protocol` is **wire-incompatible** with the running `server.ts` and will be closed on timeout.
- **The tests validate this shadow layer, not production.** When reviewing "the distributed logic," read `apps/` — that is what runs.

If you want to *use* MoreGPU, drive `apps/`. If you want to *evolve the architecture*, either adopt these packages inside `apps/` (replacing the inline implementations) or treat them as a design sketch — but do not mistake them for the shipping system.
