# Proposed ADRs — vision, training-task framework, JEPA

**Status: Accepted by the owner on 2026-09-23.**

These are numbered **0101+** so they cannot collide with the maintainer's local, git-ignored `docs/adr/` series (code
already references `ADR-0007`). If approved they can move to `docs/adr/` (which would need that path un-ignored) or stay
here.

| ADR | Title | Milestone | Decision in one line |
|---|---|---|---|
| [0101](0101-research-boundary.md) | Research boundary guard | M1 | CI test fails on study/dataset terms, copyleft headers, medical-image files, patient-ID patterns outside `docs/case-studies/`. |
| [0102](0102-test-layers.md) | Test layers for strict TDD | M1 | Add pytest for new worker modules (+`gpu`/`webgpu` markers, ≥90% coverage), vitest for new coordinator modules; keep existing standalone scripts. |
| [0103](0103-code-layout-and-release.md) | Code layout + signed release | M1 | New code in `apps/worker/moregpu_worker/` and `apps/coordinator/lib/`; release signs a hash manifest of the tree. |
| [0104](0104-training-sessions.md) | Per-session training state | M1 | Replace the global `TRAIN` slot / `trainingHome` / `diloco` with sessions keyed by id; legacy routes = session `default`. |
| [0105](0105-train-task-interface.md) | `TrainTask` interface + registry | M1 | Python protocol + built-ins + pinned entry points; `llm_lora` is the migrated existing path. |
| [0106](0106-generalised-diloco.md) | Generalised DiLoCo + tensor transport | M1 | Coordinator stays parameter server; sample-weighted mean; binary chunked sealed tensor frames; fp32 outer state; checkpoint/resume. |
| [0107](0107-heterogeneous-diloco-and-sharding.md) | Heterogeneous DiLoCo + deterministic shards | M1 | Per-worker H from measured speed; seeded global sample stream partitioned by refs; target-samples stopping. |
| [0108](0108-amp-policy.md) | AMP policy | M1 | bf16 if supported, else fp16+GradScaler, else fp32; overridable, recorded. |
| [0109](0109-jepa-diloco-ema.md) | JEPA on DiLoCo: EMA semantics | M2 | Sync online encoder+predictor; target EMA updated identically after each outer step; in-house timm-key-compatible ViT. |
| [0110](0110-vision-data-plane.md) | Vision data plane | M2 | Refs not pixels; allowlisted `file://`/`https://`/public buckets/`pushed://`; CAS cache; memmap shards. |
| [0111](0111-telemetry.md) | Telemetry schema + `moregpu bench` | M2 | Versioned JSONL written by coordinator from worker-reported timings; bench profiles via VRAM fraction / cgroups / netem (opt-in). |
| [0112](0112-vision-tasks-and-routes.md) | `segment`/`classify` tasks + `/vision/*` | M3/M5 | Tasks on the same framework; MONAI-equivalent sliding window; work-stealing scheduler for batch inference. |
| [0113](0113-model-formats.md) | Model formats, plugins, `weights_only` ban | M4 | Adapter chain state_dict/safetensors → pinned plugins → torch.export/TorchScript → ONNX; pickled full models refused. |
| [0114](0114-lowering-and-webgpu-vision.md) | Automatic lowering + WebGPU vision | M4/M6 | Deferred; torch.export → op-graph or ONNX-web, parity-probed, cached; native fallback with reason. |
