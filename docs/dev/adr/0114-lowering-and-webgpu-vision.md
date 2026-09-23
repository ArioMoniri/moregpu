# ADR-0114 — Automatic lowering and WebGPU vision

**Status:** Accepted (2026-09-23) (design only; implementation after the study) · **Milestones:** M4 (hooks), M6 (kernels)

## Decision
Native model → `torch.export` graph → MoreGPU op-graph + safetensors when every op is in the WGSL executor's table; else
in-memory ONNX → onnxruntime-web WebGPU EP; else native-only with the unsupported-op list reported. Lowered artefacts are
cached by (model sha256, lowering version, target) and must pass a parity probe vs the native forward before serving.
WGSL additions (conv2d/3d implicit GEMM, transposed conv, norms, pooling, upsample, concat, channel softmax/sigmoid,
argmax, fp16 storage, memory planner) each land with PyTorch goldens run under Deno WebGPU and Playwright (skipped where
WebGPU is unavailable). WebGPU scope is inference and JEPA feature extraction only; training stays native.
