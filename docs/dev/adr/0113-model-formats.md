# ADR-0113 — Model formats, plugins, and the `weights_only` ban

**Status:** Accepted (2026-09-23) · **Milestone:** M4

## Decision
- Adapter chain on native workers: (1) state_dict `.pth/.pt` via `torch.load(weights_only=True)` or safetensors + named
  architecture from registries (torchvision, timm, monai.networks.nets, HF vision) with `strict=True` and a reported key
  diff; (2) admin-installed plugins (`moregpu.models` entry points) pinned by name+version+wheel sha256 in a worker
  allowlist file — never code over the wire; (3) `torch.export` `.pt2` / TorchScript; (4) ONNX via onnxruntime.
  Pickled full models refused with a helpful message.
- Model spec JSON v1 (`format, source, arch, sha256, dtype, io, preprocess, inference, postprocess, placement, licence,
  citation`), validated by JSON Schema; sha256 mandatory for `https://` and `pushed://`.
- Repo-wide test bans `weights_only=False`, bare `pickle.load(s)`, `trust_remote_code=True`, and `from_pretrained` without
  `use_safetensors=True` in worker code (existing calls fixed in a separate, test-first PR).
