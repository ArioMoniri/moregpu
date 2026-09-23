# ADR-0112 — `segment`/`classify` tasks and `/vision/*`

**Status:** Accepted (2026-09-23) · **Milestones:** M3 (segment fine-tune + export), M5 (`/vision/batch`)

## Decision
- `classify` and `segment` (2D/3D) are `TrainTask`s: modes `full`, `frozen_encoder+head`, `lora` (linear/attn layers,
  reusing `LoRAWrap`); losses CE, Dice+CE; worker-side metrics (Dice, CE) with goldens vs MONAI.
- Encoder from JEPA export (safetensors) + light decoder (UNETR-lite / progressive upsampling) for segmentation.
- `/vision/load|infer|batch|jobs|unload`: sliding window with Gaussian blend + flip TTA, numerically equal to MONAI
  `sliding_window_inference` within tolerance; batch scheduler = pull-based work queue (work-stealing) with per-item
  retry on churn; heterogeneity handled by pull rate, not static split.
