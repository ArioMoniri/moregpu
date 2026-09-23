# ADR-0109 — JEPA on DiLoCo: EMA semantics and model

**Status:** Proposed · **Milestone:** M2

## Decision
- **Synced state:** online (context) encoder + predictor. **Target encoder is never transmitted.**
- **EMA timing:** the target is updated only in `after_outer_step`, from the new global:
  `target ← m_k·target + (1−m_k)·global`, identical on every worker. The per-outer-step momentum is derived from the usual
  per-step schedule as `m_k = Π_{j∈round k} m_j` (≈ `m^H`), so the effective averaging horizon in samples matches
  single-process I-JEPA. With N=1, H=1, η=1, μ=0 this is exactly per-step I-JEPA (equivalence test).
- Workers periodically report a hash of the target weights; divergence raises an alarm (it must never happen).
- **Model:** in-house ViT (Tiny/Small, patch 16, 2D; 2.5D = adjacent slices as channels; 3D = cubic patch tokens) with
  **timm-compatible state_dict keys** (tested against timm when installed) — no hard timm dependency. Predictor: narrow
  ViT (depth/width configurable). Multi-block masking per I-JEPA (configurable scales/aspect, seeded).
- **Loss:** smooth-L1 (default) or L2 on layer-normed target features.
- **Collapse monitors per round:** per-dim std of embeddings on a fixed probe batch, RankMe effective rank, loss;
  alarms when std < ε or rank < r_min.
- Gradient checkpointing and deterministic mode (`torch.use_deterministic_algorithms`) are flags.

## Consequences
The study can compare arms at matched samples without any EMA drift across workers.
