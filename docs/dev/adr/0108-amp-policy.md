# ADR-0108 — AMP policy

**Status:** Proposed · **Milestone:** M1

## Decision
`amp: auto | bf16 | fp16 | fp32` per session. `auto`: CUDA with `torch.cuda.is_bf16_supported()` → bf16 autocast (no
scaler); other CUDA (e.g. Turing) → fp16 autocast + `GradScaler`; MPS/CPU → fp32. Master weights stay fp32. The
effective mode, scaler scale and skipped-step count are reported per round. A forced mode that the device cannot run is
an error, not a silent downgrade. `llm_lora` keeps fp32 unless asked (no behaviour change).
