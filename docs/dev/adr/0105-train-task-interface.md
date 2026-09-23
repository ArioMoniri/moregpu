# ADR-0105 — `TrainTask` interface and registry

**Status:** Accepted (2026-09-23) · **Milestone:** M1

## Decision
```python
class TrainTask(Protocol):
    name: str
    def init(self, cfg: dict, ctx: TaskContext) -> TaskInfo            # build model/opt; ctx = device, amp, data, seed
    def inner_steps(self, batch_refs: list[Ref], steps: int, lr: float) -> StepReport  # losses, samples, timings
    def state_for_sync(self) -> dict[str, Tensor]                      # tensors the outer loop averages
    def load_sync_state(self, tensors: dict[str, Tensor]) -> None
    def after_outer_step(self, round: int) -> None                     # e.g. JEPA EMA (ADR-0109); default no-op
    def export(self, fmt: str) -> ExportResult                         # safetensors | torch_export | onnx
    def evaluate(self, refs: list[Ref], kind: str) -> dict             # task metrics, collapse monitors
```
- Registry: built-ins (`llm_lora`, `ijepa_2d`, `jepa_2p5d`, `jepa_3d`, `classify`, `segment`) + entry-point group
  `moregpu.train_tasks`, loaded **only** if the distribution name+version+wheel sha256 is in the worker allowlist
  (same mechanism as model plugins, ADR-0113). The coordinator can name a task; it can never ship code.
- `llm_lora` wraps today's `train_load/step/inner/set_adapter/generate` unchanged; `state_for_sync` returns the LoRA A/B
  tensors with identical names, so old and new DiLoCo paths are bit-identical (golden test).
- The inner optimizer is re-created every round (current DiLoCo semantics) unless the task sets `keep_inner_state`.

## Consequences
Token-id batches for `llm_lora` stay inline (not refs) to preserve the existing API; vision tasks use refs.
