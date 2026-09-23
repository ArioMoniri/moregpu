"""llm_lora — the pre-existing on-pool LoRA fine-tuning of an HF causal LM, as a TrainTask (ADR-0105).

Behaviour is intentionally identical to the original worker_torch.py code path (hand-written LoRA, fp32 base,
AdamW, fresh inner optimizer per DiLoCo round, adapter tensor names `<module>.A` / `<module>.B`)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..task import StepReport, TaskContext, Timer, TrainTask


class LoRAWrap(nn.Module):
    """Wrap a frozen Linear/Conv1D so y = base(x) + (x·Aᵀ)·Bᵀ·scale. B starts at 0 → the adapter is a
    no-op at step 0 (loss then equals the base model's), which makes the run reproducible against a
    seeded reference."""
    def __init__(self, base: nn.Module, in_f: int, out_f: int, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, in_f) * (1.0 / r))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def _in_out(mod: nn.Module):
    if isinstance(mod, nn.Linear):
        return mod.in_features, mod.out_features
    if hasattr(mod, "nf") and hasattr(mod, "weight"):  # transformers Conv1D (GPT-2): weight [in, out=nf]
        return mod.weight.shape[0], int(mod.nf)
    return None


def attach_lora(model: nn.Module, targets: list[str], r: int, alpha: float, dev: str = "cpu") -> int:
    """Freeze the base model, replace each target module (matched by name suffix) with a LoRAWrap.
    Returns the count of trainable adapter parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    hits = []
    for name, mod in model.named_modules():
        if any(name.split(".")[-1] == t for t in targets):
            io = _in_out(mod)
            if io:
                hits.append((name, mod, io))
    for name, mod, (in_f, out_f) in hits:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        parent.add_module(name.split(".")[-1], LoRAWrap(mod, in_f, out_f, r, alpha).to(dev))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LlmLoraTask(TrainTask):
    name = "llm_lora"

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.trainable: dict[str, nn.Parameter] = {}

    def init(self, cfg: dict, ctx: TaskContext) -> dict:
        """cfg: model (hub id) or `model_dir` (pushed/staged dir), rank, alpha, lr, seed, targets, no_dropout.
        `model_obj` may carry an already-built nn.Module (used by the worker's push path and by tests)."""
        from transformers import AutoModelForCausalLM
        self.ctx = ctx
        self.amp = None
        torch.manual_seed(int(cfg.get("seed", ctx.seed)))
        self.model = None; self.opt = None; self.trainable = {}
        if cfg.get("model_obj") is not None:
            model = cfg["model_obj"].to(ctx.device)
        elif cfg.get("model_dir"):
            model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], dtype=torch.float32,
                                                         local_files_only=True).to(ctx.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(ctx.device)
        if cfg.get("no_dropout"):
            for mod in model.modules():
                if isinstance(mod, nn.Dropout):
                    mod.p = 0.0
        targets = cfg.get("targets") or ["c_attn", "q_proj", "v_proj"]
        n_train = attach_lora(model, targets, int(cfg.get("rank", 8)), float(cfg.get("alpha", 16)), dev=ctx.device)
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-3)))
        self.model, self.step = model, 0
        self.trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        return {"ok": True, "trainable_params": n_train, "targets": targets, "device": ctx.device}

    def train_step(self, ids: torch.Tensor, labels: torch.Tensor, lr: float | None = None) -> float:
        if lr:
            for g in self.opt.param_groups:
                g["lr"] = float(lr)
        self.model.train()
        self.opt.zero_grad()
        loss = self.model(input_ids=ids, labels=labels).loss
        loss.backward()
        self.opt.step()
        self.step += 1
        return float(loss.item())

    def inner_steps(self, batch_refs: list, steps: int, lr: float) -> StepReport:
        """batch_refs: token-id windows (this worker's shard). Fresh AdamW per round (DiLoCo)."""
        dev = self.ctx.device
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.model.train()
        tm, losses = Timer(), []
        for i in range(steps):
            b = batch_refs[i % len(batch_refs)]
            ids = torch.tensor(b, dtype=torch.long, device=dev).unsqueeze(0)
            with tm.span("compute_s"):
                opt.zero_grad()
                loss = self.model(input_ids=ids, labels=ids).loss
                loss.backward(); opt.step()
            losses.append(float(loss.item()))
        self.step += steps
        return StepReport(losses, sum(len(b) for b in batch_refs[:steps]) if steps <= len(batch_refs) else steps, tm.t)

    def state_for_sync(self):
        return {n: p.detach() for n, p in self.trainable.items()}

    def load_sync_state(self, tensors):
        with torch.no_grad():
            for n, p in self.trainable.items():
                t = tensors.get(n)
                if t is not None:
                    p.copy_(t.reshape(p.shape).to(p.device))

    def close(self) -> None:
        self.model = None; self.opt = None; self.trainable = {}
