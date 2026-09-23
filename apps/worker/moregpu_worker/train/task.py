"""TrainTask interface (ADR-0105). A task owns its model, inner optimizer, data access and metrics; the DiLoCo
outer loop (coordinator or moregpu_worker.train.local) only sees `state_for_sync` / `load_sync_state`."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from . import amp as amp_mod


@dataclass
class TaskContext:
    device: str = "cpu"
    amp: str = "fp32"                 # requested AMP mode (resolved in TrainTask.setup_amp)
    seed: int = 0
    session: str = "default"
    data: object | None = None        # moregpu_worker.data.DataPlane (optional)
    deterministic: bool = False


@dataclass
class StepReport:
    losses: list[float]
    samples: int
    timings: dict = field(default_factory=lambda: {"compute_s": 0.0, "data_s": 0.0})
    metrics: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"losses": self.losses, "samples": self.samples, "timings": self.timings, "metrics": self.metrics}


class TrainTask(ABC):
    name: str = "abstract"
    keep_inner_state: bool = True     # DiLoCo keeps each worker's inner optimizer state across rounds (never synced)

    def __init__(self) -> None:
        self.ctx: TaskContext | None = None
        self.amp: amp_mod.AmpPolicy | None = None
        self.opt: torch.optim.Optimizer | None = None
        self.step = 0

    # -- lifecycle -------------------------------------------------------
    @abstractmethod
    def init(self, cfg: dict, ctx: TaskContext) -> dict: ...

    @abstractmethod
    def inner_steps(self, batch_refs: list, steps: int, lr: float) -> StepReport: ...

    @abstractmethod
    def state_for_sync(self) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def load_sync_state(self, tensors: dict[str, torch.Tensor]) -> None: ...

    def after_outer_step(self, round: int, info: dict | None = None) -> dict:
        """Called on every worker after the new global is loaded. `info` comes from the coordinator and is identical
        on all workers: {"progress": midpoint of this round in [0,1] or None, "h": global optimizer steps this round}."""
        return {}

    def extra_state(self) -> dict[str, torch.Tensor]:
        """Non-synced state that a checkpoint must carry (e.g. the JEPA EMA target and step counters)."""
        return {}

    def load_extra_state(self, tensors: dict[str, torch.Tensor]) -> None:
        return None

    def export(self, fmt: str, path: str) -> dict:
        raise NotImplementedError(f"{self.name} does not export {fmt}")

    def evaluate(self, refs: list, kind: str) -> dict:
        raise NotImplementedError(f"{self.name} has no evaluation {kind!r}")

    def describe(self) -> dict:
        return {"task": self.name, "step": self.step, "amp": self.amp.describe() if self.amp else None}

    def close(self) -> None:
        self.opt = None

    # -- helpers for subclasses -----------------------------------------
    def setup(self, ctx: TaskContext) -> None:
        self.ctx = ctx
        torch.manual_seed(ctx.seed)
        if ctx.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        self.amp = amp_mod.resolve(ctx.amp, ctx.device)

    def make_optimizer(self, params, lr: float, kind: str = "adamw", weight_decay: float = 0.0):
        params = [p for p in params if p.requires_grad]
        if kind == "adamw":
            # no weight decay on biases / norm scales / 1-D params (I-JEPA, timm convention)
            decay = [p for p in params if p.ndim >= 2]
            no_decay = [p for p in params if p.ndim < 2]
            groups = [g for g in ({"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}) if g["params"]]
            return torch.optim.AdamW(groups, lr=lr)
        if kind == "sgd":
            return torch.optim.SGD(params, lr=lr)
        raise ValueError(f"unknown optimizer {kind!r}")

    def inner_optimizer(self, params, lr: float, kind: str, weight_decay: float = 0.0):
        """DiLoCo semantics: fresh inner optimizer each round unless keep_inner_state."""
        if self.opt is None or not self.keep_inner_state:
            self.opt = self.make_optimizer(params, lr, kind, weight_decay)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return self.opt


class Timer:
    def __init__(self) -> None:
        self.t = {"compute_s": 0.0, "data_s": 0.0}

    def span(self, key: str):
        timer = self

        class _S:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()

            def __exit__(self_inner, *a):
                timer.t[key] = timer.t.get(key, 0.0) + time.perf_counter() - self_inner.t0
        return _S()
