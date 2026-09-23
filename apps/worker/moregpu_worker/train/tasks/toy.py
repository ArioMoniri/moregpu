"""toy_linear — a tiny deterministic regression task used to test the framework and DiLoCo semantics."""
from __future__ import annotations

import torch

from ..task import StepReport, TaskContext, Timer, TrainTask


class ToyLinearTask(TrainTask):
    name = "toy_linear"

    def init(self, cfg: dict, ctx: TaskContext) -> dict:
        self.setup(ctx)
        n, dim = int(cfg.get("n", 64)), int(cfg.get("dim", 5))
        self.batch = int(cfg.get("batch", 4))
        self.kind = cfg.get("optimizer", "sgd")
        self.keep_inner_state = bool(cfg.get("keep_inner_state", True))
        g = torch.Generator().manual_seed(1000 + int(cfg.get("data_seed", 0)))
        w_true = torch.randn(dim, 1, generator=g)
        self.X = torch.randn(n, dim, generator=g)
        self.Y = self.X @ w_true + 0.01 * torch.randn(n, 1, generator=g)
        torch.manual_seed(int(cfg.get("init_seed", 0)))
        self.model = torch.nn.Linear(dim, 1).to(ctx.device)
        return {"ok": True, "params": sum(p.numel() for p in self.model.parameters())}

    def _loss(self, idx):
        x, y = self.X[idx].to(self.ctx.device), self.Y[idx].to(self.ctx.device)
        return torch.nn.functional.mse_loss(self.model(x), y)

    def inner_steps(self, batch_refs: list, steps: int, lr: float) -> StepReport:
        if not batch_refs:
            raise ValueError("no samples for this worker")
        opt = self.inner_optimizer(self.model.parameters(), lr, self.kind)
        tm, losses = Timer(), []
        per = max(1, len(batch_refs) // steps)
        for s in range(steps):
            idx = batch_refs[s * per:(s + 1) * per] if s < steps - 1 else batch_refs[s * per:]
            idx = idx or batch_refs[-per:]
            with tm.span("compute_s"):
                opt.zero_grad()
                with self.amp.autocast():
                    loss = self._loss([i % len(self.X) for i in idx])
                self.amp.backward_step(loss, opt)
            losses.append(float(loss.detach()))
            self.step += 1
        return StepReport(losses, len(batch_refs), tm.t)

    def state_for_sync(self):
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def load_sync_state(self, tensors):
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                v.copy_(tensors[k].reshape(v.shape).to(v.device, v.dtype))

    def evaluate(self, refs: list, kind: str) -> dict:
        with torch.no_grad():
            return {"loss": float(self._loss([i % len(self.X) for i in refs]))}
