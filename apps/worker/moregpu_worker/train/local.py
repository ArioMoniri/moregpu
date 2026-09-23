"""In-process reference runners (no coordinator): plain training and DiLoCo with the exact same semantics as the
coordinator path (weighted average, outer Nesterov, deterministic sample stream, samples-seen accounting).
Used by tests and by study controls that need a single-process baseline."""
from __future__ import annotations

from . import registry
from .diloco import OuterState, outer_step, weighted_average
from .schedule import lr_at, progress
from .sharding import SampleStream, allocate, split
from .task import TaskContext


def _make(task: str, cfg: dict, seed: int, device: str, amp: str, keep: bool = True, wid: int = 0):
    t = registry.create(task)
    t.init({**cfg, "keep_inner_state": cfg.get("keep_inner_state", keep)}, TaskContext(device=device, amp=amp, seed=seed, session=f"local-{wid}"))
    return t


def run_plain(task: str, cfg: dict, steps: int, batch: int, lr: float, seed: int, manifest_len: int,
              device: str = "cpu", amp: str = "fp32", lr_schedule: dict | None = None,
              target_samples: int | None = None) -> dict:
    t = _make(task, cfg, seed, device, amp, keep=True)
    stream = SampleStream(manifest_len, seed)
    losses = []
    sch = {"lr": lr, "lr_schedule": lr_schedule, "target_samples": target_samples, "max_rounds": steps}
    for s in range(steps):
        p_mid = progress(sch, s * batch, batch, s)
        losses += t.inner_steps(stream.take(batch), 1, lr_at(sch, p_mid)).losses
        # per-step hook (I-JEPA EMA every step) with the same global-progress info the coordinator sends
        t.after_outer_step(s + 1, {"progress": p_mid, "h": 1.0})
    return {"state": t.state_for_sync(), "losses": losses, "samples_seen": steps * batch, "task": t}


def run_diloco(task: str, cfg: dict, n_workers: int, rounds: int, inner_steps: int, batch: int, lr: float,
               outer_lr: float, outer_momentum: float, seed: int, manifest_len: int, device: str = "cpu",
               amp: str = "fp32", keep_inner_state: bool = True, alloc_mode: str = "fixed",
               speeds: list[float] | None = None, target_samples: int | None = None,
               lr_schedule: dict | None = None) -> dict:
    workers = [_make(task, cfg, seed, device, amp, keep_inner_state, i) for i in range(n_workers)]
    st = OuterState.init(workers[0].state_for_sync())
    for w in workers[1:]:
        w.load_sync_state(st.global_)
    stream = SampleStream(manifest_len, seed)
    seen, losses = 0, []
    for _ in range(rounds):
        remaining = None if target_samples is None else target_samples - seen
        if remaining is not None and remaining <= 0:
            break
        sizes = allocate(inner_steps * batch, n_workers, speeds or [1.0] * n_workers, alloc_mode, remaining)
        sch = {"lr": lr, "lr_schedule": lr_schedule, "target_samples": target_samples, "max_rounds": rounds}
        p_mid = progress(sch, seen, sum(sizes), st.round)
        lr_r = lr_at(sch, p_mid)
        h = sum(sizes) / float(batch * n_workers)          # global optimizer steps this round (matched global batch)
        shards = split(stream.take(sum(sizes)), sizes)
        results = []
        for w, idx in zip(workers, shards):
            if not idx:
                continue
            steps = max(1, int(len(idx) / batch + 0.5))      # half-up, identical to the coordinator's Math.round
            rep = w.inner_steps(idx, min(steps, len(idx)), lr_r)
            results.append((w.state_for_sync(), rep.samples))
            losses.append(rep.losses[-1])
        seen += sum(s for _, s in results)
        outer_step(st, weighted_average(results), outer_lr, outer_momentum)
        for w in workers:
            w.load_sync_state(st.global_)
            w.after_outer_step(st.round, {"progress": p_mid, "h": h})
    return {"state": st.global_, "losses": losses, "samples_seen": seen, "rounds": st.round, "workers": workers}
