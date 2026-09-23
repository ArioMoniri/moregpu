"""In-process reference runners (no coordinator): plain training and DiLoCo with the exact same semantics as the
coordinator path (weighted average, outer Nesterov, deterministic sample stream, samples-seen accounting).
Used by tests and by study controls that need a single-process baseline."""
from __future__ import annotations

from . import registry
from .diloco import OuterState, outer_step, weighted_average
from .sharding import SampleStream, allocate, split
from .task import TaskContext


def _make(task: str, cfg: dict, seed: int, device: str, amp: str, keep: bool = False, wid: int = 0):
    t = registry.create(task)
    t.init({**cfg, "keep_inner_state": keep}, TaskContext(device=device, amp=amp, seed=seed, session=f"local-{wid}"))
    return t


def run_plain(task: str, cfg: dict, steps: int, batch: int, lr: float, seed: int, manifest_len: int,
              device: str = "cpu", amp: str = "fp32") -> dict:
    t = _make(task, cfg, seed, device, amp, keep=True)
    stream = SampleStream(manifest_len, seed)
    losses = []
    for s in range(steps):
        losses += t.inner_steps(stream.take(batch), 1, lr).losses
        t.after_outer_step(s + 1)          # per-step hook (e.g. I-JEPA EMA every step) — DiLoCo N=1,H=1 equivalence
    return {"state": t.state_for_sync(), "losses": losses, "samples_seen": steps * batch, "task": t}


def run_diloco(task: str, cfg: dict, n_workers: int, rounds: int, inner_steps: int, batch: int, lr: float,
               outer_lr: float, outer_momentum: float, seed: int, manifest_len: int, device: str = "cpu",
               amp: str = "fp32", keep_inner_state: bool = False, alloc_mode: str = "fixed",
               speeds: list[float] | None = None, target_samples: int | None = None) -> dict:
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
        shards = split(stream.take(sum(sizes)), sizes)
        results = []
        for w, idx in zip(workers, shards):
            if not idx:
                continue
            steps = max(1, round(len(idx) / batch))
            rep = w.inner_steps(idx, min(steps, len(idx)), lr)
            results.append((w.state_for_sync(), rep.samples))
            losses.append(rep.losses[-1])
        seen += sum(s for _, s in results)
        outer_step(st, weighted_average(results), outer_lr, outer_momentum)
        for w in workers:
            w.load_sync_state(st.global_)
            w.after_outer_step(st.round)
    return {"state": st.global_, "losses": losses, "samples_seen": seen, "rounds": st.round, "workers": workers}
