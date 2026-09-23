"""Per-session training state (ADR-0104): replaces the worker's single global TRAIN slot."""
from __future__ import annotations

import os

from . import registry
from .task import TaskContext, TrainTask


class SessionLimit(RuntimeError):
    pass


def default_limit(device: str) -> int:
    env = os.environ.get("MOREGPU_MAX_TRAIN_SESSIONS")
    if env:
        return max(1, int(env))
    return 2 if device == "cpu" else 1


class SessionStore:
    def __init__(self, max_sessions: int = 1):
        self.max = max_sessions
        self._s: dict[str, TrainTask] = {}

    def create(self, sid: str, task: str, cfg: dict, ctx: TaskContext, replace: bool = False) -> TrainTask:
        if sid in self._s:
            if not replace:
                raise ValueError(f"training session {sid!r} already exists")
            self.close(sid)
        if len(self._s) >= self.max:
            raise SessionLimit(f"worker holds {len(self._s)}/{self.max} training sessions; close one first "
                               f"(MOREGPU_MAX_TRAIN_SESSIONS)")
        t = registry.create(task)
        t.init(cfg, ctx)
        self._s[sid] = t
        return t

    def put(self, sid: str, t: TrainTask) -> None:
        if sid not in self._s and len(self._s) >= self.max:
            raise SessionLimit(f"worker holds {len(self._s)}/{self.max} training sessions")
        self._s[sid] = t

    def get(self, sid: str) -> TrainTask:
        if sid not in self._s:
            raise KeyError(f"no training session {sid!r} on this worker")
        return self._s[sid]

    def maybe(self, sid: str) -> TrainTask | None:
        return self._s.get(sid)

    def close(self, sid: str) -> bool:
        t = self._s.pop(sid, None)
        if t is not None:
            t.close()
        return t is not None

    def ids(self) -> list[str]:
        return list(self._s)
