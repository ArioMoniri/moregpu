"""Worker-side `task_*` train ops (ADR-0105/0106). The coordinator relays these as sealed `train` RPCs.

Sync payloads use moregpu_worker.train.tensorwire, carried as base64 chunks inside the sealed relay frames
(ADR-0106 amendment: v1 rides the existing sealed JSON relay; raw binary WS frames are a later optimisation)."""
from __future__ import annotations

import base64
import time

import torch

from . import registry, tensorwire as tw
from .sessions import SessionStore
from .task import TaskContext

DEFAULT_CHUNK = 4 << 20


class TaskRunner:
    def __init__(self, sessions: SessionStore, device: str, data_plane=None):
        self.sessions, self.device, self.data = sessions, device, data_plane
        self._out: dict[str, list[bytes]] = {}          # session -> encoded chunks awaiting pull
        self._in: dict[str, dict] = {}                   # session -> partial push {header, parts}
        self._global: dict[str, dict[str, torch.Tensor]] = {}  # last applied global (int8delta reference)

    # ------------------------------------------------------------------
    def handle(self, op: str, p: dict) -> dict:
        fn = getattr(self, "_" + op, None) if op.startswith("task_") else None
        if fn is None:
            raise ValueError(f"unknown task op {op!r}")
        return fn(p)

    def _task_list(self, p):
        return {"ok": True, "tasks": registry.available(), "sessions": self.sessions.ids()}

    def _task_init(self, p):
        sid = p["session"]
        ctx = TaskContext(device=self.device, amp=p.get("amp", "auto"), seed=int(p.get("seed", 0)), session=sid,
                          data=self.data, deterministic=bool(p.get("deterministic", False)))
        t = self.sessions.create(sid, p["task"], p.get("cfg", {}), ctx, replace=bool(p.get("replace", False)))
        self._global.pop(sid, None); self._out.pop(sid, None); self._in.pop(sid, None)
        return {"ok": True, "describe": t.describe(), "info": getattr(t, "info", None)}

    def _encode_out(self, sid: str, dtype: str, chunk_bytes: int) -> dict:
        t = self.sessions.get(sid)
        t0 = time.perf_counter()
        state = t.state_for_sync()
        if dtype == "int8delta" and sid not in self._global:
            raise ValueError("int8delta needs a previously applied global on this worker (push the global first)")
        hdr, blob = tw.encode(state, dtype, ref=self._global.get(sid))
        chunks = tw.chunk(blob, chunk_bytes)
        self._out[sid] = chunks
        return {"header": hdr, "nchunks": len(chunks), "chunk0": base64.b64encode(chunks[0]).decode(),
                "serialize_s": time.perf_counter() - t0}

    def _task_inner(self, p):
        sid = p["session"]
        t = self.sessions.get(sid)
        refs = p.get("refs") if p.get("refs") is not None else p.get("batches", [])
        rep = t.inner_steps(refs, int(p.get("steps", 1)), float(p.get("lr", 1e-3)))
        out = self._encode_out(sid, p.get("sync_dtype", "f32"), int(p.get("chunk_bytes", DEFAULT_CHUNK)))
        rep.timings["serialize_s"] = out.pop("serialize_s")
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            rep.metrics.setdefault("mem_peak_bytes", int(torch.cuda.max_memory_allocated()))
        return {"ok": True, "report": rep.to_json(), "step": t.step, **out}

    def _task_state_get(self, p):
        out = self._encode_out(p["session"], p.get("dtype", "f32"), int(p.get("chunk_bytes", DEFAULT_CHUNK)))
        return {"ok": True, **out}

    def _task_state_chunk(self, p):
        chunks = self._out.get(p["session"])
        k = int(p["k"])
        if not chunks or not 0 <= k < len(chunks):
            raise ValueError(f"no pending chunk {k} for session {p['session']!r}")
        return {"ok": True, "k": k, "data": base64.b64encode(chunks[k]).decode()}

    def _task_state_put(self, p):
        sid, k, n = p["session"], int(p["k"]), int(p["n"])
        self.sessions.get(sid)
        if k == 0:
            if not p.get("header"):
                raise ValueError("first chunk must carry the payload header")
            self._in[sid] = {"header": p["header"], "parts": [], "n": n}
        st = self._in.get(sid)
        if st is None or len(st["parts"]) != k or st["n"] != n:
            self._in.pop(sid, None)
            raise ValueError(f"state chunk {k}/{n} out of order for session {sid!r}")
        st["parts"].append(base64.b64decode(p.get("data", "")))
        if k < n - 1:
            return {"ok": True, "applied": False, "k": k}
        self._in.pop(sid, None)
        t0 = time.perf_counter()
        blob = tw.join(st["parts"], st["header"]["sha256"])
        tensors = tw.decode(st["header"], blob, ref=self._global.get(sid))
        self.sessions.get(sid).load_sync_state(tensors)
        self._global[sid] = {kk: v.clone() for kk, v in tensors.items()}
        return {"ok": True, "applied": True, "deserialize_s": time.perf_counter() - t0}

    def _task_after_outer(self, p):
        return {"ok": True, **(self.sessions.get(p["session"]).after_outer_step(int(p.get("round", 0))) or {})}

    def _task_eval(self, p):
        return {"ok": True, "metrics": self.sessions.get(p["session"]).evaluate(p.get("refs", []), p.get("kind", "loss"))}

    def _task_export(self, p):
        return {"ok": True, **self.sessions.get(p["session"]).export(p.get("fmt", "safetensors"), p["path"])}

    def _task_describe(self, p):
        return {"ok": True, "describe": self.sessions.get(p["session"]).describe()}

    def _task_close(self, p):
        sid = p["session"]
        for d in (self._out, self._in, self._global):
            d.pop(sid, None)
        return {"ok": True, "closed": self.sessions.close(sid)}
