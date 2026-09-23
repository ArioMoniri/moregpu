"""JsonlEmitter + PhaseTimer (ADR-0111): the standalone emitter external runners (DDP / single-process controls) use."""
import json
import threading

import pytest

from moregpu_worker.telemetry import emit as E
from moregpu_worker.telemetry import schema as S


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += dt


def _lines(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_emit_fills_header_and_validates(tmp_path):
    p = tmp_path / "sub" / "run.jsonl"
    em = E.JsonlEmitter(p, git_sha="cafe", config={"lr": 0.1, "batch": 8}, hw={"os": "Linux"})
    rec = em.emit("external_round", runner="ddp", world_size=2, rank=1, round=0, wall_s=1.0, compute_s=0.6,
                  data_s=0.1, serialize_s=0.0, network_s=0.2, wait_s=0.1, samples=32)
    assert rec["schema"] == "moregpu.telemetry/1" and rec["kind"] == "external_round"
    assert rec["git_sha"] == "cafe" and rec["config_hash"] == S.config_hash({"lr": 0.1, "batch": 8})
    assert rec["hw"] == {"os": "Linux"}
    assert rec["ts"].endswith("Z")
    assert S.validate(rec) == []
    assert _lines(p) == [rec]


def test_emit_only_fills_fields_the_kind_has(tmp_path):
    em = E.JsonlEmitter(tmp_path / "r.jsonl", git_sha="cafe", config_hash="b" * 64, hw={"os": "Linux"})
    rec = em.emit("round", session="s", task="toy", round=1, wall_s=1.0, reduce_s=0.1, hook_s=0.0, eval_s=0.0, workers=["w"], dropped=[],
                  samples=1, samples_seen=1, avg_last_loss=None, lr=None, bytes_up=0, bytes_down=0, alarms=[], eval=None,
                  monitors=None)
    assert "hw" not in rec                     # `round` has no hw field
    assert rec["config_hash"] == "b" * 64
    assert S.validate(rec) == []


def test_explicit_fields_win_over_defaults(tmp_path):
    em = E.JsonlEmitter(tmp_path / "r.jsonl", git_sha="cafe", hw={"os": "Linux"})
    rec = em.emit("external_round", runner="single", world_size=1, rank=0, round=0, wall_s=1.0, compute_s=1.0,
                  data_s=0, serialize_s=0, network_s=0, wait_s=0, git_sha="beef", hw=None)
    assert rec["git_sha"] == "beef" and rec["hw"] is None


def test_emit_rejects_invalid_records_strict(tmp_path):
    p = tmp_path / "r.jsonl"
    em = E.JsonlEmitter(p, git_sha=None, hw=None)
    with pytest.raises(ValueError, match="world_size"):
        em.emit("external_round", runner="ddp", world_size=0, rank=0, round=0, wall_s=1.0, compute_s=1.0,
                data_s=0, serialize_s=0, network_s=0, wait_s=0)
    assert not p.exists()
    lax = E.JsonlEmitter(p, git_sha=None, hw=None, strict=False)
    rec = lax.emit("external_round", runner="ddp", world_size=0, rank=0, round=0, wall_s=1.0, compute_s=1.0,
                   data_s=0, serialize_s=0, network_s=0, wait_s=0)
    assert rec["world_size"] == 0 and len(_lines(p)) == 1


def test_emit_is_thread_safe(tmp_path):
    p = tmp_path / "r.jsonl"
    em = E.JsonlEmitter(p, git_sha=None, hw=None)

    def work(rank):
        for i in range(25):
            em.emit("external_round", runner="ddp", world_size=4, rank=rank, round=i, wall_s=1.0, compute_s=1.0,
                    data_s=0, serialize_s=0, network_s=0, wait_s=0)
    ts = [threading.Thread(target=work, args=(r,)) for r in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    recs = _lines(p)
    assert len(recs) == 100 and all(S.validate(r) == [] for r in recs)


def test_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MOREGPU_TELEMETRY_FILE", str(tmp_path / "f.jsonl"))
    assert E.JsonlEmitter.from_env(hw=None).path == tmp_path / "f.jsonl"
    monkeypatch.delenv("MOREGPU_TELEMETRY_FILE")
    monkeypatch.setenv("MOREGPU_TELEMETRY_DIR", str(tmp_path / "d"))
    assert E.JsonlEmitter.from_env("ctl", hw=None).path == tmp_path / "d" / "ctl.jsonl"
    monkeypatch.delenv("MOREGPU_TELEMETRY_DIR")
    assert E.JsonlEmitter.from_env(hw=None) is None


def test_git_sha_default(monkeypatch):
    monkeypatch.setenv("MOREGPU_GIT_SHA", "0123abc")
    assert E.default_git_sha() == "0123abc"
    monkeypatch.delenv("MOREGPU_GIT_SHA")
    sha = E.default_git_sha()
    assert sha is None or (isinstance(sha, str) and len(sha) >= 7)


def test_git_sha_default_without_git(monkeypatch):
    monkeypatch.delenv("MOREGPU_GIT_SHA", raising=False)

    def boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(E.subprocess, "run", boom)
    assert E.default_git_sha() is None


def test_hw_auto_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "fingerprint", lambda: {"os": "Test"})
    em = E.JsonlEmitter(tmp_path / "r.jsonl", git_sha=None)
    rec = em.emit("external_round", runner="single", world_size=1, rank=0, round=0, wall_s=0, compute_s=0,
                  data_s=0, serialize_s=0, network_s=0, wait_s=0)
    assert rec["hw"] == {"os": "Test"}


# ---------------------------------------------------------------- PhaseTimer
def test_phase_timer_sums_exactly_to_wall():
    c = Clock()
    pt = E.PhaseTimer(clock=c)
    with pt.phase("data"):
        c.tick(0.25)
    with pt.phase("compute"):
        c.tick(1.0)
    c.tick(0.5)                                   # untracked time → wait
    pt.add("network", 0.125)
    with pt.phase("serialize"):
        c.tick(0.0625)
    rec = pt.record()
    assert rec == {"wall_s": 1.8125, "compute_s": 1.0, "data_s": 0.25, "serialize_s": 0.0625, "network_s": 0.125,
                   "wait_s": 0.375}
    assert sum(rec[k] for k in ("compute_s", "data_s", "serialize_s", "network_s", "wait_s")) == rec["wall_s"]
    assert S.breakdown_ok(rec)


def test_phase_timer_explicit_wait_plus_residual():
    c = Clock()
    pt = E.PhaseTimer(clock=c)
    with pt.phase("wait"):
        c.tick(0.5)
    c.tick(0.25)
    assert pt.record()["wait_s"] == 0.75


def test_phase_timer_overlap_never_negative():
    c = Clock()
    pt = E.PhaseTimer(clock=c)
    c.tick(1.0)
    pt.add("network", 3.0)                        # e.g. async comm measured elsewhere, overlapping compute
    rec = pt.record()
    assert rec["wait_s"] == 0.0 and rec["wall_s"] == 3.0
    assert S.breakdown_ok(rec)


def test_phase_timer_lap_resets():
    c = Clock()
    pt = E.PhaseTimer(clock=c)
    with pt.phase("compute"):
        c.tick(1.0)
    r1 = pt.lap()
    with pt.phase("data"):
        c.tick(2.0)
    r2 = pt.lap()
    assert r1["compute_s"] == 1.0 and r1["wall_s"] == 1.0
    assert r2["compute_s"] == 0.0 and r2["data_s"] == 2.0 and r2["wall_s"] == 2.0


def test_phase_timer_as_context_manager_and_errors():
    c = Clock()
    with E.PhaseTimer(clock=c) as pt:
        c.tick(1.0)
    c.tick(5.0)                                   # after exit: frozen
    assert pt.record()["wall_s"] == 1.0
    pt2 = E.PhaseTimer(clock=c)
    with pytest.raises(ValueError, match="unknown phase"):
        pt2.add("gpu", 1.0)
    with pytest.raises(ValueError, match="negative"):
        pt2.add("compute", -1.0)
    with pytest.raises(RuntimeError, match="nested"):
        with pt2.phase("compute"):
            with pt2.phase("data"):
                pass


def test_phase_timer_phase_records_on_exception():
    c = Clock()
    pt = E.PhaseTimer(clock=c)
    with pytest.raises(KeyError):
        with pt.phase("compute"):
            c.tick(0.5)
            raise KeyError("x")
    assert pt.record()["compute_s"] == 0.5
    with pt.phase("data"):                        # phase slot released after the exception
        c.tick(0.1)


def test_phase_timer_feeds_emitter(tmp_path):
    c = Clock()
    pt = E.PhaseTimer(clock=c)
    em = E.JsonlEmitter(tmp_path / "r.jsonl", git_sha=None, hw=None)
    with pt.phase("compute"):
        c.tick(0.5)
    rec = em.emit("external_round", runner="single", world_size=1, rank=0, round=0, samples=10,
                  samples_per_s=20.0, **pt.lap())
    assert S.validate(rec) == [] and S.breakdown_ok(rec)
