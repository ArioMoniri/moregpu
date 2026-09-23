import json

import pytest
import torch

from moregpu_worker.train import registry as R
from moregpu_worker.train.sessions import SessionStore, SessionLimit
from moregpu_worker.train.task import TaskContext, TrainTask
from moregpu_worker.train.local import run_plain, run_diloco


def ctx(seed=0):
    return TaskContext(device="cpu", amp="fp32", seed=seed, session="s")


TOY = {"n": 64, "dim": 5, "batch": 4, "optimizer": "sgd"}


def test_builtin_registry_lists_tasks():
    names = R.available()
    for n in ("toy_linear", "llm_lora", "ijepa_2d", "jepa_2p5d", "jepa_3d", "classify", "segment"):
        assert n in names


def test_unknown_task_errors():
    with pytest.raises(KeyError):
        R.create("nope")


def test_toy_task_contract():
    t = R.create("toy_linear")
    info = t.init(TOY, ctx())
    assert info["params"] > 0
    rep = t.inner_steps(list(range(8)), steps=2, lr=0.1)
    assert rep.samples == 8 and len(rep.losses) == 2 and rep.timings["compute_s"] >= 0
    st = t.state_for_sync()
    t.load_sync_state({k: torch.zeros_like(v) for k, v in st.items()})
    assert all((v == 0).all() for v in t.state_for_sync().values())
    assert "loss" in t.evaluate(list(range(8)), "loss")


def test_session_store_limits_and_isolation():
    s = SessionStore(max_sessions=2)
    a = s.create("a", "toy_linear", TOY, ctx())
    b = s.create("b", "toy_linear", {**TOY, "seed_offset": 1}, ctx(1))
    assert s.get("a") is a and s.get("b") is b and a is not b
    with pytest.raises(SessionLimit):
        s.create("c", "toy_linear", TOY, ctx())
    a.inner_steps([0, 1, 2, 3], 1, 0.1)
    # training session a must not touch session b's weights
    wb = {k: v.clone() for k, v in b.state_for_sync().items()}
    a.inner_steps([0, 1, 2, 3], 3, 0.1)
    assert all(torch.equal(wb[k], v) for k, v in b.state_for_sync().items())
    s.close("a")
    s.create("c", "toy_linear", TOY, ctx())
    with pytest.raises(KeyError):
        s.get("a")
    with pytest.raises(ValueError):
        s.create("b", "toy_linear", TOY, ctx())


def test_diloco_n1_h1_equals_plain_training():
    cfg = {**TOY, "optimizer": "adamw"}
    plain = run_plain("toy_linear", cfg, steps=12, batch=4, lr=0.05, seed=0, manifest_len=64)
    dl = run_diloco("toy_linear", cfg, n_workers=1, rounds=12, inner_steps=1, batch=4, lr=0.05,
                    outer_lr=1.0, outer_momentum=0.0, seed=0, manifest_len=64, keep_inner_state=True)
    for k in plain["state"]:
        assert torch.allclose(plain["state"][k], dl["state"][k], atol=1e-6), k


def test_two_worker_diloco_is_deterministic_and_counts_samples():
    kw = dict(n_workers=2, rounds=5, inner_steps=3, batch=4, lr=0.05, outer_lr=0.7, outer_momentum=0.9, seed=1,
              manifest_len=64)
    a = run_diloco("toy_linear", TOY, **kw)
    b = run_diloco("toy_linear", TOY, **kw)
    assert all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"])
    assert a["samples_seen"] == 2 * 5 * 3 * 4
    assert a["losses"][-1] < a["losses"][0]


def test_target_samples_stops_exactly():
    r = run_diloco("toy_linear", TOY, n_workers=2, rounds=100, inner_steps=2, batch=4, lr=0.05, outer_lr=0.7,
                   outer_momentum=0.9, seed=0, manifest_len=64, target_samples=50)
    assert r["samples_seen"] == 50


def test_plugin_allowlist_refuses_unpinned(tmp_path, monkeypatch):
    class Dist:
        def __init__(self, name, version, sha):
            self.metadata = {"Name": name}; self.version = version; self._sha = sha
        def read_text(self, f):
            if f == "direct_url.json" and self._sha:
                return json.dumps({"archive_info": {"hashes": {"sha256": self._sha}}})
            return None
    class EP:
        def __init__(self, name, dist): self.name, self.dist, self.value = name, dist, "x:y"
        def load(self): return lambda: "LOADED"
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps([{"dist": "good-plugin", "version": "1.0", "wheel_sha256": "ab" * 32}]))
    eps = [EP("good", Dist("good-plugin", "1.0", "ab" * 32)), EP("wrongver", Dist("good-plugin", "2.0", "ab" * 32)),
           EP("nohash", Dist("good-plugin", "1.0", None)), EP("unknown", Dist("evil", "1.0", "cd" * 32))]
    found, refused = R.discover_plugins(eps, allowlist_path=allow)
    assert list(found) == ["good"]
    assert set(refused) == {"wrongver", "nohash", "unknown"}
