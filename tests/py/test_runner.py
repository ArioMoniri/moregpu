"""Worker-side task op protocol (task_*), exercised as the coordinator would drive it."""
import base64

import pytest
import torch

from moregpu_worker.train import diloco as D, tensorwire as tw
from moregpu_worker.train.runner import TaskRunner
from moregpu_worker.train.sessions import SessionStore

TOY = {"n": 64, "dim": 5, "batch": 4, "optimizer": "sgd"}


def pull_state(r, sid, first):
    chunks = [base64.b64decode(first["chunk0"])]
    for k in range(1, first["nchunks"]):
        chunks.append(base64.b64decode(r.handle("task_state_chunk", {"session": sid, "k": k})["data"]))
    return tw.join(chunks, first["header"]["sha256"])


def push_state(r, sid, tensors, dtype="f32", chunk=64):
    hdr, blob = tw.encode(tensors, dtype)
    parts = tw.chunk(blob, chunk)
    out = None
    for k, p in enumerate(parts):
        out = r.handle("task_state_put", {"session": sid, "header": hdr if k == 0 else None, "k": k, "n": len(parts),
                                          "data": base64.b64encode(p).decode()})
    return out


def runner():
    return TaskRunner(SessionStore(4), device="cpu")


def test_init_inner_pull_push_roundtrip_small_chunks():
    r = runner()
    info = r.handle("task_init", {"session": "s1", "task": "toy_linear", "cfg": TOY, "amp": "fp32", "seed": 0})
    assert info["ok"] and info["describe"]["task"] == "toy_linear"
    out = r.handle("task_inner", {"session": "s1", "refs": list(range(8)), "steps": 2, "lr": 0.1,
                                  "sync_dtype": "f32", "chunk_bytes": 8})
    assert out["report"]["samples"] == 8 and out["nchunks"] > 1
    assert set(out["report"]["timings"]) >= {"compute_s", "serialize_s"}
    blob = pull_state(r, "s1", out)
    got = tw.decode(out["header"], blob)
    task = r.sessions.get("s1")
    assert all(torch.equal(got[k], v) for k, v in task.state_for_sync().items())
    zeros = {k: torch.zeros_like(v) for k, v in got.items()}
    res = push_state(r, "s1", zeros)
    assert res["applied"]
    assert all((v == 0).all() for v in task.state_for_sync().values())


def test_int8delta_uses_last_applied_global_as_reference():
    r = runner()
    r.handle("task_init", {"session": "s", "task": "toy_linear", "cfg": TOY, "seed": 0})
    g0 = {k: v.clone() for k, v in r.sessions.get("s").state_for_sync().items()}
    push_state(r, "s", g0)
    out = r.handle("task_inner", {"session": "s", "refs": list(range(8)), "steps": 2, "lr": 0.1, "sync_dtype": "int8delta"})
    blob = pull_state(r, "s", out)
    dec = tw.decode(out["header"], blob, ref=g0)
    true = r.sessions.get("s").state_for_sync()
    dmax = max((true[k] - g0[k]).abs().max().item() for k in true)
    bound = dmax / 254 + 1e-6               # half an int8 step of the largest block scale
    for k in true:
        assert (dec[k] - true[k]).abs().max().item() <= bound
    assert out["header"]["error"]["max_abs"] <= bound


def test_int8delta_without_prior_global_is_refused():
    r = runner()
    r.handle("task_init", {"session": "s", "task": "toy_linear", "cfg": TOY})
    with pytest.raises(ValueError, match="int8delta"):
        r.handle("task_inner", {"session": "s", "refs": [0, 1], "steps": 1, "lr": 0.1, "sync_dtype": "int8delta"})


def test_push_rejects_tampered_chunk_and_out_of_order():
    r = runner()
    r.handle("task_init", {"session": "s", "task": "toy_linear", "cfg": TOY})
    hdr, blob = tw.encode(r.sessions.get("s").state_for_sync(), "f32")
    bad = bytearray(blob); bad[0] ^= 1
    with pytest.raises(ValueError):
        r.handle("task_state_put", {"session": "s", "header": hdr, "k": 0, "n": 1, "data": base64.b64encode(bytes(bad)).decode()})
    with pytest.raises(ValueError, match="order"):
        r.handle("task_state_put", {"session": "s", "header": None, "k": 1, "n": 2, "data": ""})


def test_sessions_are_isolated_and_close_frees():
    r = runner()
    r.handle("task_init", {"session": "a", "task": "toy_linear", "cfg": TOY})
    r.handle("task_init", {"session": "b", "task": "toy_linear", "cfg": TOY})
    assert set(r.handle("task_list", {})["sessions"]) == {"a", "b"}
    r.handle("task_close", {"session": "a"})
    with pytest.raises(KeyError):
        r.handle("task_inner", {"session": "a", "refs": [0], "steps": 1, "lr": 0.1})


def test_eval_after_outer_describe_and_unknown_op():
    r = runner()
    r.handle("task_init", {"session": "s", "task": "toy_linear", "cfg": TOY})
    assert "loss" in r.handle("task_eval", {"session": "s", "refs": [0, 1, 2], "kind": "loss"})["metrics"]
    assert r.handle("task_after_outer", {"session": "s", "round": 1})["ok"]
    assert r.handle("task_describe", {"session": "s"})["describe"]["task"] == "toy_linear"
    with pytest.raises(ValueError):
        r.handle("task_bogus", {})


def test_runner_drives_same_result_as_local_diloco():
    """Two runners + the Python outer loop ≡ moregpu_worker.train.local.run_diloco (same stream/alloc)."""
    from moregpu_worker.train.local import run_diloco
    from moregpu_worker.train.sharding import SampleStream, allocate, split
    ref = run_diloco("toy_linear", TOY, n_workers=2, rounds=3, inner_steps=2, batch=4, lr=0.05, outer_lr=0.7,
                     outer_momentum=0.9, seed=5, manifest_len=64)
    rs = [runner(), runner()]
    for i, r in enumerate(rs):
        r.handle("task_init", {"session": "s", "task": "toy_linear", "cfg": TOY, "seed": 5})
    st = D.OuterState.init(rs[0].sessions.get("s").state_for_sync())
    push_state(rs[1], "s", st.global_)
    stream = SampleStream(64, 5)
    for _ in range(3):
        sizes = allocate(8, 2, [1, 1])
        shards = split(stream.take(16), sizes)
        res = []
        for r, idx in zip(rs, shards):
            o = r.handle("task_inner", {"session": "s", "refs": idx, "steps": 2, "lr": 0.05})
            res.append((tw.decode(o["header"], pull_state(r, "s", o)), o["report"]["samples"]))
        D.outer_step(st, D.weighted_average(res), 0.7, 0.9)
        for r in rs:
            push_state(r, "s", st.global_)
    for k in ref["state"]:
        assert torch.allclose(ref["state"][k], st.global_[k], atol=1e-6)
