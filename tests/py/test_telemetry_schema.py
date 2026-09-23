"""Telemetry schema v1 (ADR-0111): docs/telemetry.schema.json, stdlib validator, breakdown check, cross-lang agreement."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from moregpu_worker.telemetry import schema as S

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "telemetry.schema.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ts_emit_telemetry.ts"

# Field sets emitted by apps/coordinator/lib/train_session.ts emitTelemetry (kept literal on purpose: drift must fail).
WORKER_ROUND_FIELDS = {
    "schema", "kind", "ts", "session", "task", "round", "worker", "wall_s", "compute_s", "data_s", "serialize_s",
    "network_s", "wait_s", "bytes_up", "bytes_down", "samples", "samples_seen", "samples_per_s", "loss_last", "amp",
    "gpu_util", "gpu_power_w", "energy_j", "mem_peak_bytes", "hw", "wire_error", "git_sha", "config_hash",
}
ROUND_FIELDS = {
    "schema", "kind", "ts", "session", "task", "round", "wall_s", "reduce_s", "workers", "dropped", "samples",
    "samples_seen", "avg_last_loss", "bytes_up", "bytes_down", "alarms", "eval", "monitors", "git_sha", "config_hash",
}
TS = "2026-09-23T08:00:00.123Z"
H64 = "a" * 64


def worker_round(**over):
    r = {"schema": "moregpu.telemetry/1", "kind": "worker_round", "ts": TS, "session": "s1", "task": "toy", "round": 3,
         "worker": "w0", "wall_s": 1.0, "compute_s": 0.5, "data_s": 0.1, "serialize_s": 0.05, "network_s": 0.2,
         "wait_s": 0.15, "bytes_up": 1024, "bytes_down": 2048, "samples": 8, "samples_seen": 24, "samples_per_s": 16.0,
         "loss_last": 0.42, "amp": "bf16", "gpu_util": 88.0, "gpu_power_w": 210.5, "energy_j": 105.2,
         "mem_peak_bytes": 1 << 30, "hw": {"os": "Linux"}, "wire_error": {"max_abs": 0.01, "rel_l2": 0.002},
         "git_sha": "abc123", "config_hash": H64}
    r.update(over)
    return r


def round_rec(**over):
    r = {"schema": "moregpu.telemetry/1", "kind": "round", "ts": TS, "session": "s1", "task": "toy", "round": 3,
         "wall_s": 1.0, "reduce_s": 0.01, "workers": ["w0", "w1"], "dropped": [], "samples": 16, "samples_seen": 48,
         "avg_last_loss": 0.4, "bytes_up": 2048, "bytes_down": 4096, "alarms": [], "eval": {"loss": 0.3},
         "monitors": None, "git_sha": None, "config_hash": H64}
    r.update(over)
    return r


def job_rec(**over):
    r = {"schema": "moregpu.telemetry/1", "kind": "job", "ts": TS, "job": "j1", "op": "infer", "worker": "w0",
         "items": 10, "items_done": 9, "items_failed": 1, "retries": 2, "wall_s": 3.0, "compute_s": 2.0, "data_s": 0.5,
         "serialize_s": 0.1, "network_s": 0.2, "wait_s": 0.2, "items_per_s": 4.5, "gpu_util": None,
         "gpu_power_w": None, "energy_j": None, "mem_peak_bytes": None, "hw": None, "git_sha": None,
         "config_hash": None}
    r.update(over)
    return r


def bench_rec(**over):
    r = {"schema": "moregpu.telemetry/1", "kind": "bench", "ts": TS, "name": "b", "cmd": "python3 x.py",
         "repeats": 2, "seeds": [0, 1],
         "runs": [{"seed": 0, "wall_s": 1.0, "rc": 0, "records": 3, "invalid_records": 0},
                  {"seed": 1, "wall_s": 1.2, "rc": 0, "records": 3, "invalid_records": 0}],
         "failures": 0, "stats": {"wall_s": {"n": 2, "mean": 1.1, "sd": 0.14, "min": 1.0, "max": 1.2}},
         "emulation": {"label": "emulation", "limits": []}, "hw": {"os": "Linux"}, "git_sha": None,
         "config_hash": None}
    r.update(over)
    return r


def external_rec(**over):
    r = {"schema": "moregpu.telemetry/1", "kind": "external_round", "ts": TS, "runner": "ddp", "world_size": 4,
         "rank": 0, "round": 0, "wall_s": 1.0, "compute_s": 0.7, "data_s": 0.1, "serialize_s": 0.0,
         "network_s": 0.15, "wait_s": 0.05, "samples": 64, "samples_per_s": 91.4}
    r.update(over)
    return r


VALID = {"worker_round": worker_round, "round": round_rec, "job": job_rec, "bench": bench_rec,
         "external_round": external_rec}


def test_doc_file_is_the_python_schema():
    assert json.loads(DOC.read_text()) == S.SCHEMA


def test_schema_header():
    assert S.SCHEMA_ID == "moregpu.telemetry/1"
    assert S.SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert set(S.KINDS) == {"worker_round", "round", "job", "bench", "external_round"}


def test_field_sets_match_coordinator_emitter():
    d = S.SCHEMA["$defs"]
    assert set(d["worker_round"]["properties"]) == WORKER_ROUND_FIELDS
    assert set(d["worker_round"]["required"]) == WORKER_ROUND_FIELDS
    assert set(d["round"]["properties"]) == ROUND_FIELDS
    assert set(d["round"]["required"]) == ROUND_FIELDS


@pytest.mark.parametrize("kind", sorted(VALID))
def test_valid_records(kind):
    assert S.validate(VALID[kind]()) == []


@pytest.mark.parametrize("field", ["samples_per_s", "loss_last", "amp", "gpu_util", "gpu_power_w", "energy_j",
                                   "mem_peak_bytes", "hw", "wire_error", "git_sha"])
def test_worker_round_nullable(field):
    assert S.validate(worker_round(**{field: None})) == []


@pytest.mark.parametrize("field", ["avg_last_loss", "eval", "monitors", "git_sha"])
def test_round_nullable(field):
    assert S.validate(round_rec(**{field: None})) == []


BAD = [
    ("not a dict", "record must be an object"),
    ({"kind": "worker_round"}, "schema"),
    (worker_round(schema="moregpu.telemetry/2"), "schema"),
    (worker_round(kind="nope"), "kind"),
    ({**worker_round(), "kind": None}, "kind"),
    (worker_round(ts="yesterday"), "ts"),
    (worker_round(round=-1), "round"),
    (worker_round(round=1.5), "round"),
    (worker_round(round=True), "round"),
    (worker_round(wall_s="1"), "wall_s"),
    (worker_round(compute_s=-0.5), "compute_s"),
    (worker_round(compute_s=float("nan")), "compute_s"),
    (worker_round(compute_s=float("inf")), "compute_s"),
    (worker_round(bytes_up=1.5), "bytes_up"),
    (worker_round(session=None), "session"),
    (worker_round(session=""), "session"),
    (worker_round(hw=[1]), "hw"),
    (worker_round(wire_error={"max_abs": "x", "rel_l2": 0}), "wire_error.max_abs"),
    (worker_round(wire_error={"max_abs": 0.1}), "wire_error.rel_l2"),
    (worker_round(config_hash=None), "config_hash"),
    (worker_round(surprise=1), "surprise"),
    ({k: v for k, v in worker_round().items() if k != "wait_s"}, "wait_s"),
    (round_rec(workers="w0"), "workers"),
    (round_rec(workers=["w0", 3]), "workers[1]"),
    (round_rec(alarms=None), "alarms"),
    (round_rec(eval=[]), "eval"),
    (job_rec(items=-1), "items"),
    ({k: v for k, v in job_rec().items() if k != "job"}, "job"),
    (bench_rec(seeds=[0, "1"]), "seeds[1]"),
    (bench_rec(stats={"wall_s": {"n": 2, "mean": 1.0}}), "stats.wall_s.sd"),
    (bench_rec(emulation={"label": "real", "limits": []}), "emulation.label"),
    (bench_rec(emulation={"label": "emulation", "limits": [{"type": "cpus", "value": 2}]}), "emulation.limits[0].label"),
    (bench_rec(runs=[{"seed": 0}]), "runs[0].wall_s"),
    (bench_rec(repeats=0), "repeats"),
    (external_rec(world_size=0), "world_size"),
    (external_rec(runner=""), "runner"),
    (external_rec(rank=-1), "rank"),
]


@pytest.mark.parametrize("rec,needle", BAD, ids=[f"bad{i}" for i in range(len(BAD))])
def test_invalid_records_report_the_field(rec, needle):
    errs = S.validate(rec)
    assert errs, rec
    assert any(needle in e for e in errs), errs


def test_validate_accepts_integral_floats_as_integers():
    assert S.validate(worker_round(bytes_up=1024.0, round=3.0)) == []


def test_breakdown_ok():
    assert S.breakdown_ok(worker_round())
    assert S.breakdown_ok(external_rec())
    assert S.breakdown_ok(worker_round(wait_s=0.15 + 0.04))            # 4% off, within 5%
    assert not S.breakdown_ok(worker_round(wait_s=0.15 + 0.2))
    assert not S.breakdown_ok(worker_round(wait_s=0.15 + 0.04), tol=0.01)
    assert not S.breakdown_ok(worker_round(wait_s=None))
    assert not S.breakdown_ok({"wall_s": 1.0})
    assert not S.breakdown_ok(worker_round(wait_s=-0.2, network_s=0.55))  # negative phase beyond tolerance
    zero = worker_round(wall_s=0.0, compute_s=0.0, data_s=0.0, serialize_s=0.0, network_s=0.0, wait_s=0.0)
    assert S.breakdown_ok(zero)


@pytest.mark.parametrize("x,js", [(1e-7, "1e-7"), (0.000001, "0.000001"), (1e21, "1e+21"), (123.45, "123.45"),
                                  (-2.5, "-2.5"), (100.0, "100"), (1.5e300, "1.5e+300"), (float("nan"), "null"),
                                  (True, "true"), (False, "false"), (10 ** 22, "1e+22"), (0.0, "0"), (7, "7"),
                                  (0.1, "0.1"), (2.5e-7, "2.5e-7")])
def test_js_number_formatting(x, js):
    assert S.canonical_json(x) == js


def test_canonical_json_structure():
    assert S.canonical_json({"b": [1, (2.0, None)], "a": "é\n"}) == '{"a":"é\\n","b":[1,[2,null]]}'


def test_range_checks():
    assert any("gpu_util" in e for e in S.validate(worker_round(gpu_util=150.0)))


def test_config_hash_is_canonical():
    a = S.config_hash({"b": 1, "a": {"y": 2.0, "x": [1, 2]}})
    b = S.config_hash({"a": {"x": [1, 2], "y": 2}, "b": 1})
    assert a == b and len(a) == 64


# ---------------------------------------------------------------- agreement with the reference implementation
try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_is_valid_draft_2020_12():
    jsonschema.Draft202012Validator.check_schema(S.SCHEMA)


def _has_nonfinite(r):
    # JSON cannot carry NaN/inf; our validator rejects them, jsonschema (on Python floats) does not — not compared.
    return isinstance(r, dict) and any(isinstance(v, float) and (v != v or abs(v) == float("inf"))
                                       for v in r.values())


AGREEMENT = [f() for f in VALID.values()] + [copy.deepcopy(r) for r, _ in BAD if not _has_nonfinite(r)] \
    + [worker_round(bytes_up=1024.0), round_rec(avg_last_loss=None)]


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
@pytest.mark.parametrize("rec", AGREEMENT, ids=[f"agree{i}" for i in range(len(AGREEMENT))])
def test_agrees_with_jsonschema(rec):
    ref = jsonschema.Draft202012Validator(S.SCHEMA)
    assert (not S.validate(rec)) == ref.is_valid(rec)


# ---------------------------------------------------------------- what the coordinator actually writes
@pytest.mark.skipif(shutil.which("deno") is None, reason="deno not installed")
def test_coordinator_records_validate():
    out = subprocess.run(["deno", "run", "--quiet", str(FIXTURE)], cwd=ROOT, capture_output=True, text=True,
                         timeout=180)
    assert out.returncode == 0, out.stderr
    recs = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    kinds = [r["kind"] for r in recs]
    assert kinds.count("worker_round") == 4 and kinds.count("round") == 2
    for r in recs:
        assert S.validate(r) == [], (r, S.validate(r))
        if r["kind"] == "worker_round":
            assert set(r) == WORKER_ROUND_FIELDS
            assert S.breakdown_ok(r)
        else:
            assert set(r) == ROUND_FIELDS
    # python config_hash reproduces the coordinator's configHash for the same config
    cfg = {"task": "toy", "manifest_len": 32, "batch": 2, "inner_steps": 2, "lr": 0.1, "seed": 1, "chunk_bytes": 6,
           "sync_dtype": "bf16", "eval": {"refs": [0, 1], "kind": "loss", "every": 2}}
    assert recs[0]["config_hash"] == S.config_hash(cfg)
