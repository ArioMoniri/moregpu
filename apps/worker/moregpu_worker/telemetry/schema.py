"""Telemetry schema v1 — ``moregpu.telemetry/1`` (ADR-0111).

``SCHEMA`` is the single source of truth; ``docs/telemetry.schema.json`` is a byte-for-byte dump of it (a test keeps
them equal: regenerate with ``python3 -m moregpu_worker.telemetry.schema > docs/telemetry.schema.json``).

``validate`` is a stdlib-only interpreter for exactly the JSON Schema subset used here (type, const, enum, required,
properties, additionalProperties, items, minimum, maximum, minLength, pattern, $ref into $defs); it agrees with the
``jsonschema`` reference implementation on every record in the tests, except that it also rejects NaN/±inf, which
JSON cannot carry anyway.

Record kinds:
  worker_round    one per (session, round, worker), written by the coordinator (apps/coordinator/lib/train_session.ts)
  round           one per session round (aggregate), written by the coordinator
  job             one per vision batch job (or per worker share of one)
  bench           one per ``moregpu bench`` invocation: seeded repeats + stats + emulated limits
  external_round  one per round/step window of a non-MoreGPU runner (DDP, single-process control) — see emit.py
"""
from __future__ import annotations

import json
import math
import re
from decimal import Decimal

SCHEMA_ID = "moregpu.telemetry/1"
KINDS = ("worker_round", "round", "job", "bench", "external_round")
PHASES = ("compute_s", "data_s", "serialize_s", "network_s", "wait_s")
TS_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"

# ---------------------------------------------------------------- building blocks
_SECS = {"type": "number", "minimum": 0}
_NSECS = {"type": ["number", "null"], "minimum": 0}
_INT0 = {"type": "integer", "minimum": 0}
_NINT0 = {"type": ["integer", "null"], "minimum": 0}
_ID = {"type": "string", "minLength": 1}
_NSTR = {"type": ["string", "null"]}
_NOBJ = {"type": ["object", "null"]}
_STRS = {"type": "array", "items": {"type": "string"}}
_NNUM = {"type": ["number", "null"]}
_GPU = {
    "amp": {"type": ["string", "null"], "description": "resolved AMP mode (bf16|fp16|fp32) or null"},
    "gpu_util": {"type": ["number", "null"], "minimum": 0, "maximum": 100, "description": "mean NVML GPU util, %"},
    "gpu_power_w": {"type": ["number", "null"], "minimum": 0, "description": "mean NVML board power, W"},
    "energy_j": {"type": ["number", "null"], "minimum": 0,
                 "description": "NVML total-energy counter delta, else trapezoid of sampled power, J"},
    "mem_peak_bytes": {**_NINT0, "description": "torch.cuda.max_memory_allocated (or NVML used peak)"},
    "hw": {**_NOBJ, "description": "hw.fingerprint() — no PII (host is a salted hash)"},
}
_PROV = {"git_sha": _NSTR, "config_hash": {"type": ["string", "null"], "description": "sha256 of canonical config"}}


def _header(kind: str) -> dict:
    return {"schema": {"const": SCHEMA_ID}, "kind": {"const": kind},
            "ts": {"type": "string", "pattern": TS_PATTERN, "description": "ISO-8601 UTC time of emission"}}


def _phases(nullable: bool = False) -> dict:
    s = _NSECS if nullable else _SECS
    return {"wall_s": {**s, "description": "round/job wall time; = compute+data+serialize+network+wait"},
            "compute_s": s, "data_s": s, "serialize_s": s,
            "network_s": {**s, "description": "time in flight on the wire (coordinator-measured for MoreGPU)"},
            "wait_s": {**s, "description": "wall − all other phases (straggler / barrier wait)"}}


def _obj(props: dict, required, extra=False) -> dict:
    return {"type": "object", "required": sorted(required), "properties": props, "additionalProperties": extra}


_WORKER_ROUND = {
    **_header("worker_round"),
    "session": _ID, "task": _ID, "round": _INT0, "worker": _ID,
    **_phases(),
    "bytes_up": _INT0, "bytes_down": _INT0, "samples": _INT0,
    "samples_seen": {**_INT0, "description": "cumulative samples this worker has trained on in the session"},
    "samples_per_s": _NSECS, "loss_last": _NNUM,
    **_GPU,
    "wire_error": {"type": ["object", "null"], "required": ["max_abs", "rel_l2"],
                   "properties": {"max_abs": _SECS, "rel_l2": _SECS}, "additionalProperties": False,
                   "description": "lossy sync dtype reconstruction error (tensorwire header.error)"},
    "git_sha": _NSTR, "config_hash": {"type": "string"},
}
_ROUND = {
    **_header("round"),
    "session": _ID, "task": _ID, "round": _INT0, "wall_s": _SECS, "reduce_s": _SECS,
    "workers": _STRS, "dropped": _STRS, "samples": _INT0, "samples_seen": _INT0, "avg_last_loss": _NNUM,
    "bytes_up": _INT0, "bytes_down": _INT0, "alarms": _STRS, "eval": _NOBJ, "monitors": _NOBJ,
    "git_sha": _NSTR, "config_hash": {"type": "string"},
}
_JOB = {
    **_header("job"),
    "job": _ID, "op": {"type": "string"}, "session": _NSTR, "worker": _NSTR,
    "items": _INT0, "items_done": _INT0, "items_failed": _INT0, "retries": _INT0,
    **_phases(nullable=True), "wall_s": _SECS,
    "items_per_s": _NSECS, "bytes_up": _INT0, "bytes_down": _INT0,
    **_GPU, **_PROV, "metrics": _NOBJ,
}
_STAT = _obj({"n": _INT0, "mean": {"type": "number"}, "sd": _SECS, "min": {"type": "number"},
              "max": {"type": "number"}}, ["n", "mean", "sd", "min", "max"])
_LIMIT = _obj({"type": _ID, "value": {"type": ["number", "string", "null"]}, "mechanism": {"type": "string"},
               "label": {"const": "emulation"}, "applied": {"type": "boolean"}},
              ["type", "value", "label", "applied"])
_BENCH = {
    **_header("bench"),
    "name": _NSTR, "cmd": _NSTR, "repeats": {"type": "integer", "minimum": 1},
    "seeds": {"type": "array", "items": {"type": "integer"}},
    "runs": {"type": "array", "items": _obj(
        {"seed": {"type": "integer"}, "wall_s": _SECS, "rc": {"type": ["integer", "null"]}, "records": _INT0,
         "invalid_records": _INT0, "error": _NSTR}, ["seed", "wall_s", "rc", "records", "invalid_records"])},
    "failures": _INT0,
    "stats": {"type": "object", "additionalProperties": _STAT,
              "description": "per-metric summary over successful runs (wall_s, samples_per_s, …)"},
    "emulation": {"type": ["object", "null"], "required": ["label", "limits"],
                  "properties": {"label": {"const": "emulation"}, "note": {"type": "string"},
                                 "limits": {"type": "array", "items": _LIMIT}},
                  "additionalProperties": False,
                  "description": "every resource/network limit is an emulation, never a real device"},
    "hw": _NOBJ, **_PROV, "metrics": _NOBJ,
}
_EXTERNAL = {
    **_header("external_round"),
    "runner": {**_ID, "description": "ddp | single | fsdp | … (free-form, non-empty)"},
    "world_size": {"type": "integer", "minimum": 1}, "rank": _INT0, "round": _INT0,
    "session": _NSTR, "task": _NSTR,
    **_phases(),
    "samples": _INT0, "samples_seen": _INT0, "samples_per_s": _NSECS, "loss_last": _NNUM,
    "bytes_up": _INT0, "bytes_down": _INT0,
    **_GPU, **_PROV, "metrics": _NOBJ,
}

_HEAD = ("schema", "kind", "ts")
SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:moregpu:telemetry:1",
    "title": SCHEMA_ID,
    "description": "MoreGPU telemetry records (JSONL, one object per line). See docs/TELEMETRY.md and ADR-0111.",
    "type": "object",
    "required": ["schema", "kind"],
    "properties": {"schema": {"const": SCHEMA_ID}, "kind": {"enum": list(KINDS)}},
    "oneOf": [{"$ref": f"#/$defs/{k}"} for k in KINDS],
    "$defs": {
        "worker_round": _obj(_WORKER_ROUND, _WORKER_ROUND),
        "round": _obj(_ROUND, _ROUND),
        "job": _obj(_JOB, [*_HEAD, "job", "items", "wall_s"]),
        "bench": _obj(_BENCH, [k for k in _BENCH if k != "metrics"]),
        "external_round": _obj(_EXTERNAL, [*_HEAD, "runner", "world_size", "rank", "round", "wall_s", *PHASES]),
    },
}


# ---------------------------------------------------------------- validator
_TS_RE = re.compile(TS_PATTERN)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _type_ok(v, t: str) -> bool:
    if t == "null":
        return v is None
    if t == "boolean":
        return isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "object":
        return isinstance(v, dict)
    if t == "array":
        return isinstance(v, list)
    if t == "number":
        return _is_num(v)
    if t == "integer":
        return _is_num(v) and float(v).is_integer()
    raise ValueError(f"unsupported type {t!r}")  # pragma: no cover - schema is fixed


def _check(v, sch: dict, path: str, errs: list[str]) -> None:
    if "$ref" in sch:
        sch = SCHEMA["$defs"][sch["$ref"].rsplit("/", 1)[1]]
    if "const" in sch and v != sch["const"]:
        errs.append(f"{path}: must be {sch['const']!r}, got {v!r}")
        return
    if "enum" in sch and v not in sch["enum"]:
        errs.append(f"{path}: must be one of {sch['enum']}, got {v!r}")
        return
    if "type" in sch:
        types = sch["type"] if isinstance(sch["type"], list) else [sch["type"]]
        if not any(_type_ok(v, t) for t in types):
            errs.append(f"{path}: expected {'|'.join(types)}, got {type(v).__name__} {v!r:.60}")
            return
    if _is_num(v):
        if "minimum" in sch and v < sch["minimum"]:
            errs.append(f"{path}: must be ≥ {sch['minimum']}, got {v}")
        if "maximum" in sch and v > sch["maximum"]:
            errs.append(f"{path}: must be ≤ {sch['maximum']}, got {v}")
    if isinstance(v, str):
        if len(v) < sch.get("minLength", 0):
            errs.append(f"{path}: must not be empty")
        if "pattern" in sch and not re.search(sch["pattern"], v):
            errs.append(f"{path}: {v!r} does not match {sch['pattern']}")
    if isinstance(v, list) and "items" in sch:
        for i, x in enumerate(v):
            _check(x, sch["items"], f"{path}[{i}]", errs)
    if isinstance(v, dict):
        props = sch.get("properties", {})
        for k in sch.get("required", []):
            if k not in v:
                errs.append(f"{_join(path, k)}: required field missing")
        extra = sch.get("additionalProperties", True)
        for k, x in v.items():
            if k in props:
                _check(x, props[k], _join(path, k), errs)
            elif extra is False:
                errs.append(f"{_join(path, k)}: unexpected field")
            elif isinstance(extra, dict):
                _check(x, extra, _join(path, k), errs)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def validate(record) -> list[str]:
    """Return a list of human-readable problems (empty = valid ``moregpu.telemetry/1`` record)."""
    if not isinstance(record, dict):
        return ["record must be an object"]
    errs: list[str] = []
    if record.get("schema") != SCHEMA_ID:
        errs.append(f"schema: must be {SCHEMA_ID!r}, got {record.get('schema')!r}")
    kind = record.get("kind")
    if kind not in KINDS:
        errs.append(f"kind: must be one of {list(KINDS)}, got {kind!r}")
        return errs
    _check(record, SCHEMA["$defs"][kind], "", errs)
    return errs


def breakdown_ok(rec: dict, tol: float = 0.05) -> bool:
    """compute+data+serialize+network+wait ≈ wall_s (relative ``tol``), with no phase meaningfully negative."""
    vals = [rec.get(k) for k in ("wall_s", *PHASES)]
    if not all(_is_num(x) for x in vals):
        return False
    wall, parts = vals[0], vals[1:]
    slack = tol * abs(wall) + 1e-9
    return wall >= 0 and all(p >= -slack for p in parts) and abs(sum(parts) - wall) <= slack


# ---------------------------------------------------------------- config hash (== coordinator configHash)
def _js_number(x) -> str:
    """ECMAScript Number::toString, so hashes match JSON.stringify on the coordinator."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x) if abs(x) < 10 ** 21 else _js_number(float(x))
    if not math.isfinite(x):
        return "null"
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    t = Decimal(repr(abs(x))).normalize().as_tuple()
    s = "".join(map(str, t.digits))
    k, n = len(s), t.exponent + len(s)
    if k <= n <= 21:
        body = s + "0" * (n - k)
    elif 0 < n <= 21:
        body = s[:n] + "." + s[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + s
    else:
        e = n - 1
        body = s[0] + ("." + s[1:] if k > 1 else "") + "e" + ("+" if e > 0 else "-") + str(abs(e))
    return sign + body


def canonical_json(v) -> str:
    """JSON with every object's keys sorted, no whitespace, JS number formatting (coordinator-compatible)."""
    if isinstance(v, dict):
        return "{" + ",".join(json.dumps(str(k), ensure_ascii=False) + ":" + canonical_json(v[k])
                              for k in sorted(v, key=str)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(canonical_json(x) for x in v) + "]"
    if v is None:
        return "null"
    if isinstance(v, (bool, int, float)):
        return _js_number(v)
    return json.dumps(v if isinstance(v, str) else str(v), ensure_ascii=False)


def config_hash(cfg) -> str:
    import hashlib
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(SCHEMA, indent=2))
