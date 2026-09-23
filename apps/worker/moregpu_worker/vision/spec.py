"""Model spec v1 (ADR-0113): the JSON Schema plus a small stdlib validator for the subset of JSON Schema it uses.

`python3 -m moregpu_worker.vision.spec` prints the schema; `docs/model-spec.schema.json` is that output (a test keeps
them identical).
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any

FORMATS = ["state_dict", "safetensors", "plugin", "torch_export", "torchscript", "onnx"]
REGISTRIES = ["torchvision", "timm", "monai", "hf", "plugin"]
DTYPES = ["float32", "float16", "bfloat16"]

_TENSOR = {
    "type": "object",
    "required": ["name"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "shape": {"type": "array", "items": {"type": ["integer", "null"], "minimum": 1},
                  "description": "null marks a dynamic dimension (e.g. batch)"},
        "dtype": {"type": "string", "enum": ["float32", "float16", "bfloat16", "uint8", "int64"]},
    },
}

SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://moregpu.dev/schemas/model-spec-v1.json",
    "title": "MoreGPU model spec v1",
    "description": "How a worker fetches, verifies, builds and runs a published vision model (ADR-0113).",
    "type": "object",
    "additionalProperties": False,
    "required": ["format"],
    "properties": {
        "version": {"const": 1},
        "format": {"type": "string", "enum": FORMATS},
        "source": {"type": "string", "pattern": "^(hf://[^/]+/.+|https://.+|pushed://[A-Za-z0-9._-]+|file:///.+)$",
                   "description": "hf://org/repo[@rev]/file, https://… (sha256 required), pushed://<id> (sha256 "
                                  "required) or file:///path under the worker's MOREGPU_MODEL_ROOTS"},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "arch": {
            "type": "object",
            "required": ["registry", "name"],
            "additionalProperties": False,
            "properties": {
                "registry": {"type": "string", "enum": REGISTRIES},
                "name": {"type": "string", "minLength": 1},
                "kwargs": {"type": "object"},
            },
        },
        "dtype": {"type": "string", "enum": DTYPES},
        "io": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "inputs": {"type": "array", "items": {"$ref": "#/$defs/tensor"}},
                "outputs": {"type": "array", "items": {"$ref": "#/$defs/tensor"}},
            },
        },
        "preprocess": {"type": "object"},
        "inference": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string", "enum": ["full", "sliding_window"]},
                "sliding_window": {
                    "type": "object",
                    "required": ["roi"],
                    "additionalProperties": False,
                    "properties": {
                        "roi": {"type": "array", "minItems": 1, "maxItems": 3,
                                "items": {"type": "integer", "minimum": 1}},
                        "overlap": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
                        "blend": {"type": "string", "enum": ["gaussian", "constant"]},
                        "sw_batch": {"type": "integer", "minimum": 1},
                    },
                },
                "tta": {"type": "string", "enum": ["none", "flip"]},
            },
            "if": {"properties": {"mode": {"const": "sliding_window"}}, "required": ["mode"]},
            "then": {"required": ["sliding_window"]},
        },
        "postprocess": {"type": "object"},
        "placement": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "device": {"type": "string", "enum": ["auto", "cpu", "cuda", "mps"]},
                "min_vram_gb": {"type": "number", "minimum": 0},
            },
        },
        "licence": {"type": "string"},
        "citation": {"type": "string"},
    },
    "allOf": [
        {"if": {"properties": {"source": {"pattern": "^(https|pushed)://"}}, "required": ["source"]},
         "then": {"required": ["sha256"]}},
        {"if": {"properties": {"format": {"enum": ["state_dict", "safetensors", "plugin"]}}},
         "then": {"required": ["arch"]}},
        {"if": {"properties": {"format": {"const": "plugin"}}},
         "then": {"properties": {"arch": {"properties": {"registry": {"const": "plugin"}}}}},
         "else": {"required": ["source"]}},
    ],
    "$defs": {"tensor": _TENSOR},
}

DEFAULTS = {"version": 1, "dtype": "float32"}
INFERENCE_DEFAULTS = {"mode": "full", "tta": "none"}
SLIDING_DEFAULTS = {"overlap": 0.25, "blend": "gaussian", "sw_batch": 1}


class SpecError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid model spec: " + "; ".join(errors))


# ------------------------------------------------------------------ validator (JSON Schema subset)
_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _resolve(ref: str, root: dict) -> dict:
    node: Any = root
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node


def check(value: Any, schema: dict, root: dict | None = None, ptr: str = "") -> list[str]:
    """Return a list of "<json-pointer>: message" errors (empty when valid)."""
    root = schema if root is None else root
    at = ptr or "/"
    errs: list[str] = []
    if "$ref" in schema:
        errs += check(value, _resolve(schema["$ref"], root), root, ptr)
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[t](value) for t in types):
            return errs + [f"{at}: expected {' or '.join(types)}, got {json.dumps(value)[:60]}"]
    if "const" in schema and value != schema["const"]:
        errs.append(f"{at}: must be {json.dumps(schema['const'])}")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{at}: {json.dumps(value)[:60]} not one of {schema['enum']}")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{at}: {value[:60]!r} does not match {schema['pattern']}")
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{at}: shorter than {schema['minLength']}")
    if _TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{at}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{at}: {value} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errs.append(f"{at}: {value} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errs.append(f"{at}: {value} >= exclusiveMaximum {schema['exclusiveMaximum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{at}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{at}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                errs += check(item, schema["items"], root, f"{ptr}/{i}")
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errs.append(f"{at}: missing required property {req!r}")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                errs += check(v, props[k], root, f"{ptr}/{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{at}: additional property {k!r} not allowed")
            elif isinstance(schema.get("additionalProperties"), dict):
                errs += check(v, schema["additionalProperties"], root, f"{ptr}/{k}")
    for sub in schema.get("allOf", []):
        errs += check(value, sub, root, ptr)
    if "anyOf" in schema and not any(not check(value, s, root, ptr) for s in schema["anyOf"]):
        errs.append(f"{at}: matches none of anyOf")
    if "oneOf" in schema and sum(not check(value, s, root, ptr) for s in schema["oneOf"]) != 1:
        errs.append(f"{at}: must match exactly one of oneOf")
    if "if" in schema:
        branch = schema.get("then") if not check(value, schema["if"], root, ptr) else schema.get("else")
        if branch:
            errs += check(value, branch, root, ptr)
    return errs


def validate(spec: Any) -> dict:
    """Validate a model spec and return a normalised deep copy with defaults filled in."""
    if not isinstance(spec, dict):
        raise SpecError(["/: a model spec must be a JSON object"])
    errs = check(spec, SCHEMA)
    if errs:
        raise SpecError(errs)
    out = {**DEFAULTS, **copy.deepcopy(spec)}
    if "arch" in out:
        out["arch"].setdefault("kwargs", {})
    inf = {**INFERENCE_DEFAULTS, **out.get("inference", {})}
    if inf["mode"] == "sliding_window":
        inf["sliding_window"] = {**SLIDING_DEFAULTS, **inf["sliding_window"]}
    out["inference"] = inf
    return out


def schema_json() -> str:
    return json.dumps(SCHEMA, indent=2) + "\n"


if __name__ == "__main__":  # pragma: no cover
    print(schema_json(), end="")
