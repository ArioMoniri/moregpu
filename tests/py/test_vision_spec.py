"""Model spec v1 (ADR-0113): JSON Schema + stdlib validator."""
import json
from pathlib import Path

import pytest

from moregpu_worker.vision import spec as S

ROOT = Path(__file__).resolve().parents[2]
SHA = "ab" * 32


def ok(**kw):
    base = {"format": "state_dict", "source": "https://example.org/w.pt", "sha256": SHA,
            "arch": {"registry": "monai", "name": "UNet", "kwargs": {"spatial_dims": 3}}}
    base.update(kw)
    return base


def test_docs_schema_is_the_shipped_schema():
    doc = json.loads((ROOT / "docs" / "model-spec.schema.json").read_text())
    assert doc == S.SCHEMA
    assert S.SCHEMA["$schema"].startswith("https://json-schema.org/")
    assert json.loads(S.schema_json()) == S.SCHEMA


def test_minimal_valid_spec_gets_defaults():
    s = S.validate(ok())
    assert s["dtype"] == "float32"
    assert s["inference"] == {"mode": "full", "tta": "none"}
    assert s["arch"]["kwargs"] == {"spatial_dims": 3}
    assert s["version"] == 1


def test_full_spec_valid():
    s = S.validate(ok(
        dtype="float16",
        io={"inputs": [{"name": "image", "shape": [None, 1, 96, 96, 96], "dtype": "float32"}],
            "outputs": [{"name": "logits", "shape": [None, 2, 96, 96, 96]}]},
        preprocess={"intensity": {"a_min": -200, "a_max": 300}},
        inference={"mode": "sliding_window", "tta": "flip",
                   "sliding_window": {"roi": [96, 96, 96], "overlap": 0.5, "blend": "gaussian", "sw_batch": 4}},
        postprocess={"activation": "softmax", "argmax": True},
        placement={"device": "auto", "min_vram_gb": 4},
        licence="Apache-2.0", citation="Example et al. 2024"))
    assert s["inference"]["sliding_window"]["blend"] == "gaussian"


@pytest.mark.parametrize("bad, needle", [
    ({"format": "pickle"}, "format"),
    ({"source": "ftp://x/y"}, "source"),
    ({"sha256": "XYZ"}, "sha256"),
    ({"dtype": "float64"}, "dtype"),
    ({"arch": {"registry": "keras", "name": "x"}}, "registry"),
    ({"arch": {"registry": "monai"}}, "name"),
    ({"inference": {"mode": "tiled"}}, "mode"),
    ({"inference": {"mode": "sliding_window"}}, "sliding_window"),
    ({"inference": {"mode": "sliding_window", "sliding_window": {"roi": [8], "overlap": 1.5}}}, "overlap"),
    ({"inference": {"mode": "full", "tta": "rot90"}}, "tta"),
    ({"io": {"inputs": [{"shape": [1]}]}}, "name"),
    ({"io": {"inputs": [{"name": "x", "shape": ["a"]}]}}, "shape"),
    ({"unknown_field": 1}, "unknown_field"),
    ({"licence": 3}, "licence"),
])
def test_invalid_specs_are_rejected_with_a_pointer(bad, needle):
    with pytest.raises(S.SpecError) as e:
        S.validate(ok(**bad))
    assert needle in str(e.value)
    assert e.value.errors


def test_sha256_required_for_https_and_pushed():
    for src in ("https://example.org/w.pt", "pushed://blob-123"):
        s = ok(source=src)
        del s["sha256"]
        with pytest.raises(S.SpecError, match="sha256"):
            S.validate(s)


def test_sha256_optional_for_hf_and_file():
    for src in ("hf://org/repo/model.safetensors", "file:///models/w.pt"):
        s = ok(source=src, format="safetensors")
        del s["sha256"]
        S.validate(s)


def test_arch_required_for_weights_only_formats():
    s = ok()
    del s["arch"]
    with pytest.raises(S.SpecError, match="arch"):
        S.validate(s)


def test_export_and_onnx_do_not_need_arch():
    for fmt in ("torch_export", "torchscript", "onnx"):
        S.validate({"format": fmt, "source": "https://e.org/m", "sha256": SHA})


def test_plugin_format_needs_plugin_registry_source_optional():
    S.validate({"format": "plugin", "arch": {"registry": "plugin", "name": "my_seg"}})
    with pytest.raises(S.SpecError):
        S.validate({"format": "plugin", "arch": {"registry": "timm", "name": "x"}})
    with pytest.raises(S.SpecError, match="source"):
        S.validate({"format": "onnx"})


def test_not_a_dict():
    with pytest.raises(S.SpecError):
        S.validate(["format"])


def test_validator_core_keywords():
    sch = {"type": "object", "properties": {
        "n": {"type": "integer", "minimum": 1, "maximum": 3},
        "x": {"type": ["number", "null"], "exclusiveMinimum": 0},
        "l": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "boolean"}},
        "c": {"const": "k"},
        "s": {"type": "string", "minLength": 2},
        "a": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        "o": {"oneOf": [{"type": "integer"}, {"type": "number"}]},
        "r": {"$ref": "#/$defs/pos"},
        "m": {"type": "object", "additionalProperties": {"type": "integer"}},
    }, "$defs": {"pos": {"type": "integer", "minimum": 0}}}
    assert S.check({"n": 2, "x": None, "l": [True], "c": "k", "s": "ab", "a": 1, "o": 1.5, "r": 0, "m": {"q": 1}}, sch) == []
    errs = S.check({"n": 0, "x": 0, "l": [], "c": "z", "s": "a", "a": [], "o": 1, "r": -1, "m": {"q": "no"}}, sch)
    for key in ("/n", "/x", "/l", "/c", "/s", "/a", "/o", "/r", "/m/q"):
        assert any(e.startswith(key + ":") for e in errs), (key, errs)
    assert S.check(True, {"type": "integer"})  # bool is not an integer
    assert S.check([1, "x"], {"type": "array", "items": {"type": "integer"}, "maxItems": 1})
