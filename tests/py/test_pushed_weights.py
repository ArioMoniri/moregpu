"""Encoder / model weights initialised from a ``pushed://<id>`` BlobStore blob (in addition to file paths).

Same rules as file sources: sha256 checked before anything is opened, safetensors only (a pickle is refused before it
is parsed), the MOREGPU_MODEL_MAX_BYTES size cap, and MOREGPU_MODEL_ROOTS / MOREGPU_OUTPUT_DIR confinement for paths."""
import hashlib
import io
import json

import pytest
import torch
from safetensors.torch import save as st_save, save_file

from moregpu_worker import paths
from moregpu_worker.data import blobs as B
from moregpu_worker.train import registry as R
from moregpu_worker.train.task import TaskContext
from moregpu_worker.vision import weights as W
from moregpu_worker.vision.errors import IntegrityError, RefusedFormat, RefusedSource

SEG_SYN = {"kind": "2p5d", "n": 8, "size": [32, 32], "channels": 3, "seed": 0}
SEG = {"kind": "2p5d", "num_classes": 3, "decoder": {"channels": [16, 8]}, "synthetic": SEG_SYN, "mode": "full"}
JEPA = {"model": "micro", "patch": 8, "pred_dim": 16, "pred_depth": 1, "pred_heads": 2, "synthetic": SEG_SYN}


def ctx():
    return TaskContext(device="cpu", amp="fp32", seed=0)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh process-wide BlobStore (what /data/push fills on a real worker)."""
    s = B.BlobStore(stage_dir=tmp_path / "stage")
    monkeypatch.setattr(B, "_DEFAULT", s)
    yield s
    s.close()


def push(store, bid: str, data: bytes, suffix: str = ".safetensors") -> str:
    store.begin(bid, sha(data), len(data), suffix)
    for k, off in enumerate(range(0, max(1, len(data)), 1 << 16)):
        store.chunk(bid, k, data[off:off + (1 << 16)])
    store.end(bid)
    return f"pushed://{bid}"


@pytest.fixture
def jepa_export(tmp_path):
    j = R.create("jepa_2p5d")
    j.init(JEPA, ctx())
    ex = j.export("safetensors", str(tmp_path / "jepa"))
    return j, ex


# ---------------------------------------------------------------- segment/classify encoder from pushed://
def test_jepa_export_is_self_describing(jepa_export):
    j, ex = jepa_export
    from safetensors import safe_open
    with safe_open(ex["weights"], framework="pt") as f:
        meta = f.metadata()
    cfg = json.loads(meta["moregpu.encoder_config"])
    assert cfg == json.loads(open(ex["config"]).read())


def test_segment_encoder_from_pushed_blob(store, jepa_export):
    j, ex = jepa_export
    data = open(ex["weights"], "rb").read()
    uri = push(store, "enc-1", data)
    t = R.create("segment")
    info = t.init({**SEG, "encoder": {"init": "export", "path": uri, "sha256": sha(data)}}, ctx())
    for k, v in j.target.state_dict().items():
        assert torch.equal(v, t.encoder.state_dict()[k])
    assert info["encoder"] == {"source": "pushed", "sha256": sha(data)}
    assert t.inner_steps([0, 1], 1, 1e-3).losses


def test_classify_encoder_from_pushed_blob_with_explicit_config_and_source_key(store, tmp_path, jepa_export):
    j, ex = jepa_export
    sd = {k: v.contiguous() for k, v in j.target.state_dict().items()}
    data = st_save(sd)                                        # no metadata: config comes from the task cfg
    uri = push(store, "enc-2", data)
    cfg = json.loads(open(ex["config"]).read())
    t = R.create("classify")
    t.init({**SEG, "encoder": {"init": "export", "source": uri, "sha256": sha(data), "config": cfg}}, ctx())
    assert torch.equal(t.encoder.state_dict()["patch_embed.proj.weight"], sd["patch_embed.proj.weight"])
    with pytest.raises(ValueError, match="encoder config"):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": uri, "sha256": sha(data)}}, ctx())


def test_pushed_encoder_sha_mismatch_is_refused(store, jepa_export):
    _, ex = jepa_export
    data = open(ex["weights"], "rb").read()
    uri = push(store, "enc-3", data)
    with pytest.raises(IntegrityError):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": uri, "sha256": "0" * 64}}, ctx())


def test_pushed_encoder_needs_a_sha256(store, jepa_export):
    _, ex = jepa_export
    uri = push(store, "enc-4", open(ex["weights"], "rb").read())
    with pytest.raises(RefusedSource, match="sha256"):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": uri}}, ctx())


def _pickles(j):
    buf = io.BytesIO()
    torch.save(j.target.state_dict(), buf)                    # torch zip + pickle
    import pickle
    return [buf.getvalue(), pickle.dumps({"a": 1}, protocol=2), pickle.dumps([1, 2], protocol=0)]


def test_pushed_encoder_pickle_is_refused_before_it_is_parsed(store, jepa_export, monkeypatch):
    j, _ = jepa_export
    opened = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: opened.append(a))
    for i, data in enumerate(_pickles(j)):
        uri = push(store, f"pk-{i}", data)
        with pytest.raises(RefusedFormat, match="safetensors"):
            R.create("segment").init({**SEG, "encoder": {"init": "export", "path": uri, "sha256": sha(data)}}, ctx())
    assert opened == []


@pytest.mark.parametrize("junk", [b"", b"\x01\x00", b"\xff" * 8 + b"{}", (10).to_bytes(8, "little") + b"not json!!"])
def test_pushed_encoder_non_safetensors_is_refused(store, junk):
    uri = push(store, "junk", junk)
    with pytest.raises(RefusedFormat):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": uri, "sha256": sha(junk)}}, ctx())


def test_pushed_encoder_unknown_ref_is_refused(store):
    with pytest.raises(RefusedSource, match="not pushed|was not pushed"):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": "pushed://nope", "sha256": "a" * 64}}, ctx())
    with pytest.raises(RefusedSource):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": "pushed://../x", "sha256": "a" * 64}}, ctx())


def test_pushed_encoder_incomplete_blob_is_unknown(store):
    store.begin("half", "b" * 64, 10, "")
    store.chunk("half", 0, b"12345")
    with pytest.raises(RefusedSource):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": "pushed://half", "sha256": "b" * 64}}, ctx())


def test_pushed_encoder_oversized_blob_is_refused(store, jepa_export, monkeypatch):
    _, ex = jepa_export
    data = open(ex["weights"], "rb").read()
    uri = push(store, "big", data)
    monkeypatch.setenv("MOREGPU_MODEL_MAX_BYTES", str(len(data) - 1))
    with pytest.raises(RefusedSource, match="MOREGPU_MODEL_MAX_BYTES"):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": uri, "sha256": sha(data)}}, ctx())
    # and a push over the staging cap never gets staged at all
    small = B.BlobStore(stage_dir=store.stage_dir, max_bytes=16)
    with pytest.raises(ValueError, match="MOREGPU_PUSH_MAX_BYTES"):
        small.begin("big2", sha(data), len(data), ".safetensors")


# ---------------------------------------------------------------- file paths keep their confinement + gain the checks
def test_file_encoder_confinement_still_applies(tmp_path, monkeypatch, jepa_export):
    _, ex = jepa_export                                        # written under MOREGPU_OUTPUT_DIR (tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    import shutil
    shutil.copytree(tmp_path / "jepa", outside / "jepa")
    with pytest.raises(paths.ConfinementError):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": str(outside / "jepa")}}, ctx())
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(outside))
    t = R.create("segment")
    info = t.init({**SEG, "encoder": {"init": "export", "path": str(outside / "jepa")}}, ctx())
    assert info["encoder"]["source"] == "file" and info["encoder"]["sha256"] == ex["sha256"]


def test_file_encoder_sha_is_checked_when_given(tmp_path, jepa_export):
    _, ex = jepa_export
    t = R.create("segment")
    t.init({**SEG, "encoder": {"init": "export", "path": str(tmp_path / "jepa"), "sha256": ex["sha256"]}}, ctx())
    with pytest.raises(IntegrityError):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": str(tmp_path / "jepa"), "sha256": "1" * 64}}, ctx())


def test_file_encoder_pickle_and_size_cap(tmp_path, monkeypatch, jepa_export):
    j, ex = jepa_export
    monkeypatch.setenv("MOREGPU_MODEL_MAX_BYTES", "64")
    with pytest.raises(RefusedSource, match="MOREGPU_MODEL_MAX_BYTES"):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": str(tmp_path / "jepa")}}, ctx())
    monkeypatch.delenv("MOREGPU_MODEL_MAX_BYTES")
    open(ex["weights"], "wb").write(_pickles(j)[0])
    with pytest.raises(RefusedFormat):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": str(tmp_path / "jepa")}}, ctx())


def test_unknown_encoder_scheme_is_refused():
    with pytest.raises(RefusedSource):
        R.create("segment").init({**SEG, "encoder": {"init": "export", "path": "https://example.org/e.safetensors",
                                                      "sha256": "a" * 64}}, ctx())


# ---------------------------------------------------------------- the weights module directly
def test_resolve_pushed_returns_the_verified_path(store):
    data = st_save({"w": torch.ones(2)})
    push(store, "w1", data)
    p = W.resolve_pushed("pushed://w1", sha(data))
    assert p.read_bytes() == data
    W.check_safetensors(p)


def test_resolve_pushed_rehashes_the_staged_file(store):
    """A staged file changed after blob_end (disk tampering) is caught by the re-hash before it is opened."""
    data = st_save({"w": torch.ones(2)})
    push(store, "w2", data)
    store.path("w2").write_bytes(st_save({"w": torch.zeros(2)}))
    with pytest.raises(IntegrityError):
        W.resolve_pushed("pushed://w2", sha(data))


# ---------------------------------------------------------------- finetune_model: spec.source may be pushed://
ARCH = {"registry": "monai", "name": "UNet", "kwargs": {"spatial_dims": 3, "in_channels": 1, "out_channels": 2,
        "channels": [4, 8, 16], "strides": [2, 2], "num_res_units": 1}}
FT_SYN = {"kind": "3d", "n": 2, "size": [16, 16, 16], "channels": 1, "seed": 0}


def _unet():
    pytest.importorskip("monai")
    from _vision_models import tiny_monai_unet
    return tiny_monai_unet()


def _ft(spec):
    return {"spec": spec, "objective": "segment", "num_classes": 2, "synthetic": FT_SYN, "label_map": {"2": 1}}


def test_finetune_model_from_pushed_safetensors(store):
    m = _unet()
    data = st_save({k: v.contiguous() for k, v in m.state_dict().items()})
    uri = push(store, "unet", data)
    t = R.create("finetune_model")
    t.init(_ft({"format": "safetensors", "source": uri, "sha256": sha(data), "arch": ARCH}), ctx())
    for k, v in m.state_dict().items():
        assert torch.equal(v, t.model.state_dict()[k])
    assert t.inner_steps([0, 1], 1, 1e-3).losses


def test_finetune_model_pushed_refusals(store, monkeypatch):
    m = _unet()
    data = st_save({k: v.contiguous() for k, v in m.state_dict().items()})
    uri = push(store, "unet2", data)
    with pytest.raises(IntegrityError):
        R.create("finetune_model").init(_ft({"format": "safetensors", "source": uri, "sha256": "0" * 64, "arch": ARCH}), ctx())
    with pytest.raises(RefusedSource):
        R.create("finetune_model").init(_ft({"format": "safetensors", "source": "pushed://missing", "sha256": sha(data),
                                             "arch": ARCH}), ctx())
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); pk = buf.getvalue()
    puri = push(store, "unet-pk", pk)
    for fmt in ("state_dict", "safetensors"):              # a pushed pickle is refused whatever the spec calls it
        with pytest.raises(RefusedFormat, match="safetensors"):
            R.create("finetune_model").init(_ft({"format": fmt, "source": puri, "sha256": sha(pk), "arch": ARCH}), ctx())
    monkeypatch.setenv("MOREGPU_MODEL_MAX_BYTES", "100")
    with pytest.raises(RefusedSource, match="MOREGPU_MODEL_MAX_BYTES"):
        R.create("finetune_model").init(_ft({"format": "safetensors", "source": uri, "sha256": sha(data), "arch": ARCH}), ctx())


def test_finetune_model_file_source_still_confined(tmp_path, monkeypatch):
    m = _unet()
    p = tmp_path / "unet.safetensors"
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(p))
    spec = {"format": "safetensors", "source": p.resolve().as_uri(), "sha256": sha(p.read_bytes()), "arch": ARCH}
    with pytest.raises(RefusedSource, match="MOREGPU_MODEL_ROOTS"):
        R.create("finetune_model").init(_ft(spec), ctx())
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(tmp_path))
    R.create("finetune_model").init(_ft(spec), ctx())
