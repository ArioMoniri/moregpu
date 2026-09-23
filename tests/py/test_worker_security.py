"""Worker security regressions: torch>=2.6 load guard (CVE-2025-32434), export path confinement, model https/hf
fetch policy, blob staging caps, bucket download caps + wildcards, .pt2 zip-bomb caps, unload/reset hygiene,
pushed:// models through the data-plane BlobStore, per-run telemetry host salt."""
import hashlib
import http.server
import importlib
import io
import json
import os
import re
import stat
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from moregpu_worker import paths as MP
from moregpu_worker.data import blobs as B
from moregpu_worker.data.blobs import BlobStore
from moregpu_worker.data.cache import ContentCache
from moregpu_worker.data.plane import DataPlane
from moregpu_worker.data.refs import DataPolicy, Ref, RefDenied
from moregpu_worker.train import registry as R
from moregpu_worker.train.runner import TaskRunner
from moregpu_worker.train.sessions import SessionStore
from moregpu_worker.train.task import TaskContext
from moregpu_worker.vision import adapters as A
from moregpu_worker.vision import fetch as FE
from moregpu_worker.vision import ops
from moregpu_worker.vision.adapters import base as AB
from moregpu_worker.vision.adapters import export as AE
from moregpu_worker.vision.infer import InferenceStore

from _vision_models import TinyNet, save_state_dict, spec_for, tiny_monai_unet

REPO = Path(__file__).resolve().parents[2]
SEG = {"kind": "2p5d", "num_classes": 3, "encoder": {"init": "random", "model": "micro", "patch": 8},
       "decoder": {"channels": [16, 8]}, "synthetic": {"kind": "2p5d", "n": 8, "size": [32, 32], "channels": 3, "seed": 0}}
JEPA = {"model": "micro", "patch": 8, "pred_dim": 16, "pred_depth": 1, "pred_heads": 2, "n_targets": 2,
        "synthetic": {"kind": "2p5d", "n": 8, "size": [32, 32], "channels": 3, "seed": 0, "classes": 3},
        "ema": [0.9, 1.0], "total_steps": 4, "weight_decay": 0.0}


def ctx():
    return TaskContext(device="cpu", amp="fp32", seed=0)


def h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ====================================================================== P0-4: torch >= 2.6 guard (CVE-2025-32434)
@pytest.mark.parametrize("v,want", [("2.6.0", (2, 6, 0, False)), ("2.5.1+cu121", (2, 5, 1, False)),
                                    ("2.10.0", (2, 10, 0, False)), ("2.14.0+cu130", (2, 14, 0, False)),
                                    ("2.6.0.dev20241112+cpu", (2, 6, 0, True)), ("2.6.0a0+git1234abc", (2, 6, 0, True)),
                                    ("2.7.0rc1", (2, 7, 0, True)), ("2.6", (2, 6, 0, False))])
def test_torch_version_parse(v, want):
    assert AB.parse_torch_version(v) == want


@pytest.mark.parametrize("v,ok", [("2.6.0", True), ("2.6.1+cu124", True), ("2.7.0a0+gitabc", True), ("3.0.0", True),
                                  ("2.5.1", False), ("2.5.1+cu121", False), ("2.6.0.dev20241112", False),
                                  ("2.6.0rc3", False), ("1.13.1", False), ("garbage", False), ("", False)])
def test_torch_load_safe_threshold(v, ok):
    assert AB.torch_load_is_safe(v) is ok


def test_state_dict_pt_refused_on_old_torch(tmp_path, monkeypatch):
    m = tiny_monai_unet(); p = tmp_path / "w.pt"; save_state_dict(m, p)
    arch = {"registry": "monai", "name": "UNet", "kwargs": {"spatial_dims": 3, "in_channels": 1, "out_channels": 2,
            "channels": [4, 8, 16], "strides": [2, 2], "num_res_units": 1}}
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu121")
    with pytest.raises(A.RefusedFormat, match=r"2\.6"):
        A.load(spec_for(p, "state_dict", arch), FE.make_fetch(roots=[tmp_path]))
    # safetensors never goes through torch.load: still allowed on an old torch
    from safetensors.torch import save_file
    st = tmp_path / "w.safetensors"
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(st))
    A.load(spec_for(st, "safetensors", arch), FE.make_fetch(roots=[tmp_path]))


def test_torchscript_and_pt2_refused_on_old_torch(tmp_path, monkeypatch):
    m = TinyNet().eval()
    ts = tmp_path / "m.ts"
    torch.jit.save(torch.jit.trace(m, torch.randn(1, 3, 8, 8)), str(ts))
    pt2 = tmp_path / "m.pt2"
    torch.export.save(torch.export.export(m, (torch.randn(1, 3, 8, 8),)), str(pt2))
    f = FE.make_fetch(roots=[tmp_path])
    monkeypatch.setattr(torch, "__version__", "2.6.0.dev20241112+cpu")
    for p, fmt in ((ts, "torchscript"), (pt2, "torch_export")):
        with pytest.raises(A.RefusedFormat, match=r"2\.6"):
            A.load(spec_for(p, fmt), f)
    monkeypatch.setattr(torch, "__version__", "2.6.0+cpu")
    A.load(spec_for(ts, "torchscript"), f)


def test_pyproject_pins_torch_2_6():
    txt = (REPO / "apps" / "worker" / "pyproject.toml").read_text()
    assert re.search(r'"torch>=2\.6(\.\d+)?"', txt)
    readme = (REPO / "README.md").read_text()
    assert "pip install 'torch>=2.6'" in readme or 'pip install "torch>=2.6"' in readme


# ====================================================================== P1-1: export confinement
def test_confine_relative_and_absolute(tmp_path):
    root = tmp_path / "out"; root.mkdir()
    assert MP.confine("a/b", [str(root)]) == str(root / "a" / "b")
    assert MP.confine(str(root / "x"), [str(root)]) == str(root / "x")
    assert MP.confine(str(root), [str(root)]) == str(root)
    for bad in ("../x", "a/../../x", str(tmp_path / "elsewhere"), "/etc/passwd", "/"):
        with pytest.raises(PermissionError):
            MP.confine(bad, [str(root)])
    with pytest.raises((PermissionError, ValueError)):
        MP.confine("a\x00b", [str(root)])
    with pytest.raises(PermissionError):
        MP.confine("x", [])


def test_confine_symlink_escape(tmp_path):
    root = tmp_path / "out"; root.mkdir(); outside = tmp_path / "secret"; outside.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(PermissionError):
        MP.confine("link/sub", [str(root)])
    with pytest.raises(PermissionError):
        MP.confine(str(root / "link"), [str(root)])


def test_confine_second_root(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"; a.mkdir(); b.mkdir()
    assert MP.confine(str(b / "m"), [str(a), str(b)]) == str(b / "m")
    assert MP.confine("m", [str(a), str(b)]) == str(a / "m")        # relative → first root


def test_output_root_default_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("MOREGPU_OUTPUT_DIR")
    monkeypatch.chdir(tmp_path)
    assert MP.output_root() == os.path.realpath(tmp_path / "moregpu-out")
    monkeypatch.setenv("MOREGPU_OUTPUT_DIR", str(tmp_path / "o"))
    assert MP.output_root() == os.path.realpath(tmp_path / "o")
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", os.pathsep.join([str(tmp_path / "m1"), str(tmp_path / "m2")]))
    assert MP.export_read_roots() == [os.path.realpath(tmp_path / "o"), os.path.realpath(tmp_path / "m1"),
                                      os.path.realpath(tmp_path / "m2")]


def test_jepa_export_confined(tmp_path, monkeypatch):
    out = tmp_path / "out"; monkeypatch.setenv("MOREGPU_OUTPUT_DIR", str(out))
    t = R.create("jepa_2p5d"); t.init(JEPA, ctx())
    ex = t.export("safetensors", "enc")                                  # relative → inside the output dir
    assert ex["weights"] == str(out / "enc" / "encoder.safetensors") and os.path.exists(ex["weights"])
    for bad in ("../enc", str(tmp_path / "enc2")):
        with pytest.raises(PermissionError):
            t.export("safetensors", bad)
    assert not (tmp_path / "enc").exists() and not (tmp_path / "enc2").exists()


def test_segment_export_and_encoder_read_confined(tmp_path, monkeypatch):
    out = tmp_path / "out"; monkeypatch.setenv("MOREGPU_OUTPUT_DIR", str(out))
    t = R.create("segment"); t.init(SEG, ctx())
    with pytest.raises(PermissionError):
        t.export("safetensors", str(tmp_path / "evil"))
    ex = t.export("safetensors", "seg")
    assert ex["weights"].startswith(str(out) + os.sep)
    # encoder init from an export: read confined to MOREGPU_OUTPUT_DIR ∪ MOREGPU_MODEL_ROOTS
    j = R.create("jepa_2p5d"); j.init(JEPA, ctx())
    j.export("safetensors", "enc")
    elsewhere = tmp_path / "elsewhere"
    import shutil
    shutil.copytree(out / "enc", elsewhere)
    cfg = {**SEG, "encoder": {"init": "export", "path": str(elsewhere)}}
    with pytest.raises(PermissionError):
        R.create("segment").init(cfg, ctx())
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(elsewhere))
    R.create("segment").init(cfg, ctx())
    R.create("segment").init({**SEG, "encoder": {"init": "export", "path": "enc"}}, ctx())   # relative → output dir


def test_finetune_model_export_confined(tmp_path, monkeypatch):
    models = tmp_path / "models"; models.mkdir()
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(models))
    monkeypatch.setenv("MOREGPU_OUTPUT_DIR", str(tmp_path / "out"))
    m = tiny_monai_unet(); p = models / "unet.pt"; save_state_dict(m, p)
    arch = {"registry": "monai", "name": "UNet", "kwargs": {"spatial_dims": 3, "in_channels": 1, "out_channels": 2,
            "channels": [4, 8, 16], "strides": [2, 2], "num_res_units": 1}}
    t = R.create("finetune_model")
    t.init({"spec": spec_for(p, "state_dict", arch), "objective": "segment", "num_classes": 2,
            "synthetic": {"kind": "3d", "n": 2, "size": [16, 16, 16], "channels": 1, "seed": 0}}, ctx())
    with pytest.raises(PermissionError):
        t.export("safetensors", str(models / "overwrite"))               # model roots are read-only for exports
    assert t.export("safetensors", "ft")["weights"] == str(tmp_path / "out" / "ft" / "model.safetensors")


def test_runner_task_export_confined(tmp_path, monkeypatch):
    out = tmp_path / "out"; monkeypatch.setenv("MOREGPU_OUTPUT_DIR", str(out))
    r = TaskRunner(SessionStore(), "cpu")
    r.handle("task_init", {"session": "s", "task": "segment", "cfg": SEG, "amp": "fp32"})
    for bad in ("../x", str(tmp_path / "x"), "/tmp/moregpu-evil-export"):
        with pytest.raises(PermissionError):
            r.handle("task_export", {"session": "s", "path": bad})
    (out).mkdir(parents=True, exist_ok=True)
    (out / "ln").symlink_to(tmp_path)
    with pytest.raises(PermissionError):
        r.handle("task_export", {"session": "s", "path": "ln/x"})
    got = r.handle("task_export", {"session": "s", "path": "seg"})
    assert got["ok"] and got["weights"] == str(out / "seg" / "model.safetensors")


def test_infer_load_export_read_confined(tmp_path, monkeypatch):
    out = tmp_path / "out"; monkeypatch.setenv("MOREGPU_OUTPUT_DIR", str(out))
    t = R.create("segment"); t.init(SEG, ctx())
    t.export("safetensors", "m")
    import shutil
    shutil.copytree(out / "m", tmp_path / "stray")
    st = InferenceStore(device="cpu")
    assert st.out_root == str(out)
    with pytest.raises(PermissionError):
        st.handle("vision_infer_load", {"id": "x", "export": str(tmp_path / "stray")})
    with pytest.raises(PermissionError):
        st.handle("vision_infer_load", {"id": "x", "export": "../stray"})
    assert st.handle("vision_infer_load", {"id": "a", "export": "m"})["ok"]
    assert st.handle("vision_infer_load", {"id": "b", "export": str(out / "m")})["ok"]
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(tmp_path / "stray"))
    assert st.handle("vision_infer_load", {"id": "c", "export": str(tmp_path / "stray")})["ok"]


def test_e2e_and_examples_keep_exports_inside_output_dir():
    """The e2e harness gives every worker a MOREGPU_OUTPUT_DIR that contains the export dirs the tests choose."""
    pool = (REPO / "tests" / "e2e" / "_pool.py").read_text()
    assert "MOREGPU_OUTPUT_DIR" in pool
    vp = (REPO / "tests" / "e2e" / "vision_pipeline.py").read_text()
    assert 'os.path.join(out, "models")' in vp
    cli = (REPO / "tests" / "e2e" / "cli_train.py").read_text()
    assert 'os.path.join(root, "out", "enc")' in cli and 'os.path.join(root, "out", "seg")' in cli
    mixed = (REPO / "tests" / "e2e" / "mixed_fleet_vision.py").read_text()
    assert "MOREGPU_MODEL_ROOTS" in mixed or "MOREGPU_OUTPUT_DIR" in mixed


# ====================================================================== P1-2: model https / hf fetch policy
class _H(http.server.BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self):  # noqa: N802
        code, body, headers = self.routes.get(self.path, (404, b"", {}))
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        if "Content-Length" not in headers and not headers.get("_nolen"):
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def model_env(monkeypatch):
    for k in ("MOREGPU_MODEL_HOSTS", "MOREGPU_DATA_HOSTS", "MOREGPU_MODEL_MAX_BYTES"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_model_http_needs_allowlisted_host(server, tmp_path, model_env):
    srv, base = server
    _H.routes = {"/w.pt": (200, b"weights", {})}
    f = FE.make_fetch(cache_dir=tmp_path / "c")
    with pytest.raises(A.RefusedSource, match="host"):
        f(f"{base}/w.pt", h(b"weights"))
    model_env.setenv("MOREGPU_DATA_HOSTS", "127.0.0.1")                  # fallback when MODEL_HOSTS is unset
    assert f(f"{base}/w.pt", h(b"weights")).read_bytes() == b"weights"
    model_env.setenv("MOREGPU_MODEL_HOSTS", "example.org")               # MODEL_HOSTS wins over DATA_HOSTS
    with pytest.raises(A.RefusedSource, match="host"):
        FE.make_fetch(cache_dir=tmp_path / "c2")(f"{base}/w.pt", h(b"weights"))


def test_model_http_redirect_rechecked(server, tmp_path, model_env):
    srv, base = server
    _H.routes = {"/r": (302, b"", {"Location": "http://evil.invalid/w.pt"}),
                 "/ok": (302, b"", {"Location": f"{base}/w.pt"}), "/w.pt": (200, b"w", {})}
    model_env.setenv("MOREGPU_MODEL_HOSTS", "127.0.0.1")
    f = FE.make_fetch(cache_dir=tmp_path / "c")
    with pytest.raises(A.RefusedSource):
        f(f"{base}/r", h(b"w"))
    assert f(f"{base}/ok", h(b"w")).read_bytes() == b"w"


def test_model_http_size_cap_and_integrity(server, tmp_path, model_env):
    srv, base = server
    big = b"x" * 5000
    _H.routes = {"/big": (200, big, {}), "/w.pt": (200, b"w", {})}
    model_env.setenv("MOREGPU_MODEL_HOSTS", "127.0.0.1")
    model_env.setenv("MOREGPU_MODEL_MAX_BYTES", "1000")
    f = FE.make_fetch(cache_dir=tmp_path / "c")
    with pytest.raises(A.RefusedSource, match="cap"):
        f(f"{base}/big", h(big))
    assert not any(p.name.startswith(".") for p in (tmp_path / "c").iterdir())   # no partial file left behind
    with pytest.raises(A.RefusedSource, match="sha256"):
        f(f"{base}/w.pt")
    with pytest.raises(A.IntegrityError):
        f(f"{base}/w.pt", "1" * 64)
    assert FE.model_max_bytes() == 1000
    model_env.delenv("MOREGPU_MODEL_MAX_BYTES")
    assert FE.model_max_bytes() == 20 * 1024 ** 3


def test_model_plain_http_only_for_loopback(tmp_path, model_env):
    model_env.setenv("MOREGPU_MODEL_HOSTS", "example.org")
    with pytest.raises(A.RefusedSource):
        FE.make_fetch(cache_dir=tmp_path)("http://example.org/w.pt", "1" * 64)


def test_hf_requires_sha_or_commit(tmp_path, monkeypatch):
    target = tmp_path / "m.safetensors"; target.write_bytes(b"x")
    got = {}

    def fake(repo_id, filename, revision=None):
        got.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(target)

    monkeypatch.setattr(FE, "_hf_hub_download", fake)
    f = FE.make_fetch()
    for src in ("hf://org/repo/m.safetensors", "hf://org/repo@main/m.safetensors", "hf://org/repo@v1.0/m.safetensors",
                "hf://org/repo@abc123/m.safetensors"):
        with pytest.raises(A.RefusedSource, match="revision|sha256"):
            f(src)
    commit = "0123456789abcdef0123456789abcdef01234567"
    assert f(f"hf://org/repo@{commit}/m.safetensors") == target and got["revision"] == commit
    assert f("hf://org/repo@main/m.safetensors", h(b"x")) == target      # a pinned sha256 makes a mutable ref OK
    assert f("hf://org/repo/m.safetensors", h(b"x")) == target


# ====================================================================== P1-3 is in tests/security/release_verify.py
# ====================================================================== P2: telemetry host salt
def test_host_hash_salt_is_per_run():
    from moregpu_worker.telemetry import hw
    a = hw.host_hash("box")
    assert a == hw.host_hash("box")
    importlib.reload(hw)
    assert hw.host_hash("box") != a
    assert "per-run" in (hw.__doc__ or "") or "per run" in (hw.__doc__ or "")


# ====================================================================== P2: blob staging caps
def _push(bs, id, data):
    bs.begin(id, h(data), len(data)); bs.chunk(id, 0, data); return bs.end(id)


def test_blob_total_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("MOREGPU_BLOB_TOTAL_MAX_BYTES", "100")
    bs = BlobStore(stage_dir=tmp_path)
    assert bs.total_max_bytes == 100
    _push(bs, "a", b"x" * 60)
    with pytest.raises(ValueError, match="MOREGPU_BLOB_TOTAL_MAX_BYTES"):
        bs.begin("b", h(b"y" * 50), 50)
    _push(bs, "a", b"z" * 90)                                            # restarting the same id replaces it
    bs.drop("a")
    _push(bs, "b", b"y" * 50)
    monkeypatch.delenv("MOREGPU_BLOB_TOTAL_MAX_BYTES")
    assert BlobStore(stage_dir=tmp_path).total_max_bytes == 40 * 1024 ** 3


def test_blob_refused_when_stage_has_no_room(tmp_path, monkeypatch):
    bs = BlobStore(stage_dir=tmp_path)
    real = B.shutil.disk_usage
    monkeypatch.setattr(B.shutil, "disk_usage", lambda p: real(p)._replace(free=10))
    with pytest.raises(ValueError, match="free"):
        bs.begin("a", h(b"x" * 100), 100)


def test_stage_root_skips_shm_without_room(monkeypatch):
    monkeypatch.delenv("MOREGPU_STAGE_DIR", raising=False)
    monkeypatch.delenv("MOREGPU_PUSHED_DIR", raising=False)
    real = B.shutil.disk_usage
    monkeypatch.setattr(B.shutil, "disk_usage",
                        lambda p: real(p)._replace(free=3 * 1024 ** 3) if str(p) == "/dev/shm" else real(p))
    assert B.stage_root(1024) in ("/dev/shm", B.tempfile.gettempdir())
    assert B.stage_root(5 * 1024 ** 3) != "/dev/shm"


def test_pushed_dir_is_alias_for_stage_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("MOREGPU_STAGE_DIR", raising=False)
    monkeypatch.setenv("MOREGPU_PUSHED_DIR", str(tmp_path / "p"))
    assert B.stage_root() == str(tmp_path / "p")


# ====================================================================== P2: pushed:// models via the BlobStore
def test_pushed_model_reads_blob_store(tmp_path):
    bs = BlobStore(stage_dir=tmp_path / "stage")
    p = _push(bs, "w.pt", b"weights")
    f = FE.make_fetch(blobs=bs)
    assert f("pushed://w.pt", h(b"weights")) == p
    with pytest.raises(A.IntegrityError):
        f("pushed://w.pt", "1" * 64)
    bs.begin("half", h(b"ab"), 2); bs.chunk("half", 0, b"a")
    for bad in ("pushed://half", "pushed://missing", "pushed://../etc/passwd", "pushed://a/b"):
        with pytest.raises(A.RefusedSource):
            f(bad, "0" * 64)
    # a file merely placed in a directory is NOT a pushed blob
    (tmp_path / "planted").write_bytes(b"x")
    with pytest.raises(A.RefusedSource):
        FE.make_fetch(blobs=BlobStore(stage_dir=tmp_path))("pushed://planted", h(b"x"))


def test_default_fetch_and_data_plane_share_one_blob_store(tmp_path):
    plane = DataPlane(DataPolicy(roots=[str(tmp_path)]), cache=ContentCache(tmp_path / "c", 1 << 20))
    assert plane.blobs is B.default_store()
    p = _push(plane.blobs, "model-x.safetensors", b"abc")
    try:
        assert FE.default_fetch("pushed://model-x.safetensors", h(b"abc")) == p
    finally:
        plane.blobs.drop("model-x.safetensors")


# ====================================================================== P2: bucket downloads
def _fake_stream_tool(bindir, name, nbytes):
    p = bindir / name
    p.write_text(f"#!/bin/sh\necho \"$@\" > \"{bindir}/{name}.args\"\n"
                 f"for a; do last=\"$a\"; done\n"
                 f"if [ \"$last\" = \"-\" ]; then head -c {nbytes} /dev/zero; else head -c {nbytes} /dev/zero > \"$last\"; fi\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


@pytest.mark.parametrize("scheme,tool", [("s3", "s5cmd"), ("gs", "gsutil")])
def test_bucket_download_capped(tmp_path, monkeypatch, scheme, tool):
    bindir = tmp_path / "bin"; bindir.mkdir()
    _fake_stream_tool(bindir, tool, 50_000)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    plane = DataPlane(DataPolicy(roots=[str(tmp_path)], allow_buckets=True, max_download_bytes=1000),
                      cache=ContentCache(tmp_path / "c", 1 << 30))
    with pytest.raises(RefDenied, match="cap"):
        plane.resolve(Ref(f"{scheme}://pub-bucket/k/a.npy", sha256=h(b"\0" * 50_000)))
    assert not any(p.name.startswith(".dl-") for p in (tmp_path / "c").iterdir())


@pytest.mark.parametrize("uri", ["s3://pub-bucket/k/*.npy", "s3://pub-bucket/k/a?.npy", "gs://pub-bucket/**",
                                 "gs://pub-bucket/k/[ab].npy"])
def test_bucket_wildcards_refused(tmp_path, monkeypatch, uri):
    bindir = tmp_path / "bin"; bindir.mkdir()
    for t in ("s5cmd", "gsutil"):
        _fake_stream_tool(bindir, t, 10)
    monkeypatch.setenv("PATH", str(bindir))
    plane = DataPlane(DataPolicy(roots=[str(tmp_path)], allow_buckets=True), cache=ContentCache(tmp_path / "c", 1 << 20))
    with pytest.raises(RefDenied, match="wildcard"):
        plane.resolve(Ref(uri, sha256="0" * 64))
    assert not (bindir / "s5cmd.args").exists() and not (bindir / "gsutil.args").exists()


# ====================================================================== P2: .pt2 zip bomb
def _good_pt2(tmp_path):
    p = tmp_path / "good.pt2"
    torch.export.save(torch.export.export(TinyNet().eval(), (torch.randn(1, 3, 8, 8),)), str(p))
    return p


def test_pt2_member_and_total_caps(tmp_path, monkeypatch):
    good = _good_pt2(tmp_path)
    AE.check_pt2(good)
    bomb = tmp_path / "bomb.pt2"
    with zipfile.ZipFile(good) as zi, zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as zo:
        for n in zi.namelist():
            zo.writestr(n, zi.read(n))
        zo.writestr("model/extra/blob.bin", b"\0" * (8 << 20))          # 8 MiB of zeros, compresses to ~8 KiB
    assert bomb.stat().st_size < 1 << 20
    monkeypatch.setenv("MOREGPU_PT2_MEMBER_MAX_BYTES", str(4 << 20))
    with pytest.raises(A.RefusedFormat, match="member"):
        AE.check_pt2(bomb)
    monkeypatch.delenv("MOREGPU_PT2_MEMBER_MAX_BYTES")
    monkeypatch.setenv("MOREGPU_PT2_TOTAL_MAX_BYTES", str(6 << 20))
    with pytest.raises(A.RefusedFormat, match="total"):
        AE.check_pt2(bomb)


def test_pt2_does_not_read_whole_members(tmp_path, monkeypatch):
    """Non-pickle members are sniffed (magic bytes) through a stream, never z.read() in full."""
    good = _good_pt2(tmp_path)
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda *a, **k: pytest.fail("ZipFile.read used"))
    AE.check_pt2(good)


# ====================================================================== P2: unload / reset hygiene
def test_vision_unload_drops_handle_and_inference_store_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(tmp_path))
    ops.HANDLES.clear(); ops.LOWERED.clear()
    m = TinyNet().eval(); p = tmp_path / "m.ts"
    torch.jit.save(torch.jit.trace(m, torch.randn(1, 3, 8, 8)), str(p))
    st = InferenceStore(device="cpu")
    ops.handle("vision_load", {"id": "pub", "spec": spec_for(p, "torchscript")})
    st.handle("vision_infer_load", {"id": "pub", "handle": "pub", "task": "segment", "num_classes": 4, "kind": "2d"})
    st.handle("vision_infer_load", {"id": "alias", "handle": "pub", "task": "segment", "num_classes": 4, "kind": "2d"})
    assert set(st.models) == {"pub", "alias"}
    ops.handle("vision_unload", {"id": "pub"})
    assert "pub" not in ops.HANDLES and st.models == {}


def test_ops_reset_clears_handles_and_lowered(tmp_path, monkeypatch):
    monkeypatch.setenv("MOREGPU_MODEL_ROOTS", str(tmp_path))
    m = TinyNet().eval(); p = tmp_path / "m.ts"
    torch.jit.save(torch.jit.trace(m, torch.randn(1, 3, 8, 8)), str(p))
    st = InferenceStore(device="cpu")
    ops.handle("vision_load", {"id": "pub", "spec": spec_for(p, "torchscript")})
    st.handle("vision_infer_load", {"id": "pub", "handle": "pub", "task": "segment", "num_classes": 4, "kind": "2d"})
    ops.LOWERED[("pub", "x")] = object()
    ops.reset()
    assert ops.HANDLES == {} and ops.LOWERED == {} and st.models == {}


def test_welcome_reset_clears_vision_state_and_blob_staging():
    src = (REPO / "apps" / "worker" / "worker_torch.py").read_text()
    block = src[src.index("def _reset_session():"):src.index("await loop.run_in_executor(TORCH_POOL, _reset_session)")]
    for needle in ("VISION.models.clear()", "MODEL_OPS.reset()", "DATA.blobs.close()"):
        assert needle in block, needle


# ====================================================================== docs match the implementation
def test_docs_describe_the_controls():
    models = (REPO / "docs" / "MODELS.md").read_text()
    for needle in ("MOREGPU_MODEL_HOSTS", "MOREGPU_MODEL_MAX_BYTES", "MOREGPU_OUTPUT_DIR", "torch>=2.6", "40-hex"):
        assert needle in models, needle
    vision = (REPO / "docs" / "VISION.md").read_text()
    assert "Blobs are never persisted" not in vision and "MOREGPU_BLOB_TOTAL_MAX_BYTES" in vision
