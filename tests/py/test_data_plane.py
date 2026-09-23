"""DataPlane: read/slice, manifests, load_batch, https download, bucket tools, pushed blobs, stats (ADR-0110)."""
import hashlib
import http.server
import json
import os
import stat
import threading

import numpy as np
import pytest
import torch

from moregpu_worker.data.blobs import BlobStore
from moregpu_worker.data.cache import ContentCache
from moregpu_worker.data.manifest import Manifest
from moregpu_worker.data.plane import DataPlane
from moregpu_worker.data.refs import DataPolicy, IntegrityError, Ref, RefDenied


def h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"; r.mkdir()
    return r


@pytest.fixture
def plane(root, tmp_path):
    return DataPlane(DataPolicy(roots=[str(root)]), cache=ContentCache(tmp_path / "cache", 1 << 24),
                     blobs=BlobStore(stage_dir=tmp_path / "stage"))


# ---------------------------------------------------------------- read + slice
def test_read_npy_with_slice_is_memmap_view(plane, root):
    vol = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    np.save(root / "v.npy", vol)
    out = plane.read(Ref("file://v.npy", slice=(1, 4)))
    assert isinstance(out, np.memmap) and out.shape == (3, 4, 3)
    assert np.array_equal(out, vol[1:4])
    s = plane.stats()
    assert s["reads"] == 1 and s["bytes_read"] == 3 * 4 * 3 * 4 and s["read_seconds"] >= 0


def test_read_fmt_from_meta(plane, root):
    np.savez(root / "z.npz", a=np.zeros(2), b=np.ones(3))
    assert np.array_equal(plane.read(Ref("file://z.npz", meta={"fmt": "b"})), np.ones(3))


def test_read_slice_out_of_range(plane, root):
    np.save(root / "v.npy", np.zeros((3, 2)))
    with pytest.raises(ValueError):
        plane.read(Ref("file://v.npy", slice=(2, 9)))


# ---------------------------------------------------------------- manifests
def _refs_jsonl(refs):
    return "\n".join(json.dumps(r) for r in refs) + "\n"


def test_manifest_parse_and_hash_stability():
    a = _refs_jsonl([{"uri": "file://a.npy", "slice": [0, 3]}, {"meta": {"y": 1}, "uri": "file://b.npy"}])
    # same refs, different key order / whitespace / blank lines -> same canonical hash
    b = '\n{ "uri" : "file://a.npy",  "slice": [0,3], "meta": {} }\n\n{"uri":"file://b.npy","meta":{"y":1}}'
    m1, m2 = Manifest(a), Manifest.from_jsonl(b)
    assert len(m1) == 2 and m1[0].slice == (0, 3) and m1[1].meta == {"y": 1}
    assert m1.sha256 == m2.sha256 and len(m1.sha256) == 64
    assert m1.sha256 == hashlib.sha256(m1.to_jsonl().encode()).hexdigest()
    assert Manifest(_refs_jsonl([{"uri": "file://c.npy"}])).sha256 != m1.sha256
    assert list(m1)[0].uri == "file://a.npy"
    assert m1[-1].uri == "file://b.npy"


def test_manifest_from_refs_and_bytes():
    m = Manifest([Ref("file://a.npy")])
    assert Manifest(m.to_jsonl().encode()).sha256 == m.sha256


def test_manifest_bad_line():
    with pytest.raises(ValueError, match="line 2"):
        Manifest('{"uri":"file://a"}\n{not json}\n')


def test_open_manifest_cached_and_verified(plane, root):
    text = _refs_jsonl([{"uri": "file://a.npy"}])
    (root / "m.jsonl").write_text(text)
    m = plane.open_manifest("file://m.jsonl")
    assert len(m) == 1
    assert plane.open_manifest("file://m.jsonl") is m
    good = h(text.encode())
    m2 = plane.open_manifest("file://m.jsonl", sha256=good)
    assert m2 is not m and plane.open_manifest("file://m.jsonl", sha256=good) is m2
    with pytest.raises(IntegrityError):
        plane.open_manifest("file://m.jsonl", sha256="0" * 64)


# ---------------------------------------------------------------- load_batch
def _write_manifest(root, entries):
    return Manifest(_refs_jsonl(entries))


def test_load_batch_2p5d_window_and_normalize(plane, root):
    vol = np.stack([np.full((8, 8), z, np.float32) for z in range(6)])   # (Z,H,W), slice z has value z
    np.save(root / "vol.npy", vol)
    m = _write_manifest(root, [{"uri": "file://vol.npy", "slice": [s, s + 3]} for s in range(4)])
    spec = {"kind": "2p5d", "size": [8, 8], "channels": 3, "normalize": {"mean": 1.0, "std": 2.0}}
    x = plane.load_batch(m, [2, 0], spec)
    assert x.shape == (2, 3, 8, 8) and x.dtype == torch.float32
    assert torch.allclose(x[0, :, 0, 0], (torch.tensor([2.0, 3, 4]) - 1) / 2)
    assert torch.allclose(x[1, :, 0, 0], (torch.tensor([0.0, 1, 2]) - 1) / 2)
    y = plane.load_batch(m, [2, 0], spec)
    assert torch.equal(x, y)  # deterministic


def test_load_batch_2p5d_resize(plane, root):
    np.save(root / "vol.npy", np.random.default_rng(0).random((3, 16, 12)).astype(np.float32))
    m = _write_manifest(root, [{"uri": "file://vol.npy", "slice": [0, 3]}])
    x = plane.load_batch(m, [0], {"kind": "2p5d", "size": [8, 6], "channels": 3, "normalize": None})
    ref = torch.nn.functional.interpolate(torch.from_numpy(np.load(root / "vol.npy"))[None], size=(8, 6),
                                          mode="bilinear", align_corners=False)
    assert torch.allclose(x, ref)


def test_load_batch_2p5d_channel_mismatch(plane, root):
    np.save(root / "vol.npy", np.zeros((5, 4, 4), np.float32))
    m = _write_manifest(root, [{"uri": "file://vol.npy", "slice": [0, 2]}])
    with pytest.raises(ValueError):
        plane.load_batch(m, [0], {"kind": "2p5d", "size": [4, 4], "channels": 3})


def test_load_batch_2d_gray_hwc_chw(plane, root):
    np.save(root / "g.npy", np.full((4, 6), 5, np.uint8))                 # (H,W)
    np.save(root / "rgb.npy", np.stack([np.full((4, 6), c, np.uint8) for c in (1, 2, 3)], -1))  # (H,W,3)
    np.save(root / "chw.npy", np.stack([np.full((4, 6), c, np.float32) for c in (7, 8, 9)]))   # (3,H,W)
    m = _write_manifest(root, [{"uri": "file://g.npy"}, {"uri": "file://rgb.npy"}, {"uri": "file://chw.npy"}])
    g = plane.load_batch(m, [0], {"kind": "2d", "size": [4, 6]})
    assert g.shape == (1, 1, 4, 6) and float(g.max()) == 5
    g3 = plane.load_batch(m, [0], {"kind": "2d", "size": [4, 6], "channels": 3})
    assert g3.shape == (1, 3, 4, 6)                                      # gray broadcast to 3
    x = plane.load_batch(m, [1, 2], {"kind": "2d", "size": [2, 3], "channels": 3,
                                     "normalize": {"mean": [0, 0, 0], "std": [1, 1, 1]}})
    assert x.shape == (2, 3, 2, 3)
    assert x[0, :, 0, 0].tolist() == [1, 2, 3] and x[1, :, 1, 2].tolist() == [7, 8, 9]
    with pytest.raises(ValueError):
        plane.load_batch(m, [2], {"kind": "2d", "size": [4, 6], "channels": 4})


def test_load_batch_3d_trilinear_and_dtype(plane, root):
    v = np.random.default_rng(1).random((4, 6, 6)).astype(np.float32)
    np.save(root / "v.npy", v)
    m = _write_manifest(root, [{"uri": "file://v.npy"}, {"uri": "file://v.npy"}])
    x = plane.load_batch(m, [0, 1], {"kind": "3d", "size": [2, 3, 3], "dtype": "float16"})
    assert x.shape == (2, 1, 2, 3, 3) and x.dtype == torch.float16
    ref = torch.nn.functional.interpolate(torch.from_numpy(v)[None, None], size=(2, 3, 3), mode="trilinear",
                                          align_corners=False)
    assert torch.allclose(x[0].float(), ref[0], atol=1e-3)
    same = plane.load_batch(m, [0], {"kind": "3d", "size": [4, 6, 6]})
    assert torch.equal(same[0, 0], torch.from_numpy(v))


def test_load_batch_bad_spec(plane, root):
    np.save(root / "v.npy", np.zeros((2, 2, 2), np.float32))
    m = _write_manifest(root, [{"uri": "file://v.npy"}])
    for spec in ({"kind": "4d", "size": [2, 2]}, {"kind": "2d", "size": [2]}, {"kind": "3d", "size": [2, 2]},
                 {"kind": "3d", "size": [2, 2, 2], "dtype": "int7"}):
        with pytest.raises(ValueError):
            plane.load_batch(m, [0], spec)
    np.save(root / "w.npy", np.zeros((2,), np.float32))
    m2 = _write_manifest(root, [{"uri": "file://w.npy"}])
    with pytest.raises(ValueError):
        plane.load_batch(m2, [0], {"kind": "3d", "size": [2, 2, 2]})
    with pytest.raises(ValueError):
        plane.load_batch(m2, [0], {"kind": "2d", "size": [2, 2]})


# ---------------------------------------------------------------- pushed://
def test_pushed_blob_ref(plane):
    buf = __import__("io").BytesIO(); np.save(buf, np.arange(4, dtype=np.int32)); data = buf.getvalue()
    plane.blobs.begin("up1", h(data), len(data)); plane.blobs.chunk("up1", 0, data); plane.blobs.end("up1")
    assert np.array_equal(plane.read(Ref("pushed://up1", meta={"fmt": "npy"})), np.arange(4))
    assert np.array_equal(plane.read(Ref("pushed://up1", sha256=h(data), meta={"fmt": "npy"})), np.arange(4))
    with pytest.raises(IntegrityError):
        plane.resolve(Ref("pushed://up1", sha256="0" * 64))
    with pytest.raises(RefDenied):
        plane.resolve(Ref("pushed://missing"))


# ---------------------------------------------------------------- https via local server
class _Handler(http.server.BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self):  # noqa: N802
        r = self.routes.get(self.path)
        if r is None:
            self.send_response(404); self.end_headers(); return
        code, body, headers = r
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def _npy_bytes(a):
    import io
    b = io.BytesIO(); np.save(b, a); return b.getvalue()


def test_http_download_verified_and_cached(server, root, tmp_path):
    srv, base = server
    data = _npy_bytes(np.arange(10, dtype=np.float32))
    _Handler.routes = {"/a.npy": (200, data, {})}
    cache = ContentCache(tmp_path / "c", 1 << 20)
    plane = DataPlane(DataPolicy(roots=[str(root)], hosts=["127.0.0.1"]), cache=cache)
    ref = Ref(f"{base}/a.npy", sha256=h(data))
    assert np.array_equal(plane.read(ref), np.arange(10))
    _Handler.routes = {}                                     # second read served from cache
    assert np.array_equal(plane.read(ref), np.arange(10))
    s = plane.stats()
    assert s["cache_misses"] == 1 and s["cache_hits"] == 1


def test_http_sha_mismatch_refused(server, root, tmp_path):
    srv, base = server
    _Handler.routes = {"/a.npy": (200, b"tampered", {})}
    cache = ContentCache(tmp_path / "c", 1 << 20)
    plane = DataPlane(DataPolicy(roots=[str(root)], hosts=["127.0.0.1"]), cache=cache)
    with pytest.raises(IntegrityError):
        plane.resolve(Ref(f"{base}/a.npy", sha256=h(b"expected")))
    assert cache.stats()["entries"] == 0


def test_http_size_cap_and_errors(server, root, tmp_path):
    srv, base = server
    _Handler.routes = {"/big": (200, b"x" * 100, {})}
    plane = DataPlane(DataPolicy(roots=[str(root)], hosts=["127.0.0.1"], max_download_bytes=50),
                      cache=ContentCache(tmp_path / "c", 1 << 20))
    with pytest.raises(RefDenied):
        plane.resolve(Ref(f"{base}/big", sha256=h(b"x" * 100)))
    with pytest.raises(FileNotFoundError):
        plane.resolve(Ref(f"{base}/missing", sha256=h(b"")))


def test_http_redirect_to_other_host_refused(server, root, tmp_path):
    srv, base = server
    _Handler.routes = {"/r": (302, b"", {"Location": "http://evil.test/x"})}
    plane = DataPlane(DataPolicy(roots=[str(root)], hosts=["127.0.0.1"]), cache=ContentCache(tmp_path / "c", 1 << 20))
    with pytest.raises(RefDenied):
        plane.resolve(Ref(f"{base}/r", sha256=h(b"")))


def test_http_redirect_within_allowlist_followed(server, root, tmp_path):
    srv, base = server
    _Handler.routes = {"/r": (302, b"", {"Location": f"{base}/t"}), "/t": (200, b"ok", {})}
    plane = DataPlane(DataPolicy(roots=[str(root)], hosts=["127.0.0.1"]), cache=ContentCache(tmp_path / "c", 1 << 20))
    assert plane.resolve(Ref(f"{base}/r", sha256=h(b"ok"))).read_bytes() == b"ok"


# ---------------------------------------------------------------- buckets via fake CLI
def _fake_tool(bindir, name, src):
    p = bindir / name
    p.write_text(f"#!/bin/sh\n# fake {name}: copy fixed file to the last argument\n"
                 f'for a; do last="$a"; done\necho "$@" > "{bindir}/{name}.args"\ncp "{src}" "$last"\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


@pytest.mark.parametrize("scheme,tool", [("s3", "s5cmd"), ("gs", "gsutil")])
def test_bucket_download_via_cli(tmp_path, root, monkeypatch, scheme, tool):
    bindir = tmp_path / "bin"; bindir.mkdir()
    data = _npy_bytes(np.ones(3, np.float32)); src = tmp_path / "srcfile"; src.write_bytes(data)
    _fake_tool(bindir, tool, src)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    plane = DataPlane(DataPolicy(roots=[str(root)], allow_buckets=True), cache=ContentCache(tmp_path / "c", 1 << 20))
    assert np.array_equal(plane.read(Ref(f"{scheme}://pub-bucket/k/a.npy", sha256=h(data))), np.ones(3))
    args = (bindir / f"{tool}.args").read_text()
    if tool == "s5cmd":
        assert "--no-sign-request" in args
    assert f"{scheme}://pub-bucket/k/a.npy" in args
    with pytest.raises(IntegrityError):
        plane.resolve(Ref(f"{scheme}://pub-bucket/k/b.npy", sha256="0" * 64))


def test_bucket_cli_failure(tmp_path, root, monkeypatch):
    bindir = tmp_path / "bin"; bindir.mkdir()
    (bindir / "s5cmd").write_text("#!/bin/sh\nexit 3\n"); (bindir / "s5cmd").chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    plane = DataPlane(DataPolicy(roots=[str(root)], allow_buckets=True), cache=ContentCache(tmp_path / "c", 1 << 20))
    with pytest.raises(FileNotFoundError):
        plane.resolve(Ref("s3://b/k", sha256="0" * 64))
    with pytest.raises(RefDenied):
        plane.resolve(Ref("s3://-bad/k", sha256="0" * 64))


# ---------------------------------------------------------------- caps / defaults
def test_capabilities_summary(plane):
    c = plane.capabilities()
    assert c["readers"]["numpy"] is True
    assert c["n_roots"] == 1 and c["n_hosts"] == 0 and c["buckets"] is False
    assert "bytes" in c["cache"]


def test_default_construction_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MOREGPU_DATA_ROOTS", str(tmp_path))
    monkeypatch.setenv("MOREGPU_CACHE_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("MOREGPU_STAGE_DIR", str(tmp_path / "st"))
    p = DataPlane()
    assert p.policy.roots == [str(tmp_path)]
    assert p.capabilities()["cache"]["entries"] == 0
    assert p.blobs is not None
