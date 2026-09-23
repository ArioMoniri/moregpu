"""Refs, DataPolicy and DataPlane.resolve security (ADR-0110). Synthetic data only."""
import hashlib
import os

import numpy as np
import pytest

from moregpu_worker.data.refs import DataPolicy, IntegrityError, Ref, RefDenied
from moregpu_worker.data.plane import DataPlane
from moregpu_worker.data.cache import ContentCache

SHA = "a" * 64


# ---------------------------------------------------------------- Ref
def test_ref_json_round_trip():
    r = Ref("file://x/a.npy", sha256=SHA, slice=(2, 5), meta={"label": 1})
    d = r.to_json()
    assert d == {"uri": "file://x/a.npy", "sha256": SHA, "slice": [2, 5], "meta": {"label": 1}}
    assert Ref.from_json(d) == r
    assert Ref.from_json(d).slice == (2, 5)


def test_ref_minimal_json_omits_defaults():
    assert Ref("pushed://b1").to_json() == {"uri": "pushed://b1"}
    r = Ref.from_json({"uri": "pushed://b1"})
    assert r.sha256 is None and r.slice is None and r.meta == {}


def test_ref_is_frozen():
    r = Ref("pushed://b1")
    with pytest.raises(Exception):
        r.uri = "x"  # type: ignore[misc]


@pytest.mark.parametrize("bad", [
    {},
    {"uri": 3},
    {"uri": "pushed://a", "sha256": "xyz"},
    {"uri": "pushed://a", "sha256": "A" * 63},
    {"uri": "pushed://a", "slice": [5, 2]},
    {"uri": "pushed://a", "slice": [-1, 2]},
    {"uri": "pushed://a", "slice": [1]},
    {"uri": "pushed://a", "meta": [1]},
])
def test_ref_from_json_validates(bad):
    with pytest.raises(ValueError):
        Ref.from_json(bad)


def test_ref_sha_normalised_lowercase():
    assert Ref.from_json({"uri": "pushed://a", "sha256": "AB" * 32}).sha256 == "ab" * 32


# ---------------------------------------------------------------- DataPolicy
def test_policy_defaults():
    p = DataPolicy()
    assert p.roots == [] and p.hosts == [] and p.allow_buckets is False
    assert p.max_download_bytes == 20 * 1024 ** 3


def test_policy_from_env(monkeypatch, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("MOREGPU_DATA_ROOTS", os.pathsep.join([str(a), "", str(b)]))
    monkeypatch.setenv("MOREGPU_DATA_HOSTS", " Example.org, data.test ,")
    monkeypatch.setenv("MOREGPU_DATA_BUCKETS", "1")
    p = DataPolicy.from_env()
    assert p.roots == [str(a), str(b)]
    assert p.hosts == ["example.org", "data.test"]
    assert p.allow_buckets is True


def test_policy_from_env_empty(monkeypatch):
    for k in ("MOREGPU_DATA_ROOTS", "MOREGPU_DATA_HOSTS", "MOREGPU_DATA_BUCKETS"):
        monkeypatch.delenv(k, raising=False)
    p = DataPolicy.from_env()
    assert p.roots == [] and p.hosts == [] and not p.allow_buckets


def test_refdenied_is_permission_error():
    assert issubclass(RefDenied, PermissionError)


# ---------------------------------------------------------------- file:// resolve
@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    (r / "sub").mkdir(parents=True)
    np.save(r / "sub" / "a.npy", np.arange(6, dtype=np.float32))
    (tmp_path / "secret.npy").write_bytes(b"outside")
    return r


def _plane(root, **kw):
    return DataPlane(DataPolicy(roots=[str(root)], **kw), cache=ContentCache(root.parent / "cache", 1 << 20))


def test_file_absolute_inside_root(root):
    p = _plane(root).resolve(Ref(f"file://{root}/sub/a.npy"))
    assert p == (root / "sub" / "a.npy").resolve()


def test_file_relative_resolves_against_first_root(root, tmp_path):
    other = tmp_path / "other"; other.mkdir()
    plane = DataPlane(DataPolicy(roots=[str(root), str(other)]), cache=ContentCache(tmp_path / "c", 1 << 20))
    assert plane.resolve(Ref("file://sub/a.npy")) == (root / "sub" / "a.npy").resolve()
    assert plane.resolve(Ref("file:sub/a.npy")) == (root / "sub" / "a.npy").resolve()


def test_file_in_second_root_allowed(root, tmp_path):
    other = tmp_path / "other"; other.mkdir(); np.save(other / "b.npy", np.zeros(2))
    plane = DataPlane(DataPolicy(roots=[str(root), str(other)]), cache=ContentCache(tmp_path / "c", 1 << 20))
    assert plane.resolve(Ref(f"file://{other}/b.npy")) == (other / "b.npy").resolve()


@pytest.mark.parametrize("uri", [
    "file://../secret.npy",
    "file://sub/../../secret.npy",
    "file:///etc/passwd",
    "FILE_PLACEHOLDER_ABS_SECRET",
])
def test_file_traversal_denied(root, uri):
    if uri == "FILE_PLACEHOLDER_ABS_SECRET":
        uri = f"file://{root.parent}/secret.npy"
    with pytest.raises(RefDenied):
        _plane(root).resolve(Ref(uri))


def test_file_prefix_sibling_denied(root, tmp_path):
    # /tmp/x/root2 must not pass a naive startswith("/tmp/x/root") check
    sib = tmp_path / "root2"; sib.mkdir(); (sib / "f.npy").write_bytes(b"x")
    with pytest.raises(RefDenied):
        _plane(root).resolve(Ref(f"file://{sib}/f.npy"))


def test_symlink_escape_denied(root):
    os.symlink(root.parent / "secret.npy", root / "sub" / "link.npy")
    with pytest.raises(RefDenied):
        _plane(root).resolve(Ref("file://sub/link.npy"))
    os.symlink(root.parent, root / "dirlink")
    with pytest.raises(RefDenied):
        _plane(root).resolve(Ref("file://dirlink/secret.npy"))


def test_symlink_inside_root_allowed(root):
    os.symlink(root / "sub" / "a.npy", root / "alias.npy")
    assert _plane(root).resolve(Ref("file://alias.npy")) == (root / "sub" / "a.npy").resolve()


def test_file_without_roots_denied(root, tmp_path):
    plane = DataPlane(DataPolicy(), cache=ContentCache(tmp_path / "c", 1 << 20))
    with pytest.raises(RefDenied):
        plane.resolve(Ref(f"file://{root}/sub/a.npy"))
    with pytest.raises(RefDenied):
        plane.resolve(Ref("file://sub/a.npy"))


def test_missing_file_is_not_found(root):
    with pytest.raises(FileNotFoundError):
        _plane(root).resolve(Ref("file://sub/nope.npy"))


def test_file_sha256_verified_when_given(root):
    data = (root / "sub" / "a.npy").read_bytes()
    good = hashlib.sha256(data).hexdigest()
    plane = _plane(root)
    assert plane.resolve(Ref("file://sub/a.npy", sha256=good)).exists()
    with pytest.raises(IntegrityError):
        plane.resolve(Ref("file://sub/a.npy", sha256="0" * 64))


@pytest.mark.parametrize("uri", ["ftp://h/x", "/abs/path.npy", "rel.npy", "data:,abc", "", "pushed://../x"])
def test_unknown_schemes_denied(root, uri):
    with pytest.raises(RefDenied):
        _plane(root).resolve(Ref(uri))


# ---------------------------------------------------------------- https / buckets negatives (no network)
def test_https_host_not_allowlisted_denied(root):
    with pytest.raises(RefDenied):
        _plane(root, hosts=["data.test"]).resolve(Ref("https://evil.test/a.npy", sha256=SHA))


def test_https_requires_sha(root):
    with pytest.raises(RefDenied):
        _plane(root, hosts=["data.test"]).resolve(Ref("https://data.test/a.npy"))


def test_https_userinfo_trick_denied(root):
    with pytest.raises(RefDenied):
        _plane(root, hosts=["data.test"]).resolve(Ref("https://data.test@evil.test/a.npy", sha256=SHA))


def test_plain_http_to_non_loopback_denied(root):
    with pytest.raises(RefDenied):
        _plane(root, hosts=["data.test"]).resolve(Ref("http://data.test/a.npy", sha256=SHA))


@pytest.mark.parametrize("uri", ["s3://bucket/a.npy", "gs://bucket/a.npy"])
def test_bucket_refused_when_disabled(root, uri):
    with pytest.raises(RefDenied):
        _plane(root).resolve(Ref(uri, sha256=SHA))


@pytest.mark.parametrize("uri", ["s3://bucket/a.npy", "gs://bucket/a.npy"])
def test_bucket_refused_without_tool(root, uri, monkeypatch):
    monkeypatch.setenv("PATH", str(root))  # no s5cmd / gsutil here
    with pytest.raises(RefDenied):
        _plane(root, allow_buckets=True).resolve(Ref(uri, sha256=SHA))


def test_bucket_requires_sha(root):
    with pytest.raises(RefDenied):
        _plane(root, allow_buckets=True).resolve(Ref("s3://bucket/a.npy"))
