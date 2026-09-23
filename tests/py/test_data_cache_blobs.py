"""ContentCache (sha256-addressed, LRU, byte cap) and BlobStore (pushed://, RAM-staged) — ADR-0110."""
import hashlib
import os
import time

import pytest

from moregpu_worker.data.cache import ContentCache
from moregpu_worker.data.blobs import BlobStore, stage_root
from moregpu_worker.data.refs import IntegrityError


def h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- cache
def test_cache_put_get_bytes(tmp_path):
    c = ContentCache(tmp_path / "c", 1000)
    p = c.put(b"hello")
    assert p.read_bytes() == b"hello" and p.name == h(b"hello")
    assert c.get(h(b"hello")) == p
    assert c.get("0" * 64) is None
    s = c.stats()
    assert s["entries"] == 1 and s["bytes"] == 5 and s["cap_bytes"] == 1000
    assert s["hits"] == 1 and s["misses"] == 1


def test_cache_put_path_copy_and_move(tmp_path):
    c = ContentCache(tmp_path / "c", 1000)
    src = tmp_path / "src.bin"; src.write_bytes(b"abc")
    p = c.put(src, sha256=h(b"abc"))
    assert src.exists() and p.read_bytes() == b"abc"
    src2 = tmp_path / "src2.bin"; src2.write_bytes(b"xyz")
    p2 = c.put(str(src2), move=True)
    assert not src2.exists() and p2.read_bytes() == b"xyz"


def test_cache_put_verifies_hash(tmp_path):
    c = ContentCache(tmp_path / "c", 1000)
    with pytest.raises(IntegrityError):
        c.put(b"data", sha256="0" * 64)
    src = tmp_path / "s"; src.write_bytes(b"data")
    with pytest.raises(IntegrityError):
        c.put(src, sha256="1" * 64, move=True)
    assert c.stats()["entries"] == 0
    assert [f for f in os.listdir(tmp_path / "c") if not f.startswith(".")] == []


def test_cache_rejects_bad_sha_keys(tmp_path):
    c = ContentCache(tmp_path / "c", 1000)
    for bad in ("../x", "zz" * 32, "a" * 63):
        with pytest.raises(ValueError):
            c.get(bad)


def test_cache_put_same_twice_is_idempotent(tmp_path):
    c = ContentCache(tmp_path / "c", 1000)
    c.put(b"x"); c.put(b"x")
    assert c.stats()["entries"] == 1 and c.stats()["bytes"] == 1


def test_cache_lru_eviction_order_and_cap(tmp_path):
    c = ContentCache(tmp_path / "c", 30)
    a, b, d = b"a" * 10, b"b" * 10, b"d" * 10
    c.put(a); c.put(b); c.put(d)
    assert c.stats()["bytes"] == 30
    assert c.get(h(a)) is not None          # touch a -> b is now least recently used
    c.put(b"e" * 10)                        # over cap -> evict b
    assert c.get(h(b)) is None
    assert c.get(h(a)) is not None and c.get(h(d)) is not None
    s = c.stats()
    assert s["bytes"] <= 30 and s["evictions"] == 1
    c.put(b"f" * 25)                        # needs 25 -> evicts until fits
    assert c.stats()["bytes"] <= 30
    assert c.get(h(b"f" * 25)) is not None


def test_cache_entry_larger_than_cap_rejected(tmp_path):
    c = ContentCache(tmp_path / "c", 4)
    with pytest.raises(ValueError):
        c.put(b"too big")


def test_cache_reloads_existing_entries_in_mtime_order(tmp_path):
    c = ContentCache(tmp_path / "c", 100)
    old = c.put(b"old"); new = c.put(b"new")
    past = time.time() - 100
    os.utime(old, (past, past))
    (tmp_path / "c" / "junk.txt").write_bytes(b"ignored")
    c2 = ContentCache(tmp_path / "c", 6)   # smaller cap on reopen -> evicts the oldest
    assert c2.get(h(b"old")) is None and c2.get(h(b"new")) == new


def test_cache_clear(tmp_path):
    c = ContentCache(tmp_path / "c", 100)
    c.put(b"x"); c.clear()
    assert c.stats()["entries"] == 0 and c.stats()["bytes"] == 0


# ---------------------------------------------------------------- blobs
def test_stage_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MOREGPU_STAGE_DIR", str(tmp_path / "stage"))
    assert stage_root() == str(tmp_path / "stage")
    monkeypatch.delenv("MOREGPU_STAGE_DIR")
    assert os.path.isdir(stage_root())


def test_blob_round_trip(tmp_path):
    bs = BlobStore(stage_dir=tmp_path)
    data = os.urandom(1000)
    r = bs.begin("b1", h(data), len(data))
    assert r["id"] == "b1" and r["staging"] in ("ram", "disk")
    assert bs.chunk("b1", 0, data[:400])["bytes"] == 400
    bs.chunk("b1", 1, data[400:])
    p = bs.end("b1")
    assert p.read_bytes() == data and bs.path("b1") == p
    assert str(p).startswith(str(tmp_path))
    bs.drop("b1")
    assert not p.exists()
    with pytest.raises(KeyError):
        bs.path("b1")


def test_blob_strict_chunk_order(tmp_path):
    bs = BlobStore(stage_dir=tmp_path)
    data = b"0123456789"
    bs.begin("b", h(data), 10)
    with pytest.raises(ValueError):
        bs.chunk("b", 1, data[:5])          # skipped 0
    bs.chunk("b", 0, data[:5])
    with pytest.raises(ValueError):
        bs.chunk("b", 0, data[:5])          # replay
    bs.chunk("b", 1, data[5:])
    assert bs.end("b").read_bytes() == data


def test_blob_sha_mismatch_rejected_and_dropped(tmp_path):
    bs = BlobStore(stage_dir=tmp_path)
    bs.begin("b", "0" * 64, 3)
    bs.chunk("b", 0, b"abc")
    with pytest.raises(IntegrityError):
        bs.end("b")
    with pytest.raises(KeyError):
        bs.path("b")
    assert os.listdir(tmp_path) == []


def test_blob_size_cap_and_declared_size(tmp_path):
    bs = BlobStore(stage_dir=tmp_path, max_bytes=8)
    with pytest.raises(ValueError):
        bs.begin("big", "0" * 64, 9)        # declared over cap
    bs.begin("b", h(b"abcd"), 4)
    with pytest.raises(ValueError):
        bs.chunk("b", 0, b"abcdef")         # more than declared
    with pytest.raises(KeyError):
        bs.chunk("b", 1, b"x")              # aborted
    bs.begin("s", h(b"abcd"), 4)
    bs.chunk("s", 0, b"ab")
    with pytest.raises(IntegrityError):
        bs.end("s")                          # short


def test_blob_validation(tmp_path):
    bs = BlobStore(stage_dir=tmp_path)
    for bad_id in ("", "../x", "a/b", "x" * 200):
        with pytest.raises(ValueError):
            bs.begin(bad_id, "0" * 64, 1)
    with pytest.raises(ValueError):
        bs.begin("ok", "nothex", 1)
    with pytest.raises(ValueError):
        bs.begin("ok", "0" * 64, -1)
    with pytest.raises(KeyError):
        bs.chunk("never", 0, b"")
    with pytest.raises(KeyError):
        bs.end("never")
    bs.drop("never")  # no-op


def test_blob_path_before_end_is_keyerror(tmp_path):
    bs = BlobStore(stage_dir=tmp_path)
    bs.begin("b", h(b"x"), 1)
    with pytest.raises(KeyError):
        bs.path("b")
    with pytest.raises(ValueError):
        bs.chunk("b", 0, "not bytes")  # type: ignore[arg-type]


def test_blob_restart_replaces_and_close_cleans(tmp_path):
    bs = BlobStore(stage_dir=tmp_path)
    bs.begin("b", h(b"x"), 1); bs.chunk("b", 0, b"x")
    bs.begin("b", h(b"yy"), 2)               # fresh begin discards partial
    bs.chunk("b", 0, b"yy")
    assert bs.end("b").read_bytes() == b"yy"
    bs.begin("c", h(b"z"), 1)
    assert set(bs.ids()) == {"b", "c"}
    bs.close()
    assert os.listdir(tmp_path) == [] and bs.ids() == []


def test_blob_empty(tmp_path):
    bs = BlobStore(stage_dir=tmp_path)
    bs.begin("e", h(b""), 0)
    assert bs.end("e").read_bytes() == b""
