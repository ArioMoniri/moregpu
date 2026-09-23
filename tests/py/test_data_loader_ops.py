"""Fast path (pre-tiled .npy shards + DataLoader) and the worker RPC surface (ADR-0110). Synthetic data only."""
import base64
import hashlib
import json

import numpy as np
import pytest
import torch

from moregpu_worker.data import ops
from moregpu_worker.data.blobs import BlobStore
from moregpu_worker.data.cache import ContentCache
from moregpu_worker.data.loader import ShardDataset, make_loader, measure_throughput, write_shards
from moregpu_worker.data.plane import DataPlane
from moregpu_worker.data.refs import DataPolicy, IntegrityError


def _arrays(n_shards=3, per=5, shape=(2, 4, 4)):
    for s in range(n_shards):
        a = np.stack([np.full(shape, s * per + i, np.float32) for i in range(per)])
        yield f"part/{s}", a


def test_write_shards_index(tmp_path):
    idx = write_shards(_arrays(), tmp_path / "sh")
    assert idx["total"] == 15 and idx["dtype"] == "float16" and idx["sample_shape"] == [2, 4, 4]
    on_disk = json.loads((tmp_path / "sh" / "index.json").read_text())
    assert on_disk == idx
    for sh in idx["shards"]:
        f = tmp_path / "sh" / sh["file"]
        assert "/" not in sh["file"] and f.exists()
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sh["sha256"]
        assert np.load(f).dtype == np.float16


def test_write_shards_validation(tmp_path):
    with pytest.raises(ValueError):
        write_shards([("a", np.zeros((2, 3))), ("b", np.zeros((2, 4)))], tmp_path / "x")
    with pytest.raises(ValueError):
        write_shards([("a", np.zeros(()))], tmp_path / "y")
    with pytest.raises(ValueError):
        write_shards([], tmp_path / "z")
    idx = write_shards([("same", np.zeros((1, 2))), ("same", np.ones((1, 2)))], tmp_path / "w", dtype="float32")
    assert len({s["file"] for s in idx["shards"]}) == 2


def test_shard_dataset_reads_and_verifies(tmp_path):
    write_shards(_arrays(), tmp_path / "sh")
    ds = ShardDataset(tmp_path / "sh" / "index.json", verify=True)
    assert len(ds) == 15
    x = ds[7]
    assert isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.shape == (2, 4, 4) and float(x[0, 0, 0]) == 7
    assert float(ds[-1][0, 0, 0]) == 14
    with pytest.raises(IndexError):
        ds[15]
    ds2 = ShardDataset(tmp_path / "sh")  # directory accepted
    assert float(ds2[0].sum()) == 0
    f = tmp_path / "sh" / json.loads((tmp_path / "sh" / "index.json").read_text())["shards"][0]["file"]
    f.write_bytes(f.read_bytes()[:-2] + b"\x00\x01")
    with pytest.raises(IntegrityError):
        ShardDataset(tmp_path / "sh", verify=True)


def _order(loader):
    return [int(v) for b in loader for v in b[:, 0, 0, 0].tolist()]


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_deterministic_and_throughput(tmp_path, workers):
    write_shards(_arrays(), tmp_path / "sh")
    ds = ShardDataset(tmp_path / "sh")
    l1 = make_loader(ds, batch_size=4, workers=workers, seed=3)
    l2 = make_loader(ds, batch_size=4, workers=workers, seed=3)
    o1, o2 = _order(l1), _order(l2)
    assert o1 == o2 and sorted(o1) == list(range(15)) and o1 != list(range(15))
    assert _order(make_loader(ds, batch_size=4, workers=workers, seed=4)) != o1
    t = measure_throughput(l1, max_batches=3)
    assert t["batches"] == 3 and t["samples_per_s"] > 0 and t["seconds"] > 0 and t["samples"] == 12


def test_loader_no_shuffle_and_pin_flag(tmp_path):
    write_shards(_arrays(1, 4), tmp_path / "sh")
    ds = ShardDataset(tmp_path / "sh")
    loader = make_loader(ds, batch_size=2, workers=0, shuffle=False, pin_memory=False)
    assert _order(loader) == [0, 1, 2, 3] and loader.pin_memory is False
    assert make_loader(ds, 2, workers=0).pin_memory == torch.cuda.is_available()


def test_measure_throughput_tuple_batches():
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(6, 2), torch.zeros(6)), batch_size=4)
    t = measure_throughput(loader, max_batches=10)
    assert t["batches"] == 2 and t["samples"] == 6


# ---------------------------------------------------------------- ops
@pytest.fixture
def plane(tmp_path):
    return DataPlane(DataPolicy(roots=[str(tmp_path)]), cache=ContentCache(tmp_path / "c", 1 << 20),
                     blobs=BlobStore(stage_dir=tmp_path / "stage"))


def test_ops_blob_round_trip(plane):
    data = b"z" * 1000
    sha = hashlib.sha256(data).hexdigest()
    assert ops.handle(plane, "blob_begin", {"id": "b", "sha256": sha, "size": len(data)})["ok"]
    for k, i in enumerate(range(0, len(data), 300)):
        r = ops.handle(plane, "blob_chunk", {"id": "b", "k": k, "data": base64.b64encode(data[i:i + 300]).decode()})
        assert r["ok"] and r["bytes"] == min(i + 300, len(data))
    r = ops.handle(plane, "blob_end", {"id": "b"})
    assert r == {"ok": True, "id": "b", "sha256": sha, "size": 1000, "uri": "pushed://b"}
    assert plane.blobs.path("b").read_bytes() == data
    assert ops.handle(plane, "blob_drop", {"id": "b"}) == {"ok": True, "id": "b"}


def test_ops_caps_stats_and_errors(plane):
    caps = ops.handle(plane, "data_caps", {})
    assert caps["ok"] and caps["readers"]["numpy"] and caps["n_roots"] == 1
    st = ops.handle(plane, "data_stats", {})
    assert st["ok"] and st["reads"] == 0
    with pytest.raises(ValueError):
        ops.handle(plane, "nope", {})
    assert ops.OPS >= {"data_caps", "blob_begin", "blob_chunk", "blob_end", "blob_drop", "data_stats"}
    with pytest.raises(KeyError):
        ops.handle(plane, "blob_begin", {"id": "x"})
