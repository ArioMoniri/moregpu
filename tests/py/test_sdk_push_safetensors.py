"""Python SDK: push a safetensors file to the workers' BlobStore and get the pushed:// ref (encoder / model init)."""
import hashlib
import io
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save as st_save

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "clients" / "python"))
import moregpu  # noqa: E402


def client(calls):
    c = moregpu.MoreGPU("http://h:1", "T")
    c._req = lambda path, method="GET", body=None, auth=True: calls.append((path, method, body)) or {
        "ok": True, "uri": f"pushed://{body['id']}", "results": [{"worker": "w", "ok": True}]}
    return c


def test_push_safetensors_bytes_and_path(tmp_path):
    data = st_save({"w": torch.ones(3)})
    sha = hashlib.sha256(data).hexdigest()
    calls = []
    r = client(calls).push_safetensors(data, workers=["w"])
    assert r["uri"] == f"pushed://st-{sha[:16]}" and r["sha256"] == sha and r["size"] == len(data) and r["ok"]
    path, method, body = calls[0]
    assert (path, method) == ("/data/push", "POST") and body["sha256"] == sha and body["suffix"] == ".safetensors"
    assert body["workers"] == ["w"]
    p = tmp_path / "enc.safetensors"
    p.write_bytes(data)
    r2 = client(calls).push_safetensors(str(p), id="enc-1")
    assert r2["uri"] == "pushed://enc-1" and r2["sha256"] == sha
    assert r2["ref"] == {"path": "pushed://enc-1", "sha256": sha}


def test_push_safetensors_refuses_pickles_and_junk_locally():
    buf = io.BytesIO()
    torch.save({"w": torch.ones(2)}, buf)
    calls = []
    for bad in (buf.getvalue(), b"\x80\x02}q\x00.", b"", b"\x05\x00\x00\x00\x00\x00\x00\x00nope!"):
        with pytest.raises(ValueError, match="safetensors"):
            client(calls).push_safetensors(bad)
    assert calls == []
