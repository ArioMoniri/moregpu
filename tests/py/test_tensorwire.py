import numpy as np
import pytest
import torch

from moregpu_worker.train import tensorwire as tw


def _tensors(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"enc.w": torch.randn(64, 33, generator=g), "enc.b": torch.randn(33, generator=g) * 1e-3,
            "pred.scalar": torch.tensor(3.5)}


def test_f32_roundtrip_is_exact_and_hash_checked():
    t = _tensors()
    hdr, blob = tw.encode(t, "f32")
    out = tw.decode(hdr, blob)
    for k in t:
        assert out[k].dtype == torch.float32 and out[k].shape == t[k].shape
        assert torch.equal(out[k], t[k])
    bad = bytearray(blob); bad[5] ^= 1
    with pytest.raises(ValueError, match="sha256"):
        tw.decode(hdr, bytes(bad))


@pytest.mark.parametrize("dtype,tol", [("bf16", 2 ** -8), ("fp16", 2 ** -11)])
def test_half_precision_relative_error_bound(dtype, tol):
    t = _tensors()
    hdr, blob = tw.encode(t, dtype)
    out = tw.decode(hdr, blob)
    for k in t:
        normal = t[k].abs() >= 6.2e-5  # fp16 min normal; below it (subnormals) the bound is absolute
        err = (out[k] - t[k]).abs()
        rel = (err[normal] / t[k].abs()[normal]).max().item() if normal.any() else 0.0
        assert rel <= tol, (k, rel)
        assert err[~normal].max().item() <= 2 ** -24 if (~normal).any() else True
    assert len(blob) < sum(x.numel() for x in t.values()) * 4


def test_int8_delta_error_bounded_by_half_scale_and_reported():
    ref = _tensors(0)
    new = {k: v + 0.01 * torch.randn_like(v) for k, v in ref.items()}
    hdr, blob = tw.encode(new, "int8delta", ref=ref, block=16)
    out = tw.decode(hdr, blob, ref=ref)
    for k in new:
        d = new[k] - ref[k]
        scale_max = d.abs().max().item() / 127
        assert (out[k] - new[k]).abs().max().item() <= scale_max / 2 + 1e-7
    assert hdr["error"]["max_abs"] >= 0 and "rel_l2" in hdr["error"]


def test_int8_delta_requires_ref():
    with pytest.raises(ValueError):
        tw.encode(_tensors(), "int8delta")


def test_chunking_roundtrip():
    hdr, blob = tw.encode(_tensors(), "f32")
    chunks = tw.chunk(blob, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert tw.join(chunks, hdr["sha256"]) == blob


def test_golden_bytes_are_little_endian_f32():
    hdr, blob = tw.encode({"x": torch.tensor([1.0, -2.0])}, "f32")
    assert blob == np.array([1.0, -2.0], dtype="<f4").tobytes()
    assert hdr["tensors"][0] == {"name": "x", "shape": [2], "offset": 0, "nbytes": 8}
