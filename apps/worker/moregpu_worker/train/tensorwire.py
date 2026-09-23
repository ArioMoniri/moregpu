"""Tensor wire format for DiLoCo sync payloads (ADR-0106).

A payload is (header, blob): `blob` is the concatenation of each tensor's bytes (little-endian), `header` is JSON:

    {"v": 1, "dtype": "f32"|"bf16"|"fp16"|"int8delta", "sha256": <hex of blob>,
     "tensors": [{"name", "shape", "offset", "nbytes"[, "scales_offset", "nblocks", "block"]}],
     "error": {"max_abs", "rel_l2"}}            # lossy dtypes only

int8delta encodes Δ = t − ref per block of `block` elements with a symmetric absmax scale (f32) per block:
scales (nblocks × f32) are stored first, then q (numel × int8). Decoding is ref + q·scale, so both sides must hold
the identical `ref` (the last broadcast global). The coordinator decoder (apps/coordinator/lib/tensorwire.ts) must
stay byte-compatible with this module; tests/py/test_tensorwire_golden.py pins the cross-language goldens.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

DTYPES = ("f32", "bf16", "fp16", "int8delta")
DEFAULT_BLOCK = 4096


def _flat_f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float32).contiguous().reshape(-1)


def _encode_one(x: torch.Tensor, dtype: str, ref: torch.Tensor | None, block: int) -> tuple[bytes, dict, torch.Tensor]:
    """Returns (bytes, extra header fields, decoded-back f32 values for error accounting)."""
    if dtype == "f32":
        return x.numpy().astype("<f4").tobytes(), {}, x
    if dtype == "bf16":
        b = x.to(torch.bfloat16)
        return b.view(torch.int16).numpy().astype("<i2").tobytes(), {}, b.to(torch.float32)
    if dtype == "fp16":
        h = x.to(torch.float16)
        return h.numpy().astype("<f2").tobytes(), {}, h.to(torch.float32)
    # int8delta
    d = x - ref
    n = d.numel()
    nblocks = max(1, math.ceil(n / block))
    pad = torch.zeros(nblocks * block - n)
    dp = torch.cat([d, pad]).reshape(nblocks, block)
    scales = (dp.abs().amax(dim=1) / 127.0).to(torch.float32)
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    q = torch.clamp(torch.round(dp / safe[:, None]), -127, 127).to(torch.int8)
    back = (q.to(torch.float32) * scales[:, None]).reshape(-1)[:n] + ref
    raw = scales.numpy().astype("<f4").tobytes() + q.reshape(-1)[:n].numpy().tobytes()
    return raw, {"nblocks": nblocks, "block": block}, back


def encode(tensors: dict[str, torch.Tensor], dtype: str = "f32", ref: dict[str, torch.Tensor] | None = None,
           block: int = DEFAULT_BLOCK) -> tuple[dict, bytes]:
    if dtype not in DTYPES:
        raise ValueError(f"unknown wire dtype {dtype!r}; expected one of {DTYPES}")
    if dtype == "int8delta" and ref is None:
        raise ValueError("int8delta needs the reference (last broadcast) tensors")
    parts, entries, off = [], [], 0
    err_max, num, den = 0.0, 0.0, 0.0
    for name, t in tensors.items():
        x = _flat_f32(t)
        r = _flat_f32(ref[name]) if dtype == "int8delta" else None
        raw, extra, back = _encode_one(x, dtype, r, block)
        e = {"name": name, "shape": list(t.shape), "offset": off, "nbytes": len(raw)}
        if extra:
            e.update({"scales_offset": off, **extra})
        entries.append(e)
        parts.append(raw)
        off += len(raw)
        if dtype != "f32":
            diff = back - x
            err_max = max(err_max, float(diff.abs().max()) if diff.numel() else 0.0)
            num += float((diff * diff).sum())
            den += float((x * x).sum())
    blob = b"".join(parts)
    hdr = {"v": 1, "dtype": dtype, "sha256": hashlib.sha256(blob).hexdigest(), "tensors": entries}
    if dtype != "f32":
        hdr["error"] = {"max_abs": err_max, "rel_l2": math.sqrt(num / den) if den > 0 else 0.0}
    return hdr, blob


def decode(hdr: dict, blob: bytes, ref: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    if hashlib.sha256(blob).hexdigest() != hdr["sha256"]:
        raise ValueError("tensor payload sha256 mismatch")
    dtype = hdr["dtype"]
    if dtype == "int8delta" and ref is None:
        raise ValueError("int8delta decode needs the reference tensors")
    out: dict[str, torch.Tensor] = {}
    for e in hdr["tensors"]:
        raw = blob[e["offset"]: e["offset"] + e["nbytes"]]
        n = int(np.prod(e["shape"])) if e["shape"] else 1
        if dtype == "f32":
            x = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())
        elif dtype == "bf16":
            x = torch.from_numpy(np.frombuffer(raw, dtype="<i2").copy()).view(torch.bfloat16).to(torch.float32)
        elif dtype == "fp16":
            x = torch.from_numpy(np.frombuffer(raw, dtype="<f2").astype(np.float32))
        elif dtype == "int8delta":
            nb, blk = e["nblocks"], e["block"]
            scales = torch.from_numpy(np.frombuffer(raw[: 4 * nb], dtype="<f4").copy())
            q = torch.from_numpy(np.frombuffer(raw[4 * nb:], dtype=np.int8).astype(np.float32))
            idx = torch.arange(n) // blk
            x = q * scales[idx] + _flat_f32(ref[e["name"]])
        else:
            raise ValueError(f"unknown wire dtype {dtype!r}")
        if x.numel() != n:
            raise ValueError(f"{e['name']}: {x.numel()} elements, shape says {n}")
        out[e["name"]] = x.reshape(e["shape"])
    return out


def chunk(blob: bytes, size: int = 4 << 20) -> list[bytes]:
    return [blob[i: i + size] for i in range(0, len(blob), size)] or [b""]


def join(chunks: list[bytes], sha256: str) -> bytes:
    blob = b"".join(chunks)
    if hashlib.sha256(blob).hexdigest() != sha256:
        raise ValueError("reassembled payload sha256 mismatch")
    return blob
