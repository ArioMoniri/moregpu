"""Worker RPC surface of the data plane (ADR-0110), dispatched from ``worker_torch.py``'s ``train`` relay.

Ops: ``data_caps``, ``data_stats``, ``blob_begin{id,sha256,size[,suffix]}``, ``blob_chunk{id,k,data(b64)}``,
``blob_end{id}``, ``blob_drop{id}``. Replies never include local filesystem paths.
"""
from __future__ import annotations

import base64

from .plane import DataPlane

OPS = frozenset({"data_caps", "data_stats", "blob_begin", "blob_chunk", "blob_end", "blob_drop"})


def handle(plane: DataPlane, op: str, payload: dict) -> dict:
    if op == "data_caps":
        return {"ok": True, **plane.capabilities()}
    if op == "data_stats":
        return {"ok": True, **plane.stats()}
    if op == "blob_begin":
        return plane.blobs.begin(payload["id"], payload["sha256"], int(payload["size"]), payload.get("suffix") or "")
    if op == "blob_chunk":
        return plane.blobs.chunk(payload["id"], int(payload["k"]), base64.b64decode(payload["data"], validate=True))
    if op == "blob_end":
        bid = payload["id"]
        plane.blobs.end(bid)
        info = plane.blobs.info(bid)
        return {"ok": True, "id": bid, "sha256": info["sha256"], "size": info["size"], "uri": f"pushed://{bid}"}
    if op == "blob_drop":
        plane.blobs.drop(payload["id"])
        return {"ok": True, "id": payload["id"]}
    raise ValueError(f"unknown data op {op!r}")
