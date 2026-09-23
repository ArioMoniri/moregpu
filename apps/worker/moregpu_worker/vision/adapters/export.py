"""torch.export `.pt2` and TorchScript adapters (ADR-0113 priority 3). Inference-only (not trainable here).

`torch.export.load` is NOT pickle-free on its own: weights/constants flagged `use_pickle`, opaque/custom-object
constants and sample inputs can reach `torch.load(weights_only=False)` or `pickle.loads`. `check_pt2` therefore refuses
any archive with pickled payloads or non-tensor constants, and requires every embedded torch.save/pickle member to load
with weights_only=True, BEFORE torch.export.load sees the file. Members are streamed, and the archive is refused when a
member or the whole archive expands past MOREGPU_PT2_MEMBER_MAX_BYTES / MOREGPU_PT2_TOTAL_MAX_BYTES (zip bombs). Both
adapters refuse to run on torch < 2.6 (CVE-2025-32434). TorchScript archives are read by the TorchScript
deserializer, which only materialises TorchScript types (it never imports arbitrary Python globals).
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import torch

from ..errors import RefusedFormat, brief
from .base import Adapter, Fetch, Handle, fetch_verified, require_safe_torch, resolve_device

try:
    from torch.export.pt2_archive.constants import TENSOR_CONSTANT_FILENAME_PREFIX
except ImportError:  # pragma: no cover - older torch
    TENSOR_CONSTANT_FILENAME_PREFIX = "tensor_"

_PICKLE_SUFFIXES = (".pt", ".pth", ".pkl", ".pickle", ".bin")


GiB = 1024 ** 3
_JSON_MAX = 64 << 20          # a *_config.json member is metadata; anything bigger is not a real archive


def pt2_caps() -> tuple[int, int]:
    """(per-member, total) uncompressed byte caps for a .pt2 archive: MOREGPU_PT2_MEMBER_MAX_BYTES (default 8 GiB) and
    MOREGPU_PT2_TOTAL_MAX_BYTES (default 32 GiB), read at call time."""
    return (int(os.environ.get("MOREGPU_PT2_MEMBER_MAX_BYTES") or 8 * GiB),
            int(os.environ.get("MOREGPU_PT2_TOTAL_MAX_BYTES") or 32 * GiB))


def _read_capped(z: zipfile.ZipFile, info: zipfile.ZipInfo, cap: int, what: str) -> bytes:
    """Stream one member (never trusting its declared size alone) and refuse it past ``cap`` bytes."""
    buf = bytearray()
    with z.open(info) as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                return bytes(buf)
            buf += chunk
            if len(buf) > cap:
                raise RefusedFormat(f"{what}: member {info.filename} is larger than {cap} bytes uncompressed")


def check_pt2(path: Path) -> None:
    member_cap, total_cap = pt2_caps()
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise RefusedFormat(f"{path.name} is not a torch.export .pt2 archive") from None
    with z:
        infos = z.infolist()
        # zip-bomb guard BEFORE decompressing anything: declared sizes against the per-member and total caps (the
        # streamed reads below re-check the real byte count, so a lying header cannot get past the caps either)
        total = 0
        for info in infos:
            if info.file_size > member_cap:
                raise RefusedFormat(f"{path.name}: member {info.filename} is {info.file_size} bytes uncompressed, over "
                                    f"the per-member cap {member_cap} (MOREGPU_PT2_MEMBER_MAX_BYTES)")
            total += info.file_size
            if total > total_cap:
                raise RefusedFormat(f"{path.name}: archive expands to more than the total cap {total_cap} bytes "
                                    f"(MOREGPU_PT2_TOTAL_MAX_BYTES)")
        for info in infos:
            name = info.filename
            if info.is_dir():
                continue
            base = name.rsplit("/", 1)[-1]
            if base.endswith(("_weights_config.json", "_constants_config.json")):
                cfg = json.loads(_read_capped(z, info, min(member_cap, _JSON_MAX), path.name)).get("config", {})
                for fqn, meta in cfg.items():
                    if meta.get("use_pickle"):
                        raise RefusedFormat(f"{path.name}: payload {fqn!r} is stored as a pickle")
                    if base.endswith("_constants_config.json") and not str(meta.get("path_name", "")).startswith(
                            TENSOR_CONSTANT_FILENAME_PREFIX):
                        raise RefusedFormat(f"{path.name}: constant {fqn!r} is a pickled/opaque object, not a tensor")
                continue
            with z.open(info) as f:
                magic = f.read(4)                # sniff; never read a non-pickle member in full
            if base.endswith(_PICKLE_SUFFIXES) or magic == b"PK\x03\x04":
                data = _read_capped(z, info, member_cap, path.name)
                try:
                    torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                except Exception as e:
                    raise RefusedFormat(f"{path.name}: member {name} does not load with weights_only=True "
                                        f"({brief(e)})") from None


class TorchExportAdapter(Adapter):
    name = "torch_export"

    def load(self, spec: dict, fetch: Fetch) -> Handle:
        path, sha = fetch_verified(spec, fetch)
        require_safe_torch(f"torch.export archive {path.name}")
        check_pt2(path)
        try:
            ep = torch.export.load(str(path))
        except Exception as e:
            raise RefusedFormat(f"{path.name} is not a loadable torch.export archive ({e})") from None
        device = resolve_device(spec.get("placement"))
        model = ep.module().to(device)
        return Handle(spec=spec, adapter=self.name, sha256=sha, model=model, device=device,
                      extra={"exported_program": ep, "path": path})

    def forward(self, handle: Handle, x: torch.Tensor):
        return handle.model(x.to(handle.device))

    def _describe(self, handle: Handle) -> dict:
        ep = handle.extra.get("exported_program")
        return {"graph_nodes": len(ep.graph.nodes) if ep is not None else None}


class TorchScriptAdapter(Adapter):
    name = "torchscript"

    def load(self, spec: dict, fetch: Fetch) -> Handle:
        path, sha = fetch_verified(spec, fetch)
        require_safe_torch(f"TorchScript archive {path.name}")
        device = resolve_device(spec.get("placement"))
        try:
            model = torch.jit.load(str(path), map_location=device).eval()
        except Exception as e:
            raise RefusedFormat(f"{path.name} is not a TorchScript archive ({brief(e)})") from None
        return Handle(spec=spec, adapter=self.name, sha256=sha, model=model, device=device, extra={"path": path})

    def forward(self, handle: Handle, x: torch.Tensor):
        return handle.model(x.to(handle.device))
