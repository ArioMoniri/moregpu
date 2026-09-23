"""torch.export `.pt2` and TorchScript adapters (ADR-0113 priority 3). Inference-only (not trainable here).

`torch.export.load` is NOT pickle-free on its own: weights/constants flagged `use_pickle`, opaque/custom-object
constants and sample inputs can reach `torch.load(weights_only=False)` or `pickle.loads`. `check_pt2` therefore refuses
any archive with pickled payloads or non-tensor constants, and requires every embedded torch.save/pickle member to load
with weights_only=True, BEFORE torch.export.load sees the file. TorchScript archives are read by the TorchScript
deserializer, which only materialises TorchScript types (it never imports arbitrary Python globals).
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import torch

from ..errors import RefusedFormat, brief
from .base import Adapter, Fetch, Handle, fetch_verified, resolve_device

try:
    from torch.export.pt2_archive.constants import TENSOR_CONSTANT_FILENAME_PREFIX
except ImportError:  # pragma: no cover - older torch
    TENSOR_CONSTANT_FILENAME_PREFIX = "tensor_"

_PICKLE_SUFFIXES = (".pt", ".pth", ".pkl", ".pickle", ".bin")


def check_pt2(path: Path) -> None:
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise RefusedFormat(f"{path.name} is not a torch.export .pt2 archive") from None
    with z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if base.endswith(("_weights_config.json", "_constants_config.json")):
                cfg = json.loads(z.read(name)).get("config", {})
                for fqn, meta in cfg.items():
                    if meta.get("use_pickle"):
                        raise RefusedFormat(f"{path.name}: payload {fqn!r} is stored as a pickle")
                    if base.endswith("_constants_config.json") and not str(meta.get("path_name", "")).startswith(
                            TENSOR_CONSTANT_FILENAME_PREFIX):
                        raise RefusedFormat(f"{path.name}: constant {fqn!r} is a pickled/opaque object, not a tensor")
                continue
            data = z.read(name)
            if base.endswith(_PICKLE_SUFFIXES) or data[:4] == b"PK\x03\x04":
                try:
                    torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                except Exception as e:
                    raise RefusedFormat(f"{path.name}: member {name} does not load with weights_only=True "
                                        f"({brief(e)})") from None


class TorchExportAdapter(Adapter):
    name = "torch_export"

    def load(self, spec: dict, fetch: Fetch) -> Handle:
        path, sha = fetch_verified(spec, fetch)
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
        device = resolve_device(spec.get("placement"))
        try:
            model = torch.jit.load(str(path), map_location=device).eval()
        except Exception as e:
            raise RefusedFormat(f"{path.name} is not a TorchScript archive ({brief(e)})") from None
        return Handle(spec=spec, adapter=self.name, sha256=sha, model=model, device=device, extra={"path": path})

    def forward(self, handle: Handle, x: torch.Tensor):
        return handle.model(x.to(handle.device))
