"""State-dict adapter (ADR-0113 priority 1): `.pth/.pt` via torch.load(weights_only=True) or safetensors, loaded
strictly into an architecture built BY NAME from a registry (torchvision, timm, monai.networks.nets, transformers Auto*
from a config dict, or an allowlisted plugin). Nothing here can execute code from the artefact."""
from __future__ import annotations

import importlib
from pathlib import Path

import torch
import torch.nn as nn

from ..errors import KeyMismatch, RefusedFormat, UnknownArch
from .base import DTYPES, Adapter, Fetch, Handle, fetch_verified, module_sha256, resolve_device

CONTAINER_KEYS = ("state_dict", "model_state_dict", "model", "module", "net", "network")
PREFIXES = ("module.", "_orig_mod.")
HF_AUTO = {"AutoModel", "AutoModelForImageClassification", "AutoModelForSemanticSegmentation",
           "AutoModelForObjectDetection", "AutoModelForDepthEstimation", "AutoBackbone"}
_EXTRA = {"torchvision": "torchvision", "timm": "timm", "monai": "monai", "hf": "llm"}


def _all_tensors(d: dict) -> bool:
    return bool(d) and all(isinstance(v, torch.Tensor) for v in d.values())


def unwrap(obj) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Peel common checkpoint containers and DataParallel/compile prefixes; report every step."""
    steps: list[str] = []
    for _ in range(4):
        if not isinstance(obj, dict) or _all_tensors(obj):
            break
        key = next((k for k in CONTAINER_KEYS if isinstance(obj.get(k), dict)), None)
        if key is None:
            break
        obj = obj[key]
        steps.append(f"key:{key}")
    if not isinstance(obj, dict) or not _all_tensors(obj):
        raise RefusedFormat(f"checkpoint is a {type(obj).__name__}, not a dict of tensors (a state_dict)")
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if all(isinstance(k, str) and k.startswith(p) for k in obj):
                obj = {k[len(p):]: v for k, v in obj.items()}
                steps.append(f"prefix:{p}")
                changed = True
    return dict(obj), steps


def read_state_dict(path: Path, fmt: str) -> tuple[dict[str, torch.Tensor], list[str]]:
    if fmt == "safetensors" or (fmt == "plugin" and path.suffix == ".safetensors"):
        from safetensors.torch import load_file
        try:
            return unwrap(load_file(str(path), device="cpu"))
        except RefusedFormat:
            raise
        except Exception as e:
            raise RefusedFormat(f"{path.name} is not a valid safetensors file ({e})") from None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RefusedFormat(f"{path.name} could not be loaded with weights_only=True (a pickled full model or "
                            f"arbitrary objects?): {str(e).splitlines()[0][:300]}") from None
    return unwrap(obj)


def build_arch(arch: dict) -> tuple[nn.Module, dict]:
    reg, name, kw = arch["registry"], arch["name"], dict(arch.get("kwargs") or {})
    extra: dict = {}
    try:
        if reg == "torchvision":
            tvm = importlib.import_module("torchvision.models")
            if name not in tvm.list_models():
                raise UnknownArch(f"torchvision has no model {name!r}")
            return tvm.get_model(name, weights=None, **kw), extra
        if reg == "timm":
            timm = importlib.import_module("timm")
            if not timm.is_model(name):
                raise UnknownArch(f"timm has no model {name!r}")
            return timm.create_model(name, pretrained=False, **kw), extra
        if reg == "monai":
            nets = importlib.import_module("monai.networks.nets")
            cls = None if name.startswith("_") else getattr(nets, name, None)
            if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
                raise UnknownArch(f"monai.networks.nets has no network {name!r}")
            return cls(**kw), extra
        if reg == "hf":
            if name not in HF_AUTO:
                raise RefusedFormat(f"hf architectures are built from a config dict with a transformers Auto class "
                                    f"({sorted(HF_AUTO)}), not {name!r}")
            tr = importlib.import_module("transformers")
            cfg = tr.AutoConfig.for_model(**kw.get("config", {}))
            return getattr(tr, name).from_config(cfg), extra
    except ImportError as e:
        raise UnknownArch(f"registry {reg!r} is not installed on this worker "
                          f"(pip install 'moregpu-worker[{_EXTRA[reg]}]'): {e}") from None
    from . import plugins  # registry == "plugin" (the schema admits nothing else)
    factory, info = plugins.get(name)
    model = factory(**kw)
    if not isinstance(model, nn.Module):
        raise UnknownArch(f"model plugin {name!r} returned {type(model).__name__}, not an nn.Module")
    return model, {"plugin": info}


def load_strict(model: nn.Module, sd: dict[str, torch.Tensor], label: str) -> None:
    own = model.state_dict()
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    shapes = [(k, tuple(sd[k].shape), tuple(own[k].shape)) for k in own if k in sd and sd[k].shape != own[k].shape]
    if missing or unexpected or shapes:
        raise KeyMismatch(label, missing, unexpected, shapes)
    model.load_state_dict(sd, strict=True)


class StateDictAdapter(Adapter):
    name = "state_dict"
    native = True

    def __init__(self, name: str = "state_dict"):
        self.name = name

    def load(self, spec: dict, fetch: Fetch) -> Handle:
        arch = spec["arch"]
        label = f"{arch['registry']}:{arch['name']}"
        sd, steps, sha = None, [], None
        if spec.get("source"):
            path, sha = fetch_verified(spec, fetch)
            sd, steps = read_state_dict(path, spec["format"])
        model, extra = build_arch(arch)
        if sd is not None:
            load_strict(model, sd, label)
        device, dtype = resolve_device(spec.get("placement")), DTYPES[spec.get("dtype", "float32")]
        model = model.to(device=device, dtype=dtype).eval()
        info = {"arch": label, "unwrap": steps, "params": sum(p.numel() for p in model.parameters()), **extra}
        return Handle(spec=spec, adapter=self.name, sha256=sha or module_sha256(model), model=model, native=True,
                      device=device, dtype=dtype, info=info)

    def forward(self, handle: Handle, x: torch.Tensor):
        return handle.model(x.to(device=handle.device, dtype=handle.dtype))
