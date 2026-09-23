"""Refusals and load errors for model adapters (ADR-0113). Every refusal says what to do instead."""
from __future__ import annotations

REFUSAL_HINT = ("MoreGPU never unpickles arbitrary objects: export a state_dict (torch.save(model.state_dict(), p)) or "
                "safetensors plus a named architecture, a torch.export .pt2, TorchScript or ONNX; or ask the worker "
                "admin to install an allowlisted `moregpu.models` plugin")


def brief(e: BaseException, n: int = 200) -> str:
    """First line of an exception message, truncated (safe for empty messages)."""
    lines = str(e).splitlines()
    return (lines[0] if lines else type(e).__name__)[:n]


class Refused(Exception):
    """Base class: the worker refuses to load this artefact."""


class RefusedFormat(Refused):
    """The artefact is not a safe, supported format (e.g. a pickled full model)."""

    def __init__(self, detail: str):
        super().__init__(f"{detail}. {REFUSAL_HINT}.")


class RefusedSource(Refused):
    """The source URI is not allowed or not resolvable on this worker."""


class RefusedPlugin(Refused):
    """A `moregpu.models` plugin is installed but not pinned in the worker allowlist."""


class IntegrityError(Refused):
    """The artefact's sha256 does not match the spec."""


class UnknownArch(LookupError):
    """The named architecture does not exist in its registry (or the registry is not installed)."""


class NotNative(TypeError):
    """Training needs a native torch module; exported/ONNX handles are inference-only."""


class NotLoaded(RuntimeError):
    """The handle was unloaded."""


class KeyMismatch(ValueError):
    """strict=True state_dict load failed; carries the key diff."""

    def __init__(self, arch: str, missing: list[str], unexpected: list[str], shape_mismatch: list[tuple]):
        self.arch, self.missing, self.unexpected, self.shape_mismatch = arch, missing, unexpected, shape_mismatch

        def show(xs, n=20):
            return ", ".join(xs[:n]) + (f", ... (+{len(xs) - n})" if len(xs) > n else "")

        shapes = [f"{k}: checkpoint {tuple(a)} vs model {tuple(b)}" for k, a, b in shape_mismatch]
        super().__init__(
            f"state_dict does not match {arch} (strict load): "
            f"{len(missing)} missing [{show(missing)}]; {len(unexpected)} unexpected [{show(unexpected)}]; "
            f"{len(shapes)} shape mismatches [{show(shapes)}]. Check arch.name/arch.kwargs against the checkpoint")
