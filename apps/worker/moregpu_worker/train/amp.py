"""AMP policy (ADR-0108): auto → bf16 on CUDA with bf16 support, fp16 + GradScaler on other CUDA (e.g. Turing),
fp32 on CPU/MPS. Forced modes the device can't run are errors, never silent downgrades. Master weights stay fp32."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch

MODES = ("auto", "bf16", "fp16", "fp32")


def _cuda_bf16_supported() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


@dataclass
class AmpPolicy:
    mode: str
    device: str
    scaler: "torch.amp.GradScaler | None" = None
    skipped_steps: int = 0
    _last_scale: float = field(default=1.0, repr=False)

    @property
    def uses_scaler(self) -> bool:
        return self.mode == "fp16"

    def autocast(self):
        if self.mode == "fp32":
            return contextlib.nullcontext()
        dt = torch.bfloat16 if self.mode == "bf16" else torch.float16
        dev = "cuda" if self.device.startswith("cuda") else self.device
        return torch.autocast(device_type=dev, dtype=dt)

    def backward_step(self, loss: torch.Tensor, opt: torch.optim.Optimizer, clip: float | None = None,
                      params=None) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if clip:
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, clip)
            before = self.scaler.get_scale()
            self.scaler.step(opt)
            self.scaler.update()
            if self.scaler.get_scale() < before:  # inf/nan grads → step skipped and scale reduced
                self.skipped_steps += 1
            self._last_scale = self.scaler.get_scale()
        else:
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()

    def describe(self) -> dict:
        return {"mode": self.mode, "scaler": self.uses_scaler, "scale": self._last_scale if self.uses_scaler else None,
                "skipped_steps": self.skipped_steps}


def resolve(requested: str, device: str) -> AmpPolicy:
    if requested not in MODES:
        raise ValueError(f"amp must be one of {MODES}, got {requested!r}")
    cuda = device.startswith("cuda")
    if requested == "auto":
        mode = ("bf16" if _cuda_bf16_supported() else "fp16") if cuda else "fp32"
    else:
        mode = requested
    if mode == "bf16" and cuda and not _cuda_bf16_supported():
        raise ValueError("bf16 requested but this CUDA device does not support it (use fp16 or auto)")
    if mode == "fp16" and not cuda:
        raise ValueError(f"fp16 AMP needs CUDA; device is {device}")
    if mode == "bf16" and device == "mps":
        raise ValueError("bf16 autocast is not supported on MPS here; use fp32")
    scaler = torch.amp.GradScaler("cuda") if mode == "fp16" else None
    return AmpPolicy(mode, device, scaler)
