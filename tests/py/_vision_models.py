"""Tiny in-test models and helpers for the M4 model-adapter tests (ADR-0113/0114). Nothing is ever downloaded."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class TinyNet(nn.Module):
    """Conv-BN-ReLU-pool-upsample-cat-softmax-linear: every op is in the WGSL table."""

    def __init__(self, in_ch: int = 3, classes: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.gn = nn.GroupNorm(2, 8)
        self.conv2 = nn.Conv2d(16, 8, 1)
        self.fc = nn.Linear(8, classes)

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        p = F.max_pool2d(h, 2)
        u = F.interpolate(p, scale_factor=2.0, mode="nearest")
        h = self.gn(self.conv2(torch.cat([h, u], dim=1)))
        h = torch.sigmoid(h) * h + F.gelu(h)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return torch.softmax(self.fc(h), dim=-1)


class FftNet(nn.Module):
    """Uses torch.fft, which neither the WGSL table nor the ONNX exporter supports."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x):
        return torch.fft.rfft2(self.conv(x)).abs()


class CumsumNet(nn.Module):
    """aten.cumsum is outside the WGSL executor's table but ONNX-exportable → the ONNX fallback."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)

    def forward(self, x):
        return torch.cumsum(self.conv(x), dim=-1)


class InplaceReuseNet(nn.Module):
    """h.add_() mutates a tensor that is read again afterwards — not safely functionalisable by renaming."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        g = h * 2.0
        h.add_(1.0)
        return g + h


def tiny_monai_unet():
    from monai.networks.nets import UNet
    torch.manual_seed(0)
    return UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(4, 8, 16), strides=(2, 2),
                num_res_units=1).eval()


def tiny_basic_unet():
    from monai.networks.nets import BasicUNet
    torch.manual_seed(0)
    return BasicUNet(spatial_dims=3, in_channels=1, out_channels=2, features=(4, 4, 8, 8, 16, 4)).eval()


def tiny_vit():
    import timm
    torch.manual_seed(0)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=32, num_classes=10).eval()


def save_state_dict(model: nn.Module, path: Path, wrap=None) -> str:
    sd = model.state_dict()
    obj = wrap(sd) if wrap else sd
    torch.save(obj, path)
    return sha256_file(path)


def spec_for(path: Path, fmt: str, arch=None, **extra) -> dict:
    s = {"format": fmt, "source": Path(path).resolve().as_uri(), "sha256": sha256_file(path)}
    if arch:
        s["arch"] = arch
    s.update(extra)
    return s
