"""Segmentation / classification models on a ViT encoder (2D or 3D), plus loading of exported checkpoints.

SegViT: ViT tokens → progressive ×2 upsampling conv decoder, concatenated with a full-resolution conv stem of the
input (UNETR-lite skip), → 1×1 conv to classes, resized to the input size. ClsViT: mean-pooled tokens → linear."""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.vit import VisionTransformer


def _conv(nd):
    return nn.Conv2d if nd == 2 else nn.Conv3d


def _norm(nd, c):
    return nn.GroupNorm(min(8, c), c) if c >= 2 else nn.Identity()


class SegDecoder(nn.Module):
    def __init__(self, embed_dim: int, in_chans: int, num_classes: int, nd: int, grid, img_size, channels=(64, 32, 16)):
        super().__init__()
        self.nd, self.grid, self.img_size = nd, tuple(grid), tuple(img_size)
        Conv = _conv(nd)
        n_up = max(1, int(math.ceil(math.log2(max(s // g for s, g in zip(img_size, grid))))))
        chans = list(channels)[:n_up] + [channels[-1]] * max(0, n_up - len(channels))
        layers, c_in = [], embed_dim
        for c in chans:
            layers.append(nn.Sequential(Conv(c_in, c, 3, padding=1), _norm(nd, c), nn.GELU()))
            c_in = c
        self.ups = nn.ModuleList(layers)
        self.stem = nn.Sequential(Conv(in_chans, chans[-1], 3, padding=1), _norm(nd, chans[-1]), nn.GELU())
        self.fuse = nn.Sequential(Conv(2 * chans[-1], chans[-1], 3, padding=1), _norm(nd, chans[-1]), nn.GELU())
        self.head = Conv(chans[-1], num_classes, 1)

    def forward(self, tokens: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        h = tokens.transpose(1, 2).reshape((B, D) + self.grid)
        mode = "bilinear" if self.nd == 2 else "trilinear"
        for up in self.ups:
            h = up(F.interpolate(h, scale_factor=2, mode=mode, align_corners=False))
        h = F.interpolate(h, size=tuple(x.shape[2:]), mode=mode, align_corners=False)
        return self.head(self.fuse(torch.cat([h, self.stem(x)], dim=1)))


class SegViT(nn.Module):
    def __init__(self, encoder: VisionTransformer, decoder: SegDecoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, x):
        return self.decoder(self.encoder(x), x)


class ClsViT(nn.Module):
    def __init__(self, encoder: VisionTransformer, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x).mean(dim=1))


def build(task: str, vit_cfg: dict, num_classes: int, decoder_channels=(64, 32, 16)) -> nn.Module:
    enc = VisionTransformer(**{**vit_cfg, "img_size": tuple(vit_cfg["img_size"])})
    if task == "classify":
        return ClsViT(enc, num_classes)
    nd = len(vit_cfg["img_size"])
    return SegViT(enc, SegDecoder(enc.embed_dim, vit_cfg["in_chans"], num_classes, nd, enc.grid, vit_cfg["img_size"],
                                  tuple(decoder_channels)))


def load_exported(path: str) -> nn.Module:
    """Load a model written by SegmentTask/ClassifyTask.export('safetensors', path) (strict)."""
    from safetensors.torch import load_file
    meta = json.load(open(os.path.join(path, "model_config.json")))
    m = build(meta["task"], meta["encoder"], meta["num_classes"], meta.get("decoder_channels", (64, 32, 16)))
    m.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=True)
    return m.eval()
