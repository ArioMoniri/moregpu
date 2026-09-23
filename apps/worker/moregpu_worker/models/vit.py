"""Vision Transformer (2D and 3D patches) with timm-compatible parameter names, plus the narrow I-JEPA predictor.

State-dict keys match `timm` VisionTransformer(class_token=False, global_pool='') exactly — `patch_embed.proj`,
`pos_embed`, `blocks.{i}.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}`, `norm` — so exported encoders load into timm
with strict=True (tests/py/test_jepa_model.py). Positional embeddings are fixed sin-cos (stored as a frozen Parameter so
the key exists, as I-JEPA does)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SIZES = {  # name: (embed_dim, depth, heads)
    "micro": (32, 2, 2),
    "tiny": (192, 12, 3),
    "small": (384, 12, 6),
    "base": (768, 12, 12),
}


def _tuple(v, n):
    return tuple(v) if isinstance(v, (tuple, list)) else (v,) * n


def vit_config(size: str, img_size, patch, in_chans: int, **kw) -> dict:
    d, depth, heads = SIZES[size]
    return {"img_size": tuple(img_size), "patch": patch, "in_chans": in_chans, "embed_dim": d, "depth": depth,
            "heads": heads, **kw}


def _sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = 1.0 / (10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2)))
    out = pos.reshape(-1, 1).double() * omega[None]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def sincos_pos_embed(dim: int, grid: tuple[int, ...]) -> torch.Tensor:
    """Fixed sin-cos embedding for a 2D (h,w) or 3D (d,h,w) token grid, row-major token order → (N, dim) float32."""
    axes = torch.meshgrid(*[torch.arange(g) for g in grid], indexing="ij")
    k = len(grid)
    per = (dim // (2 * k)) * 2
    parts = [_sincos_1d(per, a) for a in axes]
    emb = torch.cat(parts, dim=1)
    if emb.shape[1] < dim:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], dim - emb.shape[1], dtype=emb.dtype)], dim=1)
    return emb.float()


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch, in_chans, dim):
        super().__init__()
        nd = len(img_size)
        self.patch = _tuple(patch, nd)
        self.grid = tuple(s // p for s, p in zip(img_size, self.patch))
        conv = nn.Conv2d if nd == 2 else nn.Conv3d
        self.proj = conv(in_chans, dim, kernel_size=self.patch, stride=self.patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def _rescale_blocks(blocks) -> None:
    """I-JEPA's fix_init_weight: scale residual-branch output projections by 1/sqrt(2·layer)."""
    with torch.no_grad():
        for i, blk in enumerate(blocks):
            blk.attn.proj.weight.div_(math.sqrt(2.0 * (i + 1)))
            blk.mlp.fc2.weight.div_(math.sqrt(2.0 * (i + 1)))


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch, in_chans, embed_dim, depth, heads, mlp_ratio=4.0, grad_checkpointing=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch, in_chans, embed_dim)
        self.grid = self.patch_embed.grid
        self.num_patches = math.prod(self.grid)
        self.pos_embed = nn.Parameter(sincos_pos_embed(embed_dim, self.grid)[None], requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.grad_checkpointing = grad_checkpointing
        self.apply(_init)
        _rescale_blocks(self.blocks)

    def forward(self, x, keep: torch.Tensor | None = None):
        x = self.patch_embed(x) + self.pos_embed
        if keep is not None:
            x = torch.gather(x, 1, keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.grad_checkpointing and self.training) else blk(x)
        return self.norm(x)


class Predictor(nn.Module):
    """Narrow ViT that predicts target-token representations from context tokens (I-JEPA)."""
    def __init__(self, embed_dim, pred_dim, depth, heads, grid, grad_checkpointing=False):
        super().__init__()
        self.embed = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pos_embed = nn.Parameter(sincos_pos_embed(pred_dim, tuple(grid))[None], requires_grad=False)
        self.blocks = nn.ModuleList([Block(pred_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim, eps=1e-6)
        self.proj = nn.Linear(pred_dim, embed_dim)
        self.grad_checkpointing = grad_checkpointing
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init)
        _rescale_blocks(self.blocks)

    def forward(self, z, ctx_idx, tgt_idx: list[torch.Tensor]):
        """z: (B, Kc, D) context features; ctx_idx (B, Kc); tgt_idx: M tensors (B, Kt). Returns (M·B, Kt, D)."""
        B, M = z.shape[0], len(tgt_idx)
        pos = self.pos_embed.expand(B, -1, -1)
        x = self.embed(z) + torch.gather(pos, 1, ctx_idx.unsqueeze(-1).expand(-1, -1, pos.shape[-1]))
        x = x.repeat(M, 1, 1)
        tgt = torch.cat(tgt_idx, dim=0)                                    # (M·B, Kt)
        posr = pos.repeat(M, 1, 1)
        m = self.mask_token.expand(tgt.shape[0], tgt.shape[1], -1) + torch.gather(
            posr, 1, tgt.unsqueeze(-1).expand(-1, -1, posr.shape[-1]))
        x = torch.cat([x, m], dim=1)
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.grad_checkpointing and self.training) else blk(x)
        x = self.norm(x)[:, -tgt.shape[1]:]
        return self.proj(x)
