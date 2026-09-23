import pytest
import torch

from moregpu_worker.models.vit import VisionTransformer, Predictor, vit_config, sincos_pos_embed
from moregpu_worker.train import masking as M


def test_vit_2d_shapes_and_keep_idx():
    m = VisionTransformer(**vit_config("micro", img_size=(32, 32), patch=8, in_chans=3))
    x = torch.randn(2, 3, 32, 32)
    assert m(x).shape == (2, 16, m.embed_dim)
    keep = torch.tensor([[0, 3, 5], [1, 2, 15]])
    assert m(x, keep=keep).shape == (2, 3, m.embed_dim)


def test_vit_3d_shapes():
    m = VisionTransformer(**vit_config("micro", img_size=(16, 32, 32), patch=(4, 8, 8), in_chans=1))
    assert m(torch.randn(1, 1, 16, 32, 32)).shape == (1, 4 * 4 * 4, m.embed_dim)


def test_state_dict_is_timm_compatible():
    timm = pytest.importorskip("timm")
    ours = VisionTransformer(**vit_config("tiny", img_size=(64, 64), patch=16, in_chans=3))
    ref = timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=64, in_chans=3, class_token=False,
                            global_pool="", num_classes=0, dynamic_img_size=False)
    ref.load_state_dict(ours.state_dict(), strict=True)
    ref.eval(); ours.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(ref.forward_features(x), ours(x), atol=1e-5)


def test_gradient_checkpointing_same_result():
    torch.manual_seed(0)
    a = VisionTransformer(**vit_config("micro", img_size=(32, 32), patch=8, in_chans=1))
    b = VisionTransformer(**vit_config("micro", img_size=(32, 32), patch=8, in_chans=1), grad_checkpointing=True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 1, 32, 32)
    a(x).sum().backward(); b(x).sum().backward()
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, atol=1e-5), n


def test_sincos_pos_embed_2d_3d():
    assert sincos_pos_embed(32, (4, 4)).shape == (16, 32)
    assert sincos_pos_embed(48, (2, 4, 4)).shape == (32, 48)


def test_predictor_shapes():
    p = Predictor(embed_dim=32, pred_dim=16, depth=2, heads=2, grid=(4, 4))
    z = torch.randn(2, 5, 32)
    ctx = torch.randint(0, 16, (2, 5)); tgt = [torch.randint(0, 16, (2, 3)), torch.randint(0, 16, (2, 3))]
    out = p(z, ctx, tgt)
    assert out.shape == (2 * 2, 3, 32)


# ------------------------------------------------------------------ masking
@pytest.mark.parametrize("grid", [(8, 8), (4, 8, 8)])
def test_multiblock_masks_properties(grid):
    mk = M.MultiBlockMasker(grid, n_targets=4, target_scale=(0.15, 0.2), target_aspect=(0.75, 1.5),
                            context_scale=(0.85, 1.0), min_keep=4)
    g = torch.Generator().manual_seed(0)
    ctx, tgts = mk(3, g)
    n = 1
    for d in grid:
        n *= d
    assert ctx.shape[0] == 3 and len(tgts) == 4
    kt = {t.shape[1] for t in tgts}
    assert len(kt) == 1                                  # rectangular: equal K across blocks
    for b in range(3):
        c = set(ctx[b].tolist())
        assert c and max(c) < n
        for t in tgts:
            assert not (c & set(t[b].tolist())), "context must not overlap targets"
        # each target block's size is within the scale range (before truncation it's ≥ the kept size)
        assert tgts[0].shape[1] <= int(0.2 * n) + max(grid)
    ctx2, tgts2 = mk(3, torch.Generator().manual_seed(0))
    assert torch.equal(ctx, ctx2) and all(torch.equal(a, b) for a, b in zip(tgts, tgts2))
    ctx3, _ = mk(3, torch.Generator().manual_seed(1))
    assert not torch.equal(ctx, ctx3) or ctx.shape != ctx3.shape


def test_mask_coverage_over_many_draws():
    mk = M.MultiBlockMasker((8, 8), n_targets=4)
    g = torch.Generator().manual_seed(0)
    covered = torch.zeros(64)
    for _ in range(50):
        _, tgts = mk(1, g)
        for t in tgts:
            covered[t[0]] = 1
    assert covered.mean() > 0.9     # targets eventually cover the whole grid (no dead zones)


def test_gather_tokens():
    x = torch.arange(24.).reshape(1, 6, 4)
    out = M.gather_tokens(x, torch.tensor([[5, 0]]))
    assert torch.equal(out[0, 0], x[0, 5]) and torch.equal(out[0, 1], x[0, 0])
