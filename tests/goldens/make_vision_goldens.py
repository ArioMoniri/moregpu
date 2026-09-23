"""Goldens for moregpu_worker.vision (Dice+CE loss, Dice metric, sliding-window inference) computed with MONAI.
Run: python3 tests/goldens/make_vision_goldens.py   (needs monai; outputs tests/goldens/vision_goldens.npz)"""
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

g = torch.Generator().manual_seed(0)
out = {}
# Dice+CE (softmax, one-hot target, include background, smooth 1e-5)
for nd, shape in (("2d", (2, 3, 8, 8)), ("3d", (2, 3, 4, 8, 8))):
    logits = torch.randn(shape, generator=g)
    tgt = torch.randint(0, 3, (shape[0], 1) + shape[2:], generator=g)
    loss = DiceCELoss(to_onehot_y=True, softmax=True)(logits, tgt)
    out[f"dicece_{nd}_logits"], out[f"dicece_{nd}_tgt"], out[f"dicece_{nd}_loss"] = logits.numpy(), tgt.numpy(), loss.numpy()
# Dice metric per class (no background), one-hot inputs
pred = torch.randint(0, 3, (2, 1, 6, 6, 6), generator=g); gt = torch.randint(0, 3, (2, 1, 6, 6, 6), generator=g)
oh = lambda x: torch.nn.functional.one_hot(x[:, 0], 3).permute(0, 4, 1, 2, 3)
dm = DiceMetric(include_background=False, reduction="none")(oh(pred), oh(gt))
out["dice_pred"], out["dice_gt"], out["dice_val"] = pred.numpy(), gt.numpy(), dm.numpy()
# sliding window, gaussian + constant, 2D and 3D, with a deterministic conv "model"
torch.manual_seed(1)
conv3 = torch.nn.Conv3d(1, 2, 3, padding=1); conv2 = torch.nn.Conv2d(1, 2, 3, padding=1)
x3 = torch.randn(1, 1, 20, 23, 17, generator=g); x2 = torch.randn(2, 1, 37, 29, generator=g)
with torch.no_grad():
    for mode in ("gaussian", "constant"):
        out[f"sw3_{mode}"] = sliding_window_inference(x3, (8, 8, 8), 3, conv3, overlap=0.5, mode=mode).numpy()
        out[f"sw2_{mode}"] = sliding_window_inference(x2, (16, 16), 2, conv2, overlap=0.25, mode=mode).numpy()
out["sw3_x"], out["sw2_x"] = x3.numpy(), x2.numpy()
out["conv3_w"], out["conv3_b"] = conv3.weight.detach().numpy(), conv3.bias.detach().numpy()
out["conv2_w"], out["conv2_b"] = conv2.weight.detach().numpy(), conv2.bias.detach().numpy()
np.savez_compressed(Path(__file__).with_name("vision_goldens.npz"), **out)
print("wrote vision_goldens.npz")
