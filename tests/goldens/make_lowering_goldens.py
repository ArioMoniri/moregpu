#!/usr/bin/env python3
"""Cross-component goldens: the PYTHON LOWERING's output (apps/worker/moregpu_worker/vision/lowering.py, target 'wgsl')
run by the TS WGSL executor (apps/worker/vision_wgsl.ts). Complements make_wgsl_goldens.py, whose graphs come from a
small reference exporter — these come from the production lowering, so tests/webgpu/lowering_parity.test.ts proves the
two components agree on the schema end to end.

Writes tests/goldens/wgsl/lowered/<name>.{graph.json,safetensors,io.json}:

* unet3d   — tiny MONAI BasicUNet (3D, instance norm, transposed conv, pad, leaky_relu_) at 16³
* segvit   — moregpu_worker.vision.models.build('segment', vit_config('micro', (32, 32), 8, 3), 3)
             (ViT encoder: patch-embed, SDPA, unbind → select; conv decoder: GroupNorm, bilinear upsample, skip cat)
* vit_tiny — ViT-Tiny-width encoder (dim 192, 3 heads, MLP 768), depth 1, 32×32 / 16×16 patches → tokens

The graph + safetensors bytes are exactly what `vision_lower {include_bytes: true}` returns (graph_json, weights_b64).
Deterministic: fixed seeds, CPU, float32. Run:  python3 tests/goldens/make_lowering_goldens.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps", "worker"))

from moregpu_worker.models.vit import VisionTransformer, vit_config  # noqa: E402
from moregpu_worker.vision import adapters as A  # noqa: E402
from moregpu_worker.vision import lowering as L  # noqa: E402
from moregpu_worker.vision import models as VM  # noqa: E402

OUT = os.path.join(HERE, "wgsl", "lowered")


def tjson(t: torch.Tensor) -> dict:
    a = t.detach().to(torch.float32).contiguous().numpy().astype("<f4")
    return {"shape": list(t.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def models():
    from monai.networks.nets import BasicUNet
    torch.manual_seed(0)
    unet = BasicUNet(spatial_dims=3, in_channels=1, out_channels=2, features=(4, 4, 8, 8, 16, 4)).eval()
    with torch.no_grad():  # non-trivial instance-norm affine params (default init is 1/0)
        for m in unet.modules():
            if isinstance(m, torch.nn.InstanceNorm3d) and m.affine:
                m.weight.add_(torch.randn_like(m.weight) * 0.2)
                m.bias.add_(torch.randn_like(m.bias) * 0.2)
    torch.manual_seed(5)
    seg = VM.build("segment", vit_config("micro", (32, 32), 8, 3), 3).eval()
    torch.manual_seed(6)
    vit = VisionTransformer(img_size=(32, 32), patch=16, in_chans=3, embed_dim=192, depth=1, heads=3).eval()
    return [("unet3d", unet, (1, 1, 16, 16, 16)), ("segvit", seg, (1, 3, 32, 32)), ("vit_tiny", vit, (1, 3, 32, 32))]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, model, shape in models():
        torch.manual_seed(100)
        x = torch.randn(*shape)
        art = L.lower(A.from_module(model), "wgsl", example=x, cache=False)
        if not (art.kind == "opgraph" and art.servable):
            raise SystemExit(f"{name}: not lowered to an op-graph ({art.kind}; {art.reason}; {art.unsupported_ops})")
        with torch.no_grad():
            y = model(x)
        with open(os.path.join(OUT, f"{name}.graph.json"), "w") as f:
            f.write(art.graph_json())
        with open(os.path.join(OUT, f"{name}.safetensors"), "wb") as f:
            f.write(art.weights_bytes())
        (out_name,) = art.graph["outputs"]
        (in_spec,) = art.graph["inputs"]
        with open(os.path.join(OUT, f"{name}.io.json"), "w") as f:
            json.dump({"inputs": {in_spec["name"]: tjson(x)}, "expected": {out_name: tjson(y)},
                       "parity": art.parity, "lowering_version": L.LOWERING_VERSION, "torch": torch.__version__}, f)
        print(f"  {name}: {len(art.graph['nodes'])} nodes, ops={sorted({L.base_op(n['op']) for n in art.graph['nodes']})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
