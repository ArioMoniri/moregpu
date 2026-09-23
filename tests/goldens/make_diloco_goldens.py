"""Regenerate tests/goldens/diloco_tensorwire.json — cross-language goldens shared by the Python reference
(moregpu_worker.train.{tensorwire,diloco}) and the coordinator TS lib (apps/coordinator/lib/*.ts).
Run: python3 tests/goldens/make_diloco_goldens.py"""
import base64, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "worker"))
from moregpu_worker.train import diloco as D, tensorwire as tw  # noqa: E402

g = torch.Generator().manual_seed(1234)
ref = {"a": torch.randn(3, 5, generator=g), "b": torch.randn(7, generator=g)}
new = {k: v + 0.05 * torch.randn(v.shape, generator=g) for k, v in ref.items()}
out = {"ref": {k: v.reshape(-1).tolist() for k, v in ref.items()},
       "shapes": {k: list(v.shape) for k, v in ref.items()}, "wire": {}}
for dt in ("f32", "bf16", "fp16", "int8delta"):
    hdr, blob = tw.encode(new, dt, ref=ref if dt == "int8delta" else None, block=4)
    dec = tw.decode(hdr, blob, ref=ref if dt == "int8delta" else None)
    out["wire"][dt] = {"header": hdr, "blob_b64": base64.b64encode(blob).decode(),
                       "decoded": {k: v.reshape(-1).tolist() for k, v in dec.items()}}
# f32 encoding of the "new" values (TS encode must reproduce these bytes exactly)
out["new"] = {k: v.reshape(-1).tolist() for k, v in new.items()}
# DiLoCo: 3 workers, 3 rounds, weights by samples
st = D.OuterState.init(ref)
rounds = []
for r in range(3):
    ws = [{k: v + 0.1 * torch.randn(v.shape, generator=g) for k, v in st.global_.items()} for _ in range(3)]
    samples = [4, 8, 12]
    avg = D.weighted_average(list(zip(ws, samples)))
    D.outer_step(st, avg, lr=0.7, momentum=0.9)
    rounds.append({"workers": [{k: v.reshape(-1).tolist() for k, v in w.items()} for w in ws], "samples": samples,
                   "global_after": {k: v.reshape(-1).tolist() for k, v in st.global_.items()}})
out["diloco"] = {"init": out["ref"], "lr": 0.7, "momentum": 0.9, "rounds": rounds}
(Path(__file__).with_name("diloco_tensorwire.json")).write_text(json.dumps(out))
print("wrote diloco_tensorwire.json")
