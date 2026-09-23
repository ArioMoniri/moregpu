"""Golden for ``pred_sha256`` (moregpu.pred/1), shared by the Python worker (moregpu_worker.vision.pred_hash) and the
coordinator (apps/coordinator/lib/pred_hash.ts). Hand-rolled here with hashlib + struct only, independently of both.

Preimage (bytes):  b"moregpu.pred/1\\n" + dtype + b"\\n" + shape + b"\\n" + labels
  dtype   "uint8" if every label <= 255 (also for an empty volume), else "uint16" (labels > 65535 are refused)
  shape   the label volume's shape as decimal integers joined by "," (no spaces; "" for a 0-d scalar)
  labels  the labels in C order (last axis fastest), little-endian, 1 or 2 bytes each

Labels from logits: argmax over the class axis (axis 1 of [B, C, ...]) → [B, ...]; the first maximum wins a tie and a
NaN counts as the maximum (the first NaN wins), exactly as torch.argmax / numpy.argmax.
Run: python3 tests/goldens/make_pred_golden.py"""
import hashlib
import json
import math
import struct
from pathlib import Path


def pred_sha256(labels, shape):
    dt = "uint8" if all(v <= 255 for v in labels) else "uint16"
    body = bytes(labels) if dt == "uint8" else struct.pack(f"<{len(labels)}H", *labels)
    pre = b"moregpu.pred/1\n" + dt.encode() + b"\n" + ",".join(map(str, shape)).encode() + b"\n" + body
    return dt, hashlib.sha256(pre).hexdigest(), pre.hex()


def argmax1(flat, shape):
    B, C, rest = shape[0], shape[1], shape[2:]
    inner = 1
    for d in rest:
        inner *= d
    out = []
    for b in range(B):
        for i in range(inner):
            best, bv = 0, None
            for c in range(C):
                v = flat[(b * C + c) * inner + i]
                if bv is None:
                    best, bv = c, v
                elif math.isnan(bv):
                    break
                elif math.isnan(v) or v > bv:
                    best, bv = c, v
            out.append(best)
    return out, [B] + list(rest)


cases = []
for name, labels, shape in [
    ("u8_3d", [(i * 7) % 3 for i in range(24)], [2, 3, 4]),
    ("u8_max255", [0, 255, 17, 1], [4]),
    ("u16", [0, 256, 300, 65535, 1, 2], [2, 3]),
    ("empty", [], [0, 4]),
    ("scalar", [5], []),
]:
    dt, h, pre = pred_sha256(labels, shape)
    cases.append({"name": name, "labels": labels, "shape": shape, "dtype": dt, "sha256": h, "preimage_hex": pre})

# logits [B=2, C=3, H=2, W=2] with ties (first wins) and a NaN (NaN wins)
logits = [0.1, 0.9, 0.5, 0.5,   0.2, 0.9, 0.5, -1.0,   0.3, 0.1, 0.4, 0.5,
          1.0, 1.0, float("nan"), 0.0,   2.0, 0.5, 0.5, 0.0,   1.0, 3.0, 9.0, float("nan")]
labels, lshape = argmax1(logits, [2, 3, 2, 2])
dt, h, _ = pred_sha256(labels, lshape)
logit_case = {"logits": [None if math.isnan(v) else v for v in logits], "shape": [2, 3, 2, 2], "labels": labels,
              "labels_shape": lshape, "sha256": h,
              "logits_f32le_hex": b"".join(struct.pack("<f", v) for v in logits).hex()}

Path(__file__).with_name("pred_sha256.json").write_text(json.dumps({"version": "moregpu.pred/1", "cases": cases,
                                                                    "logits": logit_case}, indent=1) + "\n")
print("wrote pred_sha256.json")
