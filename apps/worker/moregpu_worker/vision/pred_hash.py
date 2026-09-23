"""``pred_sha256`` — a deterministic hash of a predicted label volume (``moregpu.pred/1``).

Two workers that predict the same labels report the same hash, so a study can check voxel agreement across workers
(and across the torch and WebGPU paths) without reading the outputs. The coordinator computes the same value in
TypeScript (``apps/coordinator/lib/pred_hash.ts``); ``tests/goldens/pred_sha256.json`` is checked by both.

Preimage, as bytes::

    b"moregpu.pred/1\\n" + DTYPE + b"\\n" + SHAPE + b"\\n" + LABELS

* ``DTYPE``: ``uint8`` if every label is ≤ 255 (also for an empty volume), else ``uint16``. Labels above 65535,
  negative labels and non-integer values are refused.
* ``SHAPE``: the label volume's shape as decimal integers joined by ``,`` with no spaces (``""`` for a 0-d scalar).
* ``LABELS``: the labels in C order (last axis fastest), little-endian, 1 or 2 bytes each.

The dtype depends only on the label values, so the hash depends only on the labels and their shape: it does not matter
how a worker stored them. Labels from logits (:func:`labels_from_logits`) are the argmax over the class axis (axis 1 of
``[B, C, ...]``): the first maximum wins a tie and a NaN counts as the maximum, as in ``torch.argmax``.
"""
from __future__ import annotations

import hashlib

import numpy as np

VERSION = "moregpu.pred/1"


def _as_numpy(labels) -> np.ndarray:
    if hasattr(labels, "detach"):          # torch tensor
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels)


def label_dtype(labels) -> str:
    a = _as_numpy(labels)
    if a.size and not np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_):
        if not np.all(np.isfinite(a)) or not np.all(a == np.floor(a)):
            raise ValueError("labels must be integer class indices")
    if not a.size:
        return "uint8"
    lo, hi = int(a.min()), int(a.max())
    if lo < 0:
        raise ValueError(f"labels must not be negative (min {lo})")
    if hi > 65535:
        raise ValueError(f"labels above 65535 are not supported (max {hi})")
    return "uint8" if hi <= 255 else "uint16"


def canonical_labels(labels) -> np.ndarray:
    """The labels as a C-contiguous little-endian uint8 / uint16 array (the dtype :func:`label_dtype` picks)."""
    a = _as_numpy(labels)
    dt = np.dtype("u1") if label_dtype(a) == "uint8" else np.dtype("<u2")
    return np.array(a, dtype=dt, order="C")      # (np.ascontiguousarray would turn a 0-d scalar into shape (1,))


def preimage(labels) -> bytes:
    c = canonical_labels(labels)
    dt = "uint8" if c.dtype.itemsize == 1 else "uint16"
    shape = ",".join(str(int(d)) for d in c.shape)
    return f"{VERSION}\n{dt}\n{shape}\n".encode() + c.tobytes(order="C")


def pred_sha256(labels) -> str:
    return hashlib.sha256(preimage(labels)).hexdigest()


def labels_from_logits(logits) -> np.ndarray:
    """``[B, C, ...]`` logits/probabilities → ``[B, ...]`` labels (argmax over axis 1, first max wins, NaN wins)."""
    a = _as_numpy(logits)
    if a.ndim < 2:
        raise ValueError(f"logits need a class axis: shape [B, C, ...], got {list(a.shape)}")
    return np.argmax(a, axis=1)
