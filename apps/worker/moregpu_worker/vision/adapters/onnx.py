"""ONNX adapter (ADR-0113 priority 4): onnxruntime InferenceSession, providers CUDA → CoreML → CPU as available."""
from __future__ import annotations

import numpy as np
import torch

from ..errors import RefusedFormat, brief
from .base import Adapter, Fetch, Handle, fetch_verified

PREFERRED = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
_NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64,
       "tensor(uint8)": np.uint8, "tensor(int64)": np.int64}


def pick_providers(available) -> list[str]:
    out = [p for p in PREFERRED if p in available]
    return out if "CPUExecutionProvider" in out else out + ["CPUExecutionProvider"]


def available_providers() -> list[str]:
    """Providers this worker would use, in preference order ([] when onnxruntime is not installed)."""
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - onnx extra not installed
        return []
    return pick_providers(ort.get_available_providers())


def session(model, providers=None):
    import onnxruntime as ort
    return ort.InferenceSession(model, providers=providers or pick_providers(ort.get_available_providers()))


class OnnxAdapter(Adapter):
    name = "onnx"

    def load(self, spec: dict, fetch: Fetch) -> Handle:
        path, sha = fetch_verified(spec, fetch)
        try:
            sess = session(str(path))
        except Exception as e:
            raise RefusedFormat(f"{path.name} is not a loadable ONNX model ({brief(e)})") from None
        dev = "cuda" if sess.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
        return Handle(spec=spec, adapter=self.name, sha256=sha, model=sess, device=dev, extra={"path": path})

    def forward(self, handle: Handle, x: torch.Tensor):
        inp = handle.model.get_inputs()[0]
        arr = x.detach().cpu().numpy().astype(_NP.get(inp.type, np.float32), copy=False)
        return torch.from_numpy(handle.model.run(None, {inp.name: arr})[0])

    def _describe(self, handle: Handle) -> dict:
        s = handle.model

        def io(xs):
            return [{"name": a.name, "shape": [d if isinstance(d, int) else None for d in a.shape], "type": a.type}
                    for a in xs]

        return {"providers": s.get_providers(), "inputs": io(s.get_inputs()), "outputs": io(s.get_outputs())}
