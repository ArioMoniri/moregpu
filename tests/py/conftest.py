import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "worker"))


def pytest_collection_modifyitems(config, items):
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    for it in items:
        if ("cuda" in it.keywords or "gpu" in it.keywords) and not has_cuda and not os.environ.get("MOREGPU_FORCE_GPU_TESTS"):
            it.add_marker(pytest.mark.skip(reason="needs a CUDA device"))
        if "webgpu" in it.keywords and not os.environ.get("MOREGPU_WEBGPU"):
            it.add_marker(pytest.mark.skip(reason="needs WebGPU"))


def pytest_configure(config):
    for m in ("gpu: needs any accelerator", "cuda: needs CUDA", "webgpu: needs WebGPU", "slow: long-running"):
        config.addinivalue_line("markers", m)


@pytest.fixture(autouse=True)
def _output_dir(tmp_path, monkeypatch):
    """Exports are confined to MOREGPU_OUTPUT_DIR (moregpu_worker.paths): unit tests write under their tmp_path."""
    monkeypatch.setenv("MOREGPU_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("MOREGPU_MODEL_ROOTS", raising=False)
