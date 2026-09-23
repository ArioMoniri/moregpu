import pytest
import torch

from moregpu_worker.train import amp as A


def test_auto_on_cpu_is_fp32():
    p = A.resolve("auto", "cpu")
    assert p.mode == "fp32" and p.scaler is None


def test_auto_cuda_bf16_vs_fp16(monkeypatch):
    monkeypatch.setattr(A, "_cuda_bf16_supported", lambda: True)
    assert A.resolve("auto", "cuda").mode == "bf16"
    monkeypatch.setattr(A, "_cuda_bf16_supported", lambda: False)
    p = A.resolve("auto", "cuda")
    assert p.mode == "fp16" and p.uses_scaler


def test_forced_mode_the_device_cannot_run_is_an_error(monkeypatch):
    monkeypatch.setattr(A, "_cuda_bf16_supported", lambda: False)
    with pytest.raises(ValueError):
        A.resolve("bf16", "cuda")
    with pytest.raises(ValueError):
        A.resolve("fp16", "cpu")
    with pytest.raises(ValueError):
        A.resolve("nope", "cpu")


def test_cpu_bf16_allowed_explicitly():
    p = A.resolve("bf16", "cpu")
    assert p.mode == "bf16" and not p.uses_scaler
    with p.autocast():
        y = torch.nn.Linear(4, 4)(torch.randn(2, 4))
    assert y.dtype == torch.bfloat16


def test_fp32_step_and_describe():
    p = A.resolve("fp32", "cpu")
    lin = torch.nn.Linear(3, 1)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    with p.autocast():
        loss = lin(torch.ones(1, 3)).sum()
    p.backward_step(loss, opt)
    d = p.describe()
    assert d["mode"] == "fp32" and d["skipped_steps"] == 0


def test_scaler_path_counts_skipped_steps():
    class FakeScaler:
        def __init__(self): self.s = 1024.0; self.calls = []
        def scale(self, loss): self.calls.append("scale"); return loss * self.s
        def unscale_(self, opt): self.calls.append("unscale")
        def get_scale(self): return self.s
        def step(self, opt): self.calls.append("step")
        def update(self): self.s /= 2   # pretend an inf was found
    p = A.AmpPolicy("fp16", "cuda", FakeScaler())
    lin = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    p.backward_step(lin(torch.ones(1, 2)).sum(), opt, clip=1.0, params=list(lin.parameters()))
    assert p.skipped_steps == 1 and p.describe()["scale"] == 512.0
    assert p.scaler.calls == ["scale", "unscale", "step"]


def test_fp32_clip_and_bad_mps_bf16():
    p = A.resolve("fp32", "cpu")
    lin = torch.nn.Linear(2, 1); opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    p.backward_step(lin(torch.ones(1, 2)).sum() * 100, opt, clip=0.5, params=list(lin.parameters()))
    with pytest.raises(ValueError):
        A.resolve("bf16", "mps")


@pytest.mark.cuda
def test_cuda_autocast_real():
    p = A.resolve("auto", "cuda")
    with p.autocast():
        y = torch.nn.Linear(4, 4).cuda()(torch.randn(2, 4, device="cuda"))
    assert y.dtype in (torch.bfloat16, torch.float16)
