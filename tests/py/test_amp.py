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
