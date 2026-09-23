"""MOREGPU_VRAM_FRACTION: cap this worker process's share of each CUDA device (torch.cuda.set_per_process_memory_fraction),
validated at start-up and reported in the hw fingerprint (telemetry ``hw``) so run records show it was applied."""
import math
import types

import pytest

from moregpu_worker import vram
from moregpu_worker.telemetry import hw


def fake_torch(cuda=True, n=2):
    calls = []
    props = types.SimpleNamespace(name="Fake GPU", total_memory=8 << 30, major=8, minor=0)
    c = types.SimpleNamespace(
        is_available=lambda: cuda, device_count=lambda: n if cuda else 0, current_device=lambda: 0,
        get_device_properties=lambda i: props,
        set_per_process_memory_fraction=lambda f, device=None: calls.append((f, device)))
    return types.SimpleNamespace(__version__="9.9.9", version=types.SimpleNamespace(cuda="12.9"), cuda=c), calls


@pytest.fixture(autouse=True)
def _reset():
    vram.reset()
    yield
    vram.reset()


@pytest.mark.parametrize("raw, want", [("0.5", 0.5), ("1", 1.0), ("1.0", 1.0), (" 0.25 ", 0.25), ("1e-3", 0.001)])
def test_parse_accepts_fractions_in_range(raw, want):
    assert vram.parse(raw) == want


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_parse_unset_is_none(raw):
    assert vram.parse(raw) is None


@pytest.mark.parametrize("raw", ["0", "-0.1", "1.01", "2", "nan", "NaN", "inf", "-inf", "abc", "0.5x", "50%"])
def test_parse_rejects_nan_and_out_of_range(raw):
    with pytest.raises(vram.VramFractionError, match="MOREGPU_VRAM_FRACTION"):
        vram.parse(raw)
    assert issubclass(vram.VramFractionError, ValueError)


def test_apply_sets_fraction_on_every_cuda_device():
    t, calls = fake_torch(n=2)
    assert vram.apply(torch_mod=t, env={"MOREGPU_VRAM_FRACTION": "0.4"}, device="cuda") == 0.4
    assert calls == [(0.4, 0), (0.4, 1)]
    assert vram.applied() == 0.4
    assert vram.status() == {"requested": 0.4, "applied": 0.4}


def test_apply_unset_is_a_noop():
    t, calls = fake_torch()
    assert vram.apply(torch_mod=t, env={}, device="cuda") is None
    assert calls == [] and vram.applied() is None and vram.status() == {"requested": None, "applied": None}


def test_apply_without_cuda_does_not_apply_but_still_validates():
    t, calls = fake_torch(cuda=False)
    assert vram.apply(torch_mod=t, env={"MOREGPU_VRAM_FRACTION": "0.5"}, device="cpu") is None
    assert calls == [] and vram.status() == {"requested": 0.5, "applied": None}
    with pytest.raises(vram.VramFractionError):
        vram.apply(torch_mod=t, env={"MOREGPU_VRAM_FRACTION": "nan"}, device="cpu")


def test_apply_on_forced_cpu_worker_skips_cuda():
    t, calls = fake_torch(cuda=True)
    assert vram.apply(torch_mod=t, env={"MOREGPU_VRAM_FRACTION": "0.5"}, device="cpu") is None
    assert calls == []


def test_apply_rejects_bad_value_before_touching_cuda():
    t, calls = fake_torch()
    with pytest.raises(vram.VramFractionError, match="0 < f <= 1"):
        vram.apply(torch_mod=t, env={"MOREGPU_VRAM_FRACTION": "1.5"}, device="cuda")
    assert calls == [] and vram.applied() is None


def test_apply_reads_process_env(monkeypatch):
    t, calls = fake_torch(n=1)
    monkeypatch.setenv("MOREGPU_VRAM_FRACTION", "0.75")
    assert vram.apply(torch_mod=t, device="cuda:0") == 0.75 and calls == [(0.75, 0)]


def test_fingerprint_reports_applied_fraction():
    t, _ = fake_torch()
    assert hw.fingerprint(torch_mod=t)["vram_fraction"] is None
    vram.apply(torch_mod=t, env={"MOREGPU_VRAM_FRACTION": "0.3"}, device="cuda")
    fp = hw.fingerprint(torch_mod=t)
    assert fp["vram_fraction"] == 0.3 and math.isfinite(fp["vram_fraction"])


def test_worker_torch_applies_fraction_at_startup():
    """worker_torch.py calls vram.apply(device=DEV) at import (start-up), before any model is loaded."""
    src = (vram.__file__.rsplit("/moregpu_worker/", 1)[0] + "/worker_torch.py")
    text = open(src).read()
    assert "vram.apply(" in text or "_vram.apply(" in text
    # applied before the first resident-model/session state is created
    assert text.index("apply(device=DEV)") < text.index("MODELS: dict = {}")
