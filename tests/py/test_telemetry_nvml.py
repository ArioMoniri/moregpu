"""GpuSampler (ADR-0111): NVML background sampler with a no-op fallback; tested against a fake pynvml module."""
import sys
import time
import types

import pytest

from moregpu_worker.telemetry import nvml as N


class NVMLError(Exception):
    pass


def fake_nvml(util=(50, 70), power_mw=(100_000, 200_000), used=(1 << 20, 3 << 20), energy_mj=None,
              init_fails=False):
    """A pynvml stand-in. Each read advances through the given sequences (the last value repeats)."""
    st = {"u": 0, "p": 0, "m": 0, "e": 0, "init": 0, "shutdown": 0, "index": None}

    def seq(key, vals):
        i = min(st[key], len(vals) - 1)
        st[key] += 1
        return vals[i]

    m = types.SimpleNamespace()
    m.NVMLError = NVMLError
    m.state = st

    def nvmlInit():
        if init_fails:
            raise NVMLError("driver not loaded")
        st["init"] += 1
    m.nvmlInit = nvmlInit
    m.nvmlShutdown = lambda: st.__setitem__("shutdown", st["shutdown"] + 1)

    def handle(i):
        st["index"] = i
        return ("h", i)
    m.nvmlDeviceGetHandleByIndex = handle
    m.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=seq("u", util), memory=0)
    m.nvmlDeviceGetPowerUsage = lambda h: seq("p", power_mw)
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=seq("m", used), total=16 << 30, free=0)

    def energy(h):
        if energy_mj is None:
            raise NVMLError("Not Supported")
        return seq("e", energy_mj)
    m.nvmlDeviceGetTotalEnergyConsumption = energy
    return m


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_trapezoid_energy_and_means_manual_sampling():
    c = Clock()
    s = N.GpuSampler(device_index=1, period_s=0.1, nvml=fake_nvml(util=(40, 60, 80),
                                                                   power_mw=(100_000, 200_000, 300_000),
                                                                   used=(10, 30, 20)), clock=c, thread=False)
    s.start()                     # sample at t=0
    c.t = 1.0
    s.sample()                    # t=1
    c.t = 3.0
    out = s.stop()                # final sample at t=3
    assert out["gpu_util_mean"] == pytest.approx(60.0)
    assert out["gpu_power_w_mean"] == pytest.approx(200.0)
    # trapezoid: (100+200)/2*1 + (200+300)/2*2 = 150 + 500
    assert out["energy_j"] == pytest.approx(650.0)
    assert out["energy_source"] == "trapezoid"
    assert out["mem_used_peak"] == 30
    assert out["samples"] == 3
    assert s.nvml.state["index"] == 1 and s.nvml.state["shutdown"] == 1


def test_energy_counter_preferred_when_supported():
    c = Clock()
    s = N.GpuSampler(nvml=fake_nvml(energy_mj=(5_000, 9_500)), clock=c, thread=False)
    s.start()
    c.t = 2.0
    out = s.stop()
    assert out["energy_source"] == "counter"
    assert out["energy_j"] == pytest.approx(4.5)


def test_as_metrics_maps_to_report_fields():
    c = Clock()
    s = N.GpuSampler(nvml=fake_nvml(), clock=c, thread=False)
    s.start()
    c.t = 1.0
    s.stop()
    m = s.as_metrics()
    assert set(m) == {"gpu_util", "gpu_power_w", "energy_j"}
    assert m["gpu_util"] == pytest.approx(60.0)


def test_noop_when_pynvml_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)      # import pynvml → ImportError
    s = N.GpuSampler()
    assert not s.available
    s.start()
    out = s.stop()
    assert out == {"gpu_util_mean": None, "gpu_power_w_mean": None, "energy_j": None, "mem_used_peak": None,
                   "energy_source": None, "samples": 0}
    assert s.as_metrics() == {"gpu_util": None, "gpu_power_w": None, "energy_j": None}


def test_imports_real_module_name(monkeypatch):
    fake = fake_nvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert N.GpuSampler().nvml is fake


def test_init_failure_degrades_to_none():
    s = N.GpuSampler(nvml=fake_nvml(init_fails=True), thread=False)
    s.start()
    out = s.stop()
    assert out["gpu_util_mean"] is None and out["energy_j"] is None and out["samples"] == 0


def test_read_errors_mid_run_are_tolerated():
    nv = fake_nvml()
    c = Clock()
    s = N.GpuSampler(nvml=nv, clock=c, thread=False)
    s.start()

    def broken(h):
        raise NVMLError("GPU is lost")
    nv.nvmlDeviceGetPowerUsage = broken
    c.t = 1.0
    out = s.stop()
    assert out["samples"] == 1                             # only the good sample counted
    assert out["energy_j"] is None                         # one power point → no trapezoid


def test_background_thread_samples_and_stops():
    nv = fake_nvml()
    with N.GpuSampler(nvml=nv, period_s=0.005) as s:
        time.sleep(0.06)
    out = s.result
    assert out["samples"] >= 3
    assert not s._thread.is_alive()
    assert s.stop() == out                                 # idempotent


def test_stop_before_start_and_bad_period():
    s = N.GpuSampler(nvml=fake_nvml(), thread=False)
    assert s.stop()["samples"] == 0
    with pytest.raises(ValueError):
        N.GpuSampler(period_s=0)
