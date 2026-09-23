"""NVML GPU sampler (ADR-0111): utilisation, board power, energy and memory over a measured window.

Uses ``pynvml`` (PyPI package ``nvidia-ml-py``, extra ``moregpu-worker[telemetry]``) when importable; otherwise — or
when NVML fails to initialise (no driver, non-NVIDIA GPU, container without the device) — it is a no-op that reports
``None`` for everything, so callers never need to branch.

    s = GpuSampler(device_index=0, period_s=0.1).start()
    ...work...
    out = s.stop()     # {gpu_util_mean, gpu_power_w_mean, energy_j, mem_used_peak, energy_source, samples}

Energy prefers the NVML total-energy counter delta (Volta+, mJ) and falls back to the trapezoid of sampled power.
Board power includes idle draw and other processes on the same GPU; util is "a kernel was running", not SM occupancy.
"""
from __future__ import annotations

import threading
import time

_EMPTY = {"gpu_util_mean": None, "gpu_power_w_mean": None, "energy_j": None, "mem_used_peak": None,
          "energy_source": None, "samples": 0}


def _import_pynvml():
    try:
        import pynvml
        return pynvml
    except ImportError:
        return None


class GpuSampler:
    def __init__(self, device_index: int = 0, period_s: float = 0.1, *, nvml=None, clock=time.monotonic,
                 thread: bool = True):
        if period_s <= 0:
            raise ValueError("period_s must be > 0")
        self.device_index, self.period_s, self.clock, self._use_thread = device_index, period_s, clock, thread
        self.nvml = nvml if nvml is not None else _import_pynvml()
        self._h = None
        self._pts: list[tuple[float, float | None, float | None, int | None]] = []   # (t, util, power_w, used)
        self._e0 = None
        self._e1 = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.result: dict = dict(_EMPTY)
        self._started = self._stopped = False

    @property
    def available(self) -> bool:
        return self.nvml is not None

    # ------------------------------------------------------------------
    def _energy(self):
        try:
            return float(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self._h))
        except Exception:
            return None

    def sample(self) -> None:
        """Take one sample now (the background thread calls this every ``period_s``)."""
        if self._h is None:
            return
        try:
            util = float(self.nvml.nvmlDeviceGetUtilizationRates(self._h).gpu)
            power = self.nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
            used = int(self.nvml.nvmlDeviceGetMemoryInfo(self._h).used)
        except Exception:
            return                                     # transient NVML error: skip this point
        with self._lock:
            self._pts.append((self.clock(), util, power, used))

    def _loop(self):
        while not self._stop.wait(self.period_s):
            self.sample()

    def start(self) -> "GpuSampler":
        self._started = True
        if self.nvml is None:
            return self
        try:
            self.nvml.nvmlInit()
            self._h = self.nvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            self._h = None
            return self
        self._e0 = self._energy()
        self.sample()
        if self._use_thread:
            self._thread = threading.Thread(target=self._loop, name="moregpu-nvml", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> dict:
        if self._stopped or not self._started:
            return self.result
        self._stopped = True
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        if self._h is not None:
            self.sample()
            self._e1 = self._energy()
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        self.result = self._summarise()
        return self.result

    def _summarise(self) -> dict:
        pts = list(self._pts)
        if not pts:
            return dict(_EMPTY)
        out = dict(_EMPTY, samples=len(pts))
        out["gpu_util_mean"] = sum(p[1] for p in pts) / len(pts)
        out["gpu_power_w_mean"] = sum(p[2] for p in pts) / len(pts)
        out["mem_used_peak"] = max(p[3] for p in pts)
        if self._e0 is not None and self._e1 is not None and self._e1 >= self._e0:
            out["energy_j"], out["energy_source"] = (self._e1 - self._e0) / 1000.0, "counter"
        elif len(pts) >= 2:
            out["energy_j"] = sum((b[0] - a[0]) * (a[2] + b[2]) / 2 for a, b in zip(pts, pts[1:]))
            out["energy_source"] = "trapezoid"
        return out

    def as_metrics(self) -> dict:
        """The keys a worker puts into ``InnerReport.metrics`` (→ worker_round gpu_util/gpu_power_w/energy_j)."""
        r = self.result
        return {"gpu_util": r["gpu_util_mean"], "gpu_power_w": r["gpu_power_w_mean"], "energy_j": r["energy_j"]}

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
