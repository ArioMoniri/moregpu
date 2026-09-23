"""``moregpu bench`` (ADR-0111): seeded repeats of a training command with emulated resource limits.

    python3 -m moregpu_worker.bench [--repeats N | --seeds 0,1,2] [--name NAME] [--out FILE.jsonl]
        [--limit-vram FRACTION]                          torch.cuda.set_per_process_memory_fraction (in-process)
        [--limit-cpus N] [--limit-mem SIZE]              systemd-run --scope (cgroup v2) | docker run --cpus --memory
        [--backend systemd|docker] [--docker-image IMG]
        [--simulate-latency MS] [--bandwidth MBIT] [--iface IFACE]      tc netem on IFACE egress (Linux, root)
        [--dry-run | --apply]  -- CMD ...

Privileged limits (cpus/mem/netem) are opt-in: without ``--apply`` the exact commands are printed as JSON and nothing
runs. With ``--apply`` the netem qdisc is removed on every exit path (try/finally + atexit + SIGTERM/SIGINT/SIGHUP).
Every limit is labelled ``"emulation"`` in the output: a throttled big GPU is not a small GPU, and netem on one
interface is not a WAN. Each repeat runs with ``MOREGPU_SEED``, ``MOREGPU_TELEMETRY_DIR`` and
``MOREGPU_TELEMETRY_FILE`` set; telemetry the command writes there (``JsonlEmitter.from_env()``) is collected,
validated and summarised into one ``bench`` record (printed, and appended to ``--out``).
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .telemetry.schema import SCHEMA_ID, validate

EMULATION_NOTE = ("All limits are software emulation on the host that ran the bench (process VRAM cap, cgroup CPU "
                  "quota / memory max, tc netem on one interface's egress). They approximate, not reproduce, "
                  "smaller devices or real WAN links; compare only against runs with the same emulation block.")
_IFACE_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,15}$")
_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmgt]?)(i?b)?$", re.I)


class BenchError(Exception):
    """A user-facing configuration/environment error (exit code 2)."""


# ---------------------------------------------------------------- limits
def parse_size(s: str) -> int:
    """'4G', '512M', '2GiB', '1.5g', '1024' → bytes (binary multiples, as systemd and docker use)."""
    m = _SIZE_RE.match(str(s).strip())
    if not m:
        raise ValueError(f"bad size {s!r} (e.g. 512M, 4G)")
    v = int(float(m.group(1)) * (1 << (10 * " kmgt".index(m.group(2).lower() or " "))))
    if v <= 0:
        raise ValueError(f"size must be > 0: {s!r}")
    return v


def _fmt_num(x) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def cgroup_wrapper(cpus=None, mem_bytes=None, *, backend: str = "systemd", is_root: bool | None = None,
                   image: str | None = None, workdir: str | None = None) -> list[str]:
    """Command prefix that runs CMD under a CPU quota / memory max. Empty when no limit is requested."""
    if cpus is None and mem_bytes is None:
        return []
    if cpus is not None and cpus <= 0:
        raise BenchError("--limit-cpus must be > 0")
    if backend == "systemd":
        root = (os.geteuid() == 0) if is_root is None else is_root
        w = ["systemd-run"] + ([] if root else ["--user"]) + ["--scope", "--quiet"]
        if cpus is not None:
            w += ["-p", f"CPUQuota={_fmt_num(cpus * 100)}%"]
        if mem_bytes is not None:
            w += ["-p", f"MemoryMax={mem_bytes}", "-p", "MemorySwapMax=0"]
        return w + ["--"]
    if backend == "docker":
        if not image:
            raise BenchError("--backend docker needs --docker-image IMAGE (with python + torch)")
        wd = workdir or os.getcwd()
        w = ["docker", "run", "--rm"]
        if cpus is not None:
            w += ["--cpus", _fmt_num(cpus)]
        if mem_bytes is not None:
            w += ["--memory", str(mem_bytes), "--memory-swap", str(mem_bytes)]
        return w + ["-v", f"{wd}:{wd}", "-w", wd, image]
    raise BenchError(f"unknown backend {backend!r} (systemd | docker)")


def netem_commands(iface: str, latency_ms=None, bandwidth_mbit=None) -> tuple[list[str], list[str]]:
    if not _IFACE_RE.match(iface or ""):
        raise BenchError(f"bad --iface {iface!r}")
    if latency_ms is None and bandwidth_mbit is None:
        raise BenchError("netem needs --simulate-latency and/or --bandwidth")
    for v in (latency_ms, bandwidth_mbit):
        if v is not None and v <= 0:
            raise BenchError("--simulate-latency/--bandwidth must be > 0")
    add = ["tc", "qdisc", "add", "dev", iface, "root", "netem"]
    if latency_ms is not None:
        add += ["delay", f"{_fmt_num(latency_ms)}ms"]
    if bandwidth_mbit is not None:
        add += ["rate", f"{_fmt_num(bandwidth_mbit)}mbit"]
    return add, ["tc", "qdisc", "del", "dev", iface, "root"]


class NetemGuard:
    """Context manager: add a netem qdisc and guarantee its removal (finally + atexit + signal handlers)."""

    SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

    def __init__(self, iface, latency_ms=None, bandwidth_mbit=None, *, run=subprocess.run, platform=None,
                 euid=None, install_signals: bool = True):
        self.add, self.rm = netem_commands(iface, latency_ms, bandwidth_mbit)
        self.run, self.install_signals = run, install_signals
        self.platform = sys.platform if platform is None else platform
        self.euid = (os.geteuid() if hasattr(os, "geteuid") else -1) if euid is None else euid
        self._active = False
        self._prev: dict = {}

    def check(self):
        if not self.platform.startswith("linux"):
            raise BenchError(f"tc netem needs Linux (this is {self.platform}); use --dry-run to see the commands")
        if self.euid != 0:
            raise BenchError("tc netem needs root (run the bench with sudo), or use --dry-run")

    def __enter__(self):
        self.check()
        self._active = True                       # from here on, cleanup must run whatever happens
        atexit.register(self.cleanup)
        if self.install_signals:
            for s in self.SIGNALS:
                try:
                    self._prev[s] = signal.signal(s, self._on_signal)
                except (ValueError, OSError):     # not the main thread
                    pass
        try:
            self.run(self.add, check=True)
        except subprocess.CalledProcessError as e:
            self.__exit__(None, None, None)
            raise BenchError(f"tc qdisc add failed (rc={e.returncode}): {' '.join(self.add)}") from e
        return self

    def cleanup(self):
        if not self._active:
            return
        self._active = False
        self.run(self.rm, check=False)

    def _restore(self):
        for s, h in self._prev.items():
            try:
                signal.signal(s, h)
            except (ValueError, OSError, TypeError):  # pragma: no cover
                pass
        self._prev = {}

    def _on_signal(self, signum, frame):
        prev = self._prev.get(signum)
        self.cleanup()
        self._restore()
        if callable(prev):
            prev(signum, frame)
        elif prev != signal.SIG_IGN:
            os.kill(os.getpid(), signum)          # re-deliver with the default disposition

    def __exit__(self, *exc):
        try:
            self.cleanup()
        finally:
            self._restore()
            atexit.unregister(self.cleanup)
        return False


def _import_torch():
    import torch
    return torch


def apply_vram_limit(fraction: float, torch_mod=None) -> list[int]:
    """Cap this process's CUDA caching allocator at ``fraction`` of each visible device. Returns device indices."""
    if not (isinstance(fraction, (int, float)) and 0 < fraction <= 1):
        raise BenchError(f"--limit-vram fraction must be in (0, 1], got {fraction!r}")
    t = torch_mod if torch_mod is not None else _import_torch()
    if not t.cuda.is_available():
        raise BenchError("--limit-vram needs CUDA (torch.cuda.is_available() is False)")
    devs = list(range(t.cuda.device_count()))
    for d in devs:
        t.cuda.set_per_process_memory_fraction(float(fraction), d)
    return devs


def apply_vram_limit_from_env(torch_mod=None):
    """For benchmarked commands: honour MOREGPU_VRAM_FRACTION (set by ``moregpu bench --limit-vram``)."""
    v = os.environ.get("MOREGPU_VRAM_FRACTION")
    if not v:
        return None
    apply_vram_limit(float(v), torch_mod=torch_mod)
    return float(v)


def limit(type_: str, value, mechanism: str, applied: bool) -> dict:
    return {"type": type_, "value": value, "mechanism": mechanism, "label": "emulation", "applied": bool(applied)}


def emulation_block(limits: list[dict]) -> dict:
    return {"label": "emulation", "note": EMULATION_NOTE, "limits": [dict(x, label="emulation") for x in limits]}


# ---------------------------------------------------------------- repeats
def stats(xs) -> dict | None:
    xs = [float(x) for x in xs]
    if not xs:
        return None
    n, mean = len(xs), sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "min": min(xs), "max": max(xs)}


def _read_records(d: Path) -> tuple[list, int]:
    recs, bad = [], 0
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                bad += 1
    return recs, bad


def run_bench(target, repeats: int = 3, seeds=None, *, wrapper=(), env=None, name=None, emulation=None,
              runner=subprocess.run, clock=time.perf_counter, emitter=None, hw="auto", git_sha="auto",
              config_hash=None) -> dict:
    """Run ``target`` once per seed and return a validated ``bench`` record.

    ``target`` is a callable ``f(seed)`` (may return an iterable of telemetry records) or a command (list or shell
    string, run as ``wrapper + cmd`` with MOREGPU_SEED / MOREGPU_TELEMETRY_DIR / MOREGPU_TELEMETRY_FILE set)."""
    if seeds is None:
        if repeats < 1:
            raise ValueError("repeats must be ≥ 1")
        seeds = list(range(repeats))
    seeds = [int(s) for s in seeds]
    if len(seeds) != repeats:
        raise ValueError(f"got {len(seeds)} seeds for {repeats} repeats")
    is_cmd = not callable(target)
    cmd = (shlex.split(target) if isinstance(target, str) else list(target)) if is_cmd else None
    full = list(wrapper) + cmd if is_cmd else None
    runs, walls, sps = [], [], []
    for seed in seeds:
        run = {"seed": seed, "wall_s": 0.0, "rc": 0, "records": 0, "invalid_records": 0}
        recs, bad = [], 0
        if is_cmd:
            with tempfile.TemporaryDirectory(prefix="moregpu-bench-") as td:
                e = {**os.environ, **(env or {}), "MOREGPU_SEED": str(seed), "MOREGPU_TELEMETRY_DIR": td,
                     "MOREGPU_TELEMETRY_FILE": str(Path(td) / "run.jsonl")}
                t0 = clock()
                try:
                    p = runner(full, env=e, stdout=2)
                finally:
                    run["wall_s"] = clock() - t0
                run["rc"] = int(p.returncode)
                recs, bad = _read_records(Path(td))
        else:
            t0 = clock()
            try:
                out = target(seed)
                recs = list(out) if out is not None and not isinstance(out, dict) else []
            except Exception as ex:
                run["rc"], run["error"] = 1, f"{type(ex).__name__}: {ex}"
            finally:
                run["wall_s"] = clock() - t0
        valid = [r for r in recs if not validate(r)]
        run["records"], run["invalid_records"] = len(recs) + bad, len(recs) - len(valid) + bad
        runs.append(run)
        if run["rc"] == 0:
            walls.append(run["wall_s"])
            sps += [r["samples_per_s"] for r in valid if isinstance(r.get("samples_per_s"), (int, float))]
    st = {k: v for k, v in (("wall_s", stats(walls)), ("samples_per_s", stats(sps))) if v is not None}
    if hw == "auto":
        from .telemetry.hw import fingerprint
        hw = fingerprint()
    if git_sha == "auto":
        from .telemetry.emit import default_git_sha
        git_sha = default_git_sha()
    fields = {"name": name, "cmd": shlex.join(full) if is_cmd else None, "repeats": len(seeds), "seeds": seeds,
              "runs": runs, "failures": sum(1 for r in runs if r["rc"] != 0), "stats": st, "emulation": emulation,
              "hw": hw, "git_sha": git_sha, "config_hash": config_hash}
    if emitter is not None:
        return emitter.emit("bench", **fields)
    from .telemetry.emit import utc_now
    rec = {"schema": SCHEMA_ID, "kind": "bench", "ts": utc_now(), **fields}
    errs = validate(rec)
    if errs:  # pragma: no cover - construction above is schema-shaped
        raise ValueError("; ".join(errs))
    return rec


# ---------------------------------------------------------------- CLI
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="moregpu bench", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Without --apply, cpu/mem/netem limits only print the commands (dry run).")
    p.add_argument("--repeats", type=int, default=None, help="seeded repeats (default 3, or len(--seeds))")
    p.add_argument("--seeds", default=None, help="comma-separated seeds (default 0..repeats-1)")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None, help="append the bench record to this JSONL file")
    p.add_argument("--limit-vram", type=float, default=None, metavar="FRACTION")
    p.add_argument("--limit-cpus", type=float, default=None, metavar="N")
    p.add_argument("--limit-mem", default=None, metavar="SIZE")
    p.add_argument("--backend", choices=("systemd", "docker"), default="systemd")
    p.add_argument("--docker-image", default=None)
    p.add_argument("--simulate-latency", type=float, default=None, metavar="MS")
    p.add_argument("--bandwidth", type=float, default=None, metavar="MBIT")
    p.add_argument("--iface", default="eth0")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    g.add_argument("--apply", action="store_true", help="really apply cgroup/netem limits (netem: Linux + root)")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- CMD ARGS...")
    return p


def main(argv=None, *, run=subprocess.run, torch_mod=None, platform=None, euid=None) -> int:
    try:
        a = _parser().parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    platform = sys.platform if platform is None else platform
    euid = (os.geteuid() if hasattr(os, "geteuid") else -1) if euid is None else euid
    try:
        seeds = [int(s) for s in a.seeds.split(",") if s.strip()] if a.seeds else None
        repeats = a.repeats if a.repeats is not None else (len(seeds) if seeds else 3)
        if seeds is not None and len(seeds) != repeats:
            raise BenchError(f"--seeds has {len(seeds)} entries but --repeats is {repeats}")
        if repeats < 1:
            raise BenchError("--repeats must be ≥ 1")
        mem = parse_size(a.limit_mem) if a.limit_mem else None
        wrapper = cgroup_wrapper(a.limit_cpus, mem, backend=a.backend, is_root=(euid == 0), image=a.docker_image)
        netem = a.simulate_latency is not None or a.bandwidth is not None
        nadd, nrm = netem_commands(a.iface, a.simulate_latency, a.bandwidth) if netem else (None, None)
        privileged = bool(wrapper) or netem
        dry = a.dry_run or (privileged and not a.apply)
        if not dry and not cmd:
            raise BenchError("no command to benchmark (usage: moregpu bench [limits] -- CMD ARGS...)")

        def limits(applied_vram, applied_priv):
            out = []
            if a.limit_vram is not None:
                out.append(limit("vram_fraction", a.limit_vram, "torch.cuda.set_per_process_memory_fraction "
                                 "(bench process) + MOREGPU_VRAM_FRACTION for the command", applied_vram))
            mech = "systemd-run --scope (cgroup v2)" if a.backend == "systemd" else "docker run"
            if a.limit_cpus is not None:
                out.append(limit("cpus", a.limit_cpus, f"{mech} CPU quota", applied_priv))
            if mem is not None:
                out.append(limit("mem_bytes", mem, f"{mech} memory max (swap off)", applied_priv))
            if a.simulate_latency is not None:
                out.append(limit("latency_ms", a.simulate_latency, f"tc netem delay on {a.iface} egress",
                                 applied_priv))
            if a.bandwidth is not None:
                out.append(limit("bandwidth_mbit", a.bandwidth, f"tc netem rate on {a.iface} egress", applied_priv))
            return emulation_block(out) if out else None

        if dry:
            if privileged and not a.dry_run:
                print("[moregpu bench] cpu/mem/netem limits need --apply; printing the plan only.", file=sys.stderr)
            commands = ([nadd] if netem else []) + ([wrapper + cmd] if cmd else ([wrapper] if wrapper else [])) \
                + ([nrm] if netem else [])
            print(json.dumps({"dry_run": True, "repeats": repeats, "seeds": seeds or list(range(repeats)),
                              "commands": commands, "shell": [shlex.join(c) for c in commands],
                              "emulation": limits(False, False)}, indent=2))
            return 0

        env = {}
        if a.limit_vram is not None:
            apply_vram_limit(a.limit_vram, torch_mod=torch_mod)
            env["MOREGPU_VRAM_FRACTION"] = repr(a.limit_vram)
        if netem:
            guard = NetemGuard(a.iface, a.simulate_latency, a.bandwidth, run=run, platform=platform, euid=euid)
            guard.check()
        emitter = None
        if a.out:
            from .telemetry.emit import JsonlEmitter
            emitter = JsonlEmitter(a.out)

        def go():
            return run_bench(cmd, repeats, seeds, wrapper=wrapper, env=env, name=a.name, runner=run,
                             emulation=limits(a.limit_vram is not None, privileged), emitter=emitter)
        if netem:
            with guard:
                rec = go()
        else:
            rec = go()
    except (BenchError, ValueError) as e:
        print(f"[moregpu bench] {e}", file=sys.stderr)
        return 2
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
