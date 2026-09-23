"""`moregpu bench` (ADR-0111): seeded repeats + emulated resource limits (vram fraction, cgroup cpu/mem, tc netem)."""
import json
import os
import shutil
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

from moregpu_worker import bench as B
from moregpu_worker.telemetry import schema as S

ROOT = Path(__file__).resolve().parents[2]


def ok(cmd, rc=0):
    return subprocess.CompletedProcess(cmd, rc, "", "")


class FakeRun:
    """Stands in for subprocess.run; records every command. `on_bench(cmd, env)` runs for non-tc commands."""

    def __init__(self, on_bench=None, tc_rc=0, bench_raises=None):
        self.calls, self.on_bench, self.tc_rc, self.bench_raises = [], on_bench, tc_rc, bench_raises

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        if cmd[0] == "tc":
            if self.tc_rc and kw.get("check"):
                raise subprocess.CalledProcessError(self.tc_rc, cmd)
            return ok(cmd, self.tc_rc)
        if self.bench_raises:
            raise self.bench_raises
        rc = self.on_bench(cmd, kw.get("env") or {}) if self.on_bench else 0
        return ok(cmd, rc or 0)

    def tc(self):
        return [c for c in self.calls if c[0] == "tc"]


def fake_torch(cuda=True, n=2):
    calls = []
    c = types.SimpleNamespace(is_available=lambda: cuda, device_count=lambda: n,
                              set_per_process_memory_fraction=lambda f, d=None: calls.append((f, d)))
    return types.SimpleNamespace(cuda=c, calls=calls)


# ---------------------------------------------------------------- sizes and command construction
@pytest.mark.parametrize("s,v", [("1024", 1024), ("4G", 4 << 30), ("4g", 4 << 30), ("512M", 512 << 20),
                                 ("2GiB", 2 << 30), ("1.5G", 3 << 29), ("64k", 64 << 10), ("1T", 1 << 40)])
def test_parse_size(s, v):
    assert B.parse_size(s) == v


@pytest.mark.parametrize("s", ["", "abc", "-1G", "4X", "0"])
def test_parse_size_rejects(s):
    with pytest.raises(ValueError):
        B.parse_size(s)


def test_systemd_wrapper_root_and_user():
    w = B.cgroup_wrapper(cpus=2, mem_bytes=4 << 30, backend="systemd", is_root=True)
    assert w == ["systemd-run", "--scope", "--quiet", "-p", "CPUQuota=200%", "-p", f"MemoryMax={4 << 30}",
                 "-p", "MemorySwapMax=0", "--"]
    wu = B.cgroup_wrapper(cpus=1.5, mem_bytes=None, backend="systemd", is_root=False)
    assert wu == ["systemd-run", "--user", "--scope", "--quiet", "-p", "CPUQuota=150%", "--"]


def test_docker_wrapper():
    w = B.cgroup_wrapper(cpus=2, mem_bytes=1 << 30, backend="docker", image="python:3.11", workdir="/w")
    assert w == ["docker", "run", "--rm", "--cpus", "2", "--memory", str(1 << 30), "--memory-swap", str(1 << 30),
                 "-v", "/w:/w", "-w", "/w", "python:3.11"]
    with pytest.raises(B.BenchError, match="docker-image"):
        B.cgroup_wrapper(cpus=2, mem_bytes=None, backend="docker")


def test_wrapper_empty_and_invalid():
    assert B.cgroup_wrapper(cpus=None, mem_bytes=None) == []
    with pytest.raises(B.BenchError):
        B.cgroup_wrapper(cpus=0, mem_bytes=None)
    with pytest.raises(B.BenchError, match="backend"):
        B.cgroup_wrapper(cpus=1, mem_bytes=None, backend="lxc")


def test_netem_commands():
    add, rm = B.netem_commands("eth0", 50, 100)
    assert add == ["tc", "qdisc", "add", "dev", "eth0", "root", "netem", "delay", "50ms", "rate", "100mbit"]
    assert rm == ["tc", "qdisc", "del", "dev", "eth0", "root"]
    assert B.netem_commands("lo", 12.5, None)[0][-2:] == ["delay", "12.5ms"]
    assert B.netem_commands("lo", None, 10)[0][-3:] == ["netem", "rate", "10mbit"]
    with pytest.raises(B.BenchError):
        B.netem_commands("eth0", None, None)
    with pytest.raises(B.BenchError):
        B.netem_commands("eth0", -1, None)
    with pytest.raises(B.BenchError, match="iface"):
        B.netem_commands("eth0; rm -rf /", 1, None)


# ---------------------------------------------------------------- netem guard: cleanup always
def test_netem_guard_refuses_non_linux_and_non_root():
    run = FakeRun()
    with pytest.raises(B.BenchError, match="Linux"):
        with B.NetemGuard("eth0", 10, None, run=run, platform="darwin", euid=0):
            pass
    with pytest.raises(B.BenchError, match="root"):
        with B.NetemGuard("eth0", 10, None, run=run, platform="linux", euid=1000):
            pass
    assert run.calls == []


def test_netem_guard_cleans_up_on_exception(monkeypatch):
    reg = []
    monkeypatch.setattr(B.atexit, "register", lambda f: reg.append(("reg", f)))
    monkeypatch.setattr(B.atexit, "unregister", lambda f: reg.append(("unreg", f)))
    run = FakeRun()
    with pytest.raises(ZeroDivisionError):
        with B.NetemGuard("eth0", 10, 5, run=run, platform="linux", euid=0):
            assert [c[2] for c in run.tc()] == ["add"]
            1 / 0
    assert [c[2] for c in run.tc()] == ["add", "del"]
    assert [r[0] for r in reg] == ["reg", "unreg"]


def test_netem_guard_cleanup_is_idempotent_and_atexit_path():
    run = FakeRun()
    g = B.NetemGuard("eth0", 10, None, run=run, platform="linux", euid=0, install_signals=False)
    g.__enter__()
    g.cleanup()                                           # what atexit would call
    g.cleanup()
    g.__exit__(None, None, None)
    assert [c[2] for c in run.tc()] == ["add", "del"]


def test_netem_guard_cleans_up_if_add_fails():
    run = FakeRun(tc_rc=2)
    with pytest.raises(B.BenchError, match="tc qdisc add failed"):
        with B.NetemGuard("eth0", 10, None, run=run, platform="linux", euid=0):
            pass                                          # pragma: no cover
    assert [c[2] for c in run.tc()] == ["add", "del"]     # a half-applied qdisc is still removed


def test_netem_guard_signal_handler_cleans_up_and_chains():
    run = FakeRun()
    seen = []
    prev = signal.signal(signal.SIGTERM, lambda s, f: seen.append(s))
    try:
        with B.NetemGuard("eth0", 10, None, run=run, platform="linux", euid=0):
            h = signal.getsignal(signal.SIGTERM)
            h(signal.SIGTERM, None)                       # simulate delivery
            assert [c[2] for c in run.tc()] == ["add", "del"]
            assert seen == [signal.SIGTERM]
        assert [c[2] for c in run.tc()] == ["add", "del"]  # no second del
        assert signal.getsignal(signal.SIGTERM) is not h   # previous handler restored
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_netem_guard_signal_default_handler_reraises(monkeypatch):
    run = FakeRun()
    killed = []
    monkeypatch.setattr(B.os, "kill", lambda pid, s: killed.append(s))
    prev = signal.signal(signal.SIGHUP, signal.SIG_DFL)
    try:
        with B.NetemGuard("eth0", 10, None, run=run, platform="linux", euid=0):
            signal.getsignal(signal.SIGHUP)(signal.SIGHUP, None)
        assert killed == [signal.SIGHUP]
        assert [c[2] for c in run.tc()] == ["add", "del"]
    finally:
        signal.signal(signal.SIGHUP, prev)


# ---------------------------------------------------------------- vram
def test_apply_vram_limit_calls_torch_for_every_device():
    t = fake_torch(n=2)
    assert B.apply_vram_limit(0.5, torch_mod=t) == [0, 1]
    assert t.calls == [(0.5, 0), (0.5, 1)]


def test_apply_vram_limit_errors():
    with pytest.raises(B.BenchError, match="CUDA"):
        B.apply_vram_limit(0.5, torch_mod=fake_torch(cuda=False))
    for bad in (0, -0.1, 1.5):
        with pytest.raises(B.BenchError, match="fraction"):
            B.apply_vram_limit(bad, torch_mod=fake_torch())


def test_apply_vram_limit_from_env(monkeypatch):
    t = fake_torch(n=1)
    monkeypatch.delenv("MOREGPU_VRAM_FRACTION", raising=False)
    assert B.apply_vram_limit_from_env(torch_mod=t) is None
    monkeypatch.setenv("MOREGPU_VRAM_FRACTION", "0.25")
    assert B.apply_vram_limit_from_env(torch_mod=t) == 0.25
    assert t.calls == [(0.25, 0)]


def test_apply_vram_limit_real_torch_without_cuda():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("has CUDA")
    with pytest.raises(B.BenchError, match="CUDA"):
        B.apply_vram_limit(0.5)


# ---------------------------------------------------------------- run_bench
class StepClock:
    def __init__(self, durations):
        self.t, self.d, self.started = 0.0, list(durations), False

    def __call__(self):
        if self.started:
            self.t += self.d.pop(0)
        self.started = not self.started
        return self.t


def test_stats():
    st = B.stats([1.0, 2.0, 3.0])
    assert st == {"n": 3, "mean": 2.0, "sd": 1.0, "min": 1.0, "max": 3.0}
    assert B.stats([5.0])["sd"] == 0.0
    assert B.stats([]) is None


def test_run_bench_callable_seeds_and_stats():
    seen = []

    def target(seed):
        seen.append(seed)
        return [{"schema": "moregpu.telemetry/1", "kind": "external_round", "ts": "2026-01-01T00:00:00Z",
                 "runner": "single", "world_size": 1, "rank": 0, "round": 0, "wall_s": 1.0, "compute_s": 1.0,
                 "data_s": 0, "serialize_s": 0, "network_s": 0, "wait_s": 0, "samples_per_s": 10.0 * (seed + 1)},
                {"kind": "garbage"}]
    rec = B.run_bench(target, repeats=3, seeds=[7, 8, 9], clock=StepClock([1.0, 2.0, 3.0]), name="t", hw=None)
    assert seen == [7, 8, 9]
    assert rec["kind"] == "bench" and rec["repeats"] == 3 and rec["seeds"] == [7, 8, 9]
    assert rec["stats"]["wall_s"] == {"n": 3, "mean": 2.0, "sd": 1.0, "min": 1.0, "max": 3.0}
    assert rec["stats"]["samples_per_s"]["mean"] == pytest.approx(90.0)
    assert [r["records"] for r in rec["runs"]] == [2, 2, 2]
    assert [r["invalid_records"] for r in rec["runs"]] == [1, 1, 1]
    assert rec["cmd"] is None and rec["failures"] == 0
    assert S.validate(rec) == []


def test_run_bench_default_seeds_and_failures():
    def target(seed):
        if seed == 1:
            raise RuntimeError("boom")
    rec = B.run_bench(target, repeats=3, clock=StepClock([1.0, 9.0, 3.0]), hw=None)
    assert rec["seeds"] == [0, 1, 2]
    assert rec["failures"] == 1
    assert rec["runs"][1]["rc"] == 1 and "boom" in rec["runs"][1]["error"]
    assert rec["stats"]["wall_s"]["n"] == 2 and rec["stats"]["wall_s"]["mean"] == 2.0
    assert S.validate(rec) == []


def test_run_bench_all_fail():
    def target(seed):
        raise RuntimeError("x")
    rec = B.run_bench(target, repeats=2, hw=None)
    assert rec["failures"] == 2 and rec["stats"] == {}
    assert S.validate(rec) == []


def test_run_bench_command_collects_telemetry_and_wraps(tmp_path):
    def on_bench(cmd, env):
        seed = int(env["MOREGPU_SEED"])
        assert env["MOREGPU_TELEMETRY_DIR"] and os.path.isdir(env["MOREGPU_TELEMETRY_DIR"])
        with open(env["MOREGPU_TELEMETRY_FILE"], "a") as f:
            f.write(json.dumps({"schema": "moregpu.telemetry/1", "kind": "external_round",
                                "ts": "2026-01-01T00:00:00Z", "runner": "ddp", "world_size": 2, "rank": 0,
                                "round": 0, "wall_s": 1, "compute_s": 1, "data_s": 0, "serialize_s": 0,
                                "network_s": 0, "wait_s": 0}) + "\n")
            f.write("not json\n")
        return 3 if seed == 2 else 0
    run = FakeRun(on_bench)
    rec = B.run_bench(["python3", "train.py", "--x"], repeats=3, wrapper=["systemd-run", "--scope", "--"],
                      runner=run, env={"EXTRA": "1"}, hw=None,
                      emulation=B.emulation_block([B.limit("cpus", 2, "x", True)]))
    assert run.calls[0] == ["systemd-run", "--scope", "--", "python3", "train.py", "--x"]
    assert rec["cmd"] == "systemd-run --scope -- python3 train.py --x"
    assert [r["rc"] for r in rec["runs"]] == [0, 0, 3] and rec["failures"] == 1
    assert [r["records"] for r in rec["runs"]] == [2, 2, 2]              # every line seen …
    assert [r["invalid_records"] for r in rec["runs"]] == [1, 1, 1]      # … of which the non-JSON one is invalid
    assert rec["emulation"]["limits"][0]["label"] == "emulation"
    assert S.validate(rec) == []


def test_run_bench_string_command_and_emitter(tmp_path):
    from moregpu_worker.telemetry.emit import JsonlEmitter
    em = JsonlEmitter(tmp_path / "b.jsonl", git_sha=None, hw=None)
    run = FakeRun()
    rec = B.run_bench("python3 -c 'print(1)'", repeats=1, runner=run, emitter=em, hw=None)
    assert run.calls == [["python3", "-c", "print(1)"]]
    assert json.loads((tmp_path / "b.jsonl").read_text()) == rec


def test_run_bench_argument_errors():
    with pytest.raises(ValueError):
        B.run_bench(lambda s: None, repeats=0)
    with pytest.raises(ValueError, match="seeds"):
        B.run_bench(lambda s: None, repeats=2, seeds=[1])


def test_emulation_block_labels_every_limit():
    blk = B.emulation_block([B.limit("cpus", 2, "m", False), B.limit("latency_ms", 5, "m", True)])
    assert blk["label"] == "emulation" and "note" in blk
    assert all(x["label"] == "emulation" for x in blk["limits"])


# ---------------------------------------------------------------- CLI
def cli(argv, capsys, **kw):
    rc = B.main(argv, **kw)
    out = capsys.readouterr()
    return rc, (json.loads(out.out) if out.out.strip() else None), out.err


def test_cli_dry_run_prints_every_command(capsys):
    run = FakeRun()
    rc, out, _ = cli(["--dry-run", "--limit-cpus", "2", "--limit-mem", "4G", "--simulate-latency", "50",
                      "--bandwidth", "100", "--iface", "eth1", "--limit-vram", "0.5", "--repeats", "2",
                      "--", "python3", "train.py"], capsys, run=run, torch_mod=fake_torch(), euid=1000,
                     platform="darwin")
    assert rc == 0 and run.calls == []
    assert out["dry_run"] is True
    cmds = [" ".join(c) for c in out["commands"]]
    assert "tc qdisc add dev eth1 root netem delay 50ms rate 100mbit" in cmds
    assert "tc qdisc del dev eth1 root" in cmds
    assert any(c.startswith("systemd-run --user --scope") and c.endswith("-- python3 train.py") for c in cmds)
    types_ = {x["type"] for x in out["emulation"]["limits"]}
    assert types_ == {"vram_fraction", "cpus", "mem_bytes", "latency_ms", "bandwidth_mbit"}
    assert all(x["label"] == "emulation" and x["applied"] is False for x in out["emulation"]["limits"])


def test_cli_privileged_limits_without_apply_is_dry_run(capsys):
    run = FakeRun()
    rc, out, err = cli(["--simulate-latency", "20", "--", "true"], capsys, run=run, platform="linux", euid=0)
    assert rc == 0 and run.calls == [] and out["dry_run"] is True
    assert "--apply" in err


def test_cli_apply_netem_refused_when_not_root(capsys):
    run = FakeRun()
    rc, out, err = cli(["--apply", "--simulate-latency", "20", "--", "true"], capsys, run=run, platform="linux",
                       euid=1000)
    assert rc == 2 and "root" in err and run.calls == []
    rc, _, err = cli(["--apply", "--bandwidth", "20", "--", "true"], capsys, run=run, platform="darwin", euid=0)
    assert rc == 2 and "Linux" in err and run.calls == []


def test_cli_apply_runs_with_netem_and_cleans_up(capsys, tmp_path):
    run = FakeRun()
    rc, out, _ = cli(["--apply", "--simulate-latency", "20", "--bandwidth", "50", "--limit-cpus", "1",
                      "--repeats", "2", "--out", str(tmp_path / "b.jsonl"), "--name", "net",
                      "--", "python3", "x.py"], capsys, run=run, platform="linux", euid=0)
    assert rc == 0
    kinds = [c[0] if c[0] != "tc" else "tc-" + c[2] for c in run.calls]
    assert kinds == ["tc-add", "systemd-run", "systemd-run", "tc-del"]
    assert out["kind"] == "bench" and out["name"] == "net" and S.validate(out) == []
    assert all(x["applied"] and x["label"] == "emulation" for x in out["emulation"]["limits"])
    assert json.loads((tmp_path / "b.jsonl").read_text()) == out


def test_cli_apply_cleans_up_when_benchmark_crashes(capsys):
    run = FakeRun(bench_raises=OSError("exec failed"))
    with pytest.raises(OSError):
        B.main(["--apply", "--simulate-latency", "20", "--", "nope"], run=run, platform="linux", euid=0)
    assert [c[2] for c in run.tc()] == ["add", "del"]


def test_cli_limit_vram_applies_in_process_and_exports_env(capsys):
    t = fake_torch(n=1)
    envs = []
    run = FakeRun(on_bench=lambda cmd, env: envs.append(env.get("MOREGPU_VRAM_FRACTION")))
    rc, out, _ = cli(["--limit-vram", "0.5", "--repeats", "1", "--", "python3", "x.py"], capsys, run=run,
                     torch_mod=t)
    assert rc == 0 and t.calls == [(0.5, 0)] and envs == ["0.5"]
    lim = out["emulation"]["limits"]
    assert lim[0]["type"] == "vram_fraction" and lim[0]["applied"] is True and lim[0]["label"] == "emulation"


def test_cli_limit_vram_without_cuda_errors_cleanly(capsys):
    rc, out, err = cli(["--limit-vram", "0.5", "--", "true"], capsys, run=FakeRun(), torch_mod=fake_torch(False))
    assert rc == 2 and "CUDA" in err and out is None


def test_cli_argument_errors(capsys):
    assert cli(["--repeats", "1"], capsys, run=FakeRun())[0] == 2                         # no command
    assert cli(["--limit-mem", "lots", "--dry-run", "--", "x"], capsys)[0] == 2
    assert cli(["--seeds", "1,2", "--repeats", "3", "--", "x"], capsys, run=FakeRun())[0] == 2
    assert cli(["--limit-cpus", "2", "--backend", "docker", "--dry-run", "--", "x"], capsys)[0] == 2


def test_cli_plain_run_with_seeds(capsys):
    seeds = []
    run = FakeRun(on_bench=lambda cmd, env: seeds.append(env["MOREGPU_SEED"]))
    rc, out, _ = cli(["--seeds", "3,4", "--", "python3", "x.py"], capsys, run=run)
    assert rc == 0 and seeds == ["3", "4"] and out["repeats"] == 2
    assert out["emulation"] is None


def test_module_entrypoint_dry_run():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "apps" / "worker"))
    p = subprocess.run([sys.executable, "-m", "moregpu_worker.bench", "--dry-run", "--limit-cpus", "2", "--",
                        "echo", "hi"], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["dry_run"] is True


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_cli_script_bench_subcommand():
    p = subprocess.run(["bash", str(ROOT / "scripts" / "moregpu"), "bench", "--dry-run", "--limit-mem", "1G",
                        "--", "echo", "hi"], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, NO_COLOR="1"))
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert out["dry_run"] is True and out["emulation"]["limits"][0]["type"] == "mem_bytes"
    h = subprocess.run(["bash", str(ROOT / "scripts" / "moregpu"), "help"], capture_output=True, text=True,
                       timeout=60, env=dict(os.environ, NO_COLOR="1"))
    assert "moregpu bench" in h.stdout
