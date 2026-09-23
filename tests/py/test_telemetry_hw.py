"""hw.fingerprint (ADR-0111): hardware/software fingerprint without PII."""
import json
import socket
import types

from moregpu_worker.telemetry import hw


KEYS = {"host_hash", "os", "os_release", "machine", "python", "torch", "cuda", "device", "capability",
        "total_mem_bytes", "cpu_count", "ram_bytes"}


def fake_torch(cuda=True):
    props = types.SimpleNamespace(name="Fake GPU 9000", total_memory=24 << 30, major=8, minor=9)
    c = types.SimpleNamespace(is_available=lambda: cuda, current_device=lambda: 0,
                              get_device_properties=lambda i: props)
    return types.SimpleNamespace(__version__="9.9.9", version=types.SimpleNamespace(cuda="12.9"), cuda=c)


def test_fingerprint_keys_and_no_raw_hostname():
    fp = hw.fingerprint()
    assert set(fp) == KEYS
    host = socket.gethostname()
    assert host not in json.dumps(fp)
    assert len(fp["host_hash"]) == 16 and all(ch in "0123456789abcdef" for ch in fp["host_hash"])
    assert fp["cpu_count"] >= 1
    json.dumps(fp)                                         # serialisable


def test_host_hash_is_stable_and_salted():
    assert hw.host_hash("box") == hw.host_hash("box")
    assert hw.host_hash("box") != hw.host_hash("box2")
    import hashlib
    assert hw.host_hash("box") != hashlib.sha256(b"box").hexdigest()[:16]


def test_fingerprint_with_cuda():
    fp = hw.fingerprint(torch_mod=fake_torch())
    assert fp["torch"] == "9.9.9" and fp["cuda"] == "12.9"
    assert fp["device"] == "Fake GPU 9000" and fp["capability"] == "8.9"
    assert fp["total_mem_bytes"] == 24 << 30


def test_fingerprint_cpu_only():
    fp = hw.fingerprint(torch_mod=fake_torch(cuda=False))
    assert fp["device"] == "cpu" and fp["capability"] is None and fp["total_mem_bytes"] is None


def test_fingerprint_without_torch(monkeypatch):
    monkeypatch.setattr(hw, "_import_torch", lambda: None)
    fp = hw.fingerprint()
    assert fp["torch"] is None and fp["cuda"] is None and fp["device"] is None


def test_fingerprint_survives_broken_cuda():
    t = fake_torch()

    def boom(i):
        raise RuntimeError("driver")
    t.cuda.get_device_properties = boom
    fp = hw.fingerprint(torch_mod=t)
    assert fp["device"] is None and fp["capability"] is None


def test_ram_fallback(monkeypatch):
    def boom(name):
        raise ValueError(name)
    monkeypatch.setattr(hw.os, "sysconf", boom)
    assert hw.fingerprint(torch_mod=fake_torch(False))["ram_bytes"] is None
