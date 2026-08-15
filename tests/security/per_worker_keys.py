#!/usr/bin/env python3
"""
PER-WORKER KEYS + KEY-EPOCH ROTATION — proven against the REAL coordinator (no re-implemented server crypto).

The pool secret in config (`tenantKeyB64`) is no longer broadcast to every worker. It is now the coordinator's
MASTER secret from which each worker's OWN sealing key is DERIVED, and an admin revocation ROTATES a key epoch so a
captured key dies. This test stands up a real coordinator (apps/coordinator/server.ts in Deno) and connects
control-plane-only clients (like tests/scale/fake_fleet.py — a raw register→welcome→heartbeat WS, no torch) to prove,
end-to-end and against the coordinator's own runtime:

  (A) PER-WORKER DERIVATION.  Two workers that join at epoch 0 receive DIFFERENT keys in `welcome.tenantKeyB64`
      (server.ts derives `wkey = deriveWorkerKey(id, keyEpoch)` and sends `b64e(wkey)`), NOT one shared fleet key.
      Each welcome key equals HKDF-SHA256(master, "moregpu:worker:<id>:epoch:<n>") re-derived here with
      pyca/cryptography — so this also proves the Deno-WebCrypto ↔ Python-HKDF byte-compatibility the two runtimes
      depend on (HKDF is RFC 5869), exactly as seal_negatives proves the Ed25519 cross-runtime round trip.

  (C) A WORKER CANNOT OPEN ANOTHER WORKER'S COORDINATOR TRAFFIC.  We make the coordinator seal a real frame to w1
      (POST /weights {worker:"w1"} → the coordinator does `seal(home.key, …)` and sends `cache{sealed}` — server.ts
      dispatchShard/relayRPC/weights all seal with the target worker's `w.key`). w1 unseals it with its own key
      (round-trips); w2's key raises InvalidTag — the cross-worker read fails closed.

  (D) REVOCATION ROTATES THE EPOCH → THE OLD KEY IS DEAD.  An admin `remove` (POST /workers/w2/control
      {action:"remove"}) bumps `keyEpoch` (server.ts remove branch: `keyEpoch++`, echoed as `epoch` in the reply). A
      worker joining afterwards is minted at epoch 1; for the SAME id, the epoch-1 key differs from what epoch 0 would
      have produced (rotation). On the wire, a frame the coordinator seals to that epoch-1 worker CANNOT be opened by
      the pre-revocation (epoch-0) key for its id — InvalidTag. That is "bump epoch → old keys invalid", verified
      against the live coordinator's sealing rather than asserted in the abstract.

seal()/unseal() are imported from the REAL worker (apps/worker/worker_torch.py) — the exact AES-256-GCM the fleet
runs. CPU-only, no torch model, no GPU, no model download; one Deno coordinator subprocess + loopback WS clients.

Run:  python3 tests/security/per_worker_keys.py     (exit 0 = all checks passed)
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import websockets

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKER_PY = os.path.join(REPO, "apps", "worker", "worker_torch.py")


# ---- import the REAL worker crypto (seal/unseal — the exact AES-256-GCM the fleet uses) ----
def _load_worker():
    spec = importlib.util.spec_from_file_location("worker_torch", WORKER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module-level run() only fires under __main__, so importing is side-effect-free
    return mod


W = _load_worker()

# The coordinator's per-worker key derivation, re-implemented with pyca/cryptography to cross-check the REAL
# Deno/WebCrypto welcome key byte-for-byte. MUST mirror server.ts `deriveWorkerKey` + `HKDF_SALT` exactly.
HKDF_SALT = b"moregpu:hkdf:v1"


def derive_worker_key(master: bytes, worker_id: str, epoch: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=HKDF_SALT,
                info=f"moregpu:worker:{worker_id}:epoch:{epoch}".encode()).derive(master)


# ---------- tiny assertion harness (standalone script style, like tests/security/seal_negatives.py) ----------
_RESULTS: list[tuple[bool, str]] = []


def check(passed: bool, label: str) -> None:
    _RESULTS.append((bool(passed), label))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gen_pubkey() -> str:
    """A real raw-Ed25519 public key (base64), exactly like worker_torch.py:50-51 — so register.pubkey has the shape
    server.ts imports and the admin `remove` bans by key (which is what triggers the epoch bump)."""
    sk = Ed25519PrivateKey.generate()
    return base64.b64encode(sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()


def _blocking_api(port, path, method, body, admin, timeout):
    data = json.dumps(body).encode() if body is not None else None
    h = {"content-type": "application/json"}
    if admin:
        h["authorization"] = "Bearer " + admin
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        return {"httperror": e.code, "body": e.read().decode()[:400]}


async def api(loop, port, path, method="GET", body=None, admin=None, timeout=35):
    # Run blocking urllib off the event loop so the RawWorker readers stay live to receive+ack the cache frame the
    # POST /weights call is waiting on (server.ts waits up to 30s for the `cached` ack).
    return await loop.run_in_executor(None, _blocking_api, port, path, method, body, admin, timeout)


def weight_payload(wid: str, worker: str, vals: list[float]) -> dict:
    """A minimal valid /weights body: a 1×len(vals) float32 tensor pinned to `worker` (so the coordinator seals a
    cache frame to THAT worker with its per-worker key)."""
    arr = np.array(vals, dtype="<f4")
    return {"id": wid, "data": base64.b64encode(arr.tobytes()).decode(),
            "rows": 1, "cols": len(vals), "dtype": "f32", "worker": worker}


class RawWorker:
    """Control-plane-only client (no torch), like tests/scale/fake_fleet.py: register → capture the per-worker
    `welcome` key → heartbeat → stay connected, and stash any sealed `cache` frame the coordinator sends (acking ok)."""

    def __init__(self, name, url, token):
        self.name = name
        self.url = url
        self.token = token
        self.pubkey = gen_pubkey()
        self.key: bytes | None = None
        self.epoch: int | None = None
        self.caches: dict[str, dict] = {}
        self.welcomed = asyncio.Event()
        self._ws = None
        self._tasks: list = []

    def register_frame(self):
        return {"t": "register", "joinToken": self.token, "pubkey": self.pubkey,
                "node": {"id": self.name, "backend": "cpu", "label": "cpu:fake", "os": "linux"}}

    async def start(self):
        self._ws = await websockets.connect(self.url, max_size=None, ping_interval=20, ping_timeout=60)
        await self._ws.send(json.dumps(self.register_frame()))
        self._tasks.append(asyncio.ensure_future(self._reader()))
        self._tasks.append(asyncio.ensure_future(self._heartbeat()))
        await asyncio.wait_for(self.welcomed.wait(), 15)

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(2)
                await self._ws.send(json.dumps({"t": "heartbeat", "id": self.name, "load1": 0, "util": 0.0,
                                                "duty": 0.6, "ceil": 0.6, "paused": False, "pausedReason": None,
                                                "schedule": "always"}))
        except Exception:
            return

    async def _reader(self):
        try:
            async for raw in self._ws:
                try:
                    m = json.loads(raw)
                except Exception:
                    continue
                t = m.get("t")
                if t == "welcome":
                    self.key = base64.b64decode(m["tenantKeyB64"])
                    self.epoch = int(m.get("epoch", -1))
                    self.welcomed.set()
                elif t == "cache":                       # a sealed weight → stash the blob, ack so /weights returns
                    self.caches[m.get("id")] = m.get("sealed")
                    await self._ws.send(json.dumps({"t": "cached", "id": m.get("id"), "ok": True}))
                elif t == "assign":                      # a kernel shard — error so the coordinator fails it fast
                    await self._ws.send(json.dumps({"t": "result", "shardId": m.get("shardId"),
                                                    "jobId": m.get("jobId"), "ok": False, "error": "raw: no compute"}))
                elif t in ("train", "model"):
                    await self._ws.send(json.dumps({"t": f"{t}_reply", "reqId": m.get("reqId"), "ok": False,
                                                    "error": "raw: no compute"}))
                # denied / control / uncache / anything else → ignore, stay connected
        except Exception:
            return

    async def wait_cache(self, wid, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if wid in self.caches:
                return self.caches[wid]
            await asyncio.sleep(0.05)
        return None

    async def close(self):
        for tk in self._tasks:
            tk.cancel()
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass


def cannot_open(key: bytes, blob: dict, label: str) -> None:
    """Assert `key` fails CLOSED on `blob` (GCM auth failure), never silently opening another worker's traffic."""
    try:
        W.unseal(key, blob)
    except InvalidTag:
        check(True, f"{label} -> REJECTED (InvalidTag / GCM auth failure)")
        return
    except Exception as e:
        check(True, f"{label} -> REJECTED ({type(e).__name__})")
        return
    check(False, f"{label} -> SILENTLY OPENED (key isolation broken!)")


async def amain() -> int:
    loop = asyncio.get_running_loop()
    root = tempfile.mkdtemp(prefix="moregpu-pwk-")
    procs: list = []
    workers: list[RawWorker] = []
    try:
        port = free_port()
        cfg = os.path.join(root, "mg.json")
        env = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_INSECURE="1")
        env.pop("MOREGPU_PEER_TRANSPORT", None)  # coordinator<->worker path only; peer keys are covered by peer_transport.py
        procs.append(subprocess.Popen(
            ["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write", "apps/coordinator/server.ts"],
            cwd=REPO, env=env, stdout=open(os.path.join(root, "coord.log"), "w"), stderr=subprocess.STDOUT))
        for _ in range(80):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except Exception:
                await asyncio.sleep(0.5)
        conf = json.load(open(cfg))
        JOIN, ADMIN = conf["joinToken"], conf["adminToken"]
        master = base64.b64decode(conf["tenantKeyB64"])  # the coordinator's MASTER secret (server.ts: `const MASTER`)
        url = f"ws://127.0.0.1:{port}/ws"

        # ---- (A) per-worker derivation @ epoch 0: different keys, and each == HKDF(master, id, epoch) ----
        print("\n(A) per-worker derivation @ epoch 0  [server.ts deriveWorkerKey → welcome.tenantKeyB64]")
        w1 = RawWorker("w1", url, JOIN); await w1.start(); workers.append(w1)
        w2 = RawWorker("w2", url, JOIN); await w2.start(); workers.append(w2)
        check(w1.key is not None and w2.key is not None, "both workers received a welcome key")
        check(w1.epoch == 0 and w2.epoch == 0, f"welcome epoch == 0 (w1={w1.epoch} w2={w2.epoch})")
        check(w1.key != w2.key, "two workers get DIFFERENT keys (per-worker isolation, not a shared tenant key)")
        check(w1.key == derive_worker_key(master, "w1", 0),
              "w1 key == HKDF(master, 'w1', epoch 0)  (Deno WebCrypto == pyca HKDF, RFC 5869)")
        check(w2.key == derive_worker_key(master, "w2", 0), "w2 key == HKDF(master, 'w2', epoch 0)")

        # ---- (C) a worker cannot unseal ANOTHER worker's coordinator traffic ----
        print("\n(C) coordinator seals to w1 with w1's key  [POST /weights worker=w1 → cache{sealed}]")
        r = await api(loop, port, "/weights", "POST", weight_payload("probe-w1", "w1", [1.5, -2.25, 3.0, 42.0]), ADMIN)
        blob = await w1.wait_cache("probe-w1")
        check(isinstance(blob, dict) and r.get("ok"), f"coordinator delivered a sealed cache frame to w1 ({r})")
        opened = None
        try:
            opened = json.loads(W.unseal(w1.key, blob).decode())
        except Exception:
            opened = None
        check(opened is not None and int(opened.get("cols", 0)) == 4,
              "w1 UNSEALS its own coordinator traffic (round-trips to the sealed weight)")
        cannot_open(w2.key, blob, "w2 tries to unseal w1's coordinator traffic")

        # ---- (D) an admin revocation bumps the key epoch → the pre-revocation key is DEAD ----
        print("\n(D) revocation rotates the epoch  [POST /workers/w2/control remove → keyEpoch++]")
        rem = await api(loop, port, "/workers/w2/control", "POST", {"action": "remove"}, ADMIN)
        check(rem.get("ok") is True and rem.get("epoch") == 1, f"admin remove bumped key epoch 0 → 1 ({rem})")
        await w2.close()
        w4 = RawWorker("w4", url, JOIN); await w4.start(); workers.append(w4)
        check(w4.epoch == 1, f"a worker joining after the revocation is minted at epoch 1 (w4={w4.epoch})")
        stale0 = derive_worker_key(master, "w4", 0)   # the key this id WOULD have had before the revocation
        check(w4.key == derive_worker_key(master, "w4", 1), "w4 key == HKDF(master, 'w4', epoch 1)")
        check(w4.key != stale0, "SAME id yields a DIFFERENT key across epochs (the epoch bump rotated it)")
        r4 = await api(loop, port, "/weights", "POST", weight_payload("probe-w4", "w4", [7.0, 8.0]), ADMIN)
        blob4 = await w4.wait_cache("probe-w4")
        check(isinstance(blob4, dict) and r4.get("ok"), f"coordinator delivered a sealed cache frame to w4 ({r4})")
        opened4 = None
        try:
            opened4 = json.loads(W.unseal(w4.key, blob4).decode())
        except Exception:
            opened4 = None
        check(opened4 is not None, "w4 UNSEALS its own (epoch-1) coordinator traffic")
        cannot_open(stale0, blob4, "the pre-revocation (epoch-0) key for w4 tries to unseal its epoch-1 traffic")

    finally:
        for w in workers:
            try:
                await w.close()
            except Exception:
                pass
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        await asyncio.sleep(0.5)
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(1 for ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 92)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 92)
    if passed != total:
        for ok, label in _RESULTS:
            if not ok:
                print(f"  FAILED: {label}")
        return 1
    print("PER-WORKER KEYS + EPOCH ROTATION verified against the live coordinator. ✔")
    return 0


def main():
    print("=" * 92)
    print("PER-WORKER KEYS — coordinator<->worker seals use a key derived per (worker id, key epoch); a revocation")
    print("rotates the epoch so the old key is dead.")
    print(f"  coordinator : {os.path.join(REPO, 'apps', 'coordinator', 'server.ts')}")
    print(f"  worker seal : {WORKER_PY}")
    print("=" * 92)
    try:
        rc = asyncio.run(amain())
    except Exception as e:
        print(f"FATAL: {e}")
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
