#!/usr/bin/env python3
"""
fake_fleet.py — control-plane scale/soak regression for MoreGPU's coordinator (NO torch, NO GPU, CI-sane).

Regression-tests the "10k-node control plane" at a CI scale (default N=150, env-tunable) with zero GPUs: it spins
up N lightweight WebSocket clients that speak just enough of the worker protocol to REGISTER + HEARTBEAT + stay
connected, then hammers the coordinator through a connect storm, a mass disconnect, and a reconnect storm — asserting
the coordinator counts the fleet exactly, never crashes, and keeps its event loop responsive (/health stays fast).

Why this exists (docs/ROADMAP.md, Track-B "P2 scaffolds"): the coordinator is on the per-token data path and is the
bottleneck/SPOF at 10k nodes until Phase-2 peer transport. The fake-WS fleet is called out there as "the highest-
leverage scaffold" — prove the CONTROL plane (register/heartbeat/registry/reap under churn) at scale without needing
real GPUs, torch, HF downloads, or a LAN.

Protocol grounded in the REAL code (read before writing — no invented symbols):
  • register  — worker → coordinator: {"t":"register","joinToken",<pubkey>,"node":{id,backend,label,os}}
                (apps/worker/worker_torch.py:834-835 ; parsed at apps/coordinator/server.ts:193-210)
  • welcome   — coordinator → worker: {"t":"welcome","tenantKeyB64","duty"}  (server.ts:208 ; worker:911-919)
  • heartbeat — worker → coordinator ~every 4s: {"t":"heartbeat","id","load1","util","duty","ceil","paused",
                "pausedReason","schedule"}  (worker:887-893 ; consumed at server.ts:211-233)
  • denied    — coordinator → worker on rejection; "bad join token"/"removed by admin" are FATAL, everything else
                (e.g. "worker id already registered") is TRANSIENT → retry (worker:903-910 ; server.ts:195-202).
  • the coordinator removes a worker ONLY when its socket closes (server.ts:258-280 onclose) — there is no
    heartbeat-timeout reaper — so "stay connected" == "stay counted"; a socket drop == an instant fleet decrement.
Fleet is read back the way an operator would: GET /workers with Authorization: Bearer <admin token> returns the live
worker array (server.ts:1098-1108), and public GET /health returns {ok,fleet,queue} (server.ts:1048). We assert both
agree with the exact expected count at every phase.

The clients are asyncio coroutines in ONE process (no torch, ~no per-node cost) so thousands fit in CI. Each holds a
real Ed25519 public key (like worker_torch.py) so the register frame is byte-for-byte the shape the coordinator
parses; the clients never compute, so they never need the tenant key. /health latency is sampled from a dedicated
thread pool (isolated from the async load generator) — an honest read of the COORDINATOR's event-loop responsiveness.

  python3 tests/scale/fake_fleet.py            # exits non-zero on any failed assertion

Env knobs:  MOREGPU_FAKE_N (150)   MOREGPU_FAKE_DROP_FRAC (0.6)   MOREGPU_FAKE_HEALTH_MAX_MS (1500)
            MOREGPU_FAKE_HB (4.0)  MOREGPU_FAKE_UP_TIMEOUT_S (60)  MOREGPU_FAKE_DOWN_TIMEOUT_S (45)
"""
import asyncio, base64, http.client, json, os, platform, random, shutil, socket, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

import websockets

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    _HAVE_CRYPTO = True
except Exception:                                # a fake node with a bad key just registers UNSIGNED (server.ts:205)
    _HAVE_CRYPTO = False

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- env-tunable scale/soak parameters ----
N = int(os.environ.get("MOREGPU_FAKE_N", "150"))                       # fleet size
DROP_FRAC = float(os.environ.get("MOREGPU_FAKE_DROP_FRAC", "0.6"))     # fraction dropped in the mass-disconnect phase
HEALTH_MAX_MS = float(os.environ.get("MOREGPU_FAKE_HEALTH_MAX_MS", "1500"))  # /health latency ceiling (responsiveness)
HB_INTERVAL = float(os.environ.get("MOREGPU_FAKE_HB", "4.0"))         # heartbeat cadence (worker uses 4s)
UP_TIMEOUT_S = float(os.environ.get("MOREGPU_FAKE_UP_TIMEOUT_S", "60"))     # budget to reach a full fleet
DOWN_TIMEOUT_S = float(os.environ.get("MOREGPU_FAKE_DOWN_TIMEOUT_S", "45")) # budget for the reap to settle
# WAN-tolerant keepalive, matching worker_torch.py:828-832 (localhost here, but keep the shape honest).
PING_INTERVAL = float(os.environ.get("MOREGPU_WS_PING_INTERVAL", "30"))
PING_TIMEOUT = float(os.environ.get("MOREGPU_WS_PING_TIMEOUT", "90"))
BACKOFF_MIN, BACKOFF_MAX = 0.10, 0.30            # reconnect jitter after an UNEXPECTED drop (not deliberate down)

HTTP_POOL = ThreadPoolExecutor(max_workers=8)    # isolates blocking /health + /workers probes from the async loop
PID = os.getpid()


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gen_pubkey() -> str:
    """A real raw-Ed25519 public key (base64), exactly like worker_torch.py:50-51 — so `register.pubkey` is the
    shape server.ts:204-205 imports. We never sign (the fake nodes never return results), so no private key kept."""
    if _HAVE_CRYPTO:
        raw = Ed25519PrivateKey.generate().public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return base64.b64encode(raw).decode()
    return base64.b64encode(os.urandom(32)).decode()


# ---- blocking HTTP (run in HTTP_POOL so it never stalls the client event loop) ----
def _blocking_get(port, path, admin, timeout):
    t0 = time.perf_counter()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        headers = {"authorization": "Bearer " + admin} if admin else {}
        conn.request("GET", path, headers=headers)
        r = conn.getresponse(); body = r.read(); status = r.status; conn.close()
        data = None
        try: data = json.loads(body)
        except Exception: pass
        return {"ok": status == 200, "status": status, "ms": (time.perf_counter() - t0) * 1000, "data": data}
    except Exception as e:
        return {"ok": False, "status": 0, "ms": (time.perf_counter() - t0) * 1000, "data": None, "err": str(e)}


async def aget(loop, port, path, admin=None, timeout=10):
    return await loop.run_in_executor(HTTP_POOL, _blocking_get, port, path, admin, timeout)


# =========================================================================================================
# One fake worker: register → welcome → heartbeat → stay connected. Controllable up/down for churn phases.
# =========================================================================================================
class FakeWorker:
    def __init__(self, idx: int, url: str, token: str):
        self.name = f"fake-{idx}-{PID}"           # unique id (server.ts:202 rejects duplicate live ids)
        self.url = url
        self.token = token
        self.pubkey = gen_pubkey()
        self.ceil = 0.6
        # desired-state signalling: `up` set == should be connected; `down` set == tear the live socket down now.
        self.up = asyncio.Event(); self.up.set()
        self.down = asyncio.Event()
        self.stop = asyncio.Event()
        self.connected = False
        self.welcomes = 0                          # successful (re)registrations observed
        self.fatal = None                          # set on a fatal `denied` (bad token / banned)
        self.last_denied = None
        self.last_error = None
        self.hb_task = None
        self._slock = None

    # --- external controls (called from the same loop; no thread-safety needed) ---
    def bring_down(self):
        self.up.clear(); self.down.set()
    def bring_up(self):
        self.down.clear(); self.up.set()
    def shutdown(self):
        self.stop.set(); self.up.set(); self.down.set()

    def register_frame(self):
        return {"t": "register", "joinToken": self.token, "pubkey": self.pubkey,
                "node": {"id": self.name, "backend": "cpu", "label": "cpu:fake",
                         "os": platform.system().lower()}}

    async def _send(self, ws, obj):
        async with self._slock:                    # websockets requires serialized writes (worker uses a lock too)
            await ws.send(json.dumps(obj))

    async def _heartbeat(self, ws):
        # ~4s heartbeat, mirroring worker_torch.py:887-893 (consumed by server.ts:211-233).
        try:
            while True:
                await asyncio.sleep(HB_INTERVAL)
                await self._send(ws, {"t": "heartbeat", "id": self.name, "load1": 0, "cores": 4, "util": 0.0,
                                      "duty": self.ceil, "ceil": self.ceil, "paused": False,
                                      "pausedReason": None, "schedule": "always"})
        except Exception:
            return

    async def _reader(self, ws):
        async for raw in ws:
            try:
                m = json.loads(raw)
            except Exception:
                continue
            t = m.get("t")
            if t == "welcome":
                try: self.ceil = float(m.get("duty", 0.6))
                except Exception: pass
                self.connected = True; self.welcomes += 1
                if self.hb_task is None or self.hb_task.done():
                    self.hb_task = asyncio.ensure_future(self._heartbeat(ws))
            elif t == "denied":
                reason = str(m.get("reason", "")); self.last_denied = reason
                # worker_torch.py:903-910: only a permanent rejection is fatal; a transient dup-id retries.
                if ("bad join token" in reason) or ("removed by admin" in reason):
                    self.fatal = reason; self.stop.set(); self.down.set()
                return
            elif t == "control":                    # admin pause/ceil (server.ts control frame) — track ceil only
                if m.get("ceil") is not None:
                    try: self.ceil = float(m["ceil"])
                    except Exception: pass
            elif t == "assign":                     # a kernel shard — reply error so the coordinator fails it fast
                await self._send(ws, {"t": "result", "shardId": m.get("shardId"), "jobId": m.get("jobId"),
                                      "ok": False, "error": "fake-fleet: control-plane client, no compute"})
            elif t in ("train", "model"):           # relayed RPC — graceful error (never happens in this test)
                await self._send(ws, {"t": f"{t}_reply", "reqId": m.get("reqId"), "ok": False,
                                      "error": "fake-fleet: no torch"})
            elif t == "cache":
                await self._send(ws, {"t": "cached", "id": m.get("id"), "ok": False, "error": "fake-fleet: no compute"})
            # uncache / anything else → ignore, stay connected

    async def _serve_once(self):
        self._slock = asyncio.Lock(); self.hb_task = None
        async with websockets.connect(self.url, max_size=None, ping_interval=PING_INTERVAL,
                                      ping_timeout=PING_TIMEOUT, close_timeout=5) as ws:
            await self._send(ws, self.register_frame())
            reader = asyncio.ensure_future(self._reader(ws))
            downer = asyncio.ensure_future(self.down.wait())
            try:
                await asyncio.wait({reader, downer}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for tk in (reader, downer):
                    if not tk.done(): tk.cancel()
                if self.hb_task and not self.hb_task.done(): self.hb_task.cancel()
                self.connected = False
                # `async with` exit closes ws → coordinator onclose reaps this id (server.ts:258-280)

    async def run(self):
        while not self.stop.is_set():
            await self.up.wait()                    # block while deliberately brought-down
            if self.stop.is_set(): break
            try:
                await self._serve_once()
            except Exception as e:
                self.last_error = str(e)
            self.connected = False
            if self.stop.is_set(): break
            # Backoff ONLY on an unexpected drop / transient denial (up still set). A deliberate bring_down leaves
            # up cleared → no backoff, we just park on up.wait() until bring_up → an immediate reconnect storm.
            if self.up.is_set():
                await asyncio.sleep(BACKOFF_MIN + random.random() * (BACKOFF_MAX - BACKOFF_MIN))


# =========================================================================================================
# Coordinator-side observation helpers
# =========================================================================================================
async def fleet_size(loop, port, admin):
    """Fleet as an operator sees it: len(GET /workers). Returns (workers_len_or_None, health_fleet_or_None)."""
    w = await aget(loop, port, "/workers", admin=admin, timeout=10)
    h = await aget(loop, port, "/health", timeout=5)
    wl = len(w["data"]) if (w["ok"] and isinstance(w["data"], list)) else None
    hf = h["data"].get("fleet") if (h["ok"] and isinstance(h["data"], dict)) else None
    return wl, hf


async def wait_fleet(loop, port, admin, target, timeout_s, label):
    """Wait until /workers length == target for 2 consecutive reads (guards against a mid-transition sample)."""
    deadline = time.time() + timeout_s
    last, streak = None, 0
    while time.time() < deadline:
        wl, hf = await fleet_size(loop, port, admin)
        last = wl
        if wl == target:
            streak += 1
            if streak >= 2:
                return True, wl, hf
        else:
            streak = 0
        await asyncio.sleep(0.25)
    print(f"  [{label}] timed out waiting for fleet=={target}; last /workers len={last}", flush=True)
    return False, last, None


async def sample_until(loop, port, out, stop_evt, gap=0.03):
    """Continuously sample /health latency (ms) into `out` until stop_evt is set; inf marks a failed/hung probe."""
    while not stop_evt.is_set():
        r = await aget(loop, port, "/health", timeout=5)
        out.append(r["ms"] if r["ok"] else float("inf"))
        await asyncio.sleep(gap)


async def probe_burst(loop, port, n, gap=0.03):
    out = []
    for _ in range(n):
        r = await aget(loop, port, "/health", timeout=5)
        out.append(r["ms"] if r["ok"] else float("inf"))
        await asyncio.sleep(gap)
    return out


def lat_stats(samples):
    n = len(samples)
    fails = sum(1 for x in samples if x == float("inf"))
    good = sorted(x for x in samples if x != float("inf"))
    if not good:
        return {"n": n, "fails": fails, "max": None, "p50": None, "p95": None}
    p = lambda q: good[min(len(good) - 1, int(q * len(good)))]
    return {"n": n, "fails": fails, "max": round(good[-1], 1), "p50": round(p(0.5), 1), "p95": round(p(0.95), 1)}


def check_latency(fails, tag, stats, max_ms):
    if stats["fails"] > 0:
        fails.append(f"{tag}: {stats['fails']}/{stats['n']} /health probes FAILED (coordinator unresponsive/crashed)")
    if stats["max"] is not None and stats["max"] > max_ms:
        fails.append(f"{tag}: /health max latency {stats['max']}ms > {max_ms}ms budget (event loop not responsive)")


# =========================================================================================================
async def main():
    root = tempfile.mkdtemp(prefix="moregpu-fleet-")
    coord = None
    clients, tasks = [], []
    fails = []
    ok = False
    loop = asyncio.get_event_loop()
    try:
        port = free_port(); cfg = os.path.join(root, "mg.json"); coord_log = os.path.join(root, "coord.log")
        env = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_INSECURE="1",
                   MOREGPU_REGISTER_TIMEOUT_MS="30000")   # generous auth window for a large connect storm
        print(f"starting coordinator on 127.0.0.1:{port} · target fleet N={N} · drop {int(DROP_FRAC*100)}% "
              f"· /health budget {HEALTH_MAX_MS:.0f}ms", flush=True)
        coord = subprocess.Popen(["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write",
                                  "apps/coordinator/server.ts"], cwd=REPO, env=env,
                                 stdout=open(coord_log, "w"), stderr=subprocess.STDOUT)

        for _ in range(120):
            if coord.poll() is not None:
                raise RuntimeError(f"coordinator exited early (rc={coord.returncode}); see {coord_log}")
            r = await aget(loop, port, "/health", timeout=2)
            if r["ok"]: break
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("coordinator /health never came up")
        conf = json.load(open(cfg)); JOIN, ADMIN = conf["joinToken"], conf["adminToken"]
        url = f"ws://127.0.0.1:{port}/ws"

        base = lat_stats(await probe_burst(loop, port, 20))
        print(f"  baseline /health (0 nodes): p50={base['p50']}ms p95={base['p95']}ms max={base['max']}ms", flush=True)

        # -------- Phase A: CONNECT STORM — bring up N clients at once, sample /health throughout --------
        print(f"\n[A] connect storm: launching {N} clients simultaneously…", flush=True)
        t0 = time.time()
        for i in range(N):
            c = FakeWorker(i, url, JOIN)
            clients.append(c); tasks.append(asyncio.ensure_future(c.run()))
        storm, stop_evt = [], asyncio.Event()
        sampler = asyncio.ensure_future(sample_until(loop, port, storm, stop_evt))
        okA, wlA, hfA = await wait_fleet(loop, port, ADMIN, N, UP_TIMEOUT_S, "connect-storm")
        stop_evt.set(); await sampler
        settleA = lat_stats(await probe_burst(loop, port, 25))   # steady-state (all N connected, heartbeating)
        stormA = lat_stats(storm)
        dtA = time.time() - t0
        print(f"  fleet: /workers={wlA} /health.fleet={hfA} (want {N}) in {dtA:.1f}s", flush=True)
        print(f"  /health during storm: p50={stormA['p50']} p95={stormA['p95']} max={stormA['max']} "
              f"(fails={stormA['fails']}/{stormA['n']})", flush=True)
        print(f"  /health steady (N nodes): p50={settleA['p50']} p95={settleA['p95']} max={settleA['max']} "
              f"(fails={settleA['fails']}/{settleA['n']})", flush=True)
        if not okA:
            fails.append(f"connect storm: fleet reached /workers={wlA}, expected {N}")
        if hfA != wlA:
            fails.append(f"connect storm: /health.fleet ({hfA}) disagrees with /workers ({wlA})")
        check_latency(fails, "connect-storm", stormA, HEALTH_MAX_MS)
        check_latency(fails, "steady-state", settleA, HEALTH_MAX_MS)

        # -------- Phase B: MASS DISCONNECT — drop a big fraction at once, expect an exact reap --------
        K = int(N * DROP_FRAC)
        print(f"\n[B] mass disconnect: dropping {K}/{N} sockets simultaneously…", flush=True)
        for c in clients[:K]:
            c.bring_down()
        okB, wlB, hfB = await wait_fleet(loop, port, ADMIN, N - K, DOWN_TIMEOUT_S, "mass-disconnect")
        downLat = lat_stats(await probe_burst(loop, port, 20))
        print(f"  fleet after drop: /workers={wlB} /health.fleet={hfB} (want {N-K})", flush=True)
        print(f"  /health post-drop: p50={downLat['p50']} p95={downLat['p95']} max={downLat['max']} "
              f"(fails={downLat['fails']}/{downLat['n']})", flush=True)
        if not okB:
            fails.append(f"mass disconnect: fleet reaped to /workers={wlB}, expected {N-K}")
        if hfB != wlB:
            fails.append(f"mass disconnect: /health.fleet ({hfB}) disagrees with /workers ({wlB})")
        check_latency(fails, "post-drop", downLat, HEALTH_MAX_MS)
        if coord.poll() is not None:
            fails.append("coordinator process DIED during the mass disconnect")

        # -------- Phase C: RECONNECT STORM — bring the dropped nodes back, expect fleet==N again --------
        print(f"\n[C] reconnect storm: bringing {K} nodes back…", flush=True)
        for c in clients[:K]:
            c.bring_up()
        okC, wlC, hfC = await wait_fleet(loop, port, ADMIN, N, UP_TIMEOUT_S, "reconnect")
        settleC = lat_stats(await probe_burst(loop, port, 25))
        print(f"  fleet after reconnect: /workers={wlC} /health.fleet={hfC} (want {N})", flush=True)
        print(f"  /health post-reconnect: p50={settleC['p50']} p95={settleC['p95']} max={settleC['max']} "
              f"(fails={settleC['fails']}/{settleC['n']})", flush=True)
        if not okC:
            fails.append(f"reconnect storm: fleet returned to /workers={wlC}, expected {N}")
        if hfC != wlC:
            fails.append(f"reconnect storm: /health.fleet ({hfC}) disagrees with /workers ({wlC})")
        check_latency(fails, "post-reconnect", settleC, HEALTH_MAX_MS)

        # -------- Final: coordinator alive, /health answering, no fatal denials, everyone re-registered --------
        alive = coord.poll() is None
        finalh = await aget(loop, port, "/health", timeout=5)
        if not alive:
            fails.append("coordinator process is NOT alive at end of test")
        if not (finalh["ok"] and isinstance(finalh["data"], dict) and finalh["data"].get("ok") is True):
            fails.append(f"final /health did not return ok:true → {finalh}")
        fatals = [c.name for c in clients if c.fatal]
        if fatals:
            fails.append(f"{len(fatals)} client(s) got a FATAL denied (e.g. {fatals[0]}: {clients[0].fatal})")
        never = [c.name for c in clients if c.welcomes == 0]
        if never:
            fails.append(f"{len(never)} client(s) never registered (e.g. {never[0]})")
        reconnected = sum(1 for c in clients[:K] if c.welcomes >= 2)
        print(f"\nsummary: fatal_denials={len(fatals)} never_registered={len(never)} "
              f"dropped_nodes_that_re-registered={reconnected}/{K} coordinator_alive={alive}", flush=True)

        ok = not fails
    except Exception as e:
        fails.append(f"harness error: {e}")
    finally:
        for c in clients: c.shutdown()
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=20)
            except Exception:
                pass
        if coord:
            try: coord.terminate()
            except Exception: pass
        HTTP_POOL.shutdown(wait=False, cancel_futures=True)
        time.sleep(0.5)
        shutil.rmtree(root, ignore_errors=True)

    for f in fails:
        print("  FAIL:", f)
    print(f"\nverified: N={N} nodes · connect storm · mass disconnect ({int(DROP_FRAC*100)}%) · reconnect · "
          f"/health < {HEALTH_MAX_MS:.0f}ms")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
