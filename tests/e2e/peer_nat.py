#!/usr/bin/env python3
"""
peer_nat.py — cross-network peer reachability: MULTI-CANDIDATE endpoints + reflexive-address discovery (CPU-only,
no model download, loopback). This is the honest "NAT traversal" story for MoreGPU's opt-in peer transport.

MoreGPU does NOT attempt automatic NAT hole-punching (that needs STUN/TURN + coordinated simultaneous-open and
can't be made to work — or be verified — through an arbitrary/symmetric NAT). Instead a worker advertises an
ORDERED LIST of reachable candidate peer endpoints: an operator-provided public / port-forwarded / VPN address
(MOREGPU_PEER_PUBLIC) FIRST, then its raw LAN IP. The dialing peer probes them in order and uses whichever
actually answers; if none do, the coordinator keeps that edge on the relay/bridge path (unchanged fallback).
GET /whoami reports the caller's observed public IP so an operator knows what to port-forward/advertise.

This proves that machinery against the REAL server.ts + worker_torch.py:

  (1) reflexive discovery — GET /whoami returns the caller's observed source IP (127.0.0.1 on loopback).
  (2) candidate round-trip — each worker advertises TWO candidates (an unreachable public one FIRST, then its
      loopback LAN one); the coordinator surfaces both in /workers, plus the reflexive IP it observed.
  (3) in-order fallthrough is LOAD-BEARING — with an UNREACHABLE candidate advertised FIRST, a 3-stage shard
      still wires an ALL-DIRECT peer ring (every edge `direct`, 0 coordinator-relayed activations) and the
      forward + greedy decode match the un-sharded golden TOKEN-FOR-TOKEN. A naive "probe only the first
      candidate" worker would fail the probe and fall back to relay — so all-direct here proves the worker
      fell through the dead candidate to the reachable one.

Reuses tests/e2e/peer_transport.py's model-gen + Range-server + shard/forward helpers (imported, not duplicated).

Run:  python3 tests/e2e/peer_nat.py            # exits non-zero on any mismatch/failure
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peer_transport as pt  # noqa: E402  (reuse gen_models/golden/RangeHandler/api/load_shard/fwd/decode_uncached)
from http.server import ThreadingHTTPServer  # noqa: E402

MODEL = "tiny-llama"
DEAD_PUBLIC = "127.0.0.1:1"   # a closed port → the FIRST (public) candidate is unreachable, must be skipped
_RESULTS = []


def check(passed, label):
    _RESULTS.append((passed, label))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


def start_cluster_nat(root, hfport, tag):
    """Coordinator + 3 CPU torch workers, peer transport ON, each worker advertising an UNREACHABLE public
    candidate (MOREGPU_PEER_PUBLIC=127.0.0.1:1) ahead of its loopback LAN candidate."""
    procs = []
    port = pt.free_port()
    cfg = os.path.join(root, f"mg-{tag}.json")
    cenv = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_INSECURE="1",
                MOREGPU_HF_BASE=f"http://127.0.0.1:{hfport}", MOREGPU_SHARD_LOAD_DEADLINE_MS="600000",
                MOREGPU_PEER_TRANSPORT="1")
    p = subprocess.Popen(["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write",
                          "apps/coordinator/server.ts"], cwd=pt.REPO, env=cenv,
                         stdout=open(os.path.join(root, f"coord-{tag}.log"), "w"), stderr=subprocess.STDOUT)
    procs.append(p); pt.PROCS.append(p)
    for _ in range(80):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2): break
        except Exception: time.sleep(0.5)
    conf = json.load(open(cfg)); JOIN, ADMIN = conf["joinToken"], conf["adminToken"]

    wenv = dict(os.environ)
    wenv["MOREGPU_PEER_TRANSPORT"] = "1"
    wenv["MOREGPU_PEER_HOST"] = "127.0.0.1"     # loopback LAN candidate (reachable)
    wenv["MOREGPU_PEER_PUBLIC"] = DEAD_PUBLIC   # advertised FIRST, unreachable → must be skipped
    for n in pt.WORKERS:
        wp = subprocess.Popen(["python3", "apps/worker/worker_torch.py", "--server",
                               f"ws://127.0.0.1:{port}/ws", "--token", JOIN, "--name", n, "--cpu"],
                              cwd=pt.REPO, env=wenv,
                              stdout=open(os.path.join(root, f"{tag}-{n}.log"), "w"), stderr=subprocess.STDOUT)
        procs.append(wp); pt.PROCS.append(wp)
    for _ in range(120):
        w = pt.api(port, "/workers", admin=ADMIN)
        if isinstance(w, list) and len(w) >= len(pt.WORKERS): break
        time.sleep(0.5)
    return port, ADMIN, procs


def main():
    root = tempfile.mkdtemp(prefix="moregpu-nat-")
    try:
        print("generating tiny llama (4 layers → 3 stages)…", flush=True)
        pt.gen_models(root)
        g_argmax, g_logits, g_decode = pt.golden(root, MODEL)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (pt.RangeHandler,), {"root": root}))
        hfport = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        print("\n=== PEER cluster (public candidate 127.0.0.1:1 UNREACHABLE, advertised first) ===", flush=True)
        port, admin, _ = start_cluster_nat(root, hfport, "nat")

        # (1) reflexive discovery
        print("\n(1) reflexive-address discovery")
        who = pt.api(port, "/whoami")
        check(who.get("ip") == "127.0.0.1", f"GET /whoami reports the caller's observed IP (got {who.get('ip')!r})")

        # (2) candidate round-trip through the coordinator
        print("\n(2) multi-candidate advertisement round-trips to /workers")
        wl = pt.api(port, "/workers", admin=admin)
        dead_url = f"ws://{DEAD_PUBLIC}/peer"
        adv = [w for w in wl if (w.get("peer") or {}).get("candidates")]
        all_two = adv and all(len(w["peer"]["candidates"]) == 2 and w["peer"]["candidates"][0] == dead_url for w in adv)
        check(len(adv) == len(pt.WORKERS), f"all {len(pt.WORKERS)} workers advertise peer candidates ({len(adv)} seen)")
        check(bool(all_two), "each advertises [unreachable-public-first, loopback-LAN] and coordinator surfaces both")
        check(all(w.get("reflexiveIp") == "127.0.0.1" for w in adv), "coordinator recorded each worker's reflexive IP")

        # (3) in-order fallthrough is load-bearing: dead-first candidate → STILL all-direct + golden
        print("\n(3) in-order fallthrough → ALL-DIRECT peer ring despite a dead first candidate")
        sid = "nat-llama"
        s, err = pt.load_shard(port, admin, MODEL, sid)
        if err:
            check(False, f"shard load: {err}"); return finish()
        f = pt.fwd(port, sid, admin, pt.INPUT_IDS, want_logits=True)
        st = pt.api(port, f"/model/ring_stats?id={sid}", admin=admin)
        edges = st.get("edges", [])
        all_direct = bool(edges) and all(e.get("mode") == "direct" for e in edges)
        check(f.get("argmax") == g_argmax, f"peer forward argmax == golden ({f.get('argmax')} == {g_argmax})")
        check(all_direct, f"ring wired ALL-DIRECT despite dead-first candidate (edges={[e.get('mode') for e in edges]})")
        check(st.get("shard_forwards", -1) == 0, f"0 coordinator-relayed activations on the peer forward (shard_forwards={st.get('shard_forwards')})")
        dec = pt.decode_uncached(port, sid, admin)
        check(dec == g_decode, f"peer greedy decode == golden token-for-token ({dec} == {g_decode})")
        st2 = pt.api(port, f"/model/ring_stats?id={sid}", admin=admin)
        check(st2.get("shard_forwards", -1) == 0, f"still 0 coordinator relays across the whole decode (shard_forwards={st2.get('shard_forwards')})")
        return finish()
    finally:
        for p in pt.PROCS:
            try: p.terminate()
            except Exception: pass
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def finish():
    passed = sum(1 for ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 92)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 92)
    if passed != total:
        for ok, label in _RESULTS:
            if not ok: print(f"  FAILED: {label}")
        return 1
    print("MULTI-CANDIDATE peer reachability + reflexive discovery work; dead-first candidate falls through to direct. ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
