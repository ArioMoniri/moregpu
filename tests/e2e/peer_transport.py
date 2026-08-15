#!/usr/bin/env python3
"""
peer_transport.py — LAN-only worker->worker PEER activation transport EXACT-MATCH regression (no network, no model
download, CPU-only). This is the shard_parity.py harness with a THIRD worker and the OPT-IN peer transport flag.

Scope (docs/ROADMAP.md "Peer transport"): when MOREGPU_PEER_TRANSPORT=1, adjacent pipeline stages hand the sealed
hidden-state activation DIRECTLY worker->successor over a small LAN WebSocket listener instead of round-tripping every
per-token activation through the coordinator. The coordinator stays the control-plane + weight source: it hands out
the plan with successor endpoints, brokers the peer connection (wires the ring + probes reachability), kicks off
stage 0 (`inject`), and receives the last stage's result (`complete`). The sealed crypto is unchanged (AES-256-GCM
with the shared tenant key + Ed25519 origin signature by the predecessor).

It builds two tiny random-init models (GPT-2 = Conv1D, Llama-family = RMSNorm/RoPE, 4 layers each), serves them from a
local HTTP Range server (standing in for the HF hub via MOREGPU_HF_BASE), and runs TWO clusters:

  PEER  cluster  — coordinator + 3 CPU torch workers, ALL with MOREGPU_PEER_TRANSPORT=1 (and MOREGPU_PEER_HOST
                   =127.0.0.1 so the advertised peer URL is loopback). A 3-STAGE download-free shard wires a DIRECT
                   ring (w1->w2->w3). Forward + greedy-decode run over the peer path.
  RELAY cluster  — coordinator + 3 CPU torch workers, flag OFF (the DEFAULT). Same 3-stage shard, same forward +
                   greedy-decode over the unchanged coordinator-relay path (shardPipe). This is both the
                   "relay-path result" reference AND proof the default path is untouched.

For each model it asserts, TOKEN-FOR-TOKEN:
  • PEER forward argmax  == un-sharded GOLDEN argmax  == RELAY forward argmax   (and logits match numerically)
  • PEER greedy-decode   == GOLDEN greedy-decode      == RELAY greedy-decode    (NNEW tokens)
  • the coordinator did NOT relay the per-token activation on the peer path: /model/ring_stats shows the ring wired
    ALL-DIRECT and `shard_forwards` stays 0 across the whole peer forward+decode, while `injects`/`completes` climb
    by exactly (1 forward + NNEW decode). The RELAY cluster, by contrast, relays (1+NNEW)*3 stage activations — the
    observable evidence the coordinator is off the per-token data path only on the peer cluster.

  python3 tests/e2e/peer_transport.py            # exits non-zero on any mismatch/failure

Honest ceilings (see docs/ROADMAP.md §7): LAN-only. The peer listener binds a raw loopback/LAN address — there is NO
STUN/TURN/NAT traversal; a segmented/NAT'd edge fails the reachability probe and the coordinator keeps that edge on
relay (a mixed pipe with any non-direct edge falls back to shardPipe for this increment). KV-cache-over-peer is out
of scope here: a cached decode step falls back to the relay path. Greedy decode is driven UNCACHED (each token
re-sends the growing sequence — one ring inject per token), which is the path this increment moves off the
coordinator. Safe for CI: torch CPU only, models built in a tempdir and deleted; all sockets are 127.0.0.1.
"""
import base64, json, os, subprocess, sys, tempfile, threading, time, urllib.request, urllib.error, shutil, socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]   # all < vocab (256); a fixed prompt so the run is deterministic
NNEW = 8                               # greedy-decode tokens compared token-for-token
WORKERS = ["w1", "w2", "w3"]           # 3 workers → a 3-STAGE shard (2 peer edges: w1->w2, w2->w3)
PROCS = []


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gen_models(root):
    """Build tiny random-init GPT-2 and Llama models as single-file safetensors (config + weights, no tokenizer).
    4 layers each → a 3-worker shard splits into 3 contiguous stages (sizes 2,1,1)."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    specs = {}
    g = GPT2LMHeadModel(GPT2Config(vocab_size=256, n_positions=64, n_embd=32, n_layer=4, n_head=2))
    g.save_pretrained(os.path.join(root, "tiny-gpt2"), safe_serialization=True)
    specs["tiny-gpt2"] = "gpt2"
    L = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                     max_position_embeddings=64))
    L.save_pretrained(os.path.join(root, "tiny-llama"), safe_serialization=True)
    specs["tiny-llama"] = "llama"
    return specs


def golden(root, model):
    """The reference: load the model un-sharded (fp32, CPU) — same run, same transformers version. Returns the
    last-token argmax, its logits, and the greedy-decode continuation (HF's own KV cache)."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(os.path.join(root, model), dtype=torch.float32).eval()
    ids = torch.tensor([INPUT_IDS])
    with torch.no_grad():
        logits = m(ids).logits[0, -1]
        gen = m.generate(ids, max_new_tokens=NNEW, do_sample=False, use_cache=True)
    return int(logits.argmax()), logits.tolist(), gen[0, len(INPUT_IDS):].tolist()


def deb64_logits(s):
    """Decode the coordinator's base64 float32 logits blob (worker f32_to_b64) into a python list of floats."""
    import numpy as np
    return np.frombuffer(base64.b64decode(s), dtype="<f4").astype("float64").tolist()


class RangeHandler(BaseHTTPRequestHandler):
    root = None

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        # map /{model}/resolve/main/{file}?download=true -> {root}/{model}/{file}
        path = self.path.split("?", 1)[0]
        if "/resolve/main/" not in path:
            self.send_error(404); return
        model, file = path.lstrip("/").split("/resolve/main/", 1)
        file = urllib.request.unquote(file)
        fp = os.path.join(self.root, model, file)
        if not os.path.isfile(fp):
            self.send_error(404); return
        data = open(fp, "rb").read()
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            s, e = rng[6:].split("-")
            s = int(s); e = int(e) if e else len(data) - 1
            chunk = data[s:e + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {s}-{e}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers(); self.wfile.write(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers(); self.wfile.write(data)


def api(port, path, method="GET", body=None, admin=None, timeout=90):
    data = json.dumps(body).encode() if body is not None else None
    h = {"content-type": "application/json"}
    if admin: h["authorization"] = "Bearer " + admin
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        return {"httperror": e.code, "body": e.read().decode()[:400]}


def start_cluster(root, hfport, peer, tag):
    """Launch a coordinator + 3 CPU torch workers. peer=True sets MOREGPU_PEER_TRANSPORT=1 on BOTH the coordinator
    and the workers (and MOREGPU_PEER_HOST=127.0.0.1 so the advertised peer URL is loopback). Returns (port, admin,
    procs) — procs is this cluster's process list (also appended to PROCS for final cleanup)."""
    procs = []
    port = free_port(); cfg = os.path.join(root, f"mg-{tag}.json")
    cenv = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1",
                MOREGPU_HF_BASE=f"http://127.0.0.1:{hfport}", MOREGPU_SHARD_LOAD_DEADLINE_MS="600000")
    cenv.pop("MOREGPU_PEER_TRANSPORT", None)
    if peer:
        cenv["MOREGPU_PEER_TRANSPORT"] = "1"
    p = subprocess.Popen(["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write",
                          "apps/coordinator/server.ts"], cwd=REPO, env=cenv,
                         stdout=open(os.path.join(root, f"coord-{tag}.log"), "w"), stderr=subprocess.STDOUT)
    procs.append(p); PROCS.append(p)
    for _ in range(80):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2): break
        except Exception: time.sleep(0.5)
    conf = json.load(open(cfg)); JOIN, ADMIN = conf["joinToken"], conf["adminToken"]

    wenv = dict(os.environ)
    wenv.pop("MOREGPU_PEER_TRANSPORT", None); wenv.pop("MOREGPU_PEER_HOST", None)
    if peer:
        wenv["MOREGPU_PEER_TRANSPORT"] = "1"; wenv["MOREGPU_PEER_HOST"] = "127.0.0.1"  # advertise a loopback peer URL
    for n in WORKERS:
        p = subprocess.Popen(["python3", "apps/worker/worker_torch.py", "--server",
                              f"ws://127.0.0.1:{port}/ws", "--token", JOIN, "--name", n, "--cpu"],
                             cwd=REPO, env=wenv,
                             stdout=open(os.path.join(root, f"{tag}-{n}.log"), "w"), stderr=subprocess.STDOUT)
        procs.append(p); PROCS.append(p)
    for _ in range(120):
        w = api(port, "/workers", admin=ADMIN)
        if isinstance(w, list) and len(w) >= len(WORKERS): break
        time.sleep(0.5)
    return port, ADMIN, procs


def load_shard(port, admin, model, sid):
    r = api(port, "/model/shard", "POST", {"model": model, "id": sid, "push": True, "async": True,
                                           "workers": WORKERS}, admin=admin)
    if r.get("status") != "loading":
        return None, f"shard did not start: {r}"
    s = None
    for _ in range(120):
        s = api(port, f"/model/shard_status?id={sid}", admin=admin)
        if s.get("status") in ("ready", "error"): break
        time.sleep(2)
    if s.get("status") != "ready":
        return None, f"shard not ready: {s}"
    return s, None


def fwd(port, sid, admin, ids, want_logits=False):
    return api(port, "/model/shard_forward", "POST",
               {"id": sid, "input_ids": ids, **({"return_logits": True} if want_logits else {})}, admin=admin)


def decode_uncached(port, sid, admin):
    """Greedy decode by re-sending the growing sequence each token (one forward/inject per token)."""
    seq = list(INPUT_IDS); toks = []
    for _ in range(NNEW):
        r = fwd(port, sid, admin, seq)
        a = r.get("argmax")
        if a is None:
            raise RuntimeError(f"decode step failed: {r}")
        seq.append(a); toks.append(a)
    return toks


def stop(procs):
    for p in procs:
        try: p.terminate()
        except Exception: pass


def main():
    root = tempfile.mkdtemp(prefix="moregpu-peer-")
    ok = True
    try:
        print("generating tiny models (gpt2 + llama, 4 layers each → 3 stages)…", flush=True)
        specs = gen_models(root)
        goldens = {m: golden(root, m) for m in specs}

        hfport = free_port()
        RangeHandler.root = root
        httpd = ThreadingHTTPServer(("127.0.0.1", hfport), RangeHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        # ---------- PEER cluster (flag ON): forward + decode over the worker->worker ring ----------
        print("\n=== PEER cluster: coordinator + 3 workers · MOREGPU_PEER_TRANSPORT=1 ===", flush=True)
        pport, padmin, pprocs = start_cluster(root, hfport, peer=True, tag="peer")
        peer_res = {}
        for model, arch in specs.items():
            sid = f"peer-{model}"
            s, err = load_shard(pport, padmin, model, sid)
            if err:
                print(f"  FAIL {model}: {err}"); ok = False; continue
            base = api(pport, f"/model/ring_stats?id={sid}", admin=padmin)
            f = fwd(pport, sid, padmin, INPUT_IDS, want_logits=True)   # peer forward
            dec = decode_uncached(pport, sid, padmin)                  # peer greedy decode
            fin = api(pport, f"/model/ring_stats?id={sid}", admin=padmin)
            peer_res[model] = {
                "argmax": f.get("argmax"),
                "logits": deb64_logits(f["logits"]) if f.get("logits") else None,
                "decode": dec,
                "base": base, "fin": fin,
            }
        stop(pprocs); time.sleep(1)

        # ---------- RELAY cluster (flag OFF, DEFAULT): the reference relay path (shardPipe) ----------
        print("\n=== RELAY cluster: coordinator + 3 workers · flag OFF (default) ===", flush=True)
        rport, radmin, rprocs = start_cluster(root, hfport, peer=False, tag="relay")
        relay_res = {}
        for model, arch in specs.items():
            sid = f"relay-{model}"
            s, err = load_shard(rport, radmin, model, sid)
            if err:
                print(f"  FAIL {model}: {err}"); ok = False; continue
            base = api(rport, f"/model/ring_stats?id={sid}", admin=radmin)
            f = fwd(rport, sid, radmin, INPUT_IDS, want_logits=True)   # relay forward
            dec = decode_uncached(rport, sid, radmin)                  # relay greedy decode
            fin = api(rport, f"/model/ring_stats?id={sid}", admin=radmin)
            relay_res[model] = {
                "argmax": f.get("argmax"),
                "logits": deb64_logits(f["logits"]) if f.get("logits") else None,
                "decode": dec,
                "base": base, "fin": fin,
            }
        stop(rprocs); time.sleep(1)

        # ---------- compare + assert ----------
        import numpy as np
        print("\n=== parity + off-the-data-path assertions ===", flush=True)
        expect_injects = 1 + NNEW
        for model, arch in specs.items():
            g_arg, g_log, g_dec = goldens[model]
            pr, rr = peer_res.get(model), relay_res.get(model)
            if not pr or not rr:
                print(f"  FAIL {model}: missing peer/relay result"); ok = False; continue

            # (1) forward argmax: peer == golden == relay
            fwd_argmax_ok = (pr["argmax"] == g_arg == rr["argmax"])
            # (1b) forward logits: peer ~ golden (shard recompute) and peer ~ relay (same math)
            logits_ok = (pr["logits"] is not None and rr["logits"] is not None
                         and np.allclose(pr["logits"], g_log, atol=1e-3, rtol=1e-3)
                         and np.allclose(pr["logits"], rr["logits"], atol=1e-5, rtol=1e-5))
            # (2) greedy decode: peer == golden == relay (token-for-token)
            decode_ok = (pr["decode"] == g_dec == rr["decode"])

            # (3) peer ring wired ALL-DIRECT with 2 direct edges (3 stages), and the coordinator relayed ZERO
            #     per-token activations on the peer path while injects/completes climbed by (1 + NNEW).
            pb, pf = pr["base"], pr["fin"]
            edges = pf.get("edges", [])
            ring_ok = (bool(pf.get("wired")) and bool(pf.get("all_direct"))
                       and len(edges) == 2 and all(e.get("mode") == "direct" for e in edges))
            d_inj = pf.get("injects", 0) - pb.get("injects", 0)
            d_comp = pf.get("completes", 0) - pb.get("completes", 0)
            d_relay = pf.get("shard_forwards", 0) - pb.get("shard_forwards", 0)
            offpath_ok = (d_relay == 0 and d_inj == expect_injects and d_comp == expect_injects
                          and pb.get("shard_forwards", 0) == 0)

            # (contrast) the RELAY cluster DID relay every stage activation through the coordinator
            rb, rf = rr["base"], rr["fin"]
            relay_relayed = rf.get("shard_forwards", 0) - rb.get("shard_forwards", 0)
            relay_crossed_ok = (relay_relayed == expect_injects * len(WORKERS))

            model_ok = fwd_argmax_ok and logits_ok and decode_ok and ring_ok and offpath_ok and relay_crossed_ok
            ok = ok and model_ok
            edge_str = " ".join("{}->{}[{}]".format(e.get("from"), e.get("to"), e.get("mode")) for e in edges)
            print(f"  {'PASS' if model_ok else 'FAIL'} {model:10s} ({arch:5s}): "
                  f"fwd peer==golden==relay argmax={fwd_argmax_ok} logits={logits_ok} "
                  f"decode({NNEW})==golden==relay={decode_ok} ring_all_direct={ring_ok} "
                  f"coord_relayed_per_token={d_relay}(==0) injects={d_inj} completes={d_comp}", flush=True)
            print(f"       peer argmax={pr['argmax']} golden={g_arg} relay={rr['argmax']}  edges=[{edge_str}]", flush=True)
            print(f"       golden decode[:{NNEW}]={g_dec}", flush=True)
            print(f"       peer   decode[:{NNEW}]={pr['decode']}", flush=True)
            print(f"       RELAY cluster relayed {relay_relayed} stage activations through the coordinator "
                  f"(== (1+{NNEW})*{len(WORKERS)}={expect_injects*len(WORKERS)}) vs PEER cluster {d_relay}", flush=True)
            if not model_ok:
                print(f"       (debug) peer.base={pb} peer.fin={pf}", flush=True)
    finally:
        stop(PROCS)
        time.sleep(1)
        shutil.rmtree(root, ignore_errors=True)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
