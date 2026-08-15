#!/usr/bin/env python3
"""
peer_kv.py — KV-CACHE-over-PEER and MIXED-PIPE bridging EXACT-MATCH regression (no network, no model download,
CPU-only). This completes the opt-in worker->worker peer transport (docs/ROADMAP.md "Peer transport"): the two cases
the first increment still fell back on — a cached/KV decode step, and a pipe with a non-direct edge — now run over the
ring instead of the coordinator relay. It is the peer_transport.py / kv_cache_parity.py harness fused.

Scope of what this proves (MOREGPU_PEER_TRANSPORT=1):
  (1) KV-OVER-PEER — a cached decode step no longer falls back to shardPipe. The `inject`/`act`/`bridge` frames carry
      the KV {session,pos}, so EACH stage runs its OWN cached shard_forward over the ring; the coordinator injects
      stage 0 and collects the tail's `complete`, relaying ZERO per-token activations (/model/ring_stats shows
      shard_forwards flat at 0 while injects/completes climb by exactly the number of decode calls).
  (3) MIXED-PIPE BRIDGING — a pipe with SOME direct and SOME relay edges runs per-edge: the direct edges stay
      worker->worker, the relay edge is BRIDGED through the coordinator (one hop for THAT edge only), instead of the
      whole pipe falling back to shardPipe. Forced deterministically with MOREGPU_PEER_FORCE_RELAY=w1 (edge w1->w2
      relay, w2->w3 direct). /model/ring_stats shows bridged climbing (the relay edge) while shard_forwards stays 0.
  fault — a re-placed/dropped stage still falls back correctly: a streamed cached generate whose MIDDLE stage worker
      is SIGKILLed mid-stream is healed by the coordinator (re-place onto the spare + reset & re-prefill the KV) and
      the finished token sequence still equals the golden.

Three clusters over two tiny random-init models (GPT-2 = Conv1D, Llama = RMSNorm/RoPE, 4 layers → a 3-stage shard):
  PEER  (flag ON, MOREGPU_PEER_HOST=127.0.0.1, +1 spare worker for the fault case) — all-direct ring.
  RELAY (flag OFF, the DEFAULT) — the reference cached decode AND proof the default path is untouched (it relays
        every stage activation through the coordinator: shard_forwards climbs).
  MIXED (flag ON + MOREGPU_PEER_FORCE_RELAY=w1) — one relay edge bridged, the rest direct.

For each model it asserts, TOKEN-FOR-TOKEN: PEER cached-decode == GOLDEN == RELAY cached-decode, PEER relays 0
per-token activations (and is at least as fast as RELAY), MIXED cached-decode == GOLDEN with the relay edge bridged.

  python3 tests/e2e/peer_kv.py            # exits non-zero on any mismatch/failure

Honest ceilings (docs/ROADMAP.md §7): LAN-only; a segmented/NAT'd edge fails the reachability probe → that edge is
bridged (this is exactly the MIXED case). Safe for CI: torch CPU only, models built in a tempdir and deleted; all
sockets are 127.0.0.1.
"""
import base64, json, os, re, signal, socket, subprocess, sys, tempfile, threading, time, urllib.request, urllib.error, shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPT = list(range(1, 25))            # 24-token fixed prompt (all < vocab 256) — deterministic greedy run
NNEW = 8                               # cached decode tokens compared token-for-token
WORKERS = ["w1", "w2", "w3"]           # 3 workers → a 3-STAGE shard (2 edges: w1->w2, w2->w3)
PROCS = []


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gen_models(root):
    """Build tiny random-init GPT-2 and Llama models as single-file safetensors (config + weights, no tokenizer).
    4 layers each → a 3-worker shard splits into 3 contiguous stages (sizes 2,1,1). n_positions 64 leaves headroom
    for a 24-token prompt + 8 decode tokens."""
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
    ids = torch.tensor([PROMPT])
    with torch.no_grad():
        logits = m(ids).logits[0, -1]
        gen = m.generate(ids, max_new_tokens=NNEW, do_sample=False, use_cache=True)
    return int(logits.argmax()), logits.tolist(), gen[0, len(PROMPT):].tolist()


def deb64_logits(s):
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
        fp = os.path.join(self.root, model, urllib.request.unquote(file))
        if not os.path.isfile(fp):
            self.send_error(404); return
        data = open(fp, "rb").read()
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            s, e = rng[6:].split("-"); s = int(s); e = int(e) if e else len(data) - 1
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


def start_cluster(root, hfport, peer, tag, nworkers=3, extra_cenv=None):
    """Launch a coordinator + `nworkers` CPU torch workers. peer=True sets MOREGPU_PEER_TRANSPORT=1 on BOTH the
    coordinator and the workers (+ MOREGPU_PEER_HOST=127.0.0.1 → loopback peer URL). extra_cenv adds coordinator env
    (e.g. MOREGPU_PEER_FORCE_RELAY, the failover deadlines). Returns (port, admin, worker_procs, coord_log)."""
    port = free_port(); cfg = os.path.join(root, f"mg-{tag}.json")
    cenv = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_INSECURE="1",
                MOREGPU_HF_BASE=f"http://127.0.0.1:{hfport}", MOREGPU_SHARD_LOAD_DEADLINE_MS="600000")
    cenv.pop("MOREGPU_PEER_TRANSPORT", None); cenv.pop("MOREGPU_PEER_FORCE_RELAY", None)
    if peer:
        cenv["MOREGPU_PEER_TRANSPORT"] = "1"
    if extra_cenv:
        cenv.update(extra_cenv)
    coord_log = os.path.join(root, f"coord-{tag}.log")
    p = subprocess.Popen(["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write",
                          "apps/coordinator/server.ts"], cwd=REPO, env=cenv,
                         stdout=open(coord_log, "w"), stderr=subprocess.STDOUT)
    PROCS.append(p)
    for _ in range(80):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2): break
        except Exception: time.sleep(0.5)
    conf = json.load(open(cfg)); JOIN, ADMIN = conf["joinToken"], conf["adminToken"]

    wenv = dict(os.environ)
    wenv.pop("MOREGPU_PEER_TRANSPORT", None); wenv.pop("MOREGPU_PEER_HOST", None)
    if peer:
        wenv["MOREGPU_PEER_TRANSPORT"] = "1"; wenv["MOREGPU_PEER_HOST"] = "127.0.0.1"
    names = [f"w{i+1}" for i in range(nworkers)]
    wprocs = {}
    for n in names:
        wp = subprocess.Popen(["python3", "apps/worker/worker_torch.py", "--server",
                               f"ws://127.0.0.1:{port}/ws", "--token", JOIN, "--name", n, "--cpu"],
                              cwd=REPO, env=wenv,
                              stdout=open(os.path.join(root, f"{tag}-{n}.log"), "w"), stderr=subprocess.STDOUT)
        wprocs[n] = wp; PROCS.append(wp)
    for _ in range(160):
        w = api(port, "/workers", admin=ADMIN)
        if isinstance(w, list) and len(w) >= nworkers: break
        time.sleep(0.5)
    return port, ADMIN, wprocs, coord_log


def load_shard(port, admin, model, sid, workers=WORKERS):
    r = api(port, "/model/shard", "POST", {"model": model, "id": sid, "push": True, "async": True,
                                           "workers": workers}, admin=admin)
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


def decode_cached(port, sid, admin, session):
    """Cached greedy decode: step 0 PREFILLS the whole prompt (pos 0); every later step sends ONLY the new token
    (pos = tokens already cached). With the flag on this drives KV-over-peer; with it off, the relay path."""
    seq = list(PROMPT); toks = []; t0 = time.perf_counter()
    fr = api(port, "/model/shard_forward", "POST",
             {"id": sid, "input_ids": seq, "session": session, "cached": True, "pos": 0}, admin=admin)
    a = fr.get("argmax")
    if a is None:
        raise RuntimeError(f"cached prefill failed: {fr}")
    seq.append(a); toks.append(a)
    for _ in range(NNEW - 1):
        fr = api(port, "/model/shard_forward", "POST",
                 {"id": sid, "input_ids": [seq[-1]], "session": session, "cached": True, "pos": len(seq) - 1},
                 admin=admin)
        a = fr.get("argmax")
        if a is None:
            raise RuntimeError(f"cached decode failed: {fr}")
        seq.append(a); toks.append(a)
    return toks, (time.perf_counter() - t0) * 1000


# Coordinator re-place log line (server.ts replaceStage) — reused from postload_failover.py.
REPLACE_RE = re.compile(r"re-placed stage\s+(\d+)\s+\(layers\s+(\d+)-(\d+)\)\s+(\S+)\s*(?:->|→)\s*(\S+)")


def stream_generate(port, sid, admin, on_first, kill_after, max_new):
    """POST /model/shard_generate (streamed NDJSON, cached KV internally). After `kill_after` tokens, fire on_first()
    once (the test SIGKILLs the middle worker), keep reading to the end. Returns (tokens, done?, error?)."""
    body = json.dumps({"id": sid, "input_ids": PROMPT, "max_new_tokens": max_new}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/model/shard_generate", data=body, method="POST",
                                 headers={"content-type": "application/json", "authorization": "Bearer " + admin})
    tokens, done, error, fired = [], None, None, False
    resp = urllib.request.urlopen(req, timeout=180)
    for raw in resp:
        line = raw.decode().strip()
        if not line:
            continue
        obj = json.loads(line)
        if "token" in obj:
            tokens.append(int(obj["token"]))
            if not fired and len(tokens) >= kill_after:
                fired = True; on_first()
        elif "error" in obj:
            error = obj
        elif obj.get("done"):
            done = obj
    return tokens, done, error


def stop(wprocs):
    for p in wprocs.values():
        try: p.terminate()
        except Exception: pass


def main():
    root = tempfile.mkdtemp(prefix="moregpu-peerkv-")
    ok = True
    fails = []
    try:
        import numpy as np
        print("generating tiny models (gpt2 + llama, 4 layers each → 3 stages)…", flush=True)
        specs = gen_models(root)
        goldens = {m: golden(root, m) for m in specs}

        hfport = free_port()
        RangeHandler.root = root
        httpd = ThreadingHTTPServer(("127.0.0.1", hfport), RangeHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        # ---------- PEER cluster (flag ON): cached decode over the worker->worker ring; +1 spare for the fault case ----------
        print("\n=== PEER cluster: coordinator + 4 workers · MOREGPU_PEER_TRANSPORT=1 ===", flush=True)
        pport, padmin, pw, pcoord = start_cluster(root, hfport, peer=True, tag="peer", nworkers=4,
                                                  extra_cenv={"MOREGPU_SHARD_STAGE_TRIES": "12",
                                                              "MOREGPU_SHARD_RECONNECT_WAIT_MS": "30000"})
        peer_res = {}
        for model in specs:
            sid = f"peer-{model}"
            s, err = load_shard(pport, padmin, model, sid)
            if err:
                fails.append(f"PEER {model}: {err}"); continue
            f = fwd(pport, sid, padmin, PROMPT, want_logits=True)          # one uncached forward (logits check)
            base = api(pport, f"/model/ring_stats?id={sid}", admin=padmin)  # measure the cached-decode window only
            dec, ms = decode_cached(pport, sid, padmin, f"{sid}-s1")        # cached decode OVER PEER
            fin = api(pport, f"/model/ring_stats?id={sid}", admin=padmin)
            peer_res[model] = {"argmax": f.get("argmax"),
                               "logits": deb64_logits(f["logits"]) if f.get("logits") else None,
                               "decode": dec, "ms": ms, "base": base, "fin": fin}

        # fault case: a fresh shard on w1/w2/w3, streamed cached generate, SIGKILL the MIDDLE stage → heal onto w4.
        fault_ok = fault_done = False; fault_tokens = []
        FMODEL = "tiny-llama"; fsid = "peer-fault"
        s, err = load_shard(pport, padmin, FMODEL, fsid)
        if err:
            fails.append(f"PEER fault-shard: {err}")
        else:
            mid = next((x for x in (s.get("stages") or []) if not x.get("first") and not x.get("last")), None)
            killed = {"done": False}

            def kill_mid():
                killed["done"] = True
                os.kill(pw["w2"].pid, signal.SIGKILL)
                print(f"  >>> SIGKILL w2 (pid {pw['w2'].pid}) mid cached-generate over peer", flush=True)

            gmax = 24
            fgolden = list(goldens[FMODEL][2])  # first NNEW golden ids …
            # extend golden to gmax tokens (recompute greedily, mirroring the shard path)
            import torch
            from transformers import AutoModelForCausalLM
            gm = AutoModelForCausalLM.from_pretrained(os.path.join(root, FMODEL), dtype=torch.float32).eval()
            gseq = list(PROMPT)
            with torch.no_grad():
                gen = gm.generate(torch.tensor([gseq]), max_new_tokens=gmax, do_sample=False, use_cache=True)
            fgolden = gen[0, len(PROMPT):].tolist()
            fault_tokens, done, ferr = stream_generate(pport, fsid, padmin, kill_mid, 1, gmax)
            time.sleep(1.0)
            reps = REPLACE_RE.findall(open(pcoord, "r", errors="replace").read())
            mid_reps = [t for t in reps if t[0] == "1" and t[3] == "w2"]
            w2_dead = pw["w2"].poll() is not None
            fault_done = done is not None and ferr is None
            fault_ok = (killed["done"] and w2_dead and bool(mid_reps) and fault_done and fault_tokens == fgolden)
            new_w = mid_reps[-1][4] if mid_reps else None
            print(f"  fault: w2_dead={w2_dead} re-placed stage1 w2→{new_w} done={fault_done} "
                  f"tokens={len(fault_tokens)} match_golden={fault_tokens == fgolden}", flush=True)
            if not fault_ok:
                fails.append(f"PEER fault fallback: w2_dead={w2_dead} reps={mid_reps} done={fault_done} "
                             f"match={fault_tokens == fgolden}")
        stop(pw); time.sleep(1)

        # ---------- RELAY cluster (flag OFF, DEFAULT): the reference cached decode + the relay-count contrast ----------
        print("\n=== RELAY cluster: coordinator + 3 workers · flag OFF (default) ===", flush=True)
        rport, radmin, rw, _ = start_cluster(root, hfport, peer=False, tag="relay", nworkers=3)
        relay_res = {}
        for model in specs:
            sid = f"relay-{model}"
            s, err = load_shard(rport, radmin, model, sid)
            if err:
                fails.append(f"RELAY {model}: {err}"); continue
            base = api(rport, f"/model/ring_stats?id={sid}", admin=radmin)
            dec, ms = decode_cached(rport, sid, radmin, f"{sid}-s1")
            fin = api(rport, f"/model/ring_stats?id={sid}", admin=radmin)
            relay_res[model] = {"decode": dec, "ms": ms, "base": base, "fin": fin}
        stop(rw); time.sleep(1)

        # ---------- MIXED cluster (flag ON + MOREGPU_PEER_FORCE_RELAY=w1): one relay edge bridged, the rest direct ----------
        print("\n=== MIXED cluster: coordinator + 3 workers · MOREGPU_PEER_TRANSPORT=1 MOREGPU_PEER_FORCE_RELAY=w1 ===", flush=True)
        mport, madmin, mw, _ = start_cluster(root, hfport, peer=True, tag="mixed", nworkers=3,
                                             extra_cenv={"MOREGPU_PEER_FORCE_RELAY": "w1"})
        mixed_res = {}
        for model in specs:
            sid = f"mixed-{model}"
            s, err = load_shard(mport, madmin, model, sid)
            if err:
                fails.append(f"MIXED {model}: {err}"); continue
            base = api(mport, f"/model/ring_stats?id={sid}", admin=madmin)
            dec, ms = decode_cached(mport, sid, madmin, f"{sid}-s1")
            fin = api(mport, f"/model/ring_stats?id={sid}", admin=madmin)
            mixed_res[model] = {"decode": dec, "base": base, "fin": fin}
        stop(mw); time.sleep(1)

        # ---------- compare + assert ----------
        print("\n=== KV-over-peer + mixed-pipe assertions ===", flush=True)
        NCALLS = NNEW  # cached decode = 1 prefill + (NNEW-1) decode = NNEW forward calls
        for model, arch in specs.items():
            g_arg, g_log, g_dec = goldens[model]
            pr, rr, mr = peer_res.get(model), relay_res.get(model), mixed_res.get(model)
            if not (pr and rr and mr):
                fails.append(f"{model}: missing peer/relay/mixed result"); continue

            # (1) KV-over-peer cached decode == golden == relay (token-for-token)
            kv_match = (pr["decode"] == g_dec == rr["decode"])
            logits_ok = pr["logits"] is not None and np.allclose(pr["logits"], g_log, atol=1e-3, rtol=1e-3)
            # peer forward argmax == golden
            fwd_ok = (pr["argmax"] == g_arg)
            # (2) peer relayed ZERO per-token activations over the cached decode; injects/completes climbed by NCALLS
            pb, pf = pr["base"], pr["fin"]
            d_inj = pf.get("injects", 0) - pb.get("injects", 0)
            d_comp = pf.get("completes", 0) - pb.get("completes", 0)
            d_relay = pf.get("shard_forwards", 0) - pb.get("shard_forwards", 0)
            ring_ok = bool(pf.get("all_direct")) and len(pf.get("edges", [])) == 2 and all(e.get("mode") == "direct" for e in pf.get("edges", []))
            offpath_ok = (d_relay == 0 and d_inj == NCALLS and d_comp == NCALLS)
            # (contrast) RELAY relayed every stage activation through the coordinator
            rb, rf = rr["base"], rr["fin"]
            relay_relayed = rf.get("shard_forwards", 0) - rb.get("shard_forwards", 0)
            relay_crossed_ok = relay_relayed == NCALLS * len(WORKERS)
            # peer at least as fast as relay (fewer coordinator round-trips per token; tolerance for CI jitter)
            faster_ok = pr["ms"] <= rr["ms"] * 1.5

            # (3) MIXED pipe cached decode == golden; the relay edge (w1->w2) bridged, w2->w3 direct, shard_forwards 0
            mb, mf = mr["base"], mr["fin"]
            mixed_match = (mr["decode"] == g_dec)
            medges = mf.get("edges", [])
            mixed_edges_ok = (len(medges) == 2
                              and any(e.get("from") == "w1" and e.get("to") == "w2" and e.get("mode") == "relay" for e in medges)
                              and any(e.get("from") == "w2" and e.get("to") == "w3" and e.get("mode") == "direct" for e in medges)
                              and not mf.get("all_direct"))
            d_bridged = mf.get("bridged", 0) - mb.get("bridged", 0)
            d_mrelay = mf.get("shard_forwards", 0) - mb.get("shard_forwards", 0)
            mixed_proof_ok = (d_bridged == NCALLS and d_mrelay == 0)

            model_ok = (kv_match and logits_ok and fwd_ok and ring_ok and offpath_ok
                        and relay_crossed_ok and faster_ok and mixed_match and mixed_edges_ok and mixed_proof_ok)
            ok = ok and model_ok
            if not model_ok:
                fails.append(f"{model}: kv_match={kv_match} logits={logits_ok} fwd={fwd_ok} ring_all_direct={ring_ok} "
                             f"peer_relayed={d_relay}(==0) inj={d_inj} comp={d_comp} relay_relayed={relay_relayed} "
                             f"faster={faster_ok} mixed_match={mixed_match} mixed_edges={mixed_edges_ok} "
                             f"bridged={d_bridged}(=={NCALLS}) mixed_relay={d_mrelay}")
            mstr = " ".join("{}->{}[{}]".format(e.get("from"), e.get("to"), e.get("mode")) for e in medges)
            print(f"  {'PASS' if model_ok else 'FAIL'} {model:10s} ({arch:5s}): "
                  f"KV-over-peer==golden==relay={kv_match} logits={logits_ok} "
                  f"coord_relayed_per_token={d_relay}(==0) injects={d_inj} completes={d_comp} "
                  f"| RELAY relayed {relay_relayed} (==NCALLS*{len(WORKERS)}) "
                  f"| peer {pr['ms']:.0f}ms vs relay {rr['ms']:.0f}ms faster={faster_ok}", flush=True)
            print(f"       golden decode[:{NNEW}]={g_dec}", flush=True)
            print(f"       peer   decode[:{NNEW}]={pr['decode']}", flush=True)
            print(f"       MIXED  cached==golden={mixed_match} edges=[{mstr}] bridged={d_bridged}(=={NCALLS}) "
                  f"shard_forwards={d_mrelay}(==0)", flush=True)
        ok = ok and fault_ok and not fails
        print(f"\n  fault fallback (dropped/re-placed stage still == golden): {'PASS' if fault_ok else 'FAIL'}", flush=True)
    finally:
        stop({i: p for i, p in enumerate(PROCS)})
        time.sleep(1)
        shutil.rmtree(root, ignore_errors=True)
    for f in fails:
        print("  FAIL:", f)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
