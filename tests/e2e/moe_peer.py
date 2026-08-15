#!/usr/bin/env python3
"""
moe_peer.py — MoE ALL-TO-ALL-over-PEER EXACT-MATCH regression (no network, no model download, CPU-only). This is the
moe_expert_parity.py harness with the OPT-IN peer transport: the routed-expert dispatch/combine that the first MoE
increment RELAYED THROUGH THE COORDINATOR now goes worker->worker (backbone node <-> expert-holder nodes over sealed
+ Ed25519-signed peer channels). The coordinator only kicks the forward off at the backbone and collects the final
logits — it relays ZERO expert activations. See docs/ROADMAP.md "MoE expert parallelism (the Kimi path)" §4.

Generates a tiny random-init routed-MoE (OLMoE: RMSNorm + RoPE + a top-k routed expert MLP per layer), serves it from
a local Range server (HF-hub stand-in via MOREGPU_HF_BASE), and runs TWO clusters, each placing it as an EP×pipeline
hybrid — the DENSE backbone on ONE worker, the routed experts SPLIT across the other two ({0,1} and {2,3}):

  PEER  cluster (flag ON, MOREGPU_PEER_HOST=127.0.0.1) — the backbone drives the whole forward and dispatches each
        layer's routed experts DIRECTLY to their holder workers over the peer mesh, combining locally. It asserts
        /model/moe_ring_stats shows the mesh wired ALL-DIRECT and `dispatches` (coordinator-relayed expert
        activations) STAYS 0 across the forward while injects/completes climb by 1 — the observable evidence the
        all-to-all is off the coordinator.
  RELAY cluster (flag OFF, DEFAULT) — the reference: the same forward with the router dispatch/combine relayed
        through the coordinator (moePipe). Proof the default path is untouched — it relays every expert activation
        (`dispatches` > 0).

For each of a SINGLE-file and a MULTI-file checkpoint it asserts, EXACTLY: PEER moe_forward argmax == un-sharded
GOLDEN argmax == RELAY argmax (and logits match numerically), with the coordinator relaying 0 expert activations on
the peer path. The golden is computed IN THIS RUN on the SAME transformers version.

  python3 tests/e2e/moe_peer.py            # exits non-zero on any mismatch/failure

Honest ceilings (docs/ROADMAP.md §7): LAN-only; an unreachable holder falls back to the relayed moePipe. Safe for
CI: torch CPU only, model built in a tempdir and deleted; all sockets are 127.0.0.1.
"""
import base64, json, os, socket, subprocess, sys, tempfile, threading, time, urllib.request, urllib.error, shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]   # all < vocab (256); a fixed prompt so the run is deterministic
PROCS = []


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gen_models(root):
    """Build one tiny random-init OLMoE (4 layers, 4 routed experts, top-2) as BOTH a single-file safetensors and a
    many-file sharded checkpoint (+ index.json) — the tiny shard size scatters each expert's gate/up/down_proj across
    files, exercising the download-free multi-file loader on genuinely non-contiguous per-expert addressing."""
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(0)
    cfg = OlmoeConfig(vocab_size=256, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      num_experts=4, num_experts_per_tok=2, max_position_embeddings=64)
    m = OlmoeForCausalLM(cfg).eval()
    m.save_pretrained(os.path.join(root, "tiny-olmoe"), safe_serialization=True)                              # single-file
    m.save_pretrained(os.path.join(root, "tiny-olmoe-multi"), safe_serialization=True, max_shard_size="48KB")  # multi-file
    return {"tiny-olmoe": "single-file", "tiny-olmoe-multi": "multi-file"}


def golden_argmax(root, model):
    """The reference: load the MoE un-sharded (fp32, CPU) and take the last-token argmax — same run, same versions."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(os.path.join(root, model), dtype=torch.float32).eval()
    with torch.no_grad():
        logits = m(torch.tensor([INPUT_IDS])).logits[0, -1]
    return int(logits.argmax()), logits


class RangeHandler(BaseHTTPRequestHandler):
    root = None

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
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


def api(port, path, method="GET", body=None, admin=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    h = {"content-type": "application/json"}
    if admin: h["authorization"] = "Bearer " + admin
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        return {"httperror": e.code, "body": e.read().decode()[:400]}


def b64_to_logits(b64):
    import numpy as np
    return np.frombuffer(base64.b64decode(b64), dtype="<f4")


def start_cluster(root, hfport, peer, tag):
    """Launch a coordinator + 3 CPU torch workers (1 dense backbone + 2 expert holders). peer=True sets
    MOREGPU_PEER_TRANSPORT=1 on the coordinator + workers (+ MOREGPU_PEER_HOST=127.0.0.1 → loopback peer URL)."""
    port = free_port(); cfg = os.path.join(root, f"mg-{tag}.json")
    cenv = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_INSECURE="1",
                MOREGPU_HF_BASE=f"http://127.0.0.1:{hfport}", MOREGPU_SHARD_LOAD_DEADLINE_MS="600000")
    cenv.pop("MOREGPU_PEER_TRANSPORT", None)
    if peer:
        cenv["MOREGPU_PEER_TRANSPORT"] = "1"
    p = subprocess.Popen(["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write",
                          "apps/coordinator/server.ts"], cwd=REPO, env=cenv,
                         stdout=open(os.path.join(root, f"coord-{tag}.log"), "w"), stderr=subprocess.STDOUT)
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
    wprocs = {}
    for n in ("w1", "w2", "w3"):
        wp = subprocess.Popen(["python3", "apps/worker/worker_torch.py", "--server",
                               f"ws://127.0.0.1:{port}/ws", "--token", JOIN, "--name", n, "--cpu"],
                              cwd=REPO, env=wenv,
                              stdout=open(os.path.join(root, f"{tag}-{n}.log"), "w"), stderr=subprocess.STDOUT)
        wprocs[n] = wp; PROCS.append(wp)
    for _ in range(160):
        w = api(port, "/workers", admin=ADMIN)
        if isinstance(w, list) and len(w) >= 3: break
        time.sleep(0.5)
    return port, ADMIN, wprocs


def stop(wprocs):
    for p in wprocs.values():
        try: p.terminate()
        except Exception: pass


def main():
    root = tempfile.mkdtemp(prefix="moregpu-moe-peer-")
    ok = True
    fails = []
    try:
        import numpy as np
        print("generating tiny OLMoE (4 layers · 4 experts · top-2) single-file + multi-file…", flush=True)
        specs = gen_models(root)
        goldens = {m: golden_argmax(root, m) for m in specs}

        hfport = free_port()
        RangeHandler.root = root
        httpd = ThreadingHTTPServer(("127.0.0.1", hfport), RangeHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        # ---------- PEER cluster (flag ON): the backbone<->holder all-to-all over the peer mesh ----------
        print("\n=== PEER cluster: coordinator + 3 workers · MOREGPU_PEER_TRANSPORT=1 ===", flush=True)
        pport, padmin, pw = start_cluster(root, hfport, peer=True, tag="peer")
        peer_res = {}
        for model in specs:
            sid = f"moe-peer-{model}"
            r = api(pport, "/model/moe_shard", "POST", {"model": model, "id": sid, "push": True}, admin=padmin)
            if not r.get("ok"):
                fails.append(f"PEER {model}: moe_shard failed: {r}"); continue
            place = f"backbone={r['backbone']['worker']} holders=" + " ".join(
                f"{h['worker']}{{{','.join(map(str, h['experts']))}}}" for h in r["holders"])
            base = api(pport, f"/model/moe_ring_stats?id={sid}", admin=padmin)
            fr = api(pport, "/model/moe_forward", "POST", {"id": sid, "input_ids": INPUT_IDS, "return_logits": True}, admin=padmin)
            fin = api(pport, f"/model/moe_ring_stats?id={sid}", admin=padmin)
            peer_res[model] = {"argmax": fr.get("argmax"),
                               "logits": b64_to_logits(fr["logits"]) if isinstance(fr.get("logits"), str) else None,
                               "place": place, "base": base, "fin": fin}
            api(pport, "/model/moe_unload", "POST", {"id": sid}, admin=padmin)
        stop(pw); time.sleep(1)

        # ---------- RELAY cluster (flag OFF, DEFAULT): the reference relayed dispatch/combine (moePipe) ----------
        print("\n=== RELAY cluster: coordinator + 3 workers · flag OFF (default) ===", flush=True)
        rport, radmin, rw = start_cluster(root, hfport, peer=False, tag="relay")
        relay_res = {}
        for model in specs:
            sid = f"moe-relay-{model}"
            r = api(rport, "/model/moe_shard", "POST", {"model": model, "id": sid, "push": True}, admin=radmin)
            if not r.get("ok"):
                fails.append(f"RELAY {model}: moe_shard failed: {r}"); continue
            base = api(rport, f"/model/moe_ring_stats?id={sid}", admin=radmin)
            fr = api(rport, "/model/moe_forward", "POST", {"id": sid, "input_ids": INPUT_IDS, "return_logits": True}, admin=radmin)
            fin = api(rport, f"/model/moe_ring_stats?id={sid}", admin=radmin)
            relay_res[model] = {"argmax": fr.get("argmax"),
                                "logits": b64_to_logits(fr["logits"]) if isinstance(fr.get("logits"), str) else None,
                                "base": base, "fin": fin}
            api(rport, "/model/moe_unload", "POST", {"id": sid}, admin=radmin)
        stop(rw); time.sleep(1)

        # ---------- compare + assert ----------
        print("\n=== MoE-over-peer parity + off-the-data-path assertions ===", flush=True)
        for model, kind in specs.items():
            gi, glog = goldens[model]
            pr, rr = peer_res.get(model), relay_res.get(model)
            if not (pr and rr):
                fails.append(f"{model}: missing peer/relay result"); continue

            # (1) PEER argmax == golden == RELAY argmax; logits match numerically
            argmax_ok = (pr["argmax"] == gi == rr["argmax"])
            pdiff = float(np.abs(pr["logits"] - glog.numpy()).max()) if pr["logits"] is not None else 9e9
            rdiff = float(np.abs(pr["logits"] - rr["logits"]).max()) if (pr["logits"] is not None and rr["logits"] is not None) else 9e9
            logits_ok = (pdiff < 1e-3 and rdiff < 1e-5)

            # (2) mesh wired all-direct; coordinator relayed ZERO expert activations while injects/completes climbed by 1
            pb, pf = pr["base"], pr["fin"]
            mesh_ok = bool(pf.get("wired")) and bool(pf.get("all_direct")) and len(pf.get("holders", [])) == 2
            d_disp = pf.get("dispatches", 0) - pb.get("dispatches", 0)
            d_inj = pf.get("injects", 0) - pb.get("injects", 0)
            d_comp = pf.get("completes", 0) - pb.get("completes", 0)
            offpath_ok = (d_disp == 0 and d_inj == 1 and d_comp == 1 and pb.get("dispatches", 0) == 0)

            # (contrast) RELAY relayed the expert activations through the coordinator
            rb, rf = rr["base"], rr["fin"]
            relay_disp = rf.get("dispatches", 0) - rb.get("dispatches", 0)
            relay_crossed_ok = relay_disp > 0

            model_ok = argmax_ok and logits_ok and mesh_ok and offpath_ok and relay_crossed_ok
            ok = ok and model_ok
            if not model_ok:
                fails.append(f"{model}: argmax={argmax_ok} logits(pΔ={pdiff:.2e},rΔ={rdiff:.2e})={logits_ok} "
                             f"mesh_all_direct={mesh_ok} peer_dispatches={d_disp}(==0) inj={d_inj} comp={d_comp} "
                             f"relay_dispatches={relay_disp}(>0)")
            print(f"  {'PASS' if model_ok else 'FAIL'} {model:16s} ({kind:11s}): "
                  f"peer argmax={pr['argmax']} golden={gi} relay={rr['argmax']} match={argmax_ok} "
                  f"max|Δlogit| peer-vs-golden={pdiff:.2e} peer-vs-relay={rdiff:.2e}", flush=True)
            print(f"       placement: {pr['place']}", flush=True)
            print(f"       PEER mesh all_direct={mesh_ok} coord_relayed_experts={d_disp}(==0) injects={d_inj} "
                  f"completes={d_comp}  |  RELAY relayed {relay_disp} expert activations through the coordinator", flush=True)
        ok = ok and not fails
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
