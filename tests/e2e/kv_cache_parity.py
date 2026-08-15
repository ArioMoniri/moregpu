#!/usr/bin/env python3
"""
kv_cache_parity.py — incremental-KV-cache EXACT-MATCH regression for the pipeline-shard decode path
(no network, no model download, CPU-only).

The shard path used to have NO KV cache: every decode token re-ran the whole growing prefix through every stage
(O(n^2)). This test proves the new per-stage incremental cache is (a) numerically exact and (b) actually skips the
prefix re-run. It is the style of tests/e2e/shard_parity.py: tiny random-init models (GPT-2 = Conv1D and a
Llama-family = RMSNorm/RoPE), served from a local HTTP Range server standing in for the HF hub (MOREGPU_HF_BASE),
a Deno coordinator, and two CPU torch workers doing a download-free 2-stage shard. Golden is computed IN THIS RUN
on the SAME transformers version (numerics drift across versions).

For each model it drives the coordinator's /model/shard_forward three ways over ~16 decode tokens and asserts they
agree TOKEN-FOR-TOKEN:
  • GOLDEN     — the un-sharded model's greedy generate (HF's own KV cache), computed in-process.
  • UNCACHED   — sharded decode with NO session: each step re-sends the full growing sequence (the old O(n^2) path).
  • CACHED     — sharded decode with a session id: step 0 PREFILLS the prompt, every later step sends ONLY the new
                 token; each stage keeps its own layers' past_key_values (HF DynamicCache) and attends to its cache.

It then exercises the session-evict op:
  • /model/shard_reset drops the live KV; a FRESH cached session still reproduces golden (clean restart), and a
    decode step (pos>0) on the EVICTED session is rejected with an explicit "re-prefill" error — the honest
    post-load fault-tolerance surface (a dropped/evicted stage loses its live KV → the request must re-prefill).

Per-token wall-clock is logged for cached vs uncached to show the cached path does NOT re-run the prefix (uncached
per-token time climbs with sequence length; cached stays roughly flat). Timing is logged, not asserted (CI-safe).

  python3 tests/e2e/kv_cache_parity.py            # exits non-zero on any mismatch/failure

Safe for CI: torch CPU only, models built in a tempdir and deleted; all sockets are 127.0.0.1.
"""
import json, os, subprocess, sys, tempfile, threading, time, urllib.request, urllib.error, shutil, socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPT = list(range(1, 49))   # 48-token fixed prompt (all < vocab 256) → a long-enough prefix that the O(n^2)
NNEW = 16                     # re-run cost of the uncached path is visibly larger than a 1-token cached decode
PROCS = []


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gen_models(root):
    """Build tiny random-init GPT-2 and Llama models as single-file safetensors (config + weights, no tokenizer).
    n_positions 128 leaves headroom for a 48-token prompt + 16 decode tokens (max seq 64)."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    specs = {}
    g = GPT2LMHeadModel(GPT2Config(vocab_size=256, n_positions=128, n_embd=64, n_layer=4, n_head=4))
    g.save_pretrained(os.path.join(root, "tiny-gpt2"), safe_serialization=True)
    specs["tiny-gpt2"] = "gpt2"
    L = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                                     num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                     max_position_embeddings=128))
    L.save_pretrained(os.path.join(root, "tiny-llama"), safe_serialization=True)
    specs["tiny-llama"] = "llama"
    return specs


def golden_tokens(root, model):
    """The reference: load the model un-sharded (fp32, CPU) and greedily generate NNEW tokens with HF's own KV
    cache — same run, same transformers version. Returns the list of new token ids."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(os.path.join(root, model), dtype=torch.float32).eval()
    ids = torch.tensor([PROMPT])
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=NNEW, do_sample=False, use_cache=True)
    return out[0, len(PROMPT):].tolist()


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


def decode_uncached(port, sid, admin):
    """Sharded decode with NO KV cache: every step re-sends the full growing sequence (the old O(n^2) path)."""
    seq = list(PROMPT); toks = []; times = []
    for _ in range(NNEW):
        t0 = time.perf_counter()
        fr = api(port, "/model/shard_forward", "POST", {"id": sid, "input_ids": seq}, admin=admin)
        times.append((time.perf_counter() - t0) * 1000)
        a = fr.get("argmax")
        if a is None:
            raise RuntimeError(f"uncached shard_forward failed: {fr}")
        seq.append(a); toks.append(a)
    return toks, times


def decode_cached(port, sid, admin, session):
    """Sharded decode WITH the incremental KV cache: step 0 prefills the whole prompt; every later step sends only
    the new token and each stage attends to its cached prefix (pos = tokens already cached on the stage)."""
    seq = list(PROMPT); toks = []; times = []
    t0 = time.perf_counter()
    fr = api(port, "/model/shard_forward", "POST",
             {"id": sid, "input_ids": seq, "session": session, "cached": True, "pos": 0}, admin=admin)
    times.append((time.perf_counter() - t0) * 1000)
    a = fr.get("argmax")
    if a is None:
        raise RuntimeError(f"cached prefill failed: {fr}")
    seq.append(a); toks.append(a)
    for _ in range(NNEW - 1):
        t0 = time.perf_counter()
        fr = api(port, "/model/shard_forward", "POST",
                 {"id": sid, "input_ids": [seq[-1]], "session": session, "cached": True, "pos": len(seq) - 1},
                 admin=admin)
        times.append((time.perf_counter() - t0) * 1000)
        a = fr.get("argmax")
        if a is None:
            raise RuntimeError(f"cached decode failed: {fr}")
        seq.append(a); toks.append(a)
    return toks, times


def main():
    root = tempfile.mkdtemp(prefix="moregpu-kvparity-")
    ok = True
    try:
        print("generating tiny models (gpt2 + llama)…", flush=True)
        specs = gen_models(root)

        hfport = free_port()
        RangeHandler.root = root
        httpd = ThreadingHTTPServer(("127.0.0.1", hfport), RangeHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        port = free_port(); cfg = os.path.join(root, "mg.json")
        env = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1",
                   MOREGPU_HF_BASE=f"http://127.0.0.1:{hfport}", MOREGPU_SHARD_LOAD_DEADLINE_MS="600000")
        PROCS.append(subprocess.Popen(["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write",
                                       "apps/coordinator/server.ts"], cwd=REPO, env=env,
                                      stdout=open(os.path.join(root, "coord.log"), "w"), stderr=subprocess.STDOUT))
        for _ in range(80):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2): break
            except Exception: time.sleep(0.5)
        conf = json.load(open(cfg)); JOIN, ADMIN = conf["joinToken"], conf["adminToken"]

        for n in ("w1", "w2"):
            PROCS.append(subprocess.Popen(["python3", "apps/worker/worker_torch.py", "--server",
                                           f"ws://127.0.0.1:{port}/ws", "--token", JOIN, "--name", n, "--cpu"],
                                          cwd=REPO, env=os.environ,
                                          stdout=open(os.path.join(root, f"{n}.log"), "w"), stderr=subprocess.STDOUT))
        for _ in range(120):
            w = api(port, "/workers", admin=ADMIN)
            if isinstance(w, list) and len(w) >= 2: break
            time.sleep(0.5)

        for model, arch in specs.items():
            golden = golden_tokens(root, model)
            sid = f"kv-{model}"
            r = api(port, "/model/shard", "POST", {"model": model, "id": sid, "push": True, "async": True,
                                                   "workers": ["w1", "w2"]}, admin=ADMIN)
            if r.get("status") != "loading":
                print(f"  FAIL {model}: shard did not start: {r}"); ok = False; continue
            st = None
            for _ in range(120):
                s = api(port, f"/model/shard_status?id={sid}", admin=ADMIN); st = s.get("status")
                if st in ("ready", "error"): break
                time.sleep(2)
            if st != "ready":
                print(f"  FAIL {model}: shard not ready: {s}"); ok = False; continue

            # warm the workers' torch path for this model (first forward pays import/JIT/alloc) so the timings
            # below reflect per-token compute, not one-off warmup landing on the first uncached step.
            api(port, "/model/shard_forward", "POST", {"id": sid, "input_ids": PROMPT}, admin=ADMIN)

            # (1) uncached sharded decode == golden
            unc, unc_ms = decode_uncached(port, sid, ADMIN)
            # (2) cached sharded decode == uncached == golden
            cac, cac_ms = decode_cached(port, sid, ADMIN, session=f"{sid}-s1")

            unc_ok = (unc == golden)
            cac_ok = (cac == unc == golden)

            # (3) session-evict op: reset drops the live KV → a FRESH session still reproduces golden…
            rr = api(port, "/model/shard_reset", "POST", {"id": sid, "session": f"{sid}-s1"}, admin=ADMIN)
            reset_ok = bool(rr.get("ok"))
            cac2, _ = decode_cached(port, sid, ADMIN, session=f"{sid}-s2")
            restart_ok = (cac2 == golden)
            # …and a decode step (pos>0) on the EVICTED session is rejected with an explicit re-prefill error
            # (the honest post-load fault-tolerance surface: lost live KV → must re-prefill).
            guard = api(port, "/model/shard_forward", "POST",
                        {"id": sid, "input_ids": [PROMPT[0]], "session": f"{sid}-s1", "cached": True, "pos": 7},
                        admin=ADMIN)
            guard_ok = (guard.get("httperror") is not None and "re-prefill" in str(guard.get("body", "")))

            model_ok = unc_ok and cac_ok and reset_ok and restart_ok and guard_ok
            ok = ok and model_ok
            # cached: split prefill (step 0) from the single-token decode steps to show decode is ~flat vs the
            # uncached path whose per-step cost grows because it re-runs the whole prefix each token.
            cac_decode = cac_ms[1:] or cac_ms
            unc_total = sum(unc_ms); cac_total = sum(cac_ms)
            print(f"  {'PASS' if model_ok else 'FAIL'} {model:10s} ({arch:5s}): "
                  f"uncached==golden={unc_ok} cached==uncached==golden={cac_ok} "
                  f"reset={reset_ok} fresh-session==golden={restart_ok} evicted-decode-rejected={guard_ok}", flush=True)
            print(f"       golden[:6]={golden[:6]}  cached[:6]={cac[:6]}  ({NNEW} tokens each)")
            print(f"       per-token ms  UNCACHED (re-runs prefix): first={unc_ms[0]:.1f} last={unc_ms[-1]:.1f} "
                  f"avg={unc_total/len(unc_ms):.1f} total={unc_total:.0f}", flush=True)
            print(f"                     CACHED   (1 token/step):   prefill={cac_ms[0]:.1f} "
                  f"decode-avg={sum(cac_decode)/len(cac_decode):.1f} total={cac_total:.0f} "
                  f"→ {unc_total/max(cac_total,1e-6):.1f}x faster, prefix re-run skipped", flush=True)
            if not guard_ok:
                print(f"       (evict-guard response was: {guard})")
    finally:
        for p in PROCS:
            try: p.terminate()
            except Exception: pass
        time.sleep(1)
        shutil.rmtree(root, ignore_errors=True)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
