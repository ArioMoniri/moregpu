#!/usr/bin/env python3
"""
shard_parity.py — download-free pipeline-shard EXACT-MATCH regression test (no network, no model download, CPU-only).

Generates two tiny random-init models (GPT-2 = Conv1D, and Llama-family = RMSNorm/RoPE), serves them from a local
HTTP server with Range support (standing in for the HF hub via MOREGPU_HF_BASE), starts a coordinator + N CPU torch
workers, then asserts that a download-free pipeline shard reproduces the un-sharded model's next-token argmax and
logits. The golden is computed IN THIS RUN on the SAME transformers version (numerics drift across versions), and it
also guards the Llama-family `create_causal_mask` shard path. Safe for CI: torch CPU only, models are built in a
tempdir and deleted.

  python3 tests/e2e/shard_parity.py            # exits non-zero on any mismatch/failure
"""
import json, os, subprocess, sys, tempfile, threading, time, urllib.request, urllib.error, shutil, socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]   # all < vocab (256); a fixed prompt so the run is deterministic
PROCS = []


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gen_models(root):
    """Build tiny random-init GPT-2 and Llama models as single-file safetensors (config + weights, no tokenizer)."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    specs = {}
    g = GPT2LMHeadModel(GPT2Config(vocab_size=256, n_positions=64, n_embd=32, n_layer=4, n_head=2))
    g.save_pretrained(os.path.join(root, "tiny-gpt2"), safe_serialization=True)
    specs["tiny-gpt2"] = "gpt2"
    # same weights, but split across many files + a model.safetensors.index.json — exercises the multi-file loader
    g.save_pretrained(os.path.join(root, "tiny-gpt2-multi"), safe_serialization=True, max_shard_size="64KB")
    specs["tiny-gpt2-multi"] = "gpt2·multi"
    L = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                     max_position_embeddings=64))
    L.save_pretrained(os.path.join(root, "tiny-llama"), safe_serialization=True)
    specs["tiny-llama"] = "llama"
    return specs


def golden_argmax(root, model):
    """The reference: load the model un-sharded (fp32, CPU) and take the last-token argmax — same run, same versions."""
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


def main():
    root = tempfile.mkdtemp(prefix="moregpu-parity-")
    ok = True
    try:
        print("generating tiny models (gpt2 + llama)…", flush=True)
        specs = gen_models(root)

        hfport = free_port()
        RangeHandler.root = root
        httpd = ThreadingHTTPServer(("127.0.0.1", hfport), RangeHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        port = free_port(); cfg = os.path.join(root, "mg.json")
        env = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_INSECURE="1",
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
            gi, glogits = golden_argmax(root, model)
            sid = f"shard-{model}"
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
            fr = api(port, "/model/shard_forward", "POST", {"id": sid, "input_ids": INPUT_IDS,
                                                            "return_logits": True}, admin=ADMIN)
            si = fr.get("argmax")
            match = (si == gi)
            # We requested return_logits — actually CHECK them (argmax alone can miss a sub-argmax numeric drift).
            ldiff = "n/a"
            if fr.get("logits"):
                import numpy as _np, base64 as _b64
                pl = _np.frombuffer(_b64.b64decode(fr["logits"]), dtype="<f4").astype("float64")
                gl = glogits.detach().cpu().numpy().astype("float64").reshape(-1)
                ldiff = float(_np.abs(pl - gl).max()) if pl.shape == gl.shape else 9e9
                match = match and (ldiff < 1e-3)
            else:
                match = False  # requested logits but the shard path returned none
            print(f"  {'PASS' if match else 'FAIL'} {model:10s} ({arch:5s}): shard argmax={si} golden={gi} max|Δlogit|={ldiff}", flush=True)
            ok = ok and match
    finally:
        for p in PROCS:
            try: p.terminate()
            except Exception: pass
        time.sleep(1)
        try:
            import torch  # free any cached model
        except Exception:
            pass
        shutil.rmtree(root, ignore_errors=True)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
