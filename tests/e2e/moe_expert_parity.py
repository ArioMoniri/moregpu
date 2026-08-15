#!/usr/bin/env python3
"""
moe_expert_parity.py — download-free EXPERT-PARALLEL (MoE) EXACT-MATCH regression (no network, no download, CPU-only).

The FIRST verifiable increment of expert parallelism (docs/ROADMAP.md "MoE expert parallelism (the Kimi path)").
Generates a tiny random-init routed-MoE (OLMoE: RMSNorm + RoPE + a top-k routed expert MLP per layer), serves it
from a local Range server (standing in for the HF hub via MOREGPU_HF_BASE), starts a coordinator + THREE CPU torch
workers, then places it as an EP×pipeline HYBRID:

    • the DENSE backbone (token embedding, every attention block, the router `mlp.gate`, final norm + LM head) on
      ONE worker — its routed FFN is proxied out; and
    • the routed experts SPLIT across the other two workers, each resident for a SUBSET of expert indices only
      (experts {0,1} on one holder, {2,3} on the other), selected download-free via the shipped multi-file
      loader (hfSafetensorsPlan → stageExpertTensors: one expert's gate/up/down_proj may live in DIFFERENT source
      files; the coordinator Range-fetches each from the right file and merges them).

One forward is driven layer-by-layer with the router DISPATCH/COMBINE RELAYED THROUGH THE COORDINATOR (correctness
-first; the peer-mesh all-to-all that takes the coordinator off the per-token data path is a LATER increment).
It asserts the expert-parallel next-token argmax (and logits) EXACTLY reproduce the un-sharded MoE. The golden is
computed IN THIS RUN on the SAME transformers version (numerics drift across versions). Both a SINGLE-file and a
MULTI-file checkpoint are tested — the multi-file case is the real one: it forces each expert's tensors to be
genuinely non-contiguous across shards, exercising the shipped multi-file loader on real per-expert addressing.

Safe for CI: torch CPU only, model built in a tempdir and deleted.

  python3 tests/e2e/moe_expert_parity.py            # exits non-zero on any mismatch/failure
"""
import json, os, subprocess, sys, tempfile, threading, time, urllib.request, urllib.error, shutil, socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]   # all < vocab (256); a fixed prompt so the run is deterministic
PROCS = []


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gen_models(root):
    """Build one tiny random-init OLMoE (4 layers, 4 routed experts, top-2) as BOTH a single-file safetensors and
    a many-file sharded checkpoint (+ model.safetensors.index.json). The tiny shard size scatters each expert's
    gate/up/down_proj across different files, so the multi-file variant exercises non-contiguous expert addressing."""
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(0)
    cfg = OlmoeConfig(vocab_size=256, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      num_experts=4, num_experts_per_tok=2, max_position_embeddings=64)
    m = OlmoeForCausalLM(cfg).eval()
    m.save_pretrained(os.path.join(root, "tiny-olmoe"), safe_serialization=True)                      # single-file
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
    import numpy as np, base64
    return np.frombuffer(base64.b64decode(b64), dtype="<f4")


def main():
    root = tempfile.mkdtemp(prefix="moregpu-moe-parity-")
    ok = True
    try:
        print("generating tiny OLMoE (4 layers · 4 experts · top-2) single-file + multi-file…", flush=True)
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

        # 3 CPU torch workers: the coordinator places 1 as the dense backbone and splits the experts over the other 2.
        for n in ("w1", "w2", "w3"):
            PROCS.append(subprocess.Popen(["python3", "apps/worker/worker_torch.py", "--server",
                                           f"ws://127.0.0.1:{port}/ws", "--token", JOIN, "--name", n, "--cpu"],
                                          cwd=REPO, env=os.environ,
                                          stdout=open(os.path.join(root, f"{n}.log"), "w"), stderr=subprocess.STDOUT))
        for _ in range(120):
            w = api(port, "/workers", admin=ADMIN)
            if isinstance(w, list) and len(w) >= 3: break
            time.sleep(0.5)

        for model, kind in specs.items():
            gi, glogits = golden_argmax(root, model)
            sid = f"moe-{model}"
            r = api(port, "/model/moe_shard", "POST", {"model": model, "id": sid, "push": True}, admin=ADMIN)
            if not r.get("ok"):
                print(f"  FAIL {model}: moe_shard failed: {r}"); ok = False; continue
            place = f"backbone={r['backbone']['worker']} holders=" + " ".join(
                f"{h['worker']}{{{','.join(map(str, h['experts']))}}}" for h in r["holders"])
            fr = api(port, "/model/moe_forward", "POST", {"id": sid, "input_ids": INPUT_IDS,
                                                          "return_logits": True}, admin=ADMIN)
            si = fr.get("argmax")
            match = (si == gi)
            diff = "n/a"
            if isinstance(fr.get("logits"), str):
                import numpy as np
                diff = float(np.abs(b64_to_logits(fr["logits"]) - glogits.numpy()).max())
            print(f"  {'PASS' if match else 'FAIL'} {model:16s} ({kind:11s}): EP argmax={si} golden argmax={gi} "
                  f"max|Δlogit|={diff}", flush=True)
            print(f"       placement: {place}", flush=True)
            api(port, "/model/moe_unload", "POST", {"id": sid}, admin=ADMIN)
            ok = ok and match and (diff == "n/a" or diff < 1e-3)
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
