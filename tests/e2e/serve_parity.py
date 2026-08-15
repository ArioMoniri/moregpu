#!/usr/bin/env python3
"""
serve_parity.py — SINGLE-NODE resident-model serving is token-for-token identical to un-sharded transformers,
including the DOWNLOAD-FREE (`push`) path. CPU-only, no model download, loopback.

This closes a real coverage gap: every other parity test drives the *sharded* path (`/model/shard` →
`shard_forward`). The non-sharded resident-model endpoints — `POST /model/load {push:true}` (the coordinator
streams the weights to the worker, which stages them in RAM and never touches the hub), `POST /model/forward`,
and `POST /model/generate` (the WHOLE greedy decode runs on the worker via HF's internal KV cache, one round-trip)
— are a shipped, documented feature (README: "download-free serving … verified token-for-token identical"), and
this is the test that actually backs that claim.

For GPT-2 and Llama-family it asserts, against an independent in-process un-sharded golden on the SAME transformers:
  • /model/forward argmax == golden argmax, and its logits match numerically (max|Δ| < 1e-3)
  • /model/generate greedy continuation == golden greedy decode, token-for-token (NNEW tokens)
  • the load ran download-free (staging: ram|disk, the fleet fetched nothing from the hub)

Reuses tests/e2e/peer_transport.py's model-gen + Range-server + golden + api helpers (imported, not duplicated).

Run:  python3 tests/e2e/serve_parity.py            # exits non-zero on any mismatch/failure
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
import peer_transport as pt  # noqa: E402  (gen_models / golden / deb64_logits / RangeHandler / api / free_port)
from http.server import ThreadingHTTPServer  # noqa: E402

_RESULTS = []


def check(passed, label):
    _RESULTS.append((passed, label))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


def start_serve_cluster(root, hfport, tag):
    """Coordinator + ONE CPU torch worker (no peer). Returns (port, admin)."""
    port = pt.free_port()
    cfg = os.path.join(root, f"mg-{tag}.json")
    cenv = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_INSECURE="1",
                MOREGPU_HF_BASE=f"http://127.0.0.1:{hfport}", MOREGPU_SHARD_LOAD_DEADLINE_MS="600000")
    pt.PROCS.append(subprocess.Popen(
        ["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write", "apps/coordinator/server.ts"],
        cwd=pt.REPO, env=cenv, stdout=open(os.path.join(root, f"coord-{tag}.log"), "w"), stderr=subprocess.STDOUT))
    for _ in range(80):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2): break
        except Exception: time.sleep(0.5)
    conf = json.load(open(cfg)); JOIN, ADMIN = conf["joinToken"], conf["adminToken"]
    pt.PROCS.append(subprocess.Popen(
        ["python3", "apps/worker/worker_torch.py", "--server", f"ws://127.0.0.1:{port}/ws", "--token", JOIN, "--name", "w1", "--cpu"],
        cwd=pt.REPO, env=dict(os.environ), stdout=open(os.path.join(root, f"{tag}-w1.log"), "w"), stderr=subprocess.STDOUT))
    for _ in range(120):
        w = pt.api(port, "/workers", admin=ADMIN)
        if isinstance(w, list) and len(w) >= 1: break
        time.sleep(0.5)
    return port, ADMIN


def load_download_free(port, admin, model, mid):
    """POST /model/load {push:true, async:true} then poll /model/status until ready. Returns (status_dict, err)."""
    r = pt.api(port, "/model/load", "POST", {"model": model, "id": mid, "push": True, "async": True}, admin=admin)
    if r.get("status") not in ("loading", "ready"):
        return None, f"load did not start: {r}"
    s = r
    for _ in range(120):
        s = pt.api(port, f"/model/status?id={mid}", admin=admin)
        if s.get("status") in ("ready", "error"): break
        time.sleep(2)
    if s.get("status") != "ready":
        return None, f"load not ready: {s}"
    return s, None


def main():
    root = tempfile.mkdtemp(prefix="moregpu-serve-")
    try:
        print("generating tiny models (gpt2 + llama)…", flush=True)
        specs = pt.gen_models(root)
        goldens = {m: pt.golden(root, m) for m in specs}

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (pt.RangeHandler,), {"root": root}))
        hfport = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        port, admin = start_serve_cluster(root, hfport, "serve")

        for model in specs:
            print(f"\n=== {model} — download-free resident serving ==")
            g_argmax, g_logits, g_decode = goldens[model]
            mid = f"serve-{model}"
            st, err = load_download_free(port, admin, model, mid)
            if err:
                check(False, f"{model}: {err}"); continue
            staging = str(st.get("info", {}).get("staging") or st.get("staging") or "?")
            check(staging in ("ram", "disk"), f"{model}: loaded download-free (staging={staging}, fleet fetched nothing from the hub)")

            f = pt.api(port, "/model/forward", "POST", {"id": mid, "input_ids": pt.INPUT_IDS, "return_logits": True}, admin=admin)
            check(f.get("argmax") == g_argmax, f"{model}: /model/forward argmax == golden ({f.get('argmax')} == {g_argmax})")
            if f.get("logits"):
                pl = pt.deb64_logits(f["logits"])
                md = max(abs(a - b) for a, b in zip(pl, g_logits))
                check(md < 1e-3, f"{model}: /model/forward logits match golden (max|Δ|={md:.2e} < 1e-3)")
            else:
                check(False, f"{model}: /model/forward returned no logits despite return_logits=True")

            gen = pt.api(port, "/model/generate", "POST", {"id": mid, "input_ids": pt.INPUT_IDS, "max_new_tokens": pt.NNEW}, admin=admin)
            toks = gen.get("tokens")
            check(toks == g_decode, f"{model}: /model/generate greedy decode == golden token-for-token ({toks} == {g_decode})")
            pt.api(port, "/model/unload", "POST", {"id": mid}, admin=admin)
    finally:
        for p in pt.PROCS:
            try: p.terminate()
            except Exception: pass
        import shutil
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(1 for ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 92)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 92)
    if passed != total:
        for ok, label in _RESULTS:
            if not ok: print(f"  FAILED: {label}")
        return 1
    print("Single-node download-free serving is token-for-token identical to un-sharded transformers. ✔")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    finally:
        for p in pt.PROCS:
            try: p.terminate()
            except Exception: pass
