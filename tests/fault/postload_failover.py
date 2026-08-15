#!/usr/bin/env python3
"""
postload_failover.py — POST-LOAD stage-failover regression test (no network, no model download, CPU-only).

Proves the shipped post-load failover feature end to end: a download-free shard whose MIDDLE stage's worker PROCESS
is KILLED mid-generation is HEALED by the coordinator — it re-places that stage's exact layer range onto a spare
torch worker, re-streams the stage download-free, swaps the live plan, and RESUMES the greedy loop — so the final
generated sequence still equals the un-sharded greedy golden.

This is the strictly harder sibling of tests/fault/churn_resume.py: churn_resume drops a worker's *socket* mid-LOAD
while the PROCESS survives (resume the staging). Here the PROCESS is DESTROYED mid-GENERATION (SIGKILL) — its shard
is gone for good — and recovery must move the stage to a DIFFERENT node and keep the in-flight request going.

Harness is the tests/e2e/shard_parity.py style: a tiny random-init Llama served from a local HTTP Range server
(HF-hub stand-in via MOREGPU_HF_BASE — no download), a Deno coordinator, and CPU torch workers. Four workers join:
w1/w2/w3 carry the 3 pipeline stages, w4 is an idle SPARE not in the plan. We drive /model/shard_generate (streams
one NDJSON line per token), read a couple of tokens to confirm generation is live, then `kill -9` w2 (the MIDDLE
stage). The coordinator's shardPipe detects the dead stage worker, re-places layers [start,end) onto w4, re-streams
that one stage (streamStageToWorker → the same download-free path the initial load uses), and restarts the forward
from stage 0 (no KV cache → the growing seq is simply recomputed). The stream then finishes on the healed plan.

What the real code does (read before writing this test):
  • apps/coordinator/server.ts  shardPipe(): on a stage whose worker is gone (pre-check) or a shard_forward RPC that
    fails after the worker left, calls replaceStage() and RESTARTS the pipe from stage 0 (bounded by SHARD_STAGE_TRIES).
  • apps/coordinator/server.ts  replaceStage(): picks a torch worker not already carrying a stage of this plan (else
    waits SHARD_RECONNECT_WAIT_MS for one), (re)streams the stage via streamStageToWorker(resume=false), then swaps
    plan.stages[idx].worker and logs `re-placed stage <i> (layers a-b) <old> → <new> after post-load disconnect`.
  • apps/coordinator/server.ts  ws.onclose: a departed worker's READY shard plan is KEPT (not deleted) so shardPipe
    can heal it; only a load-time (empty-stages) plan is left to the load path.

Asserts (all must hold):
  1. the killed worker's PROCESS actually died (poll() is not None) — its shard is truly gone, so this tests real
     re-placement onto another node, not a socket-only reconnect;
  2. the coordinator log shows the re-place line, moving the middle stage's layer range OFF w2 ONTO the spare w4;
  3. the streamed generation COMPLETED (a {done:...} line, no {error}) despite the mid-flight kill;
  4. the full generated token sequence equals the un-sharded greedy golden (correctness survived the failover).

  python3 tests/fault/postload_failover.py            # exits non-zero on any mismatch/failure

Safe for CI: torch CPU only, one tiny model built in a tempdir and deleted; all sockets are 127.0.0.1.
"""
import json, os, re, shutil, signal, socket, subprocess, sys, tempfile, threading, time
import urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "tiny-failover-llama"                     # single-segment ref → matches the coordinator's HF_REPO_RE
PROMPT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]            # all < vocab (256); fixed prompt → deterministic greedy run
MAX_NEW = 40                                     # long enough that many tokens remain to route through the killed node
KILL_AFTER_TOKENS = 1                            # kill w2 right after the 1st streamed token → deep mid-generation
PROCS = []


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gen_model(root):
    """Build a tiny random-init Llama (RoPE + causal mask → the harder shard forward path) as single-file
    safetensors, no tokenizer. 4 layers so a 3-stage split gives a real MIDDLE stage (sizes 2,1,1 → stage 1 =
    layers [2,3)). hidden=256 keeps each per-token forward non-trivial so the kill lands mid-generation."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=256, intermediate_size=512,
                                     num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=8,
                                     max_position_embeddings=128))
    m.save_pretrained(os.path.join(root, MODEL), safe_serialization=True)
    size = os.path.getsize(os.path.join(root, MODEL, "model.safetensors"))
    print(f"built {MODEL}: model.safetensors = {size/1e6:.2f} MB (4 layers → 3 stages: sizes 2,1,1)", flush=True)


def golden_greedy(root, n):
    """Reference: un-sharded (fp32, CPU) greedy decode that MIRRORS the shard path — recompute the FULL growing
    sequence each step and take the last-token argmax (the shard path has no KV cache, so it does the same). Same
    run, same transformers version → the sharded argmax must match at every step, so the sequences stay in lockstep."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(os.path.join(root, MODEL), dtype=torch.float32).eval()
    seq = list(PROMPT_IDS)
    with torch.no_grad():
        for _ in range(n):
            logits = m(torch.tensor([seq])).logits[0, -1]
            seq.append(int(logits.argmax()))
    return seq[len(PROMPT_IDS):]                  # just the newly generated ids


class RangeHandler(BaseHTTPRequestHandler):
    """Serves {root}/{model}/{file} for /{model}/resolve/main/{file}, with HTTP Range — the HF hub stand-in."""
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
            self.send_error(404); return               # e.g. tokenizer.json 404 → coordinator simply skips it
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


# Coordinator re-place log line (server.ts replaceStage): "re-placed stage <i> (layers a-b) <old> → <new> after ...".
# The message is uncolored; tolerate the unicode arrow.
REPLACE_RE = re.compile(r"re-placed stage\s+(\d+)\s+\(layers\s+(\d+)-(\d+)\)\s+(\S+)\s*(?:->|→)\s*(\S+)")


def replaced_lines(logpath):
    try:
        return REPLACE_RE.findall(open(logpath, "r", errors="replace").read())
    except OSError:
        return []


def stream_generate(port, sid, admin, on_first_tokens, kill_after):
    """POST /model/shard_generate and read the NDJSON stream token by token. After `kill_after` token lines have
    arrived, invoke on_first_tokens() ONCE (the test uses it to SIGKILL the middle worker) and keep reading to the
    end. Returns (tokens, done_obj_or_None, error_obj_or_None)."""
    body = json.dumps({"id": sid, "input_ids": PROMPT_IDS, "max_new_tokens": MAX_NEW}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/model/shard_generate", data=body, method="POST",
                                 headers={"content-type": "application/json", "authorization": "Bearer " + admin})
    tokens, done, error, fired = [], None, None, False
    resp = urllib.request.urlopen(req, timeout=180)     # streaming NDJSON; each readline blocks up to the timeout
    for raw in resp:                                     # iterates line-by-line as chunks arrive
        line = raw.decode().strip()
        if not line:
            continue
        obj = json.loads(line)
        if "token" in obj:
            tokens.append(int(obj["token"]))
            if not fired and len(tokens) >= kill_after:
                fired = True
                on_first_tokens()                        # kill the middle worker mid-generation
        elif "error" in obj:
            error = obj
        elif obj.get("done"):
            done = obj
    return tokens, done, error


def main():
    root = tempfile.mkdtemp(prefix="moregpu-failover-")
    ok = True
    fails = []
    try:
        print("generating tiny llama…", flush=True)
        gen_model(root)
        golden = golden_greedy(root, MAX_NEW)
        print(f"golden greedy ({MAX_NEW} tok): {golden}", flush=True)

        # local HF-hub stand-in (Range server)
        hfport = free_port()
        RangeHandler.root = root
        httpd = ThreadingHTTPServer(("127.0.0.1", hfport), RangeHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        # coordinator — small SHARD_RECONNECT_WAIT so the "no spare" branch (not exercised here: w4 IS a spare)
        # can't hang CI, generous stage tries so re-placement + restart has headroom.
        port = free_port(); cfg = os.path.join(root, "mg.json")
        env = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1",
                   MOREGPU_HF_BASE=f"http://127.0.0.1:{hfport}",
                   MOREGPU_SHARD_LOAD_DEADLINE_MS="600000",
                   MOREGPU_SHARD_STAGE_TRIES="12",
                   MOREGPU_SHARD_RECONNECT_WAIT_MS="30000")
        coord_log = os.path.join(root, "coord.log")
        PROCS.append(subprocess.Popen(
            ["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write",
             "apps/coordinator/server.ts"], cwd=REPO, env=env,
            stdout=open(coord_log, "w"), stderr=subprocess.STDOUT))
        for _ in range(80):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2): break
            except Exception: time.sleep(0.5)
        conf = json.load(open(cfg)); JOIN, ADMIN = conf["joinToken"], conf["adminToken"]

        # four CPU torch workers: w1/w2/w3 → the 3 pipeline stages, w4 → an idle SPARE the failover re-places onto
        workers = {}
        for n in ("w1", "w2", "w3", "w4"):
            workers[n] = subprocess.Popen(
                ["python3", "apps/worker/worker_torch.py", "--server", f"ws://127.0.0.1:{port}/ws",
                 "--token", JOIN, "--name", n, "--cpu"], cwd=REPO, env=os.environ,
                stdout=open(os.path.join(root, f"{n}.log"), "w"), stderr=subprocess.STDOUT)
            PROCS.append(workers[n])
        for _ in range(160):                          # torch+transformers import at worker start is slow
            w = api(port, "/workers", admin=ADMIN)
            if isinstance(w, list) and len({x.get("id") for x in w} & set(workers)) == 4: break
            time.sleep(0.5)

        # download-free shard across w1/w2/w3 (explicit order → stage i = wI+1); w4 stays out of the plan
        sid = "failover-shard"
        r = api(port, "/model/shard", "POST",
                {"model": MODEL, "id": sid, "push": True, "async": True, "workers": ["w1", "w2", "w3"]}, admin=ADMIN)
        if r.get("status") != "loading":
            fails.append(f"shard did not start: {r}"); raise SystemExit
        st = None; s = {}
        for _ in range(180):
            s = api(port, f"/model/shard_status?id={sid}", admin=ADMIN); st = s.get("status")
            if st in ("ready", "error"): break
            time.sleep(1)
        if st != "ready":
            fails.append(f"shard never reached ready: status={st} detail={s}"); raise SystemExit
        stages = s.get("stages") or []
        mid = next((x for x in stages if not x.get("first") and not x.get("last")), None)
        print(f"shard ready: stages={[ (x['worker'], x['start'], x['end']) for x in stages ]}", flush=True)
        if not mid or mid.get("worker") != "w2":
            fails.append(f"expected middle stage on w2, got {mid}"); raise SystemExit

        # drive a streamed generation and SIGKILL w2 (the MIDDLE stage) right after the first token
        killed = {"done": False}

        def kill_middle():
            killed["done"] = True
            os.kill(workers["w2"].pid, signal.SIGKILL)
            print(f"  >>> SIGKILL w2 (pid {workers['w2'].pid}) after {KILL_AFTER_TOKENS} token(s) — mid-generation", flush=True)

        tokens, done, error = stream_generate(port, sid, ADMIN, kill_middle, KILL_AFTER_TOKENS)

        time.sleep(1.0)                               # let the coordinator log the re-place line + w2 fully reap
        w2_dead = workers["w2"].poll() is not None
        reps = replaced_lines(coord_log)

        # ---- assertions ----
        if not killed["done"]:
            fails.append("kill hook never fired — generation finished before the first token was read (raise MAX_NEW)")
        if not w2_dead:
            fails.append(f"w2 PROCESS still alive (pid {workers['w2'].pid}) — SIGKILL failed; not a real node loss")
        mid_reps = [t for t in reps if t[0] == "1" and t[3] == "w2"]   # stage index 1, moved off w2
        if not mid_reps:
            fails.append(f"coordinator did not log a re-place of stage 1 off w2 — failover did not fire (reps={reps})")
        else:
            _, a, b, old, new = mid_reps[-1]
            if new == "w2" or new in ("w1", "w3"):
                fails.append(f"middle stage re-placed onto {new} (not the spare w4): {mid_reps[-1]}")
            if (int(a), int(b)) != (int(mid["start"]), int(mid["end"])):
                fails.append(f"re-placed layer range {a}-{b} != original middle range {mid['start']}-{mid['end']}")
        if error is not None:
            fails.append(f"generation reported an in-band error (failover did not recover): {error}")
        if done is None:
            fails.append(f"generation did not complete with a done line (got {len(tokens)} tokens)")
        if tokens != golden:
            fails.append(f"generated sequence != un-sharded golden\n    got   ={tokens}\n    golden={golden}")

        ok = not fails
        new_worker = mid_reps[-1][4] if mid_reps else None
        print(f"  w2_dead={w2_dead}  re-placed stage1 w2 → {new_worker}  tokens={len(tokens)}  done={bool(done)}")
        print(f"  match_golden={tokens == golden}", flush=True)
    except SystemExit:
        ok = False
    finally:
        for p in PROCS:
            try: p.terminate()
            except Exception: pass
        time.sleep(1)
        shutil.rmtree(root, ignore_errors=True)

    for f in fails:
        print("  FAIL:", f)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
