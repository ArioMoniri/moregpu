#!/usr/bin/env python3
"""
churn_resume.py — download-free churn-RESUME regression test (no network, no model download, CPU-only).

Proves the shipped churn-resume feature end to end: a download-free shard whose stage worker keeps losing its
socket mid-stream RE-STREAMS ONLY THE TAIL on each reconnect (resume from the last flushed PUSH_CHUNK), so the
load makes forward progress across repeated drops, reaches `ready`, and the final sharded model reproduces the
un-sharded model's next-token argmax.

Harness is the style of tests/e2e/shard_parity.py: a tiny random-init model served from a local HTTP Range server
(standing in for the HF hub via MOREGPU_HF_BASE — no download), a Deno coordinator, and a CPU torch worker. The
twist: the flaky worker connects through a KILLABLE in-process TCP proxy. The proxy relays the worker's WebSocket
to the coordinator and, after every ~DROP_AFTER_BYTES of streamed weight bytes, tears the tunnel down (closes both
sockets) — dropping the worker's WebSocket WITHOUT killing the worker PROCESS. That is the whole point: the worker's
in-memory staging (PUSH[id] → a tempdir, since macOS has no /dev/shm) survives the socket drop because the process
lives on, and that staged partial is exactly what `push_begin(resume=true)` reports so the coordinator resumes.

What the real code does (read before writing this test):
  • apps/coordinator/server.ts:1315  logs `shard <id> stage <w>: resuming — <X>MB already staged, streaming the rest`
    ONLY when the reconnected worker reports sizes['model.safetensors'] > 0 (i.e. a genuine tail-resume).
  • apps/coordinator/server.ts:1340   retry loop calls loadStage(st, w, attempt>1); the partial staging is NOT
    unloaded between attempts (that partial is what enables resume); waitForWorker() polls for a same-id reconnect.
  • apps/coordinator/server.ts:622    streamStageSafetensors(..., resumeFrom) skips whole segments before the
    already-staged offset (no HF re-fetch, no re-push) and appends only the tail.
  • apps/worker/worker_torch.py:377    model_push_begin(resume=true) keeps PUSH[id]'s dir and returns per-file
    byte sizes; model_push_chunk appends ("ab"); on reconnect `welcome` clears models but KEEPS PUSH staging.

Asserts (all must hold):
  1. the proxy actually dropped the link >= MIN_DROPS times (the churn really happened);
  2. the coordinator log shows >= 2 "resuming — <X>MB already staged" lines with X STRICTLY GROWING (tail-resume
     from a growing offset — a broken resume would report 0 and log nothing, or a constant offset);
  3. the worker PROCESS stayed alive across every drop (poll() is None) — so its staging survived, which is what
     resume needs; killing the process would wipe it and would not test resume;
  4. shard_status reached `ready` despite the repeated drops;
  5. shard_forward's argmax equals the un-sharded golden argmax (final sharded model == un-sharded model).

  python3 tests/fault/churn_resume.py            # exits non-zero on any mismatch/failure

Safe for CI: torch CPU only, one tiny model built in a tempdir and deleted; all sockets are 127.0.0.1.
"""
import asyncio, json, os, re, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "tiny-churn-llama"                       # single-segment ref → matches the coordinator's HF_REPO_RE
INPUT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]             # all < vocab (256); fixed prompt → deterministic run
PUSH_CHUNK = 1 << 18                             # 256 KiB — the resume granularity we force on the coordinator
DROP_AFTER_BYTES = 2_000_000                     # relay ~2 MB of weight stream, then kill the tunnel (mid-stream)
MAX_DROPS = 3                                    # kill the tunnel this many times, then let it stream to completion
MIN_DROPS = 2                                    # ... but require at least this many real drops to have happened
PROCS = []


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gen_model(root):
    """Build a tiny random-init Llama as single-file safetensors — big enough (~8 MB fp32) that model.safetensors
    spans MANY 256 KiB chunks, so a mid-stream drop leaves a real partial and the resume offset can grow across
    drops. Llama (RoPE + causal-mask) also exercises the more complex shard forward path. No tokenizer files."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=256, intermediate_size=512,
                                     num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=8,
                                     max_position_embeddings=64))
    m.save_pretrained(os.path.join(root, MODEL), safe_serialization=True)
    size = os.path.getsize(os.path.join(root, MODEL, "model.safetensors"))
    print(f"built {MODEL}: model.safetensors = {size/1e6:.2f} MB (single file)", flush=True)
    return size


def golden_argmax(root):
    """Reference: load the model UN-sharded (fp32, CPU) and take the last-token argmax — same run, same versions."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(os.path.join(root, MODEL), dtype=torch.float32).eval()
    with torch.no_grad():
        logits = m(torch.tensor([INPUT_IDS])).logits[0, -1]
    return int(logits.argmax())


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


class ChurnProxy:
    """A killable in-process TCP tunnel: worker → (this proxy) → coordinator. Runs its own asyncio loop in a
    daemon thread. For each worker connection it opens an upstream socket to the coordinator and pumps bytes both
    ways. It COUNTS the coordinator→worker direction (that carries the push_chunk weight stream); after every
    DROP_AFTER_BYTES on a connection it "kills the tunnel" — closes both sockets — while the first MAX_DROPS
    drops remain. Killing the tunnel drops the worker's WebSocket but never touches the worker PROCESS, so the
    worker's staged partial survives and the next connection triggers a coordinator-side resume."""

    def __init__(self, up_host, up_port, drop_after_bytes, max_drops):
        self.up = (up_host, up_port)
        self.drop_after_bytes = drop_after_bytes
        self.max_drops = max_drops
        self.drops = 0            # how many times we've torn the tunnel down
        self.total_c2w = 0        # cumulative coordinator→worker bytes across ALL connections (tail-only proof)
        self.port = None
        self._loop = None
        self._server = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        self._ready.wait(10)
        return self.port

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._shutdown)

    def _shutdown(self):
        if self._server:
            self._server.close()
        self._loop.stop()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
            self._loop.run_forever()
        finally:
            # loop stopped → cancel any still-open tunnels and let them unwind, so no "Task destroyed while
            # pending" noise leaks past the RESULT line.
            pending = asyncio.all_tasks(self._loop)
            for t in pending:
                t.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    async def _serve(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        self._ready.set()

    async def _handle(self, creader, cwriter):
        try:
            ureader, uwriter = await asyncio.open_connection(*self.up)
        except Exception:
            try: cwriter.close()
            except Exception: pass
            return
        done = asyncio.Event()
        seen = {"c2w": 0}         # coordinator→worker bytes on THIS connection

        async def pump(src, dst, count):
            try:
                while True:
                    data = await src.read(65536)
                    if not data:
                        break
                    if count:
                        seen["c2w"] += len(data); self.total_c2w += len(data)
                    dst.write(data)
                    await dst.drain()
                    # Kill the tunnel mid-stream once this connection has relayed enough of the weight stream.
                    if count and seen["c2w"] >= self.drop_after_bytes and self.drops < self.max_drops:
                        self.drops += 1
                        break
            except Exception:
                pass
            finally:
                done.set()

        w2c = asyncio.ensure_future(pump(creader, uwriter, False))  # worker→coordinator (acks) — not counted
        c2w = asyncio.ensure_future(pump(ureader, cwriter, True))   # coordinator→worker (weights) — counted/dropped
        await done.wait()
        for wtr in (cwriter, uwriter):
            try: wtr.close()
            except Exception: pass
        for t in (w2c, c2w):
            t.cancel()


def api(port, path, method="GET", body=None, admin=None, timeout=90):
    data = json.dumps(body).encode() if body is not None else None
    h = {"content-type": "application/json"}
    if admin: h["authorization"] = "Bearer " + admin
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        return {"httperror": e.code, "body": e.read().decode()[:400]}


# Match the coordinator's resume line (server.ts:1315). The message text is uncolored; tolerate the em dash.
RESUME_RE = re.compile(r"resuming\s*[—-]\s*([0-9.]+)MB already staged")


def resume_offsets(logpath):
    """All staged-MB values the coordinator logged on resume, in order — should be strictly increasing."""
    try:
        return [float(x) for x in RESUME_RE.findall(open(logpath, "r", errors="replace").read())]
    except OSError:
        return []


def main():
    root = tempfile.mkdtemp(prefix="moregpu-churn-")
    proxy = None
    ok = True
    fails = []
    try:
        print("generating tiny llama…", flush=True)
        st_size = gen_model(root)
        golden = golden_argmax(root)

        # local HF-hub stand-in (Range server)
        hfport = free_port()
        RangeHandler.root = root
        httpd = ThreadingHTTPServer(("127.0.0.1", hfport), RangeHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        # coordinator — small PUSH_CHUNK makes the resume granularity fine (many flushed chunks); generous churn caps
        port = free_port(); cfg = os.path.join(root, "mg.json")
        env = dict(os.environ, PORT=str(port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_INSECURE="1",
                   MOREGPU_HF_BASE=f"http://127.0.0.1:{hfport}",
                   MOREGPU_PUSH_CHUNK=str(PUSH_CHUNK),
                   MOREGPU_SHARD_LOAD_DEADLINE_MS="600000",
                   MOREGPU_SHARD_STAGE_TRIES="12",
                   MOREGPU_SHARD_RECONNECT_WAIT_MS="60000")
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

        # killable tunnel in front of the coordinator's WS; the flaky worker connects to THIS, not the coordinator
        proxy = ChurnProxy("127.0.0.1", port, DROP_AFTER_BYTES, MAX_DROPS)
        proxy_port = proxy.start()
        print(f"churn proxy on 127.0.0.1:{proxy_port} → coordinator 127.0.0.1:{port} "
              f"(drop every {DROP_AFTER_BYTES/1e6:.1f} MB, up to {MAX_DROPS}×)", flush=True)

        # the flaky stage worker — behind the proxy, so a tunnel kill drops its socket but not its process
        worker = subprocess.Popen(
            ["python3", "apps/worker/worker_torch.py", "--server", f"ws://127.0.0.1:{proxy_port}/ws",
             "--token", JOIN, "--name", "flaky", "--cpu"], cwd=REPO, env=os.environ,
            stdout=open(os.path.join(root, "flaky.log"), "w"), stderr=subprocess.STDOUT)
        PROCS.append(worker)
        worker_pid = worker.pid
        for _ in range(120):                      # torch+transformers import at worker start is slow
            w = api(port, "/workers", admin=ADMIN)
            if isinstance(w, list) and any(x.get("id") == "flaky" for x in w): break
            time.sleep(0.5)

        # download-free single-stage shard onto the flaky worker (it holds the WHOLE model → max stream = max churn)
        sid = "churn-shard"
        r = api(port, "/model/shard", "POST",
                {"model": MODEL, "id": sid, "push": True, "async": True, "workers": ["flaky"]}, admin=ADMIN)
        if r.get("status") != "loading":
            fails.append(f"shard did not start: {r}"); ok = False
        else:
            st = None; s = {}
            for _ in range(240):                   # ~8 MB stream + several reconnect cycles
                s = api(port, f"/model/shard_status?id={sid}", admin=ADMIN); st = s.get("status")
                if st in ("ready", "error"): break
                time.sleep(2)

            resumes = resume_offsets(coord_log)
            worker_alive = worker.poll() is None

            # ---- assertions ----
            if proxy.drops < MIN_DROPS:
                fails.append(f"proxy dropped the link only {proxy.drops}× (< {MIN_DROPS}) — churn did not exercise resume")
            growing = len(resumes) >= 2 and all(b > a for a, b in zip(resumes, resumes[1:]))
            if not growing:
                fails.append(f"resume offsets not strictly growing across ≥2 drops: {resumes}")
            if not worker_alive:
                fails.append(f"worker PROCESS died (pid {worker_pid}) — its staging would be wiped; resume untested")
            if st != "ready":
                fails.append(f"shard never reached ready: status={st} detail={s}")
            si = None
            if st == "ready":
                fr = api(port, "/model/shard_forward", "POST",
                         {"id": sid, "input_ids": INPUT_IDS, "return_logits": True}, admin=ADMIN)
                si = fr.get("argmax")
                if si != golden:
                    fails.append(f"sharded argmax {si} != un-sharded golden {golden}: {fr}")
            # tail-only sanity: a broken resume that re-streamed from zero each attempt would move FAR more bytes
            if proxy.total_c2w > st_size * 3.0:
                fails.append(f"relayed {proxy.total_c2w/1e6:.1f} MB for an {st_size/1e6:.1f} MB model — "
                             f"not tail-only (re-streaming from zero?)")

            ok = not fails
            print(f"  drops={proxy.drops}  resume_offsets_MB={resumes}  worker_alive={worker_alive} (pid {worker_pid})")
            print(f"  status={st}  shard_argmax={si}  golden_argmax={golden}  relayed={proxy.total_c2w/1e6:.1f}MB", flush=True)
    finally:
        for p in PROCS:
            try: p.terminate()
            except Exception: pass
        if proxy: proxy.stop()
        time.sleep(1)
        shutil.rmtree(root, ignore_errors=True)

    for f in fails:
        print("  FAIL:", f)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
