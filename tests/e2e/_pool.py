"""Shared e2e harness: a Deno coordinator + N CPU torch workers on localhost (no network, no downloads)."""
import json, os, shutil, socket, subprocess, sys, tempfile, time, urllib.error, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def tail(path, n=30):
    try:
        return "".join(open(path).readlines()[-n:])
    except Exception:
        return "(no log)"


class Pool:
    def __init__(self, names, coord_env=None, worker_env=None):
        self.names, self.procs, self.wprocs = list(names), [], {}
        self.root = tempfile.mkdtemp(prefix="moregpu-e2e-")
        self.coord_env, self.worker_env = coord_env or {}, worker_env or {}

    def __enter__(self):
        self.port = free_port(); cfg = os.path.join(self.root, "mg.json")
        self.coord_log = os.path.join(self.root, "coord.log")
        env = dict(os.environ, PORT=str(self.port), MOREGPU_CONFIG=cfg, MOREGPU_BIND="127.0.0.1", MOREGPU_INSECURE="1",
                   MOREGPU_TRAIN_DIR=os.path.join(self.root, "train"), **self.coord_env)
        self.procs.append(subprocess.Popen(
            ["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write", "apps/coordinator/server.ts"],
            cwd=REPO, env=env, stdout=open(self.coord_log, "w"), stderr=subprocess.STDOUT))
        for _ in range(120):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("FAIL: coordinator did not come up\n" + tail(self.coord_log))
        conf = json.load(open(cfg)); self.join, self.admin = conf["joinToken"], conf["adminToken"]
        for n in self.names:
            self.start_worker(n)
        self.wait_workers(self.names)
        return self

    def start_worker(self, n):
        # exports are confined to MOREGPU_OUTPUT_DIR on the worker: default it to the pool's temp root (tests pick
        # export dirs under pool.root); a test's worker_env overrides it
        wenv = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MOREGPU_OUTPUT_DIR": self.root,
                **self.worker_env}
        p = subprocess.Popen(["python3", "apps/worker/worker_torch.py", "--server", f"ws://127.0.0.1:{self.port}/ws",
                              "--token", self.join, "--name", n, "--cpu"], cwd=REPO, env=wenv,
                             stdout=open(os.path.join(self.root, f"{n}.log"), "w"), stderr=subprocess.STDOUT)
        self.procs.append(p); self.wprocs[n] = p

    def kill_worker(self, n):
        self.wprocs[n].kill(); self.wprocs[n].wait()

    def wait_workers(self, names, timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            w = self.api("/workers")
            if isinstance(w, list) and set(names) <= {x.get("id") for x in w if "torch" in (x.get("label") or "")}:
                return
            time.sleep(0.5)
        raise SystemExit("FAIL: workers did not join\n" + tail(self.coord_log) +
                         "".join(f"\n--- {n} ---\n" + tail(os.path.join(self.root, f"{n}.log")) for n in names))

    def api(self, path, method="GET", body=None, timeout=600):
        data = json.dumps(body).encode() if body is not None else None
        h = {"content-type": "application/json", "authorization": "Bearer " + self.admin}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers=h)
        try:
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            return {"httperror": e.code, "body": e.read().decode()[:600]}

    def __exit__(self, *a):
        for p in self.procs:
            try:
                p.terminate(); p.wait(timeout=5)
            except Exception:
                p.kill()
        if not os.environ.get("MOREGPU_KEEP_E2E"):
            shutil.rmtree(self.root, ignore_errors=True)


class Checks:
    def __init__(self):
        self.n = self.ok = 0; self.failed = []

    def __call__(self, cond, msg):
        self.n += 1
        if cond:
            self.ok += 1; print(f"  [PASS] {msg}", flush=True)
        else:
            self.failed.append(msg); print(f"  [FAIL] {msg}", flush=True)

    def finish(self):
        print(f"RESULT: {self.ok}/{self.n} checks passed")
        sys.exit(0 if not self.failed else 1)
