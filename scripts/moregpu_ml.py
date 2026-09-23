#!/usr/bin/env python3
"""
moregpu_ml.py — no-code fine-tuning + inference on a MoreGPU pool.

The whole lifecycle from a text file to chatting with your fine-tuned model, in two commands and zero
glue code. Talks to a running coordinator's admin HTTP API; a native torch worker must be in the fleet
(apps/worker/worker_torch.py) — that is the worker that has autograd. Self-contained: no repo imports, so
it runs from a clone OR piped straight from GitHub (curl … | python3 - finetune …).

  finetune  MODEL [--data FILE] [--steps N] [--out FILE]   LoRA fine-tune on your text; saves the adapter
  generate  MODEL --prompt "…"  [--from-training]          run the model and print its continuation
  chat      MODEL [--from-training]                        interactive REPL (one turn per line)

  --from-training uses the LIVE just-fine-tuned model (base + adapter, still resident) instead of loading
  a fresh base model — so `finetune` then `chat --from-training` chats with what you just trained.

Connection (auto-discovered, override with flags/env):
  --url    coordinator base URL   (env MOREGPU_BASE, else MOREGPU_SERVER, else http://localhost:8787)
  --token  admin token            (env MOREGPU_ADMIN_TOKEN, else adminToken from ./.moregpu-server.json)

Example (one machine):
  moregpu serve                          # start a pool (prints the join + admin tokens)
  MOREGPU_SERVER=ws://localhost:8787/ws MOREGPU_TOKEN=<join> \
    python3 apps/worker/worker_torch.py  # a NATIVE TORCH worker (serve --worker gives a WebGPU slot, which
                                         # cannot train/serve models — fine-tuning needs autograd, i.e. torch)
  moregpu finetune gpt2 --data notes.txt # LoRA fine-tune on notes.txt, saves gpt2-lora.json
  moregpu chat gpt2 --from-training      # chat with the fine-tuned model
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

DEMO_TEXT = (
    "The compute pool learns. A gpu shares its cycles, a worker signs its work, and the coordinator "
    "seals every byte. Machines join, machines leave, and the model still trains. More gpus, more speed; "
    "fewer round trips, less waiting. The pool remembers what it computed, and every worker is one "
    "virtual gpu. "
) * 8
# Union of common LoRA target module names — the worker attaches to whichever EXIST on the model, so this
# is architecture-agnostic (not name-matched): c_attn = GPT-2's fused QKV Conv1D; q_proj/v_proj = the
# Llama/Qwen/SmolLM attention projections. Covers the families the pool can fine-tune today.
TARGETS = ["c_attn", "q_proj", "v_proj"]


# ----------------------------------------------------------------------------- tiny admin-API client
class Pool:
    def __init__(self, url: str, token: str, timeout: int = 900):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _req(self, path: str, method: str = "GET", body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header("authorization", "Bearer " + self.token)
        if data is not None:
            req.add_header("content-type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:  # server answered 4xx/5xx — surface its JSON error, not a bare 502
            try:
                return json.loads(e.read().decode())
            except Exception:
                raise RuntimeError(f"{method} {path} → HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            # Never reached the coordinator: connection refused / DNS / timeout / TLS-verify. Exit clean (this
            # is a no-code CLI — a raw traceback would be user-hostile). Note the self-signed-https limit.
            reason = getattr(e, "reason", e)
            sys.exit(f"cannot reach the coordinator at {self.url} ({reason}).\n"
                     f"  Is it running, and is the URL right? Set MOREGPU_SERVER or pass --url.\n"
                     f"  (A self-signed https:// coordinator isn't supported here — use a real-cert tunnel or ws://.)")

    def workers(self) -> list:
        w = self._req("/workers")
        return w if isinstance(w, list) else []

    def train_load(self, model, rank=8, alpha=16, lr=1e-3, seed=0, push=False):
        # force: a fresh `finetune` replaces any prior session (the user asked to fine-tune THIS now).
        # push: download-free — the coordinator streams the base so THIS worker never touches HF (a
        # no-download node can fine-tune). Needs a model.safetensors on HF.
        return self._req("/train/load", "POST",
                         {"model": model, "rank": rank, "alpha": alpha, "lr": lr, "seed": seed,
                          "targets": TARGETS, "force": True, "push": push})

    def train_step(self, ids):
        return self._req("/train/step", "POST", {"input_ids": list(ids), "labels": list(ids)})

    def train_adapter(self):
        return self._req("/train/adapter", "POST", {})

    def train_generate(self, ids, max_new_tokens=40):
        return self._req("/train/generate", "POST",
                         {"input_ids": list(ids), "max_new_tokens": max_new_tokens}).get("tokens", [])

    def model_load(self, model, push=False, fp16=False):
        # fp16 halves VRAM/bandwidth on a GPU worker (CUDA/MPS) — the difference between a multi-B model
        # fitting a single GPU or not; a CPU worker downgrades to fp32 (no half GEMM) and says so.
        return self._req("/model/load", "POST", {"model": model, "push": push, "fp16": fp16})

    def generate(self, ids, max_new_tokens=40):
        return self._req("/model/generate", "POST",
                         {"input_ids": list(ids), "max_new_tokens": max_new_tokens}).get("tokens", [])

    # ---- pipeline sharding (split a model across workers; optionally quantized) ----
    def shard_load(self, model, push=True, quant=None, split=None, layers=None, actq=None):
        body = {"model": model, "push": bool(push), "async": True}
        if quant:
            body["quant"] = quant
        if split:
            body["split"] = split
        if layers:
            body["layers"] = layers
        if actq:
            body["actq"] = actq
        return self._req("/model/shard", "POST", body)

    def shard_status(self, sid=None):
        return self._req("/model/shard_status" + (f"?id={sid}" if sid else ""))

    def shard_chat(self, prompt, sid=None, max_new_tokens=64):
        body = {"prompt": prompt, "max_new_tokens": max_new_tokens}
        if sid:
            body["id"] = sid
        return self._req("/model/shard_chat", "POST", body)

    def shard_unload(self, sid=None):
        return self._req("/model/shard_unload", "POST", {"id": sid} if sid else {})


# ----------------------------------------------------------------------------- helpers
def discover(args, need_torch: bool = True) -> Pool:
    url = args.url or os.environ.get("MOREGPU_BASE")
    if not url:
        srv = os.environ.get("MOREGPU_SERVER", "http://localhost:8787")
        url = srv.rstrip("/")[:-3] if srv.endswith("/ws") else srv
        url = url.replace("wss:", "https:").replace("ws:", "http:")
    token = args.token or os.environ.get("MOREGPU_ADMIN_TOKEN", "")
    if not token:  # read adminToken from the serve config on this machine
        cfg = os.environ.get("MOREGPU_CONFIG", "./.moregpu-server.json")
        try:
            token = json.load(open(cfg)).get("adminToken", "")
        except Exception:
            pass
    if not token:
        sys.exit("no admin token — set MOREGPU_ADMIN_TOKEN, pass --token, or run this next to the "
                 "coordinator's .moregpu-server.json (created by `moregpu serve`).")
    pool = Pool(url, token)
    ws = pool.workers()
    if not ws:
        sys.exit(f"no workers in the fleet at {url} — start one (`moregpu serve --worker`, or a torch worker).")
    if need_torch and not any("torch" in (w.get("label") or "") for w in ws):
        sys.exit("no native torch worker in the fleet — fine-tuning/serving needs apps/worker/worker_torch.py "
                 "(WebGPU workers are inference-only kernels). Start a torch worker and retry.")
    # sharding runs on any 'shard'-capable worker (WebGPU or torch); the coordinator's /model/shard is the authority
    # and returns a clear error if none is connected, so we don't hard-gate here (older coordinators omit caps).
    return pool


def load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
    except Exception:
        sys.exit("this command needs `transformers` (and `torch` on the worker). pip install transformers torch")
    tok = AutoTokenizer.from_pretrained(model)
    return tok


def read_corpus(path: str | None, tok) -> str:
    """Turn a dataset into one training string. .txt → verbatim; .jsonl/.json → {text} | {prompt,completion} |
    {messages:[…]} rendered with the chat template if the tokenizer has one. No file → a built-in demo text."""
    if not path:
        return DEMO_TEXT
    raw = open(path, encoding="utf-8").read()
    if path.endswith(".txt") or path.endswith(".md"):
        return raw
    rows = []
    if path.endswith(".json"):
        obj = json.loads(raw)
        rows = obj if isinstance(obj, list) else [obj]
    else:  # .jsonl (default for anything else)
        rows = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    out = []
    for r in rows:
        if isinstance(r, str):
            out.append(r)
        elif "text" in r:
            out.append(str(r["text"]))
        elif "messages" in r:
            if getattr(tok, "chat_template", None):
                out.append(tok.apply_chat_template(r["messages"], tokenize=False))
            else:  # no chat template on this base model → train on the raw turn TEXT, not the JSON envelope
                out.append("\n".join(str(m.get("content", "")) for m in r["messages"] if isinstance(m, dict)))
        elif "prompt" in r or "completion" in r:
            out.append(str(r.get("prompt", "")) + str(r.get("completion", "")))
        else:
            out.append(json.dumps(r))
    return "\n".join(out)


def windows(ids: list[int], t: int, n: int) -> list[list[int]]:
    out, i = [], 0
    if len(ids) < 2:
        ids = ids * 2
    while len(out) < n:
        if i + t + 1 > len(ids):
            i = 0
        out.append(ids[i:i + t] or ids[:t])
        i += t
    return out


def slope(ys: list[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2
    ym = sum(ys) / n
    num = sum((i - xm) * (y - ym) for i, y in enumerate(ys))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den if den else 0.0


# ----------------------------------------------------------------------------- commands
def cmd_finetune(args) -> int:
    if args.steps < 1:  # else windows() yields no batches, losses stays empty → IndexError below
        sys.exit("--steps must be >= 1")
    pool = discover(args)
    tok = load_tokenizer(args.model)
    text = read_corpus(args.data, tok)
    ids = tok(text)["input_ids"]
    batches = windows(ids, args.window, args.steps)
    src = args.data or "built-in demo text"
    print(f"== fine-tune {args.model} ==  data: {src}  ·  {len(ids)} tokens  ·  "
          f"{args.steps} steps × window {args.window}  ·  rank {args.rank}", flush=True)

    info = pool.train_load(args.model, rank=args.rank, alpha=args.alpha, lr=args.lr, seed=0, push=args.push)
    if not info.get("ok"):
        sys.exit(f"train/load failed: {info.get('error', info)}")
    print(f"   loaded on {info.get('worker', '?')} · trainable params "
          f"{info.get('trainable_params', '?'):,} · device {info.get('device', '?')}", flush=True)

    sample = args.sample or " ".join(text.split()[:4])  # first few words → a prompt to show the effect
    sids = tok(sample)["input_ids"]
    try:
        before = tok.decode(pool.train_generate(sids, 24)).strip()
    except Exception:
        before = None

    losses = []
    for i, b in enumerate(batches):
        r = pool.train_step(b)
        if "loss" not in r:
            sys.exit(f"train/step {i} failed: {r.get('error', r)}")
        losses.append(float(r["loss"]))
        if i < 3 or (i + 1) % max(1, args.steps // 8) == 0 or i == len(batches) - 1:
            print(f"   step {r.get('step', i + 1):>4}  loss {r['loss']:.4f}", flush=True)

    ad = pool.train_adapter()
    tensors = ad.get("tensors", {})
    out_path = args.out or (re.sub(r"[^A-Za-z0-9._-]", "_", args.model.split("/")[-1]) + "-lora.json")
    json.dump({"format": "moregpu-lora-v1", "model": args.model, "step": ad.get("step"),
               "rank": args.rank, "alpha": args.alpha, "targets": TARGETS, "tensors": tensors},
              open(out_path, "w"))

    print("\n== result ==", flush=True)
    print(f"   loss {losses[0]:.4f} → {losses[-1]:.4f}  (min {min(losses):.4f}, slope {slope(losses):+.4f})")
    print(f"   adapter: {len(tensors)} tensors saved → {out_path}")
    if before is not None:
        after = tok.decode(pool.train_generate(sids, 24)).strip()
        print(f"\n   prompt : {sample!r}")
        print(f"   before : {before!r}")
        print(f"   after  : {after!r}")
        print("\n   ↳ chat with it now:  moregpu chat " + args.model + " --from-training")
    learned = losses[-1] < losses[0] - 0.02 and slope(losses) < 0
    print(f"\n{'done — the model learned your text.' if learned else 'done (loss did not clearly drop — try more --steps or a higher --lr).'}")
    return 0 if learned else 1


def _decode_new(tok, prompt_ids, new_ids) -> str:
    return tok.decode(new_ids, skip_special_tokens=True)


def cmd_generate(args) -> int:
    pool = discover(args)
    tok = load_tokenizer(args.model)
    if not args.from_training:
        info = pool.model_load(args.model, push=args.push, fp16=args.fp16)
        if not info.get("ok"):
            sys.exit(f"model/load failed: {info.get('error', info)}")
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    ids = tok(prompt)["input_ids"]
    gen = pool.train_generate(ids, args.max_new) if args.from_training else pool.generate(ids, args.max_new)
    print(prompt + _decode_new(tok, ids, gen))
    return 0


def cmd_chat(args) -> int:
    pool = discover(args)
    tok = load_tokenizer(args.model)
    if not args.from_training:
        info = pool.model_load(args.model, push=args.push, fp16=args.fp16)
        if not info.get("ok"):
            sys.exit(f"model/load failed: {info.get('error', info)}")
    where = "fine-tuned (live)" if args.from_training else "base"
    if args.prompt is not None or not sys.stdin.isatty():  # one-shot (piped or --prompt)
        for line in ([args.prompt] if args.prompt is not None else sys.stdin.read().splitlines()):
            if not line.strip():
                continue
            ids = _chat_ids(tok, line)
            gen = pool.train_generate(ids, args.max_new) if args.from_training else pool.generate(ids, args.max_new)
            print(_decode_new(tok, ids, gen).strip())
        return 0
    print(f"== chat with {args.model} [{where}] ==  (empty line or Ctrl-D to quit)")
    while True:
        try:
            line = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        ids = _chat_ids(tok, line)
        gen = pool.train_generate(ids, args.max_new) if args.from_training else pool.generate(ids, args.max_new)
        print("bot › " + _decode_new(tok, ids, gen).strip())
    return 0


def cmd_shard(args) -> int:
    import time
    pool = discover(args, need_torch=False)  # sharding runs on WebGPU workers too (not just torch)
    if args.unload:
        print(json.dumps(pool.shard_unload(None if args.model in ("", "-") else args.model), indent=2))
        return 0
    # wq8/wq4 are push (coordinator quantizes + streams); int8/nf4/auto are non-push (worker self-loads via bnb).
    push = not args.no_push
    if args.quant in ("int8", "nf4", "auto"):
        push = False
    split = [int(x) for x in args.split.split(",")] if args.split else None
    print(f"sharding {args.model}" + (f" · quant={args.quant}" if args.quant else "") + (f" · actq={args.actq}" if args.actq else "") + (" · download-free" if push else "") + " …")
    r = pool.shard_load(args.model, push=push, quant=args.quant, split=split, layers=args.layers, actq=args.actq)
    sid = r.get("id", args.model)
    if r.get("error") and not r.get("stages"):
        print("shard failed:", r.get("error"))
        return 1
    for _ in range(900):  # async load → poll until ready
        st = pool.shard_status(sid)
        status = st.get("status")
        if status in ("ready", "error"):
            r = st
            break
        if status == "unknown" and st.get("error"):
            r = st
            break
        time.sleep(1)
    if r.get("status") == "error" or r.get("error"):
        print("shard failed:", r.get("error", "unknown error"))
        return 1
    stages = r.get("stages", [])
    print(f"sharded '{sid}' into {len(stages)} stage(s):")
    for i, s in enumerate(stages):
        role = "first+last" if s.get("first") and s.get("last") else "first" if s.get("first") else "last" if s.get("last") else "middle"
        ph = s.get("params_held")
        print(f"  stage {i}: worker {s.get('worker')}  layers [{s.get('start')},{s.get('end')})  {role}" + (f"  · {ph:,} params" if ph else ""))
    if args.prompt:
        print(f"\nchat> {args.prompt}")
        cr = pool.shard_chat(args.prompt, sid=sid, max_new_tokens=args.max_new)
        if cr.get("error"):
            print("chat failed:", cr["error"])
            return 1
        print(cr.get("text", ""))
        print(f"[{cr.get('n')} tokens · {cr.get('ms')}ms · workers {cr.get('workers')}]")
    return 0


def _chat_ids(tok, line: str) -> list[int]:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": line}], add_generation_prompt=True)
    return tok(line)["input_ids"]


# ----------------------------------------------------------------------------- training sessions / vision (v0.7)
def _die_on_error(r: dict) -> dict:
    if isinstance(r, dict) and r.get("error"):
        sys.exit(f"[moregpu] {r['error']}")
    return r


def _csv(v):
    return [x for x in (v or "").split(",") if x] or None


def _manifest_len(path: str | None) -> int | None:
    if path and os.path.exists(path):
        return sum(1 for line in open(path) if line.strip())
    return None


def _drive(pool: Pool, sid: str, rounds: int | None, quiet: bool = False) -> dict:
    """Run one round per request until the stopping rule (or `rounds`), printing progress."""
    last, i = None, 0
    while rounds is None or i < rounds:
        r = _die_on_error(pool._req(f"/train/sessions/{sid}/round", "POST", {"rounds": 1}))
        last = r["rounds"][-1] if r.get("rounds") else None
        i += 1
        if last and not quiet:
            mon = last.get("monitors") or {}
            extra = f" rankme {mon['rankme']:.1f} std {mon['std_mean']:.3f}" if "rankme" in mon else ""
            ev = f" eval {json.dumps(last['eval'])}" if last.get("eval") else ""
            print(f"round {last['round']:4d}  loss {last['avg_last_loss']:.4f}  lr {last['lr']:.2e}  samples {last['samples_seen']}"
                  f"  {last['wall_s']:.1f}s{extra}{ev}", flush=True)
            for a in last.get("alarms") or []:
                print(f"  ALARM: {a}", flush=True)
        if not last or last.get("done"):
            break
    return last or {}


def _session_common(a) -> dict:
    body = {"manifest_len": a.manifest_len, "batch": a.batch, "inner_steps": a.inner_steps, "lr": a.lr, "amp": a.amp,
            "seed": a.seed, "outer_lr": a.outer_lr, "outer_momentum": a.outer_momentum, "alloc": a.alloc,
            "sync_dtype": a.sync_dtype}
    if a.target_samples:
        body["target_samples"] = a.target_samples
    if a.max_rounds:
        body["max_rounds"] = a.max_rounds
    if a.cosine:
        body["lr_schedule"] = {"kind": "cosine", "warmup_frac": a.warmup, "min_lr": a.min_lr}
    if a.checkpoint_every:
        body["checkpoint_every"] = a.checkpoint_every
    if _csv(a.workers):
        body["workers"] = _csv(a.workers)
    if a.id:
        body["id"] = a.id
    return body


def cmd_train(a) -> int:
    pool = discover(a)
    if a.kind in ("jepa", "segment", "classify"):
        data = None
        if a.data:
            data = {"manifest": a.data if "://" in a.data else f"file://{a.data}", "spec": {"size": [int(x) for x in a.size.split(",")],
                                                                                         "channels": a.channels}}
            if a.sha256:
                data["sha256"] = a.sha256
        a.manifest_len = a.manifest_len or _manifest_len(a.data) or (a.synthetic_n if a.synthetic else None)
        if not a.manifest_len:
            sys.exit("[moregpu] --manifest-len is required (number of lines in the refs manifest on the workers)")
        synthetic = {"kind": {"jepa_3d": "3d", "ijepa_2d": "2d"}.get(a.task, "2p5d"), "n": a.synthetic_n,
                     "size": [int(x) for x in a.size.split(",")], "channels": a.channels, "seed": 0} if a.synthetic else None
        if a.kind == "jepa":
            cfg = {"model": a.model, "patch": int(a.patch) if a.patch.isdigit() else [int(x) for x in a.patch.split(",")],
                   "n_targets": a.n_targets, "ema": [a.ema, 1.0], "grad_checkpointing": a.grad_checkpointing}
            if a.target_samples:
                cfg["total_steps"] = max(1, a.target_samples // a.batch)
            task = a.task
        else:
            enc = {"init": "export", "path": a.encoder} if a.encoder else {"init": "random", "model": a.model, "patch": int(a.patch)}
            cfg = {"kind": a.task.split("_")[-1] if a.task in ("seg_2d", "seg_3d") else "2p5d", "num_classes": a.num_classes,
                   "encoder": enc, "mode": a.mode}
            task = a.kind
        if data:
            cfg["data"] = data
        if synthetic:
            cfg["synthetic"] = synthetic
        body = {"task": task, "cfg": cfg, **_session_common(a)}
        r = _die_on_error(pool._req("/train/sessions", "POST", body))
        sid = r["session"]
        print(f"[moregpu] session {sid}: {task} on {', '.join(r['workers'])} · {r['params']:,} synced params", flush=True)
        if a.background:
            _die_on_error(pool._req(f"/train/sessions/{sid}/run", "POST", {}))
            print(f"[moregpu] running in the background — `moregpu train status {sid}`")
            return 0
        last = _drive(pool, sid, a.rounds)
        if a.export:
            ex = _die_on_error(pool._req(f"/train/sessions/{sid}/export", "POST", {"fmt": a.export_format, "path": a.export}))
            print(f"[moregpu] exported → {ex.get('weights') or ex.get('path')} (on worker)")
        if a.telemetry_out:
            recs = pool._req(f"/train/sessions/{sid}/telemetry?n=2000").get("records", [])
            with open(a.telemetry_out, "w") as f:
                f.writelines(json.dumps(x) + "\n" for x in recs)
            print(f"[moregpu] {len(recs)} telemetry records → {a.telemetry_out}")
        print(json.dumps({"session": sid, "round": last.get("round"), "samples_seen": last.get("samples_seen"),
                          "loss": last.get("avg_last_loss")}))
        return 0
    if a.kind == "status":
        r = pool._req(f"/train/sessions/{a.sid}") if a.sid else pool._req("/train/sessions")
        print(json.dumps(r, indent=2)); return 0
    if a.kind == "round":
        _drive(pool, a.sid, a.rounds); return 0
    if a.kind in ("stop", "checkpoint"):
        print(json.dumps(_die_on_error(pool._req(f"/train/sessions/{a.sid}/{a.kind}", "POST", {})))); return 0
    if a.kind == "resume":
        print(json.dumps(_die_on_error(pool._req("/train/sessions/resume", "POST", {"id": a.sid})), indent=2)); return 0
    if a.kind == "rm":
        print(json.dumps(pool._req(f"/train/sessions/{a.sid}", "DELETE"))); return 0
    if a.kind == "export":
        print(json.dumps(_die_on_error(pool._req(f"/train/sessions/{a.sid}/export", "POST", {"fmt": a.export_format, "path": a.export})), indent=2))
        return 0
    if a.kind == "telemetry":
        recs = pool._req(f"/train/sessions/{a.sid}/telemetry?n=2000").get("records", [])
        out = open(a.telemetry_out, "w") if a.telemetry_out else sys.stdout
        out.writelines(json.dumps(x) + "\n" for x in recs)
        return 0
    sys.exit(f"unknown train command {a.kind}")


def cmd_vision(a) -> int:
    pool = discover(a)
    if a.kind == "load":
        body = {"id": a.id}
        if a.spec:
            body.update(spec=json.load(open(a.spec)), task=a.task, num_classes=a.num_classes, kind=a.vkind)
        else:
            body["export"] = a.export
        if _csv(a.workers):
            body["workers"] = _csv(a.workers)
        print(json.dumps(_die_on_error(pool._req("/vision/load", "POST", body)), indent=2)); return 0
    if a.kind == "batch":
        items = [{"ref": {"uri": v if "://" in v else f"file://{v}"}, "out": f"pred_{os.path.splitext(os.path.basename(v))[0]}"} for v in a.volumes]
        for i, m in enumerate(a.masks or []):
            items[i]["mask"] = {"uri": m if "://" in m else f"file://{m}"}
        r = _die_on_error(pool._req("/vision/batch", "POST", {"id": a.id, "items": items, "split": a.split, "tta": a.tta}))
        import time
        while True:
            j = pool._req(f"/vision/jobs/{r['job']}")
            print(f"\r[moregpu] {j['done']}/{j['items']} done · {j['failed']} failed · retries {j['retries']} · stolen {j['stolen']}", end="", flush=True)
            if j["status"] not in ("running", "pending"):
                print(); break
            time.sleep(1)
        print(json.dumps(pool._req(f"/vision/jobs/{r['job']}?results=1"), indent=2)); return 0
    if a.kind == "jobs":
        print(json.dumps(pool._req(f"/vision/jobs/{a.id}?results=1" if a.id else "/vision/jobs"), indent=2)); return 0
    if a.kind == "unload":
        print(json.dumps(pool._req("/vision/unload", "POST", {"id": a.id}))); return 0
    if a.kind == "models":
        print(json.dumps(pool._req("/vision/models"), indent=2)); return 0
    sys.exit(f"unknown vision command {a.kind}")


def cmd_models(a) -> int:
    pool = discover(a)
    if a.kind == "describe":
        print(json.dumps(pool._req("/vision/capabilities"), indent=2)); return 0
    if a.kind == "lower":
        print(json.dumps(_die_on_error(pool._req("/vision/lower", "POST", {"id": a.id, "target": a.target})), indent=2)); return 0
    sys.exit(f"unknown models command {a.kind}")


def cmd_net(a) -> int:
    pool = discover(a, need_torch=False)
    print(json.dumps(pool._req(f"/net?pings={a.pings}&sustained_mb={a.sustained_mb}"), indent=2)); return 0



def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="moregpu_ml", description="no-code fine-tuning + inference on a MoreGPU pool")
    p.add_argument("--url"); p.add_argument("--token")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("finetune", help="LoRA fine-tune a model on your text")
    f.add_argument("model")
    f.add_argument("--data", help="a .txt/.md file, or .jsonl/.json of {text}|{prompt,completion}|{messages}")
    f.add_argument("--steps", type=int, default=40)
    f.add_argument("--window", type=int, default=32)
    f.add_argument("--rank", type=int, default=8)
    f.add_argument("--alpha", type=float, default=16)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--out", help="where to save the adapter (default <model>-lora.json)")
    f.add_argument("--sample", help="a prompt to show before/after the fine-tune")
    f.add_argument("--push", action="store_true", help="download-free: the coordinator streams the base to the "
                   "worker, so a no-download node (e.g. a laptop worker) fine-tunes a model it never fetched")
    f.set_defaults(fn=cmd_finetune)

    g = sub.add_parser("generate", help="run the model and print its continuation")
    g.add_argument("model")
    g.add_argument("--prompt")
    g.add_argument("--max-new", dest="max_new", type=int, default=40)
    g.add_argument("--from-training", action="store_true", help="use the live just-fine-tuned model")
    g.add_argument("--push", action="store_true", help="download-free: coordinator streams weights to the "
                   "worker (needs a model.safetensors on HF; RAM-staged where /dev/shm exists, else a transient disk dir)")
    g.add_argument("--fp16", action="store_true", help="load in fp16 on a GPU worker — halves VRAM so a bigger "
                   "model fits (CPU workers downgrade to fp32)")
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("chat", help="interactive chat (one turn per line)")
    c.add_argument("model")
    c.add_argument("--prompt")
    c.add_argument("--max-new", dest="max_new", type=int, default=40)
    c.add_argument("--from-training", action="store_true", help="chat with the live just-fine-tuned model")
    c.add_argument("--push", action="store_true", help="download-free: coordinator streams weights to the worker")
    c.add_argument("--fp16", action="store_true", help="load in fp16 on a GPU worker — halves VRAM (CPU downgrades to fp32)")
    c.set_defaults(fn=cmd_chat)

    sh = sub.add_parser("shard", help="pipeline-shard a model across workers (optionally quantized), then optionally chat it")
    sh.add_argument("model", help="HF model id (Llama/Qwen2-family for a WebGPU worker); or the shard id when --unload")
    sh.add_argument("--quant", choices=["wq8", "wq4", "int8", "nf4", "auto"],
                    help="wq8/wq4: the coordinator quantizes each stage to int8/int4 + scale and streams it (any "
                         "WebGPU worker, download-free); int8/nf4/auto: bitsandbytes on a CUDA torch worker (non-push)")
    sh.add_argument("--actq", choices=["int8"], help="int8 activation wire between MIDDLE stages (~4x smaller "
                    "per-hop; lossy, kept fp32 into the last stage) — cuts the recurring per-token bandwidth")
    sh.add_argument("--split", help="explicit per-stage layer counts for a heterogeneous fleet, e.g. 12,24")
    sh.add_argument("--layers", type=int, help="override the model's layer count (if config can't be read)")
    sh.add_argument("--prompt", help="after sharding, run this prompt through the pipeline (text in → text out)")
    sh.add_argument("--max-new", dest="max_new", type=int, default=64)
    sh.add_argument("--no-push", action="store_true", help="disable download-free streaming (the worker self-loads from HF)")
    sh.add_argument("--unload", action="store_true", help="unload the sharded model (pass its id/model) instead of loading")
    sh.set_defaults(fn=cmd_shard)

    t = sub.add_parser("train", help="training sessions: JEPA pretraining, segment/classify fine-tuning (DiLoCo across torch workers)")
    t.add_argument("kind", choices=["jepa", "segment", "classify", "status", "round", "stop", "checkpoint", "resume", "rm", "export", "telemetry"])
    t.add_argument("sid", nargs="?", help="session id (status/round/stop/checkpoint/resume/rm/export/telemetry)")
    t.add_argument("--task", default="jepa_2p5d", help="jepa_2p5d | ijepa_2d | jepa_3d (jepa); seg_2d | seg_2p5d | seg_3d (segment)")
    t.add_argument("--data", help="refs.jsonl manifest path/URI ON THE WORKERS (file:// under MOREGPU_DATA_ROOTS, or pushed://id)")
    t.add_argument("--sha256"); t.add_argument("--manifest-len", dest="manifest_len", type=int)
    t.add_argument("--synthetic", action="store_true", help="synthetic structured data (smoke test, no data needed)")
    t.add_argument("--synthetic-n", dest="synthetic_n", type=int, default=64)
    t.add_argument("--size", default="224,224"); t.add_argument("--channels", type=int, default=3)
    t.add_argument("--model", default="tiny"); t.add_argument("--patch", default="16")
    t.add_argument("--n-targets", dest="n_targets", type=int, default=4); t.add_argument("--ema", type=float, default=0.996)
    t.add_argument("--grad-checkpointing", dest="grad_checkpointing", action="store_true")
    t.add_argument("--encoder", help="(segment/classify) JEPA encoder export dir on the workers")
    t.add_argument("--num-classes", dest="num_classes", type=int, default=3); t.add_argument("--mode", default="full", choices=["full", "frozen", "lora"])
    t.add_argument("--workers", help="comma-separated worker ids (default: all torch workers)")
    t.add_argument("--rounds", type=int, help="stop after N rounds (default: until the stopping rule)")
    t.add_argument("--inner-steps", dest="inner_steps", type=int, default=50)
    t.add_argument("--batch", type=int, default=32); t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--target-samples", dest="target_samples", type=int); t.add_argument("--max-rounds", dest="max_rounds", type=int)
    t.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "fp32"]); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--outer-lr", dest="outer_lr", type=float, default=0.7); t.add_argument("--outer-momentum", dest="outer_momentum", type=float, default=0.9)
    t.add_argument("--alloc", default="proportional", choices=["fixed", "proportional"])
    t.add_argument("--sync-dtype", dest="sync_dtype", default="f32", choices=["f32", "bf16", "fp16", "int8delta"])
    t.add_argument("--cosine", action="store_true"); t.add_argument("--warmup", type=float, default=0.05); t.add_argument("--min-lr", dest="min_lr", type=float, default=1e-6)
    t.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=0)
    t.add_argument("--id"); t.add_argument("--background", action="store_true")
    t.add_argument("--export", help="export directory ON THE WORKER after training (inside its MOREGPU_OUTPUT_DIR; relative paths resolve inside it)"); t.add_argument("--export-format", dest="export_format", default="safetensors")
    t.add_argument("--telemetry-out", dest="telemetry_out", help="write the session's telemetry JSONL here")
    t.set_defaults(fn=cmd_train)

    v = sub.add_parser("vision", help="vision inference on the pool: load exports/published models, batch-segment volumes")
    v.add_argument("kind", choices=["load", "batch", "jobs", "unload", "models"])
    v.add_argument("id", nargs="?", help="model id (load/batch/unload) or job id (jobs)")
    v.add_argument("--export", help="MoreGPU export dir on the workers (inside MOREGPU_OUTPUT_DIR or MOREGPU_MODEL_ROOTS)"); v.add_argument("--spec", help="published-model spec JSON (docs/MODELS.md)")
    v.add_argument("--task", default="segment"); v.add_argument("--num-classes", dest="num_classes", type=int); v.add_argument("--kind", dest="vkind", default="3d")
    v.add_argument("--workers"); v.add_argument("--volumes", nargs="*", default=[]); v.add_argument("--masks", nargs="*", default=[])
    v.add_argument("--split", default="cases", choices=["cases", "tiles"]); v.add_argument("--tta", default="flip", choices=["none", "flip"])
    v.set_defaults(fn=cmd_vision)

    mo = sub.add_parser("models", help="published-model capabilities per worker; lower a loaded model for WebGPU")
    mo.add_argument("kind", choices=["describe", "lower"]); mo.add_argument("id", nargs="?"); mo.add_argument("--target", default="wgsl")
    mo.set_defaults(fn=cmd_models)

    ne = sub.add_parser("net", help="per-worker RTT percentiles + (sustained) bandwidth")
    ne.add_argument("--pings", type=int, default=20); ne.add_argument("--sustained-mb", dest="sustained_mb", type=int, default=0)
    ne.set_defaults(fn=cmd_net)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
