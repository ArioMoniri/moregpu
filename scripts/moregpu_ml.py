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

    def train_load(self, model, rank=8, alpha=16, lr=1e-3, seed=0):
        # force: a fresh `finetune` replaces any prior session (the user asked to fine-tune THIS now).
        return self._req("/train/load", "POST",
                         {"model": model, "rank": rank, "alpha": alpha, "lr": lr, "seed": seed,
                          "targets": TARGETS, "force": True})

    def train_step(self, ids):
        return self._req("/train/step", "POST", {"input_ids": list(ids), "labels": list(ids)})

    def train_adapter(self):
        return self._req("/train/adapter", "POST", {})

    def train_generate(self, ids, max_new_tokens=40):
        return self._req("/train/generate", "POST",
                         {"input_ids": list(ids), "max_new_tokens": max_new_tokens}).get("tokens", [])

    def model_load(self, model, push=False):
        return self._req("/model/load", "POST", {"model": model, "push": push})

    def generate(self, ids, max_new_tokens=40):
        return self._req("/model/generate", "POST",
                         {"input_ids": list(ids), "max_new_tokens": max_new_tokens}).get("tokens", [])


# ----------------------------------------------------------------------------- helpers
def discover(args) -> Pool:
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
    if not any("torch" in (w.get("label") or "") for w in ws):
        sys.exit("no native torch worker in the fleet — fine-tuning/serving needs apps/worker/worker_torch.py "
                 "(WebGPU workers are inference-only kernels). Start a torch worker and retry.")
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

    info = pool.train_load(args.model, rank=args.rank, alpha=args.alpha, lr=args.lr, seed=0)
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
        info = pool.model_load(args.model, push=args.push)
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
        info = pool.model_load(args.model, push=args.push)
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


def _chat_ids(tok, line: str) -> list[int]:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": line}], add_generation_prompt=True)
    return tok(line)["input_ids"]


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
    f.set_defaults(fn=cmd_finetune)

    g = sub.add_parser("generate", help="run the model and print its continuation")
    g.add_argument("model")
    g.add_argument("--prompt")
    g.add_argument("--max-new", dest="max_new", type=int, default=40)
    g.add_argument("--from-training", action="store_true", help="use the live just-fine-tuned model")
    g.add_argument("--push", action="store_true", help="download-free: coordinator streams weights to the "
                   "worker (needs a model.safetensors on HF; RAM-staged where /dev/shm exists, else a transient disk dir)")
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("chat", help="interactive chat (one turn per line)")
    c.add_argument("model")
    c.add_argument("--prompt")
    c.add_argument("--max-new", dest="max_new", type=int, default=40)
    c.add_argument("--from-training", action="store_true", help="chat with the live just-fine-tuned model")
    c.add_argument("--push", action="store_true", help="download-free: coordinator streams weights to the worker")
    c.set_defaults(fn=cmd_chat)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
