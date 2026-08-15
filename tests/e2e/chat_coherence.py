#!/usr/bin/env python3
"""
chat_coherence.py — download-free SHARDED CHAT EXACT-MATCH regression (no network, no model download, CPU-only).

shard_parity.py proves a sharded forward reproduces the un-sharded next-token argmax; this test carries that all the
way through the /model/shard_chat text↔text path — the one that needs a real TOKENIZER on the first stage. It builds
tiny random-init models (GPT-2 = Conv1D, and a Llama-family = RMSNorm/RoPE) EACH BUNDLED WITH a tiny byte-level-BPE
tokenizer whose vocab is sized to the model (len(tokenizer) == vocab_size, so every prompt id and every argmax id
round-trips), serves them from a local HTTP Range server standing in for the HF hub (MOREGPU_HF_BASE), starts a
coordinator + three CPU torch workers, and download-free shards each model across 2–3 stages. The coordinator streams
the tokenizer files (tokenizer.json / tokenizer_config.json / special_tokens_map.json) to the FIRST stage exactly as
in production, so shard_tok / shard_detok work over the wire.

For each model it then asserts the SHARDED greedy chat equals the UN-SHARDED greedy decode of the same prompt, two ways:

  • TEXT  — POST /model/shard_chat {prompt} (first stage tokenizes → shardPipe runs the cached greedy loop stage→stage,
            stopping on the tokenizer's eos → first stage detokenizes with skip_special_tokens) MUST equal the golden
            text: tokenize the SAME way (raw prompt — these tokenizers carry no chat template, so shard_tok uses the
            raw text), run the identical greedy argmax loop un-sharded with the same eos-stop, decode the same way.
  • TOKENS — POST /model/shard_generate {input_ids} (coordinator-side cached greedy loop, one NDJSON line/token) MUST
            equal, token-for-token, the un-sharded greedy argmax token stream over the same ids (no eos-stop, fixed N).

The golden is computed IN THIS RUN on the SAME transformers version (numerics drift across versions). do_sample is off
everywhere; greedy argmax is deterministic, and the per-stage incremental KV cache is proven token-exact vs the
un-sharded decode by the sibling kv_cache_parity.py — so the sharded chat is required to be character-identical here.

  python3 tests/e2e/chat_coherence.py            # exits non-zero on any mismatch/failure

Safe for CI: torch CPU only, models built in a tempdir and deleted; all sockets are 127.0.0.1.
"""
import json, os, subprocess, sys, tempfile, threading, time, urllib.request, urllib.error, shutil, socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # workers fork after tokenizer use → silence the fork warning

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPT = "the quick brown fox jumps over"   # fixed prompt → the whole run is deterministic
CHAT_NEW = 8      # max_new_tokens for the text path (eos may stop it earlier — golden mirrors that exactly)
GEN_NEW = 12      # fixed token count for the token-stream path (no eos-stop, both sides run all N)
PROCS = []


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def build_tokenizer():
    """A tiny byte-level BPE tokenizer built from scratch (no hub). Byte-level ⇒ it can encode ANY text without an
    <unk> and DECODE any id in [0, vocab) — so a random model's argmax always maps back to bytes. `<|endoftext|>` is
    added AFTER training so it lands at the TOP id (a random model rarely emits it ⇒ non-trivial chat output), and is
    reused as eos/bos/pad/unk. Returned with its exact length so each model's vocab_size is set to match it."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a tiny sharded model coherence test",
        "more gpu pools model weights across a fleet of cpu workers",
        "greedy decoding is deterministic when do sample is off",
        "pipeline sharding pipes hidden states from stage to stage",
    ] * 40
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus, trainer)   # 256 byte ids + merges, no special tokens yet
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", bos_token="<|endoftext|>",
                                   unk_token="<|endoftext|>", pad_token="<|endoftext|>")  # eos = top id
    return fast, len(fast)


def gen_models(root):
    """Build tiny random-init GPT-2 and Llama models AS SINGLE-FILE safetensors, each with the SAME tiny tokenizer
    saved alongside (config + weights + tokenizer.json/…). vocab_size == len(tokenizer). Geometry mirrors the
    kv_cache_parity.py models (proven token-exact cached==uncached), sized so prompt + decode fit the 64-pos window."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    tok, V = build_tokenizer()
    specs = {}
    g = GPT2LMHeadModel(GPT2Config(vocab_size=V, n_positions=64, n_embd=64, n_layer=4, n_head=4))
    dg = os.path.join(root, "tiny-gpt2"); g.save_pretrained(dg, safe_serialization=True); tok.save_pretrained(dg)
    specs["tiny-gpt2"] = ("gpt2", ["w1", "w2", "w3"])   # 4 layers → 3 stages (2,1,1): exercises a 3-hop pipe
    L = LlamaForCausalLM(LlamaConfig(vocab_size=V, hidden_size=64, intermediate_size=128,
                                     num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                     max_position_embeddings=64))
    dl = os.path.join(root, "tiny-llama"); L.save_pretrained(dl, safe_serialization=True); tok.save_pretrained(dl)
    specs["tiny-llama"] = ("llama", ["w1", "w2"])       # 4 layers → 2 stages (2,2)
    return specs, V


def _greedy(m, ids, n, eos):
    """The un-sharded reference greedy loop — the exact semantics the sharded path runs: take last-token argmax
    (do_sample off), append it, and when eos is not None stop AFTER appending the eos token (mirrors shard_chat's
    'push tok then break on eos'). Returns the list of NEW token ids. Stateless full re-run == the sharded cached
    decode token-for-token (proven by kv_cache_parity.py) at these fp32 tiny sizes."""
    import torch
    seq = list(ids); new = []
    with torch.no_grad():
        for _ in range(n):
            logits = m(torch.tensor([seq])).logits[0, -1]
            t = int(logits.argmax()); seq.append(t); new.append(t)
            if eos is not None and t == eos:
                break
    return new


def golden(root, model):
    """Load the model un-sharded (fp32, CPU) WITH its tokenizer and produce the two goldens the sharded path must
    match: (chat_text, chat_new_count) via the eos-stopping greedy loop + skip-special decode (mirrors shard_chat),
    and (input_ids, gen_tokens) — the fixed-length greedy token stream (mirrors shard_generate). No chat template on
    these tokenizers ⇒ the prompt is tokenized raw, exactly as shard_tok does."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = os.path.join(root, model)
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    tk = AutoTokenizer.from_pretrained(d)
    assert getattr(tk, "chat_template", None) is None, "test assumes no chat template (shard_tok tokenizes raw)"
    ids = tk(PROMPT, return_tensors=None)["input_ids"]
    eos = tk.eos_token_id
    chat_new = _greedy(m, ids, CHAT_NEW, eos)                       # eos-stopping (matches shard_chat)
    chat_text = tk.decode(chat_new, skip_special_tokens=True)       # matches shard_detok
    gen_tokens = _greedy(m, ids, GEN_NEW, None)                     # fixed N, no eos-stop (matches shard_generate)
    return {"ids": ids, "eos": eos, "chat_text": chat_text, "chat_n": len(chat_new), "gen_tokens": gen_tokens}


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


def api_ndjson(port, path, body, admin, timeout=120):
    """POST and read an NDJSON stream (one JSON object per line) — for /model/shard_generate."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST",
                                 headers={"content-type": "application/json", "authorization": "Bearer " + admin})
    lines = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                raw = raw.strip()
                if raw:
                    lines.append(json.loads(raw))
    except urllib.error.HTTPError as e:
        return [{"httperror": e.code, "body": e.read().decode()[:400]}]
    return lines


def main():
    root = tempfile.mkdtemp(prefix="moregpu-chat-")
    ok = True
    try:
        print("generating tiny models (gpt2 + llama) WITH a tiny byte-level tokenizer…", flush=True)
        specs, V = gen_models(root)
        print(f"tokenizer/model vocab = {V}", flush=True)

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

        # three CPU torch workers → gpt2 shards across all 3, llama across 2 (covers the "2–3 workers" split)
        for n in ("w1", "w2", "w3"):
            PROCS.append(subprocess.Popen(["python3", "apps/worker/worker_torch.py", "--server",
                                           f"ws://127.0.0.1:{port}/ws", "--token", JOIN, "--name", n, "--cpu"],
                                          cwd=REPO, env=os.environ,
                                          stdout=open(os.path.join(root, f"{n}.log"), "w"), stderr=subprocess.STDOUT))
        for _ in range(120):
            w = api(port, "/workers", admin=ADMIN)
            if isinstance(w, list) and len(w) >= 3: break
            time.sleep(0.5)

        for model, (arch, wkrs) in specs.items():
            g = golden(root, model)
            sid = f"chat-{model}"
            r = api(port, "/model/shard", "POST", {"model": model, "id": sid, "push": True, "async": True,
                                                   "workers": wkrs}, admin=ADMIN)
            if r.get("status") != "loading":
                print(f"  FAIL {model}: shard did not start: {r}"); ok = False; continue
            st = s = None
            for _ in range(120):
                s = api(port, f"/model/shard_status?id={sid}", admin=ADMIN); st = s.get("status")
                if st in ("ready", "error"): break
                time.sleep(2)
            if st != "ready":
                print(f"  FAIL {model}: shard not ready: {s}"); ok = False; continue
            stages = "→".join(f"{x['worker']}[{x['start']}-{x['end']})" for x in s.get("stages", []))

            # (1) TEXT: sharded chat (tokenize → cached greedy w/ eos-stop → detokenize) == un-sharded greedy decode
            ch = api(port, "/model/shard_chat", "POST", {"id": sid, "prompt": PROMPT, "max_new_tokens": CHAT_NEW},
                     admin=ADMIN)
            chat_text = ch.get("text"); chat_n = ch.get("n")
            text_ok = (chat_text == g["chat_text"]) and (chat_n == g["chat_n"])

            # (2) TOKENS: sharded coordinator-side greedy stream over the same ids == un-sharded greedy token stream
            lines = api_ndjson(port, "/model/shard_generate", {"id": sid, "input_ids": g["ids"],
                                                               "max_new_tokens": GEN_NEW}, ADMIN)
            done = next((x for x in lines if x.get("done")), None)
            errline = next((x for x in lines if x.get("error") or x.get("httperror")), None)
            gen_tokens = done.get("tokens") if done else None
            tok_ok = (errline is None) and (gen_tokens == g["gen_tokens"])

            model_ok = text_ok and tok_ok
            ok = ok and model_ok
            print(f"  {'PASS' if model_ok else 'FAIL'} {model:10s} ({arch:5s}) {len(wkrs)} stages [{stages}]:",
                  flush=True)
            print(f"       TEXT   shard_chat=={'golden' if text_ok else 'MISMATCH'}  "
                  f"n={chat_n}(golden {g['chat_n']})  text={chat_text!r}")
            print(f"       TOKENS shard_generate=={'golden' if tok_ok else 'MISMATCH'}  "
                  f"shard={gen_tokens}  golden={g['gen_tokens']}")
            if not text_ok:
                print(f"       (golden text={g['chat_text']!r}  chat resp={ch})")
            if not tok_ok and errline is not None:
                print(f"       (shard_generate error line: {errline})")
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
