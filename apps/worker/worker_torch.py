#!/usr/bin/env python3
"""
MoreGPU NATIVE TORCH worker — a drop-in peer of the Deno/WebGPU worker (apps/worker/worker.ts) that
computes with PyTorch on the best local device (CUDA → MPS → CPU) instead of WebGPU.

It speaks the EXACT same sealed WebSocket protocol, so it joins the SAME pool with the SAME join token
and shares the SAME tenant key as the Deno workers — the coordinator is backend-agnostic. What it adds:
native BLAS / Apple-MPS speed per kernel, resident weights held as on-device tensors (incl. fp16), and
— because torch has autograd — the substrate for on-pool fine-tuning (see the train_* verbs).

This is ADR-0007's native-accelerator tier: a signed, admin-installed binary on owned hardware, NOT the
zero-install invisible WebGPU worker. Run it where you trust the machine.

    pip install torch cryptography websockets
    python3 apps/worker/worker_torch.py --server ws://ADMIN:8787/ws --token <join-token> [--cpu]

Protocol (matches apps/worker/worker.ts): register{joinToken,pubkey,node} → welcome{tenantKeyB64,duty} ;
sealed assign{shardId,jobId,sealedIn} → result{shardId,jobId,ok,sealedOut,sig,ms,backend} ;
cache{id,sealed}→cached{id,ok} ; uncache{id} ; control{...} ; ~4s heartbeat. Payloads are AES-256-GCM
sealed with the tenant key; every result is Ed25519-signed over `${shardId}|${iv}|${ct}`.
"""
from __future__ import annotations
import argparse, asyncio, base64, json, os, platform, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import websockets

ap = argparse.ArgumentParser()
ap.add_argument("--server", default=os.environ.get("MOREGPU_SERVER", "ws://localhost:8787/ws"))
ap.add_argument("--token", default=os.environ.get("MOREGPU_TOKEN", ""))
ap.add_argument("--name", default=os.environ.get("MOREGPU_NAME", f"torch-{platform.node().split('.')[0][:12]}-{os.getpid()%100000}"))
ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
A, _ = ap.parse_known_args()  # parse_known_args so this module can also be imported (e.g. by a training reference)

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if A.cpu:
    DEV = "cpu"
# The coordinator tallies GPU vs CPU shards by whether the result `backend` string startsWith('gpu')
# (server.ts:399) and counts GPU *slots* by node.backend==='gpu' (server.ts:518). A real accelerator
# must therefore register backend='gpu' AND label/result-backend starting with 'gpu:'.
NODE_BACKEND = "cpu" if DEV == "cpu" else "gpu"
BACKEND = f"cpu:torch" if DEV == "cpu" else f"gpu:torch-{DEV}"
NAME = A.name

_sk = Ed25519PrivateKey.generate()
PUBKEY_B64 = base64.b64encode(_sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()

def b64e(b: bytes) -> str: return base64.b64encode(b).decode()
def b64d(s: str) -> bytes: return base64.b64decode(s)

# ---- wire <-> tensor: every payload is flat, row-major, little-endian raw Float32 base64 ----
def f32_to_b64(t: torch.Tensor) -> str:
    return b64e(t.detach().to("cpu", torch.float32).contiguous().numpy().astype("<f4").tobytes())
def b64_to_t(s: str) -> torch.Tensor:  # flat float32 tensor (on CPU; caller moves to DEV)
    return torch.from_numpy(np.frombuffer(b64d(s), dtype="<f4").copy())

# ---- sealing: AES-256-GCM, 12-byte IV, ct = ciphertext||16-byte tag (byte-compatible w/ WebCrypto) ----
def seal(key: bytes, plain: bytes) -> dict:
    iv = os.urandom(12)
    return {"iv": b64e(iv), "ct": b64e(AESGCM(key).encrypt(iv, plain, None))}
def unseal(key: bytes, blob: dict) -> bytes:
    return AESGCM(key).decrypt(b64d(blob["iv"]), b64d(blob["ct"]), None)
def sign_result(shard_id: str, blob: dict) -> str:
    return b64e(_sk.sign(f"{shard_id}|{blob['iv']}|{blob['ct']}".encode()))

resident: dict[str, torch.Tensor] = {}  # bRef id -> weight tensor on DEV (float32 or float16)

def compute_shard(req: dict) -> torch.Tensor:
    """Run one kernel. Returns a flat CPU/​DEV tensor of the row-major output."""
    k = req["kernel"]
    a = b64_to_t(req["a"]).to(DEV)
    if k == "matmul":
        rows, N, K = int(req["rows"]), int(req["N"]), int(req["K"])
        Am = a.reshape(rows, K)
        if req.get("bRef") is not None:
            W = resident[req["bRef"]]  # [K, N], stored as uploaded (no transpose — GPT-2 Conv1D is [in,out])
            if W.dtype == torch.float16:
                # match worker.ts f16 GEMM: round A→f16, keep W f16, accumulate in f32
                out = Am.to(torch.float16).float() @ W.float()
            else:
                out = Am @ W
        else:
            out = Am @ b64_to_t(req["b"]).to(DEV).reshape(K, N)
        res = out.reshape(-1)
    elif k in ("vector_add", "vector_mul", "saxpy", "relu", "scale", "gelu"):
        s = float(req.get("scalar", 1.0))
        if k == "relu":
            res = torch.relu(a)
        elif k == "scale":
            res = a * s
        elif k == "gelu":  # tanh approximation, to match worker.ts:137 / server.ts CPU reference
            res = torch.nn.functional.gelu(a, approximate="tanh")
        else:
            b = b64_to_t(req["b"]).to(DEV)
            res = a + b if k == "vector_add" else a * b if k == "vector_mul" else s * a + b
    elif k in ("softmax", "layernorm"):
        cols = int(req["cols"]); x = a.reshape(-1, cols)
        if k == "softmax":
            res = torch.softmax(x, dim=1).reshape(-1)
        else:  # normalize only (unbiased=False, eps 1e-5, no affine); LN weight/bias applied client-side
            m = x.mean(1, keepdim=True); v = ((x - m) ** 2).mean(1, keepdim=True)
            res = ((x - m) / torch.sqrt(v + 1e-5)).reshape(-1)
    else:
        raise ValueError(f"unknown kernel {k}")
    if DEV == "cuda": torch.cuda.synchronize()
    elif DEV == "mps": torch.mps.synchronize()
    return res

# All torch work (kernels + training) runs on ONE background thread: keeps the event loop free for
# heartbeats/other frames, and serializes MPS access (a single Metal context is not safe across threads).
TORCH_POOL = ThreadPoolExecutor(max_workers=1)
MAX_RESIDENT_MODELS = int(os.environ.get("MOREGPU_MAX_RESIDENT_MODELS", "2"))  # bound VRAM: LRU-evict beyond this

def _empty_cache():
    try:
        if DEV == "cuda": torch.cuda.empty_cache()
        elif DEV == "mps": torch.mps.empty_cache()
    except Exception:
        pass

def _check_ids(vocab, *seqs):
    """Reject out-of-range token ids BEFORE the embedding lookup — an out-of-vocab index is a device-side
    assert on CUDA that poisons the whole process context (bricks TORCH_POOL), so fail cleanly instead."""
    for s in seqs:
        if s and (max(s) >= vocab or min(s) < 0):
            raise ValueError(f"token id out of range [0,{vocab}) — check the tokenizer matches the model")

# ---------------------------------------------------------------------------
# On-pool fine-tuning (ADR-0007 native-accelerator tier): the WHOLE train step
# (forward + cross-entropy + backward + optimizer.step) runs LOCALLY here in torch.
# The base model is frozen; a small LoRA adapter is the only trainable tensor, so
# gradients never leave the worker — only a tiny sealed microbatch in / scalar loss
# out, and (on demand) the MB-scale adapter. This is single-worker fine-tuning; the
# pool hosts one training job, it does not (yet) parallelize training across workers.
# ---------------------------------------------------------------------------
class LoRAWrap(nn.Module):
    """Wrap a frozen Linear/Conv1D so y = base(x) + (x·Aᵀ)·Bᵀ·scale. B starts at 0 → the adapter is a
    no-op at step 0 (loss then equals the base model's), which makes the run reproducible against a
    seeded reference."""
    def __init__(self, base: nn.Module, in_f: int, out_f: int, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, in_f) * (1.0 / r))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

def _in_out(mod: nn.Module):
    if isinstance(mod, nn.Linear):
        return mod.in_features, mod.out_features
    if hasattr(mod, "nf") and hasattr(mod, "weight"):  # transformers Conv1D (GPT-2): weight [in, out=nf]
        return mod.weight.shape[0], int(mod.nf)
    return None

def attach_lora(model: nn.Module, targets: list[str], r: int, alpha: float) -> int:
    """Freeze the base model, replace each target module (matched by name suffix) with a LoRAWrap.
    Returns the count of trainable adapter parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    hits = []
    for name, mod in model.named_modules():
        if any(name.split(".")[-1] == t for t in targets):
            io = _in_out(mod)
            if io:
                hits.append((name, mod, io))
    for name, mod, (in_f, out_f) in hits:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        parent.add_module(name.split(".")[-1], LoRAWrap(mod, in_f, out_f, r, alpha).to(DEV))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

TRAIN: dict = {"model": None, "opt": None, "step": 0, "trainable": {}}

def train_load(cfg: dict) -> dict:
    from transformers import AutoModelForCausalLM
    torch.manual_seed(int(cfg.get("seed", 0)))
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(DEV)
    if cfg.get("no_dropout"):  # deterministic training (e.g. DiLoCo) → reproducible against a reference
        for mod in model.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
    targets = cfg.get("targets") or ["c_attn", "q_proj", "v_proj"]
    n_train = attach_lora(model, targets, int(cfg.get("rank", 8)), float(cfg.get("alpha", 16)))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(cfg.get("lr", 1e-3)))
    TRAIN.update(model=model, opt=opt, step=0,
                 trainable={n: p for n, p in model.named_parameters() if p.requires_grad})
    return {"ok": True, "trainable_params": n_train, "targets": targets, "device": DEV}

def train_step(batch: dict) -> dict:
    model, opt = TRAIN["model"], TRAIN["opt"]
    if model is None:
        raise RuntimeError("no training session — call /train/load first")
    if "lr" in batch and batch["lr"]:
        for g in opt.param_groups:
            g["lr"] = float(batch["lr"])
    _check_ids(getattr(model.config, "vocab_size", 1 << 30), batch["input_ids"], batch.get("labels", []))
    ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=DEV).unsqueeze(0)
    labels = torch.tensor(batch.get("labels", batch["input_ids"]), dtype=torch.long, device=DEV).unsqueeze(0)
    model.train()
    opt.zero_grad()
    loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    opt.step()
    if DEV == "mps": torch.mps.synchronize()
    TRAIN["step"] += 1
    return {"ok": True, "loss": float(loss.item()), "step": TRAIN["step"]}

def train_adapter() -> dict:
    """Return the (small) trainable adapter tensors as flat f32 base64 — grads/base weights never leave."""
    out = {n: {"data": f32_to_b64(p), "shape": list(p.shape)} for n, p in TRAIN["trainable"].items()}
    return {"ok": True, "step": TRAIN["step"], "tensors": out}

# --- DiLoCo / local-SGD building blocks (distributed LoRA across many workers) ---
# Each round: every worker starts from the SAME broadcast adapter, runs H local AdamW steps on its OWN
# data shard (train_inner), and returns its adapter; the coordinator averages them + applies an outer
# Nesterov step and broadcasts the result (train_set_adapter). Inner optimizer state resets each round
# (standard DiLoCo), so a round is a pure function of {broadcast adapter, this worker's batches}.
def train_inner(payload: dict) -> dict:
    model = TRAIN["model"]
    if model is None:
        raise RuntimeError("no training session — call train_load first")
    batches = payload["batches"]          # list of token-id windows (this worker's shard)
    if not batches:
        raise RuntimeError("no batches supplied for this worker's DiLoCo shard")
    _check_ids(getattr(model.config, "vocab_size", 1 << 30), *batches)
    steps = int(payload.get("steps", len(batches)))
    lr = float(payload.get("lr", 1e-3))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)   # fresh inner optimizer each round (DiLoCo)
    model.train()
    losses = []
    for i in range(steps):
        b = batches[i % len(batches)]
        ids = torch.tensor(b, dtype=torch.long, device=DEV).unsqueeze(0)
        opt.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward(); opt.step()
        losses.append(float(loss.item()))
    if DEV == "mps": torch.mps.synchronize()
    TRAIN["step"] += steps
    out = {n: {"data": f32_to_b64(p), "shape": list(p.shape)} for n, p in TRAIN["trainable"].items()}
    return {"ok": True, "losses": losses, "tensors": out}

def train_set_adapter(payload: dict) -> dict:
    """Overwrite the resident LoRA adapter with broadcast tensors (the coordinator's averaged global)."""
    with torch.no_grad():
        for n, p in TRAIN["trainable"].items():
            t = payload["tensors"].get(n)
            if t is not None:
                p.copy_(b64_to_t(t["data"]).reshape(p.shape).to(DEV))
    return {"ok": True, "step": TRAIN["step"]}

def train_dispatch(op: str, payload: dict) -> dict:
    if op == "load": return train_load(payload)
    if op == "step": return train_step(payload)
    if op == "adapter": return train_adapter()
    if op == "inner": return train_inner(payload)
    if op == "set_adapter": return train_set_adapter(payload)
    raise ValueError(f"unknown train op {op}")

# ---------------------------------------------------------------------------
# RESIDENT-MODEL SERVING: hold a whole model on-device and run the ENTIRE forward
# per call. This is what makes serving fast — the fine-grained kernel path drives
# ~500 sealed round-trips PER TOKEN; here the client sends token ids ONCE and gets
# logits back in ONE round-trip (native GPT-2 forward is ~9 ms on MPS). The LM head
# runs on the worker's device (not sent over the wire); only the final-token logits
# (or just the argmax) travel back.
# ---------------------------------------------------------------------------
MODELS: dict = {}  # model_id -> torch model (eval, frozen)

def model_load(cfg: dict) -> dict:
    from transformers import AutoModelForCausalLM, AutoConfig
    mid = cfg.get("id") or cfg["model"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float16 if cfg.get("fp16") else torch.float32).to(DEV).eval()
    if mid not in MODELS and len(MODELS) >= MAX_RESIDENT_MODELS:  # LRU-evict to bound VRAM
        old = next(iter(MODELS)); del MODELS[old]; _empty_cache()
    MODELS.pop(mid, None); MODELS[mid] = model  # (re)insert as most-recent
    c = AutoConfig.from_pretrained(cfg["model"])
    return {"ok": True, "id": mid, "n_layer": getattr(c, "n_layer", getattr(c, "num_hidden_layers", None)),
            "n_params": sum(p.numel() for p in model.parameters()), "dtype": "f16" if cfg.get("fp16") else "f32", "device": DEV}

def model_forward(payload: dict) -> dict:
    mid = payload.get("id")
    model = MODELS.get(mid)
    if model is None:
        raise RuntimeError(f"model {mid} not loaded — call /model/load first")
    _check_ids(getattr(model.config, "vocab_size", 1 << 30), payload["input_ids"])
    ids = torch.tensor(payload["input_ids"], dtype=torch.long, device=DEV).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1].float()  # last-token logits [vocab]
    if DEV == "mps": torch.mps.synchronize()
    res = {"ok": True, "argmax": int(logits.argmax().item())}
    if payload.get("return_logits"):
        res["logits"] = f32_to_b64(logits)
    if payload.get("topk"):
        v, i = torch.topk(logits, int(payload["topk"]))
        res["top"] = [[int(a), float(b)] for a, b in zip(i.tolist(), v.tolist())]
    return res

def model_generate(payload: dict) -> dict:
    """Greedy generation entirely on the worker (HF's internal KV cache) — the whole decode in ONE
    round-trip. Returns just the new token ids."""
    mid = payload.get("id"); model = MODELS.get(mid)
    if model is None:
        raise RuntimeError(f"model {mid} not loaded — call /model/load first")
    _check_ids(getattr(model.config, "vocab_size", 1 << 30), payload["input_ids"])
    ids = torch.tensor(payload["input_ids"], dtype=torch.long, device=DEV).unsqueeze(0)
    n = max(1, min(int(payload.get("max_new_tokens", 16)), 1024))  # cap to keep a single call bounded
    attn = torch.ones_like(ids)
    kw = {"max_new_tokens": n, "do_sample": False, "use_cache": True, "attention_mask": attn}
    eos = getattr(model.config, "eos_token_id", None)
    if eos is not None:
        kw["pad_token_id"] = eos[0] if isinstance(eos, (list, tuple)) else eos
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, **kw)
    if DEV == "mps": torch.mps.synchronize()
    elif DEV == "cuda": torch.cuda.synchronize()
    new = out[0, ids.shape[1]:].tolist()
    return {"ok": True, "tokens": new, "n": len(new), "ms": (time.perf_counter() - t0) * 1000}

# ---------------------------------------------------------------------------
# PIPELINE-PARALLEL model SHARDING (GPT-2 family ONLY — gpt2 / gpt2-medium / …).
# Split the transformer's h[] blocks into contiguous STAGES, one per worker; each
# worker holds ONLY its stage's params resident and computes only its blocks. A
# forward pipes the hidden state [1, seq, 768] stage→stage (only the activations
# cross the wire — the low-bandwidth path — never the weights). This is the "model
# too big for one machine" story; here it's demoed with gpt2 across ≥2 workers.
# NOTE: GPT-2 ties lm_head.weight to wte.weight, so keeping lm_head on the last stage
# retains the head matrix even after wte is dropped there; the first stage keeps wte.
# ---------------------------------------------------------------------------
SHARDS: dict = {}  # shard id -> kept stage modules (blocks slice + optional embeddings / final head)

def shard_load(cfg: dict) -> dict:
    """Load a GPT-2 model, KEEP ONLY this stage's contiguous block slice transformer.h[start:end]
    (plus wte/wpe/drop if `first`, and ln_f/lm_head if `last`), delete everything else to actually
    free memory, and store the kept modules under SHARDS[id]. GPT-2 family only."""
    from transformers import GPT2LMHeadModel
    sid = cfg["id"]; start, end = int(cfg["start"]), int(cfg["end"])
    first, last = bool(cfg.get("first")), bool(cfg.get("last"))
    model = GPT2LMHeadModel.from_pretrained(cfg["model"], dtype=torch.float32).to(DEV).eval()
    tr = model.transformer
    n_layer = int(model.config.n_layer)
    shard: dict = {"first": first, "last": last, "start": start, "end": end, "n_layer": n_layer,
                   "blocks": tr.h[start:end],  # a GPT2Block ModuleList slice (references, not copies)
                   "vocab_size": int(model.config.vocab_size), "hidden_dim": int(model.config.n_embd),
                   "n_positions": int(model.config.n_positions)}
    if first:  # first stage embeds token ids → hidden states
        shard["wte"], shard["wpe"], shard["drop"] = tr.wte, tr.wpe, tr.drop
    if last:  # last stage applies the final norm + LM head (lm_head.weight is tied to wte.weight)
        shard["ln_f"], shard["lm_head"] = tr.ln_f, model.lm_head
    # count params THIS stage actually holds, deduping the tied wte/lm_head weight by tensor identity
    held = [shard["blocks"]] + [shard[k] for k in ("wte", "wpe", "drop", "ln_f", "lm_head") if k in shard]
    seen: set = set(); params_held = 0
    for mod in held:
        for p in mod.parameters():
            if id(p) not in seen:
                seen.add(id(p)); params_held += p.numel()
    # Actually free the unkept weights: point the model's block list at our slice, drop the model wrapper.
    # Our SHARDS references keep the kept modules alive; every unkept module loses its last reference and is
    # collected (the tied head matrix survives on the last stage via lm_head even though wte is dropped).
    tr.h = shard["blocks"]
    SHARDS.pop(sid, None)  # replacing a live shard: drop the old modules first
    if len(SHARDS) >= MAX_RESIDENT_MODELS:  # bound VRAM, same as resident models
        SHARDS.pop(next(iter(SHARDS)), None)
    del tr, model
    _empty_cache()
    SHARDS[sid] = shard
    return {"ok": True, "id": sid, "layers": [start, end], "first": first, "last": last, "n_layer": n_layer, "params_held": params_held}

def shard_forward(payload: dict) -> dict:
    """Run ONE stage's blocks. If first: embed input_ids → hidden; else: reshape the piped-in hidden.
    If last: final norm + LM head → {argmax, logits?}; else: return the hidden state for the next stage."""
    sid = payload.get("id"); shard = SHARDS.get(sid)
    if shard is None:
        raise RuntimeError(f"shard {sid} not loaded — call shard_load first")
    first, last = shard["first"], shard["last"]
    hidden_dim = shard["hidden_dim"]
    with torch.no_grad():
        if first:
            ids_list = payload["input_ids"]
            _check_ids(shard["vocab_size"], ids_list)
            if len(ids_list) > shard["n_positions"]:  # position ids ≥ n_positions overflow wpe (CUDA device-assert)
                raise ValueError(f"sequence length {len(ids_list)} exceeds the model's context window {shard['n_positions']}")
            ids = torch.tensor(ids_list, dtype=torch.long, device=DEV).unsqueeze(0)  # [1, seq]
            seq = ids.shape[1]
            pos = torch.arange(seq, dtype=torch.long, device=DEV)
            h = shard["drop"](shard["wte"](ids) + shard["wpe"](pos))  # [1, seq, 768]
        else:
            seq = int(payload["seq"])
            h = b64_to_t(payload["hidden"]).reshape(1, seq, hidden_dim).to(DEV)
        for blk in shard["blocks"]:  # GPT2Block applies causal self-attention internally (no mask needed)
            h = blk(h)[0]
        if last:
            h = shard["ln_f"](h)
            logits = shard["lm_head"](h)[0, -1].float()  # last-token logits [vocab]
            res = {"ok": True, "argmax": int(logits.argmax().item())}
            if payload.get("return_logits"):
                res["logits"] = f32_to_b64(logits)
        else:
            res = {"ok": True, "hidden": f32_to_b64(h.flatten()), "seq": seq, "hidden_dim": hidden_dim}
    if DEV == "cuda": torch.cuda.synchronize()
    elif DEV == "mps": torch.mps.synchronize()
    return res

def model_arch(cfg: dict) -> dict:
    """Cheap: read a model's layer count + context window from config.json (no weights) so the coordinator
    can size a shard plan correctly for any GPT-2 variant (base/medium/large/xl)."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(cfg["model"])
    return {"ok": True, "n_layer": int(getattr(c, "n_layer", getattr(c, "num_hidden_layers", 0))),
            "n_positions": int(getattr(c, "n_positions", getattr(c, "max_position_embeddings", 0)))}

def model_dispatch(op: str, payload: dict) -> dict:
    if op == "load": return model_load(payload)
    if op == "forward": return model_forward(payload)
    if op == "generate": return model_generate(payload)
    if op == "arch": return model_arch(payload)
    if op == "unload": MODELS.pop(payload.get("id"), None); _empty_cache(); return {"ok": True}
    if op == "shard_load": return shard_load(payload)
    if op == "shard_forward": return shard_forward(payload)
    if op == "shard_unload": SHARDS.pop(payload.get("id"), None); _empty_cache(); return {"ok": True}
    raise ValueError(f"unknown model op {op}")

async def run():
    loop = asyncio.get_event_loop()
    paused = False; pause_reason = ""; ceil = 0.6
    while True:
        try:
            async with websockets.connect(A.server, max_size=None) as ws:
                await ws.send(json.dumps({"t": "register", "joinToken": A.token, "pubkey": PUBKEY_B64,
                                          "node": {"id": NAME, "backend": NODE_BACKEND, "label": BACKEND, "os": platform.system().lower()}}))
                key = None

                async def heartbeat():
                    while True:
                        await asyncio.sleep(4)
                        try:
                            await ws.send(json.dumps({"t": "heartbeat", "id": NAME, "load1": 0, "cores": os.cpu_count() or 4,
                                                      "util": 0.0, "duty": 0.0 if paused else ceil, "ceil": ceil,
                                                      "paused": paused, "pausedReason": pause_reason, "schedule": "always"}))
                        except Exception:
                            return
                hb = asyncio.ensure_future(heartbeat())
                try:
                    async for raw in ws:
                        m = json.loads(raw); t = m.get("t")
                        if t == "denied":
                            print(f"[torch-worker] rejected: {m.get('reason')}"); return
                        elif t == "welcome":
                            key = b64d(m["tenantKeyB64"]); ceil = float(m.get("duty", 0.6))
                            # fresh session: the coordinator forgets our resident state on disconnect, so a
                            # reconnecting worker starts clean too (no stale models/weights pinned in VRAM).
                            resident.clear(); MODELS.clear(); SHARDS.clear(); TRAIN.update(model=None, opt=None, step=0, trainable={}); _empty_cache()
                            print(f"[torch-worker] joined pool on {DEV} · duty ceiling {int(ceil*100)}%")
                        elif t == "control":
                            if "pause" in m: paused = bool(m["pause"]); pause_reason = "admin" if paused else ""
                            if "ceil" in m and m["ceil"] is not None: ceil = float(m["ceil"])
                            print(f"[torch-worker] admin control → {'PAUSED' if paused else 'active'} · ceiling {int(ceil*100)}%")
                        elif t == "uncache":
                            resident.pop(m.get("id"), None)
                        elif t == "cache":
                            try:
                                w = json.loads(unseal(key, m["sealed"]).decode())
                                rows, cols = int(w["rows"]), int(w["cols"])
                                if w.get("dtype") == "f16":
                                    ten = torch.from_numpy(np.frombuffer(b64d(w["data"]), dtype="<f2").copy()).reshape(rows, cols).to(DEV)
                                else:
                                    ten = b64_to_t(w["data"]).reshape(rows, cols).to(DEV)
                                resident[m["id"]] = ten
                                await ws.send(json.dumps({"t": "cached", "id": m["id"], "ok": True}))
                            except Exception as e:
                                await ws.send(json.dumps({"t": "cached", "id": m["id"], "ok": False, "error": str(e)}))
                        elif t == "assign":
                            t0 = time.perf_counter()
                            try:
                                req = json.loads(unseal(key, m["sealedIn"]).decode())
                                out = await loop.run_in_executor(TORCH_POOL, compute_shard, req)  # off the event loop
                                sealed = seal(key, json.dumps({"out": f32_to_b64(out)}).encode())
                                await ws.send(json.dumps({"t": "result", "shardId": m["shardId"], "jobId": m["jobId"], "ok": True,
                                                          "sealedOut": sealed, "sig": sign_result(m["shardId"], sealed),
                                                          "ms": (time.perf_counter() - t0) * 1000, "backend": BACKEND}))
                            except Exception as e:
                                await ws.send(json.dumps({"t": "result", "shardId": m["shardId"], "jobId": m["jobId"], "ok": False, "error": str(e)}))
                        elif t in ("train", "model"):  # relayed RPCs: fine-tuning (op=load|step|adapter) / resident-model serving (op=load|forward) / pipeline sharding (op=shard_load|shard_forward|shard_unload)
                            reqId = m.get("reqId"); reply = f"{t}_reply"
                            dispatch = train_dispatch if t == "train" else model_dispatch
                            try:
                                payload = json.loads(unseal(key, m["sealed"]).decode())
                                res = await loop.run_in_executor(TORCH_POOL, dispatch, m["op"], payload)
                                sealed = seal(key, json.dumps(res).encode())
                                await ws.send(json.dumps({"t": reply, "reqId": reqId, "ok": True, "sealed": sealed}))
                            except Exception as e:
                                await ws.send(json.dumps({"t": reply, "reqId": reqId, "ok": False, "error": str(e)}))
                finally:
                    hb.cancel()
        except Exception as e:
            print(f"[torch-worker] disconnected ({e}); retrying in 2s"); await asyncio.sleep(2)

if __name__ == "__main__":
    print(f"[torch-worker] {NAME} · device={DEV} · backend={BACKEND} · server={A.server}")
    if not A.token:
        print("[torch-worker] warning: no --token / MOREGPU_TOKEN set; the coordinator will reject me")
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[torch-worker] bye")
