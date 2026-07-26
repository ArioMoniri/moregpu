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

def _check_labels(vocab, labels):
    """Labels (unlike input_ids) may carry HF's ignore index -100 to mask a position out of the loss; every
    other value must be a valid token id. Kept separate from _check_ids so -100 is allowed ONLY on labels."""
    for x in labels:
        if not (x == -100 or 0 <= x < vocab):
            raise ValueError(f"label id out of range — must be -100 (ignore) or in [0,{vocab})")

def _ctx_window(model) -> int:
    """The model's max sequence length (positions). GPT-2 exposes n_positions; Llama-style exposes
    max_position_embeddings. Defaults huge so an unknown config never falsely rejects."""
    n = getattr(model.config, "n_positions", None)
    if n is None:
        n = getattr(model.config, "max_position_embeddings", 1 << 30)
    return int(n)

def _check_ctx(model, seq_len: int, extra: int = 0):
    """Guard the context window BEFORE the forward — a position index past n_positions is an out-of-range
    embedding lookup → CUDA device-assert that bricks the single-thread TORCH_POOL. Mirrors the guard
    shard_forward already applies; `extra` covers the tokens a generate call will append."""
    n_pos = _ctx_window(model)
    if seq_len + extra > n_pos:
        raise ValueError(f"sequence length {seq_len + extra} exceeds the model's context window {n_pos}")

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

def attach_lora(model: nn.Module, targets: list[str], r: int, alpha: float, dev: str | None = None) -> int:
    """Freeze the base model, replace each target module (matched by name suffix) with a LoRAWrap.
    Returns the count of trainable adapter parameters. `dev` overrides the placement device (used by an
    out-of-band verification reference so it can reproduce a worker running on a different device)."""
    dev = dev or DEV
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
        parent.add_module(name.split(".")[-1], LoRAWrap(mod, in_f, out_f, r, alpha).to(dev))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

TRAIN: dict = {"model": None, "opt": None, "step": 0, "trainable": {}}

def train_load(cfg: dict) -> dict:
    from transformers import AutoModelForCausalLM
    torch.manual_seed(int(cfg.get("seed", 0)))
    # Free the previous training model BEFORE allocating the replacement so we never transiently hold two
    # full models in VRAM (a single global TRAIN slot; loading first would double-allocate).
    TRAIN.update(model=None, opt=None, trainable={}); _empty_cache()
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
    vocab = getattr(model.config, "vocab_size", 1 << 30)
    _check_ids(vocab, batch["input_ids"])            # input ids: strictly in-vocab
    _check_labels(vocab, batch.get("labels", []))    # labels: in-vocab OR -100 (HF ignore index)
    _check_ctx(model, len(batch["input_ids"]))       # context-window guard (before the forward)
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
    _check_ctx(model, max(len(b) for b in batches))  # longest window must fit the context (before the forward)
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
MODEL_NAMES: dict = {}  # model_id -> HF name (for the tokenizer in model_chat)
_TOKS: dict = {}  # model_id -> tokenizer (cached, for text↔text chat)

def model_load(cfg: dict) -> dict:
    from transformers import AutoModelForCausalLM, AutoConfig
    mid = cfg.get("id") or cfg["model"]
    # fp16 has no CPU matmul kernel in torch ("not implemented for 'Half'") — force f32 on CPU workers and
    # report the effective dtype so the caller knows the request was downgraded.
    fp16_requested = bool(cfg.get("fp16"))
    fp16_effective = fp16_requested and DEV != "cpu"
    # Evict the LRU victim (MODELS is kept most-recent-last) and free VRAM BEFORE allocating the replacement,
    # so peak residency never exceeds the cap (loading first would briefly hold MAX+1 models on-device).
    if mid not in MODELS and len(MODELS) >= MAX_RESIDENT_MODELS:
        MODELS.pop(next(iter(MODELS)), None); _empty_cache()
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float16 if fp16_effective else torch.float32).to(DEV).eval()
    MODELS.pop(mid, None); MODELS[mid] = model  # (re)insert as most-recent
    MODEL_NAMES[mid] = cfg["model"]; _TOKS.pop(mid, None)  # remember the HF name for the tokenizer (model_chat)
    c = AutoConfig.from_pretrained(cfg["model"])
    return {"ok": True, "id": mid, "n_layer": getattr(c, "n_layer", getattr(c, "num_hidden_layers", None)),
            "n_params": sum(p.numel() for p in model.parameters()),
            "dtype": "f16" if fp16_effective else "f32", "fp16_requested": fp16_requested,
            "fp16_effective": fp16_effective, "device": DEV}

# ---------------------------------------------------------------------------
# DOWNLOAD-FREE resident load ("weight push"): the WORKER never contacts the HF hub. The coordinator
# (the admin box, where "all weights live") fetches config + safetensors + tokenizer over HTTPS ONCE and
# streams the raw file bytes here in sealed chunks. We stage them in a RAM-backed dir (/dev/shm when
# present → zero SSD) and load with from_pretrained(local_files_only=True) — HF's OWN loader, so
# key-remapping / tied weights / sharded checkpoints / tokenizer all "just work" for any model family —
# then delete the staging dir, leaving only the on-device model. The fleet stays a pure GPU compute
# provider: no hub download, and no SSD write on a tmpfs-backed box. Unlike the per-token pipeline pipe,
# this is a ONE-TIME transfer, so it survives a slow/flaky tunnel far better.
# ---------------------------------------------------------------------------
PUSH: dict = {}  # id -> {"dir": staging path, "bytes": total written, "ram": bool}
# Safety rail: refuse to stage more than this many bytes into the (possibly RAM-backed) staging dir, so a
# runaway/oversized push can't OOM this donated machine. Env-tunable; matches the coordinator's cap.
PUSH_MAX_BYTES = int(os.environ.get("MOREGPU_PUSH_MAX_BYTES", str(20 * 1024 ** 3)))

def _stage_root() -> str:
    """Prefer a RAM-backed filesystem so streamed weights never touch SSD; fall back to the OS temp dir."""
    import tempfile
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return tempfile.gettempdir()

def _push_cleanup(mid) -> None:
    import shutil
    st = PUSH.pop(mid, None)
    if st and os.path.isdir(st["dir"]):
        shutil.rmtree(st["dir"], ignore_errors=True)

def model_push_begin(payload: dict) -> dict:
    import tempfile
    mid = payload.get("id") or payload.get("model")
    _push_cleanup(mid)  # drop any stale half-streamed staging for this id
    root = _stage_root()
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in str(mid))[:40]
    d = tempfile.mkdtemp(prefix=f"moregpu-{safe}-", dir=root)
    PUSH[mid] = {"dir": d, "bytes": 0, "ram": root == "/dev/shm"}
    return {"ok": True, "id": mid, "staging": "ram" if root == "/dev/shm" else "disk"}

def model_push_chunk(payload: dict) -> dict:
    mid = payload.get("id"); st = PUSH.get(mid)
    if st is None:
        raise RuntimeError("push not begun — call push_begin first")
    name = os.path.basename(str(payload["name"]))  # sanitize: strip any path components (no traversal)
    if not name or name in (".", ".."):
        raise RuntimeError(f"bad staged file name {payload.get('name')!r}")
    raw = base64.b64decode(payload["data"])
    if st["bytes"] + len(raw) > PUSH_MAX_BYTES:  # rail against a runaway push OOM-ing this donated box
        _push_cleanup(mid)
        raise RuntimeError(f"push exceeds {PUSH_MAX_BYTES}-byte staging cap (raise MOREGPU_PUSH_MAX_BYTES) — aborted")
    with open(os.path.join(st["dir"], name), "ab") as f:  # fresh mkdtemp dir → 'ab' == create; chunks arrive in order
        f.write(raw)
    st["bytes"] += len(raw)
    return {"ok": True, "bytes": st["bytes"]}

def model_push_end(payload: dict) -> dict:
    """Assemble the model from the staged files (HF's loader, hub disabled), pin it on-device, cache its
    tokenizer from the SAME staged files (so model_chat never has to reach the hub), then delete staging."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    mid = payload.get("id"); st = PUSH.get(mid)
    if st is None:
        raise RuntimeError("push not begun — call push_begin first")
    d = st["dir"]
    fp16_requested = bool(payload.get("fp16")); fp16_effective = fp16_requested and DEV != "cpu"
    try:
        if mid not in MODELS and len(MODELS) >= MAX_RESIDENT_MODELS:
            MODELS.pop(next(iter(MODELS)), None); _empty_cache()  # evict LRU BEFORE allocating (peak ≤ cap)
        model = AutoModelForCausalLM.from_pretrained(
            d, dtype=torch.float16 if fp16_effective else torch.float32, local_files_only=True).to(DEV).eval()
        MODELS.pop(mid, None); MODELS[mid] = model  # (re)insert as most-recent
        MODEL_NAMES[mid] = payload.get("model", mid)
        _TOKS.pop(mid, None)
        try:  # pre-load the tokenizer from the staged files; if none were shipped, forward/generate still work
            _TOKS[mid] = AutoTokenizer.from_pretrained(d, local_files_only=True)
        except Exception:
            _TOKS[mid] = None  # model_chat raises a clear error; token-id paths are unaffected
        c = AutoConfig.from_pretrained(d, local_files_only=True)
        return {"ok": True, "id": mid, "device": DEV, "mode": "download-free",
                "n_layer": getattr(c, "n_layer", getattr(c, "num_hidden_layers", None)),
                "n_params": sum(p.numel() for p in model.parameters()),
                "dtype": "f16" if fp16_effective else "f32", "fp16_requested": fp16_requested,
                "fp16_effective": fp16_effective, "bytes": st["bytes"],
                "staging": "ram" if st["ram"] else "disk", "tokenizer": _TOKS.get(mid) is not None}
    finally:
        _push_cleanup(mid)  # weights are resident on-device now — drop the staged copy (RAM or disk)

def model_forward(payload: dict) -> dict:
    mid = payload.get("id")
    model = MODELS.get(mid)
    if model is None:
        raise RuntimeError(f"model {mid} not loaded — call /model/load first")
    _check_ids(getattr(model.config, "vocab_size", 1 << 30), payload["input_ids"])
    _check_ctx(model, len(payload["input_ids"]))     # context-window guard (before the forward)
    MODELS[mid] = MODELS.pop(mid)                     # mark most-recently-used → genuine LRU eviction order
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
    n = max(1, min(int(payload.get("max_new_tokens", 16)), 1024))  # cap to keep a single call bounded
    _check_ctx(model, len(payload["input_ids"]), extra=n)  # prompt + decode must fit the context window
    MODELS[mid] = MODELS.pop(mid)                           # mark most-recently-used → genuine LRU
    ids = torch.tensor(payload["input_ids"], dtype=torch.long, device=DEV).unsqueeze(0)
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
# PIPELINE-PARALLEL model SHARDING (two families: GPT-2 and Llama-style — the latter
# covers Llama / SmolLM / TinyLlama / Qwen2 / Qwen3, anything with model.layers +
# RMSNorm + RoPE). Split the transformer's decoder blocks into contiguous STAGES, one
# per worker; each worker holds ONLY its stage's params resident and computes only its
# blocks. A forward pipes the hidden state [1, seq, hidden] stage→stage (only the
# activations cross the wire — the low-bandwidth path — never the weights). This is the
# "model too big for one machine" story; demoed with gpt2 / SmolLM-135M across ≥2 workers.
# The two families differ in the per-stage forward: GPT-2 has a learned positional
# embedding (wpe) and a GPT2Block that masks internally; Llama-style uses RoPE, so every
# stage recomputes cos/sin from position_ids and passes a causal mask to each layer.
# NOTE: both families tie lm_head.weight to the input embedding, so keeping lm_head on the
# last stage retains the head matrix even after the embedding is dropped there; the first
# stage keeps the embedding. That one tied matrix is duplicated across the two end stages.
# ---------------------------------------------------------------------------
SHARDS: dict = {}  # shard id -> kept stage modules (blocks slice + optional embeddings / final head)


def _shard_arch(model: nn.Module) -> str:
    """Detect the sharding family from a loaded causal LM: 'gpt2' (transformer.h + learned wpe) vs
    'llama' (model.layers + RMSNorm + RoPE — Llama / SmolLM / TinyLlama / Qwen2 / Qwen3)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama"
    raise ValueError(f"sharding unsupported for {type(model).__name__}: need GPT-2 (transformer.h) "
                     f"or Llama-style (model.layers)")

def shard_load(cfg: dict) -> dict:
    """Load a causal LM via AutoModelForCausalLM, KEEP ONLY this stage's contiguous decoder-block slice
    (plus the input embedding if `first`, and the final norm + LM head if `last`), delete everything else
    to actually free memory, and store the kept modules under SHARDS[id]. Handles two families, detected
    from the loaded model: GPT-2 (transformer.h + learned wpe) and Llama-style (model.layers + RMSNorm +
    RoPE — Llama / SmolLM / TinyLlama / Qwen2 / Qwen3)."""
    from transformers import AutoModelForCausalLM
    sid = cfg["id"]; start, end = int(cfg["start"]), int(cfg["end"])
    first, last = bool(cfg.get("first")), bool(cfg.get("last"))
    if cfg.get("push"):
        # DOWNLOAD-FREE: the coordinator streamed this stage's config + a per-stage safetensors (only the
        # tensors this stage needs) into PUSH[sid]'s dir. Build the arch from config and load ONLY the present
        # (this-stage) weights — missing layers random-init but get dropped below when we slice to our blocks —
        # so the worker never touches the HF hub. (No low_cpu_mem_usage: with a partial checkpoint the missing
        # tensors must be real-initialized on CPU, not left on meta, or .to(DEV) can't move them.)
        st = PUSH.get(sid)
        if st is None:
            raise RuntimeError("shard weights not staged — push_begin/push_chunk must precede a push shard_load")
        model = AutoModelForCausalLM.from_pretrained(st["dir"], dtype=torch.float32, local_files_only=True).to(DEV).eval()
    else:
        # (non-push) materializes the FULL model on-device then slices — simple, but the worker downloads it.
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    arch = _shard_arch(model)
    mc = model.config
    shard: dict = {"arch": arch, "first": first, "last": last, "start": start, "end": end,
                   "vocab_size": int(mc.vocab_size)}
    if arch == "gpt2":
        base = model.transformer
        shard.update(n_layer=int(mc.n_layer), hidden_dim=int(mc.n_embd), n_positions=int(mc.n_positions),
                     blocks=base.h[start:end])  # a GPT2Block ModuleList slice (references, not copies)
        if first:  # first stage embeds token ids → hidden states (token + learned positional embedding)
            shard["wte"], shard["wpe"], shard["drop"] = base.wte, base.wpe, base.drop
        if last:  # last stage applies the final norm + LM head (lm_head.weight is tied to wte.weight)
            shard["ln_f"], shard["lm_head"] = base.ln_f, model.lm_head
        base.h = shard["blocks"]  # point the block list at our slice so the unkept blocks are freed
        held_keys = ("wte", "wpe", "drop", "ln_f", "lm_head")
    else:  # llama-style: RoPE (no learned wpe), RMSNorm final norm; keep config + rotary_emb on EVERY
        base = model.model  # stage so any stage can recompute cos/sin from position_ids (they're tiny)
        shard.update(n_layer=int(mc.num_hidden_layers), hidden_dim=int(mc.hidden_size),
                     n_positions=int(getattr(mc, "max_position_embeddings", 1 << 30)),
                     blocks=base.layers[start:end], config=mc, rotary_emb=base.rotary_emb)
        if first:  # first stage embeds token ids (RoPE is applied inside each layer, not here)
            shard["embed_tokens"] = base.embed_tokens
        if last:  # last stage applies the final RMSNorm + LM head (may be tied to embed_tokens.weight)
            shard["norm"], shard["lm_head"] = base.norm, model.lm_head
        base.layers = shard["blocks"]  # point the layer list at our slice so the unkept layers are freed
        held_keys = ("embed_tokens", "norm", "lm_head")  # rotary_emb holds no params (just an inv_freq buffer)
    # count params THIS stage actually holds, deduping the tied embedding/lm_head weight by tensor identity
    held = [shard["blocks"]] + [shard[k] for k in held_keys if k in shard]
    seen: set = set(); params_held = 0
    for mod in held:
        for p in mod.parameters():
            if id(p) not in seen:
                seen.add(id(p)); params_held += p.numel()
    # Actually free the unkept weights: we re-pointed the block list at our slice above, so dropping the
    # model wrapper collects every unkept module. Our SHARDS references keep the kept modules alive (the
    # tied head matrix survives on the last stage via lm_head even though the embedding is dropped there).
    SHARDS.pop(sid, None)  # replacing a live shard: drop the old modules first
    if len(SHARDS) >= MAX_RESIDENT_MODELS:  # bound VRAM, same as resident models
        SHARDS.pop(next(iter(SHARDS)), None)
    del base, model
    _empty_cache()
    SHARDS[sid] = shard
    if cfg.get("push"):
        _push_cleanup(sid)  # stage weights are now sliced into SHARDS — drop the staged partial safetensors
    return {"ok": True, "id": sid, "layers": [start, end], "first": first, "last": last,
            "n_layer": shard["n_layer"], "arch": arch, "params_held": params_held, "mode": "download-free" if cfg.get("push") else "download"}

def shard_forward(payload: dict) -> dict:
    """Run ONE stage's blocks. If first: embed input_ids → hidden; else: reshape the piped-in hidden.
    If last: final norm + LM head → {argmax, logits?}; else: return the hidden state for the next stage.
    Branches on the stored arch: GPT-2 blocks mask causally on their own; Llama-style layers need the
    RoPE cos/sin (recomputed here from position_ids) and an explicit causal mask each stage."""
    sid = payload.get("id"); shard = SHARDS.get(sid)
    if shard is None:
        raise RuntimeError(f"shard {sid} not loaded — call shard_load first")
    arch = shard["arch"]; first, last = shard["first"], shard["last"]
    hidden_dim = shard["hidden_dim"]
    with torch.no_grad():
        if first:
            ids_list = payload["input_ids"]
            _check_ids(shard["vocab_size"], ids_list)
            if len(ids_list) > shard["n_positions"]:  # positions past the context window overflow the model
                raise ValueError(f"sequence length {len(ids_list)} exceeds the model's context window {shard['n_positions']}")
            ids = torch.tensor(ids_list, dtype=torch.long, device=DEV).unsqueeze(0)  # [1, seq]
            seq = ids.shape[1]
        else:
            seq = int(payload["seq"])
            h = b64_to_t(payload["hidden"]).reshape(1, seq, hidden_dim).to(DEV)
        if arch == "gpt2":
            if first:
                pos = torch.arange(seq, dtype=torch.long, device=DEV)
                h = shard["drop"](shard["wte"](ids) + shard["wpe"](pos))  # [1, seq, n_embd]
            for blk in shard["blocks"]:  # GPT2Block applies causal self-attention internally (no mask needed)
                h = blk(h)[0]
            if last:
                h = shard["ln_f"](h)
        else:  # llama-style: recompute RoPE cos/sin + a causal mask exactly as LlamaModel.forward does,
            # TODO(review): this feeds ONE full-causal mask to every layer, which is correct for standard
            # Llama/Qwen/SmolLM/TinyLlama but WRONG for sliding-window / mixed-attention models (Mistral,
            # Gemma-2/3, sliding Qwen2): those need per-layer masks dispatched by decoder_layer.attention_type
            # (create_sliding_window_causal_mask for the sliding layers). Reject such configs until supported.
            from transformers.masking_utils import create_causal_mask  # so the layers see identical inputs
            if first:
                h = shard["embed_tokens"](ids)  # [1, seq, hidden] — RoPE is applied per-layer, not here
            cache_position = torch.arange(seq, dtype=torch.long, device=DEV)
            position_ids = cache_position.unsqueeze(0)  # [1, seq]
            pos_emb = shard["rotary_emb"](h, position_ids)  # (cos, sin) — depends only on positions + head_dim
            causal_mask = create_causal_mask(config=shard["config"], input_embeds=h, attention_mask=None,
                                             cache_position=cache_position, past_key_values=None,
                                             position_ids=position_ids)
            for blk in shard["blocks"]:  # LlamaDecoderLayer.forward returns the hidden tensor directly
                h = blk(h, attention_mask=causal_mask, position_ids=position_ids, position_embeddings=pos_emb)
            if last:
                h = shard["norm"](h)
        if last:
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

def model_chat(payload: dict) -> dict:
    """Text in → text out. Tokenizes on the worker (it has the model's tokenizer), so a plain browser chat
    page can talk to a served model without a JS tokenizer. Applies the model's chat template when present."""
    mid = payload.get("id"); model = MODELS.get(mid)
    if model is None:
        raise RuntimeError(f"model {mid} not loaded — call /model/load first")
    from transformers import AutoTokenizer
    if mid not in _TOKS:
        _TOKS[mid] = AutoTokenizer.from_pretrained(MODEL_NAMES.get(mid, mid))
    tk = _TOKS[mid]
    if tk is None:  # download-free push shipped no tokenizer files — don't reach the hub; steer to token-id API
        raise RuntimeError("no tokenizer for this download-free model — use /model/generate with token ids, "
                           "or re-push including the tokenizer files")
    prompt = str(payload.get("prompt", ""))[:8000]
    if getattr(tk, "chat_template", None):  # instruct models (e.g. Qwen-Instruct) → use their chat format
        text = tk.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    enc = tk(text, return_tensors="pt").to(DEV)
    seq = enc["input_ids"].shape[1]
    n = max(1, min(int(payload.get("max_new_tokens", 96)), 1024))
    _check_ctx(model, seq, n)
    MODELS[mid] = MODELS.pop(mid)
    kw = {"max_new_tokens": n, "pad_token_id": tk.eos_token_id}
    if payload.get("do_sample"):
        kw.update(do_sample=True, temperature=float(payload.get("temperature", 0.7)), top_p=float(payload.get("top_p", 0.9)))
    else:
        kw["do_sample"] = False
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    if DEV == "cuda": torch.cuda.synchronize()
    elif DEV == "mps": torch.mps.synchronize()
    reply = tk.decode(out[0, seq:], skip_special_tokens=True)
    return {"ok": True, "text": reply, "n": int(out.shape[1] - seq), "ms": (time.perf_counter() - t0) * 1000}

def model_dispatch(op: str, payload: dict) -> dict:
    if op == "load": return model_load(payload)
    if op == "push_begin": return model_push_begin(payload)
    if op == "push_chunk": return model_push_chunk(payload)
    if op == "push_end": return model_push_end(payload)
    if op == "forward": return model_forward(payload)
    if op == "generate": return model_generate(payload)
    if op == "chat": return model_chat(payload)
    if op == "arch": return model_arch(payload)
    if op == "unload": _push_cleanup(payload.get("id")); MODELS.pop(payload.get("id"), None); MODEL_NAMES.pop(payload.get("id"), None); _TOKS.pop(payload.get("id"), None); _empty_cache(); return {"ok": True}
    if op == "shard_load": return shard_load(payload)
    if op == "shard_forward": return shard_forward(payload)
    if op == "shard_unload": _push_cleanup(payload.get("id")); SHARDS.pop(payload.get("id"), None); _empty_cache(); return {"ok": True}
    raise ValueError(f"unknown model op {op}")

async def run():
    loop = asyncio.get_event_loop()
    paused = False; pause_reason = ""; ceil = 0.6
    while True:
        try:
            # WAN-tolerant keepalive: over a high-latency tunnel the default 20s ping timeout drops the
            # socket (the very low-bandwidth/WAN case MoreGPU targets). Tune generously + env-overridable.
            async with websockets.connect(
                A.server, max_size=None,
                ping_interval=float(os.environ.get("MOREGPU_WS_PING_INTERVAL", "30")),
                ping_timeout=float(os.environ.get("MOREGPU_WS_PING_TIMEOUT", "90")),
                close_timeout=10,
            ) as ws:
                await ws.send(json.dumps({"t": "register", "joinToken": A.token, "pubkey": PUBKEY_B64,
                                          "node": {"id": NAME, "backend": NODE_BACKEND, "label": BACKEND, "os": platform.system().lower()}}))
                key = None
                # websockets requires writes to be serialized, and handlers now run concurrently (below), so
                # every send goes through this lock.
                send_lock = asyncio.Lock()
                inflight_tasks: set = set()
                async def ws_send(obj):
                    async with send_lock:
                        await ws.send(json.dumps(obj))
                def spawn(coro):
                    # Dispatch a handler OFF the read loop so one long generate/train can't head-of-line-block
                    # frame reception or heartbeats. Compute still funnels through the single-thread TORCH_POOL,
                    # so MPS access stays serialized — only the awaiting is concurrent, keeping the loop live.
                    task = asyncio.ensure_future(coro)
                    inflight_tasks.add(task); task.add_done_callback(inflight_tasks.discard)

                async def handle_assign(mm):
                    t0 = time.perf_counter()
                    try:
                        req = json.loads(unseal(key, mm["sealedIn"]).decode())
                        out = await loop.run_in_executor(TORCH_POOL, compute_shard, req)  # off the event loop
                        sealed = seal(key, json.dumps({"out": f32_to_b64(out)}).encode())
                        await ws_send({"t": "result", "shardId": mm["shardId"], "jobId": mm["jobId"], "ok": True,
                                       "sealedOut": sealed, "sig": sign_result(mm["shardId"], sealed),
                                       "ms": (time.perf_counter() - t0) * 1000, "backend": BACKEND})
                    except Exception as e:
                        await ws_send({"t": "result", "shardId": mm.get("shardId"), "jobId": mm.get("jobId"), "ok": False, "error": str(e)})

                async def handle_relay(kind, mm):  # relayed RPCs: fine-tuning ('train') / resident serving + pipeline sharding ('model')
                    reqId = mm.get("reqId"); reply = f"{kind}_reply"
                    dispatch = train_dispatch if kind == "train" else model_dispatch
                    try:
                        payload = json.loads(unseal(key, mm["sealed"]).decode())
                        res = await loop.run_in_executor(TORCH_POOL, dispatch, mm["op"], payload)
                        sealed = seal(key, json.dumps(res).encode())
                        await ws_send({"t": reply, "reqId": reqId, "ok": True, "sealed": sealed})
                    except Exception as e:
                        await ws_send({"t": reply, "reqId": reqId, "ok": False, "error": str(e)})

                async def handle_cache(mm):
                    try:
                        w = json.loads(unseal(key, mm["sealed"]).decode())
                        rows, cols = int(w["rows"]), int(w["cols"])
                        if w.get("dtype") == "f16":
                            ten = torch.from_numpy(np.frombuffer(b64d(w["data"]), dtype="<f2").copy()).reshape(rows, cols).to(DEV)
                        else:
                            ten = b64_to_t(w["data"]).reshape(rows, cols).to(DEV)
                        resident[mm["id"]] = ten
                        await ws_send({"t": "cached", "id": mm["id"], "ok": True})
                    except Exception as e:
                        await ws_send({"t": "cached", "id": mm.get("id"), "ok": False, "error": str(e)})

                async def heartbeat():
                    while True:
                        await asyncio.sleep(4)
                        try:
                            await ws_send({"t": "heartbeat", "id": NAME, "load1": 0, "cores": os.cpu_count() or 4,
                                           "util": 0.0, "duty": 0.0 if paused else ceil, "ceil": ceil,
                                           "paused": paused, "pausedReason": pause_reason, "schedule": "always"})
                        except Exception:
                            return
                hb = asyncio.ensure_future(heartbeat())
                try:
                    async for raw in ws:
                        # A single malformed/unexpected frame must NOT tear down the session (that would wipe
                        # all resident models/weights) — parse + dispatch under a guard, log and skip on error.
                        try:
                            m = json.loads(raw); t = m.get("t")
                            if t == "denied":
                                reason = str(m.get("reason", ""))
                                # Only a permanent rejection is fatal. A transient 'worker id already registered'
                                # (e.g. after a network blip, before the coordinator reaped our old socket) must
                                # retry the connect loop instead of exiting the worker permanently.
                                if ("bad join token" in reason) or ("removed by admin" in reason):
                                    print(f"[torch-worker] rejected (fatal): {reason}"); return
                                print(f"[torch-worker] rejected: {reason} — retrying in 5s"); await asyncio.sleep(5); break
                            elif t == "welcome":
                                key = b64d(m["tenantKeyB64"]); ceil = float(m.get("duty", 0.6))
                                # fresh session: the coordinator forgets our resident state on disconnect, so a
                                # reconnecting worker starts clean too (no stale models/weights pinned in VRAM).
                                for _pid in list(PUSH): _push_cleanup(_pid)  # drop any staged (un-finished) weight pushes
                                resident.clear(); MODELS.clear(); SHARDS.clear(); TRAIN.update(model=None, opt=None, step=0, trainable={}); _empty_cache()
                                print(f"[torch-worker] joined pool on {DEV} · duty ceiling {int(ceil*100)}%")
                            elif t == "control":
                                if "pause" in m: paused = bool(m["pause"]); pause_reason = "admin" if paused else ""
                                if "ceil" in m and m["ceil"] is not None: ceil = float(m["ceil"])
                                print(f"[torch-worker] admin control → {'PAUSED' if paused else 'active'} · ceiling {int(ceil*100)}%")
                            elif t == "uncache":
                                resident.pop(m.get("id"), None)
                            elif t == "cache":
                                spawn(handle_cache(m))
                            elif t == "assign":
                                spawn(handle_assign(m))
                            elif t in ("train", "model"):
                                spawn(handle_relay(t, m))
                        except Exception as e:
                            print(f"[torch-worker] skipped bad frame: {e}")
                finally:
                    hb.cancel()
                    for task in list(inflight_tasks): task.cancel()
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
