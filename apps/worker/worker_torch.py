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
import argparse, asyncio, base64, json, os, platform, socket, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
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

# OPT-IN peer worker->worker activation transport (default OFF → relay path is unchanged). See docs/ROADMAP.md.
PEER_TRANSPORT = os.environ.get("MOREGPU_PEER_TRANSPORT", "").lower() in ("1", "true", "yes", "on")

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

# Compute ops that count toward the node's DUTY (the admin slider's ceiling throttles these — a kernel shard,
# a resident/pipeline forward, a training step). Transfer/control ops (push_*/ping/load/unload/arch) are never
# throttled (throttling a weight stream would just make loading slower for no benefit).
_PACED_OPS = {"forward", "generate", "chat", "shard_forward", "inner", "step"}
def _paced(fn, args, ceil_val, pace_it=True):
    """Run a compute op on the TORCH_POOL thread, then, if the duty ceiling < 100%, sleep so the thread is busy
    at most `ceil_val` of the time — a real duty-cycle throttle. Since the pool is single-threaded, sleeping here
    genuinely paces ALL of this node's compute (kernels + serving + sharding), so the admin slider now DOES
    something: a lower ceiling → the node contributes more slowly, leaving the rest of its time to its owner."""
    t0 = time.perf_counter(); r = fn(*args); dt = time.perf_counter() - t0
    if pace_it and ceil_val < 0.999:
        pace = min(10.0, dt * (1.0 / max(0.05, ceil_val) - 1.0))  # idle time to hold busy-fraction ≈ ceil (capped 10s)
        if pace > 0.001:
            time.sleep(pace)
    return r
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
    st = PUSH.get(mid)
    # RESUME: a stage that dropped mid-stream reconnects and we keep its partial staging — report the bytes
    # already written per file so the coordinator re-streams only the tail (append continues each file).
    if payload.get("resume") and st is not None and os.path.isdir(st["dir"]):
        sizes = {}
        for fn in os.listdir(st["dir"]):
            try: sizes[fn] = os.path.getsize(os.path.join(st["dir"], fn))
            except OSError: pass
        st["bytes"] = sum(sizes.values())  # re-sync the cap counter to what's actually on disk
        return {"ok": True, "id": mid, "staging": "ram" if st["ram"] else "disk", "resumed": True, "sizes": sizes}
    _push_cleanup(mid)  # fresh push: drop any stale half-streamed staging for this id
    root = _stage_root()
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in str(mid))[:40]
    d = tempfile.mkdtemp(prefix=f"moregpu-{safe}-", dir=root)
    PUSH[mid] = {"dir": d, "bytes": 0, "ram": root == "/dev/shm"}
    return {"ok": True, "id": mid, "staging": "ram" if root == "/dev/shm" else "disk", "resumed": False, "sizes": {}}

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
SHARD_TOKS: dict = {}  # shard id -> tokenizer (only on the FIRST stage) so a browser can chat a sharded model
# --- per-stage incremental KV CACHE (the fix for the O(n^2) shard decode) ---------------------------------
# In "cached" mode each stage keeps its OWN layers' past_key_values (an HF DynamicCache) across shard_forward
# calls, keyed by (shard id, session id). A decode step then feeds only the NEW token (first stage) / its hidden
# (later stages) instead of the whole growing prefix, and each stage attends to its cached past — turning the
# re-run-the-prefix decode into a true single-token step. The stage's decoder blocks are re-indexed 0-based at
# shard_load (see shard_load) so this per-stage cache is self-contained: DynamicCache indexes layers from 0 and
# create_causal_mask reads get_mask_sizes(cache_position, layer_idx=0) off THIS stage's first real layer.
SHARD_KV: dict = {}  # (shard id, session id) -> HF DynamicCache holding just this stage's layers' K/V
MAX_KV_SESSIONS = int(os.environ.get("MOREGPU_MAX_KV_SESSIONS", "8"))  # bound VRAM: LRU-evict live decode caches

def _kv_get(sid: str, session: str, fresh: bool):
    """Fetch (or lazily create) THIS stage's DynamicCache for (sid, session); `fresh` drops any prior cache first
    (a pos-0 prefill restarts the session). LRU-evicts the oldest session when over MAX_KV_SESSIONS."""
    from transformers import DynamicCache
    key = (sid, session)
    if fresh:
        SHARD_KV.pop(key, None)
    kv = SHARD_KV.pop(key, None)  # pop+reinsert = mark most-recently-used (dict preserves insertion order)
    if kv is None:
        while len(SHARD_KV) >= MAX_KV_SESSIONS:
            SHARD_KV.pop(next(iter(SHARD_KV)), None)  # evict the LRU (oldest) session's cache
        kv = DynamicCache()
    SHARD_KV[key] = kv
    return kv

def _kv_drop(sid, session=None) -> int:
    """Evict a session's cache (or, if session is None, every session of this shard). Returns count evicted."""
    keys = [k for k in SHARD_KV if k[0] == sid and (session is None or k[1] == session)]
    for k in keys:
        SHARD_KV.pop(k, None)
    return len(keys)

def shard_tok(payload: dict) -> dict:
    """Tokenize a text prompt for a sharded model (runs on the FIRST stage, which holds the tokenizer). Applies
    the chat template when present. Returns input_ids so the coordinator can pipe them through the shard."""
    sid = payload.get("id"); tk = SHARD_TOKS.get(sid)
    if tk is None:
        raise RuntimeError("no tokenizer for this shard — re-shard (the tokenizer is streamed to the first stage)")
    prompt = str(payload.get("prompt", ""))[:8000]
    text = tk.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) \
        if getattr(tk, "chat_template", None) else prompt
    ids = tk(text, return_tensors=None)["input_ids"]
    return {"ok": True, "input_ids": ids, "eos": tk.eos_token_id}

def shard_detok(payload: dict) -> dict:
    """Decode generated token ids back to text (runs on the FIRST stage's tokenizer)."""
    sid = payload.get("id"); tk = SHARD_TOKS.get(sid)
    if tk is None:
        raise RuntimeError("no tokenizer for this shard")
    return {"ok": True, "text": tk.decode(payload.get("tokens", []), skip_special_tokens=True)}


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
        # Loading a PARTIAL checkpoint (only this stage's tensors) makes transformers log a scary "checkpoint
        # seems corrupted / keys MISSING" report — but that's BY DESIGN (the other stages' weights live on other
        # nodes; the missing ones here are random-init then dropped when we slice to our blocks). Quiet it so it
        # doesn't look like a failure; a genuine load error still raises.
        try:
            from transformers.utils import logging as _tflog; _tflog.set_verbosity_error()
        except Exception:
            pass
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
        for _li, _blk in enumerate(shard["blocks"]):  # re-index this stage's blocks 0-based so a per-stage
            _blk.attn.layer_idx = _li                 # DynamicCache (incremental decode) is self-contained
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
        for _li, _blk in enumerate(shard["blocks"]):  # re-index 0-based (see gpt2 branch) so this stage's KV
            _blk.self_attn.layer_idx = _li            # cache indexes from 0 and create_causal_mask reads it
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
    SHARDS.pop(sid, None); _kv_drop(sid)  # replacing a live shard: drop the old modules + any live KV sessions
    if len(SHARDS) >= MAX_RESIDENT_MODELS:  # bound VRAM, same as resident models
        SHARDS.pop(next(iter(SHARDS)), None)
    del base, model
    _empty_cache()
    SHARDS[sid] = shard
    tok_ok = False
    if first:
        # The FIRST stage keeps the tokenizer so a browser can chat a SHARDED model (tokenize prompt → ids here,
        # pipe ids through the stages, decode the output here). from a push shard it's in the staged dir; else HF.
        SHARD_TOKS.pop(sid, None)
        try:
            from transformers import AutoTokenizer
            src = PUSH[sid]["dir"] if (cfg.get("push") and sid in PUSH) else cfg["model"]
            SHARD_TOKS[sid] = AutoTokenizer.from_pretrained(src, local_files_only=bool(cfg.get("push")))
            tok_ok = True
        except Exception as e:
            print(f"[torch-worker] shard {sid}: tokenizer unavailable ({e}) — sharded chat disabled, token-id path still works")
    if cfg.get("push"):
        _push_cleanup(sid)  # stage weights are now sliced into SHARDS — drop the staged partial safetensors
    return {"ok": True, "id": sid, "layers": [start, end], "first": first, "last": last, "tokenizer": tok_ok,
            "n_layer": shard["n_layer"], "arch": arch, "params_held": params_held, "mode": "download-free" if cfg.get("push") else "download"}

def shard_forward(payload: dict) -> dict:
    """Run ONE stage's blocks. If first: embed input_ids → hidden; else: reshape the piped-in hidden.
    If last: final norm + LM head → {argmax, logits?}; else: return the hidden state for the next stage.
    Branches on the stored arch: GPT-2 blocks mask causally on their own; Llama-style layers need the
    RoPE cos/sin (recomputed here from position_ids) and an explicit causal mask each stage.

    INCREMENTAL KV CACHE: when payload carries a session id in "cached" mode, this stage keeps its own layers'
    past_key_values across calls (SHARD_KV[(sid, session)]). A pos-0 call is the PREFILL (whole prompt, cache
    restarts empty); a pos>0 call is a DECODE step feeding only the NEW token / hidden (seq==1). Positions are
    taken from the cache length so RoPE / wpe / the causal mask all offset by the cached prefix — the cached
    greedy stream is therefore token-identical to the uncached one, at O(n) instead of O(n^2). When cached is
    off (no session), the code path below is byte-identical to the original stateless forward."""
    sid = payload.get("id"); shard = SHARDS.get(sid)
    if shard is None:
        raise RuntimeError(f"shard {sid} not loaded — call shard_load first")
    arch = shard["arch"]; first, last = shard["first"], shard["last"]
    hidden_dim = shard["hidden_dim"]
    session = payload.get("session")
    cached = bool(payload.get("cached")) and session is not None
    kv = None; past_len = 0
    if cached:
        pos = int(payload.get("pos", 0))
        kv = _kv_get(sid, session, fresh=(pos == 0))  # pos 0 == prefill → restart this session's cache
        past_len = kv.get_seq_length(0)               # tokens already cached on THIS stage (0 for a fresh prefill)
        # FAULT TOLERANCE: a stage that dropped AFTER load lost its LIVE KV (a fresh `welcome` clears SHARD_KV)
        # while the coordinator still holds the shard plan → a decode step then arrives with pos>0 but an empty
        # cache here. Surface it clearly so the coordinator re-prefills, instead of silently emitting garbage.
        if past_len != pos:
            raise RuntimeError(f"KV desync for session {session!r}: stage holds {past_len} cached tokens but the "
                               f"coordinator sent pos={pos} — this stage lost its live KV (reconnect/evict); re-prefill")
    with torch.no_grad():
        if first:
            ids_list = payload["input_ids"]
            _check_ids(shard["vocab_size"], ids_list)
            if past_len + len(ids_list) > shard["n_positions"]:  # positions past the context window overflow the model
                raise ValueError(f"sequence length {past_len + len(ids_list)} exceeds the model's context window {shard['n_positions']}")
            ids = torch.tensor(ids_list, dtype=torch.long, device=DEV).unsqueeze(0)  # [1, seq]
            seq = ids.shape[1]
        else:
            seq = int(payload["seq"])
            h = b64_to_t(payload["hidden"]).reshape(1, seq, hidden_dim).to(DEV)
        # absolute positions of THIS call's tokens: [past_len .. past_len+seq). For an uncached call past_len==0,
        # so cache_position == arange(seq) and every path below reduces to the original stateless numerics.
        cache_position = torch.arange(past_len, past_len + seq, dtype=torch.long, device=DEV)
        if arch == "gpt2":
            if first:
                h = shard["drop"](shard["wte"](ids) + shard["wpe"](cache_position))  # [1, seq, n_embd]
            for blk in shard["blocks"]:  # GPT2Block masks causally on its own: attn_mask None → is_causal for a
                kw = {"past_key_values": kv, "use_cache": True, "cache_position": cache_position} if cached else {}
                out = blk(h, **kw)  # multi-token prefill, and is_causal off for a 1-token decode (attends to all cached kv)
                h = out[0] if isinstance(out, (tuple, list)) else out  # older transformers: (hidden, …); ≥4.54: tensor
            if last:
                h = shard["ln_f"](h)
        else:  # llama-style: recompute RoPE cos/sin + a causal mask exactly as LlamaModel.forward does,
            # TODO(review): this feeds ONE full-causal mask to every layer, which is correct for standard
            # Llama/Qwen/SmolLM/TinyLlama but WRONG for sliding-window / mixed-attention models (Mistral,
            # Gemma-2/3, sliding Qwen2): those need per-layer masks dispatched by decoder_layer.attention_type
            # (create_sliding_window_causal_mask for the sliding layers). Reject such configs until supported.
            from transformers.masking_utils import create_causal_mask  # so the layers see identical inputs
            import inspect
            if first:
                h = shard["embed_tokens"](ids)  # [1, seq, hidden] — RoPE is applied per-layer, not here
            position_ids = cache_position.unsqueeze(0)  # [1, seq] — absolute (offset by the cached prefix)
            pos_emb = shard["rotary_emb"](h, position_ids)  # (cos, sin) — depends only on positions + head_dim
            # transformers renamed input_embeds -> inputs_embeds (and trims kwargs across versions) — pick by signature.
            # Passing the cache lets create_causal_mask size the mask as [q=seq, kv=past_len+seq] for a decode step.
            _msig = inspect.signature(create_causal_mask).parameters
            _mkw = {"config": shard["config"], "attention_mask": None, "cache_position": cache_position,
                    "past_key_values": (kv if cached else None), "position_ids": position_ids,
                    ("inputs_embeds" if "inputs_embeds" in _msig else "input_embeds"): h}
            causal_mask = create_causal_mask(**{k: v for k, v in _mkw.items() if k in _msig})
            for blk in shard["blocks"]:  # LlamaDecoderLayer.forward returns the hidden tensor directly
                kw = {"attention_mask": causal_mask, "position_ids": position_ids, "position_embeddings": pos_emb}
                if cached:
                    kw.update(past_key_values=kv, use_cache=True, cache_position=cache_position)
                h = blk(h, **kw)
            if last:
                h = shard["norm"](h)
        if last:
            logits = shard["lm_head"](h)[0, -1].float()  # last-token logits [vocab]
            res = {"ok": True, "argmax": int(logits.argmax().item())}
            if payload.get("return_logits"):
                res["logits"] = f32_to_b64(logits)
        else:
            res = {"ok": True, "hidden": f32_to_b64(h.flatten()), "seq": seq, "hidden_dim": hidden_dim}
    if cached:
        res["past"] = int(past_len + seq)  # cache length after this call == the coordinator's next pos
    if DEV == "cuda": torch.cuda.synchronize()
    elif DEV == "mps": torch.mps.synchronize()
    return res

def shard_reset(payload: dict) -> dict:
    """Evict a live decode KV cache: {id, session?} — drop one session, or (session omitted) every session of the
    shard. The coordinator calls this after a cached chat/generate finishes, or to force a clean re-prefill."""
    sid = payload.get("id")
    n = _kv_drop(sid, payload.get("session"))
    _empty_cache()
    return {"ok": True, "id": sid, "evicted": n}

# ── EXPERT PARALLELISM (MoE) — see docs/ROADMAP.md "MoE expert parallelism (the Kimi path)". FIRST verifiable
# increment. A worker plays ONE of two roles for a routed-MoE model, both loaded via the SAME partial-checkpoint
# push path as shard_load (missing tensors random-init, then dropped/ignored — the worker never touches the hub):
#   • BACKBONE holder — the DENSE lane: token embedding, every attention block, both layernorms, the router
#     `mlp.gate`, the final norm + LM head. Its routed FFN is NOT resident; each MoE layer's expert `mlp` is
#     swapped for a proxy that keeps the real router, captures the block's normed input, and returns ZEROS for
#     the FFN — so re-running the REAL decoder layer yields exactly the post-attention residual and the routed
#     FFN is supplied later by the remote holders.
#   • EXPERT holder — the SPARSE plane: a SUBSET of the routed experts (the same expert-index set across every
#     layer), resident. It runs the routed FFN for ONLY its resident experts on the tokens routed to them.
# The coordinator drives one forward LAYER-BY-LAYER (moe_embed → per-layer moe_route → dispatch to holders'
# expert_forward → moe_apply → moe_head), RELAYING the router dispatch/combine through ITSELF. A peer-mesh
# all-to-all (coordinator off the per-token data path) is a LATER increment — this coordinator relay is exactly
# the SPOF/throughput bottleneck the mesh will remove (roadmap §4). Correctness-first: numerics identical to the
# un-sharded model, comms not yet optimal.
MOE_BB: dict = {}       # sid -> backbone state (the whole model + swapped-in router proxies)
MOE_EXPERTS: dict = {}  # sid -> {"H": hidden, "experts": {(layer, E): nn.Module}, "ids": [E...]}

class _MoEBackboneProxy(nn.Module):
    """Stand-in for a routed-MoE layer's expert block on the BACKBONE. Keeps the real router `gate`, captures the
    block's (normed) input and the router logits, and returns ZEROS for the routed-FFN output. Mirrors the
    (final_hidden_states, router_logits) return contract of OlmoeSparseMoeBlock / Qwen2MoeSparseMoeBlock, so the
    UNCHANGED decoder layer runs its attention + residual normally and the FFN contribution is added back later
    from the remote expert holders (via moe_apply)."""
    def __init__(self, gate: nn.Module):
        super().__init__(); self.gate = gate; self.cap = None; self.router_logits = None
    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        flat = hidden_states.reshape(-1, H)
        self.cap = flat                       # [B*S, H] — the routed-FFN input (post_attention_layernorm output)
        self.router_logits = self.gate(flat)  # [B*S, n_experts] — router, kept dense on the backbone
        return torch.zeros(B, S, H, dtype=hidden_states.dtype, device=hidden_states.device), self.router_logits

def _moe_arch_ok(model) -> bool:
    return hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model, "rotary_emb")

def moe_backbone_load(cfg: dict) -> dict:
    """Load a routed-MoE causal LM (Llama-style: model.layers + RoPE + RMSNorm — OLMoE / Qwen2-MoE), KEEP the
    dense backbone, and swap every layer's expert block for a router-only proxy (see _MoEBackboneProxy). Loaded
    from the streamed BACKBONE-only partial checkpoint (experts absent → random-init, then discarded by the
    proxy swap) so the worker downloads nothing. dtype float32 (parity reference is fp32; real deploys use fp16)."""
    from transformers import AutoModelForCausalLM
    sid = cfg["id"]
    if cfg.get("push"):
        st = PUSH.get(sid)
        if st is None:
            raise RuntimeError("moe backbone weights not staged — push_begin/push_chunk must precede a push moe_backbone_load")
        try:
            from transformers.utils import logging as _tflog; _tflog.set_verbosity_error()  # quiet the partial-checkpoint report (by design)
        except Exception:
            pass
        model = AutoModelForCausalLM.from_pretrained(st["dir"], dtype=torch.float32, local_files_only=True).to(DEV).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    if not _moe_arch_ok(model):
        raise ValueError(f"MoE EP needs a Llama-style routed-MoE (model.layers + rotary_emb), got {type(model).__name__}")
    mm = model.model; mc = model.config
    proxies = []
    for layer in mm.layers:
        if not hasattr(layer.mlp, "gate"):
            raise ValueError("a decoder layer's mlp has no router `gate` — not a routed-MoE model")
        p = _MoEBackboneProxy(layer.mlp.gate); layer.mlp = p; proxies.append(p)  # drop the (random) experts; keep the router
    n_experts = int(getattr(mc, "num_experts", getattr(mc, "n_routed_experts", getattr(mc, "num_local_experts", 0))))
    topk = int(getattr(mc, "num_experts_per_tok", getattr(mc, "num_experts_per_token", 0)))
    if not (n_experts > 0 and topk > 0):
        raise ValueError(f"could not read routed-expert count / top-k from config ({n_experts=}, {topk=})")
    MOE_BB.pop(sid, None)
    if len(MOE_BB) >= MAX_RESIDENT_MODELS:
        MOE_BB.pop(next(iter(MOE_BB)), None)
    MOE_BB[sid] = {"model": model, "mm": mm, "layers": mm.layers, "proxies": proxies, "config": mc,
                   "n_layer": int(mc.num_hidden_layers), "hidden": int(mc.hidden_size), "vocab": int(mc.vocab_size),
                   "n_experts": n_experts, "topk": topk, "norm_topk": bool(getattr(mc, "norm_topk_prob", False))}
    if cfg.get("push"):
        _push_cleanup(sid)
    _empty_cache()
    return {"ok": True, "id": sid, "role": "backbone", "n_layer": int(mc.num_hidden_layers), "n_experts": n_experts,
            "topk": topk, "hidden": int(mc.hidden_size), "params_held": sum(p.numel() for p in model.parameters())}

def expert_load(cfg: dict) -> dict:
    """Load ONLY a SUBSET of a routed-MoE's experts, resident. Built from the streamed EXPERT-subset partial
    checkpoint (all non-expert tensors + non-resident experts random-init → dropped when we keep just our expert
    modules). {id, experts:[E...], model?, push?}. This is the download-free per-expert placement the roadmap's
    stageExpertTensors selects (one expert's gate/up/down_proj may live in different source files; the coordinator
    merged the selected tensors into this stage's model.safetensors)."""
    from transformers import AutoModelForCausalLM
    sid = cfg["id"]; experts = sorted(int(e) for e in cfg["experts"])
    if cfg.get("push"):
        st = PUSH.get(sid)
        if st is None:
            raise RuntimeError("expert weights not staged — push_begin/push_chunk must precede a push expert_load")
        try:
            from transformers.utils import logging as _tflog; _tflog.set_verbosity_error()
        except Exception:
            pass
        model = AutoModelForCausalLM.from_pretrained(st["dir"], dtype=torch.float32, local_files_only=True).to(DEV).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    if not _moe_arch_ok(model):
        raise ValueError(f"MoE EP needs a Llama-style routed-MoE, got {type(model).__name__}")
    mm = model.model; mc = model.config; n_layer = int(mc.num_hidden_layers)
    kept: dict = {}
    for L in range(n_layer):
        experts_ml = mm.layers[L].mlp.experts  # nn.ModuleList indexed by GLOBAL expert id
        for E in experts:
            kept[(L, E)] = experts_ml[E]       # reference the resident expert module (survives the model drop below)
    MOE_EXPERTS.pop(sid, None)
    if len(MOE_EXPERTS) >= MAX_RESIDENT_MODELS:
        MOE_EXPERTS.pop(next(iter(MOE_EXPERTS)), None)
    params = sum(p.numel() for m in kept.values() for p in m.parameters())
    MOE_EXPERTS[sid] = {"H": int(mc.hidden_size), "experts": kept, "ids": experts}
    del model, mm  # our `kept` references keep only the resident experts alive; the rest (attn/embed/other experts) is freed
    if cfg.get("push"):
        _push_cleanup(sid)
    _empty_cache()
    return {"ok": True, "id": sid, "role": "expert_holder", "experts": experts, "params_held": params}

def _moe_bb(sid):
    bb = MOE_BB.get(sid)
    if bb is None:
        raise RuntimeError(f"moe backbone {sid} not loaded — call moe_backbone_load first")
    return bb

def moe_embed(payload: dict) -> dict:
    """BACKBONE: token ids → hidden [1,S,H] (the input embedding). Start of one MoE forward."""
    bb = _moe_bb(payload["id"]); mm = bb["mm"]
    ids_list = payload["input_ids"]; _check_ids(bb["vocab"], ids_list)
    if len(ids_list) > bb["config"].max_position_embeddings:
        raise ValueError(f"sequence length {len(ids_list)} exceeds context window {bb['config'].max_position_embeddings}")
    with torch.no_grad():
        ids = torch.tensor(ids_list, dtype=torch.long, device=DEV).unsqueeze(0)
        h = mm.embed_tokens(ids)
    return {"ok": True, "hidden": f32_to_b64(h.flatten()), "seq": int(ids.shape[1])}

def moe_route(payload: dict) -> dict:
    """BACKBONE, one MoE layer: run the REAL decoder layer's attention (RoPE + causal mask recomputed exactly as
    the un-sharded model does, mirroring shard_forward's llama path) with the expert FFN proxied to ZERO — so the
    returned hidden is the post-attention residual (attn_hidden). Also returns the routed-FFN input (moe_in) and
    the top-k routing (ids + weights, softmaxed in fp32 + optional norm, exactly like the reference sparse block).
    The coordinator dispatches moe_in + routing to the expert holders and combines via moe_apply."""
    bb = _moe_bb(payload["id"]); mm = bb["mm"]; H = bb["hidden"]
    L = int(payload["layer"]); seq = int(payload["seq"])
    from transformers.masking_utils import create_causal_mask
    import inspect
    with torch.no_grad():
        h = b64_to_t(payload["hidden"]).reshape(1, seq, H).to(DEV)
        cache_position = torch.arange(0, seq, dtype=torch.long, device=DEV)
        position_ids = cache_position.unsqueeze(0)
        pos_emb = mm.rotary_emb(h, position_ids)  # (cos, sin) — depends only on positions + head_dim
        _msig = inspect.signature(create_causal_mask).parameters
        _mkw = {"config": bb["config"], "attention_mask": None, "cache_position": cache_position,
                "past_key_values": None, "position_ids": position_ids,
                ("inputs_embeds" if "inputs_embeds" in _msig else "input_embeds"): h}
        cmask = create_causal_mask(**{k: v for k, v in _mkw.items() if k in _msig})
        p = bb["proxies"][L]; p.cap = None; p.router_logits = None
        out = bb["layers"][L](h, attention_mask=cmask, position_ids=position_ids, position_embeddings=pos_emb)
        attn_hidden = out[0]                # [1,S,H] — mlp proxy returned 0, so this IS the post-attn residual base
        moe_in = p.cap                      # [S,H] — post_attention_layernorm output (the routed-FFN input)
        routing = torch.softmax(p.router_logits, dim=1, dtype=torch.float)  # fp32 softmax, exactly like the ref block
        topw, topi = torch.topk(routing, bb["topk"], dim=-1)
        if bb["norm_topk"]:
            topw = topw / topw.sum(dim=-1, keepdim=True)
        topw = topw.to(moe_in.dtype)        # cast back to hidden dtype, like the reference
    return {"ok": True, "attn_hidden": f32_to_b64(attn_hidden.flatten()), "moe_in": f32_to_b64(moe_in.flatten()),
            "topk_i": [int(x) for x in topi.flatten().tolist()], "topk_w": f32_to_b64(topw.flatten()),
            "seq": seq, "k": int(bb["topk"]), "n_experts": bb["n_experts"]}

def expert_forward(payload: dict) -> dict:
    """EXPERT holder, one MoE layer: run the routed FFN for ONLY this holder's resident experts (subset given in
    `experts`) on the tokens routed to them, weighted by the router weight, and return the per-token PARTIAL sum
    [S,H] (zero for tokens/slots not routed to a resident expert). Summing all holders' partials reconstructs the
    layer's full routed-FFN output. This mirrors the reference sparse block's per-expert gather → weight →
    index_add, restricted to the resident experts (index_add in ascending expert order, matching the reference)."""
    st = MOE_EXPERTS.get(payload["id"])
    if st is None:
        raise RuntimeError(f"expert holder {payload['id']} not loaded — call expert_load first")
    H = st["H"]; kept = st["experts"]
    L = int(payload["layer"]); seq = int(payload["seq"]); k = int(payload["k"])
    experts = sorted(int(e) for e in payload["experts"])
    with torch.no_grad():
        moe_in = b64_to_t(payload["hidden"]).reshape(seq, H).to(DEV)
        topi = torch.tensor(payload["topk_i"], dtype=torch.long, device=DEV).reshape(seq, k)
        topw = b64_to_t(payload["topk_w"]).reshape(seq, k).to(DEV)
        partial = torch.zeros(seq, H, dtype=moe_in.dtype, device=DEV)
        for E in experts:
            mod = kept.get((L, E))
            if mod is None:
                raise RuntimeError(f"expert (layer={L}, E={E}) not resident on this holder (has {st['ids']})")
            hit = (topi == E).nonzero(as_tuple=False)  # [(token, slot)...] routed to this expert
            if hit.numel() == 0:
                continue
            t = hit[:, 0]; j = hit[:, 1]
            out = mod(moe_in[t]) * topw[t, j, None]    # OlmoeMLP: down_proj(silu(gate_proj(x)) * up_proj(x)) · w
            partial.index_add_(0, t, out.to(moe_in.dtype))
    return {"ok": True, "partial": f32_to_b64(partial.flatten()), "seq": seq}

def moe_apply(payload: dict) -> dict:
    """BACKBONE, one MoE layer: COMBINE — sum the holders' routed-FFN partials (in the order sent = ascending
    expert range, matching the reference index_add order) and add the post-attention residual, all in fp32 on the
    worker. Returns the layer output hidden [1,S,H] for the next layer's moe_route."""
    bb = _moe_bb(payload["id"]); H = bb["hidden"]; seq = int(payload["seq"])
    with torch.no_grad():
        attn_hidden = b64_to_t(payload["attn_hidden"]).reshape(1, seq, H).to(DEV)
        moe_out = torch.zeros(seq, H, dtype=attn_hidden.dtype, device=DEV)
        for pb in payload.get("partials", []):
            moe_out = moe_out + b64_to_t(pb).reshape(seq, H).to(DEV)
        h = attn_hidden + moe_out.reshape(1, seq, H)
    return {"ok": True, "hidden": f32_to_b64(h.flatten()), "seq": seq}

def moe_head(payload: dict) -> dict:
    """BACKBONE: final RMSNorm + LM head on the last layer's hidden → last-token argmax (+ optional logits)."""
    bb = _moe_bb(payload["id"]); H = bb["hidden"]; seq = int(payload["seq"])
    with torch.no_grad():
        h = b64_to_t(payload["hidden"]).reshape(1, seq, H).to(DEV)
        h = bb["mm"].norm(h)
        logits = bb["model"].lm_head(h)[0, -1].float()
        res = {"ok": True, "argmax": int(logits.argmax().item())}
        if payload.get("return_logits"):
            res["logits"] = f32_to_b64(logits)
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
    if op == "ping": return {"ok": True, "pong": True, "n": len(payload.get("blob", ""))}  # echo → coordinator times RTT (empty blob) or throughput (large blob)
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
    if op == "shard_reset": return shard_reset(payload)
    if op == "shard_tok": return shard_tok(payload)
    if op == "shard_detok": return shard_detok(payload)
    if op == "shard_unload": _push_cleanup(payload.get("id")); SHARDS.pop(payload.get("id"), None); SHARD_TOKS.pop(payload.get("id"), None); _kv_drop(payload.get("id")); _empty_cache(); return {"ok": True}
    # ── EXPERT PARALLELISM (MoE) — one forward driven by the coordinator: moe_backbone_load / expert_load (place),
    #    then moe_embed → per-layer (moe_route → holders' expert_forward → moe_apply) → moe_head. moe_unload frees.
    if op == "moe_backbone_load": return moe_backbone_load(payload)
    if op == "expert_load": return expert_load(payload)
    if op == "moe_embed": return moe_embed(payload)
    if op == "moe_route": return moe_route(payload)
    if op == "expert_forward": return expert_forward(payload)
    if op == "moe_apply": return moe_apply(payload)
    if op == "moe_head": return moe_head(payload)
    if op == "moe_unload": _push_cleanup(payload.get("id")); MOE_BB.pop(payload.get("id"), None); MOE_EXPERTS.pop(payload.get("id"), None); _empty_cache(); return {"ok": True}
    raise ValueError(f"unknown model op {op}")

# ============================================================================
# PROTOTYPE: LAN-only worker->worker peer activation transport (OPT-IN).
# Enabled only when MOREGPU_PEER_TRANSPORT=1. Adjacent pipeline stages hand the sealed
# hidden state DIRECTLY to their successor over a small LAN WebSocket listener instead of
# round-tripping every per-token activation through the coordinator. Reuses seal()/unseal()
# + the shared tenant key (confidentiality) and _sk / sign scheme (Ed25519 origin auth) — no
# new crypto. The coordinator stays the control-plane + weight source: it wires the ring,
# injects stage 0, and collects the last stage's `complete`. See docs/ROADMAP.md "Peer
# transport". LAN-ONLY caveat: the advertised URL is a raw LAN IPv4 — no STUN/TURN/NAT
# traversal; a segmented/NAT'd edge fails the probe and the coordinator keeps it on relay.
# ============================================================================
RING: dict = {}         # sid -> {epoch, token, succ:{id,url,pub}|None, pred_pub:bytes|None, conn}
_KEY = None             # tenant key (set from `welcome`; the SAME key the coordinator hands every worker)
_CEIL = [0.6]           # current duty ceiling, shared with the async peer handlers
_LOOP = None            # the worker's asyncio loop (run_in_executor from peer handlers)
_CTRL_SEND = [None]     # bound ws_send of the LIVE control connection (for complete/edge_fault frames)
_PEER_URL = None        # ws://<lan-ip>:<port>/peer advertised to same-tenant peers at register
_PEER_SERVER = None


def _lan_ip() -> str:
    """Best-effort primary LAN IPv4 (no packets are actually sent). MOREGPU_PEER_HOST overrides it
    (set 127.0.0.1 for a single-host test). This is the honest LAN caveat — a raw subnet address."""
    override = os.environ.get("MOREGPU_PEER_HOST")
    if override:
        return override
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9)); return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


async def _ctrl_send(obj):
    fn = _CTRL_SEND[0]
    if fn is None:
        return
    try:
        await fn(obj)
    except Exception as e:
        print(f"[torch-worker] peer ctrl send failed: {e}")


async def _peer_probe(url: str, sid) -> bool:
    """Reachability probe: dial the successor's listener, expect a signed peer_pong. Success → the coordinator
    marks this edge `direct`; any failure (refused/timeout/isolation) → the coordinator keeps it on relay."""
    try:
        async with websockets.connect(url, open_timeout=3) as ws:
            n = b64e(os.urandom(12))
            await ws.send(json.dumps({"t": "peer_ping", "sid": sid, "nonce": n}))
            p = json.loads(await asyncio.wait_for(ws.recv(), 3))
            return p.get("nonce") == n
    except Exception:
        return False


async def _run_stage(payload):
    return await _LOOP.run_in_executor(TORCH_POOL, _paced, shard_forward, (payload,), _CEIL[0], True)


def _ring_last(r) -> bool:
    """This stage is the pipe TAIL iff it has no successor AND its outgoing edge isn't a bridged relay hop."""
    return r.get("succ") is None and not r.get("relay")


async def ring_send(sid, seq, epoch, out, want_logits, cache=None):
    """Hand this stage's output onward: a stage ERROR (or the pipe TAIL) delivers `complete` to the coordinator over
    the control WS; a DIRECT edge seals+signs the hidden state to the successor's peer WS; a RELAY edge (mixed pipe)
    BRIDGES the same sealed+signed frame through the coordinator, which routes it into the successor. `cache`, when
    set, carries the KV session/pos so each stage runs its cached shard_forward over the ring (KV-over-peer)."""
    r = RING.get(sid)
    if out.get("ok") is False:                               # any stage's error → straight to the coordinator
        await _ctrl_send({"t": "complete", "sid": sid, "seq": seq, "epoch": epoch, "ok": False, "error": out.get("error")})
        return
    if not r or _ring_last(r):                               # pipe TAIL → coordinator control WS
        frame = {"t": "complete", "sid": sid, "seq": seq, "epoch": epoch, "ok": True, "argmax": out.get("argmax")}
        if out.get("logits") is not None:
            frame["logits"] = out["logits"]
        await _ctrl_send(frame)
        return
    sealed = seal(_KEY, json.dumps(out).encode())            # same seal() the coordinator uses — any tenant peer can open it
    sig = b64e(_sk.sign(f'{sid}|{seq}|{sealed["iv"]}|{sealed["ct"]}'.encode()))
    frame = {"sid": sid, "seq": seq, "epoch": epoch, "token": r["token"], "sealed": sealed, "sig": sig, "return_logits": want_logits}
    if cache is not None:
        frame.update(cached=True, session=cache["session"], pos=cache["pos"])
    if r.get("relay"):                                       # RELAY edge → BRIDGE through the coordinator (it knows the successor)
        frame["t"] = "bridge"; frame["from"] = NAME
        await _ctrl_send(frame)
        return
    frame["t"] = "act"                                       # DIRECT edge → straight to the successor's peer listener
    try:
        if r.get("conn") is None:
            r["conn"] = await websockets.connect(r["succ"]["url"], max_size=None)
        await r["conn"].send(json.dumps(frame))
    except Exception as e:                                    # peer unreachable mid-stream → let coordinator flip to bridge
        r["conn"] = None
        await _ctrl_send({"t": "edge_fault", "sid": sid, "seq": seq, "epoch": epoch,
                          "succ": (r.get("succ") or {}).get("id"), "error": str(e)})


async def _handle_act(m):
    """Inbound activation from a predecessor (arriving over the peer WS, or via `bridge_in` from the coordinator on a
    relay edge): epoch/token fence → Ed25519 origin-verify → unseal → run this stage's blocks (cached iff the frame
    carries a KV session) → forward to the successor (or `complete` if this is the pipe tail)."""
    sid = m.get("sid"); r = RING.get(sid)
    if not r or m.get("epoch") != r.get("epoch") or m.get("token") != r.get("token"):
        return                                               # stale epoch / wrong ring token → drop
    b = m["sealed"]; msg = f'{sid}|{m["seq"]}|{b["iv"]}|{b["ct"]}'.encode()
    try:
        Ed25519PublicKey.from_public_bytes(r["pred_pub"]).verify(b64d(m["sig"]), msg)
    except Exception:
        print("[torch-worker] peer act: Ed25519 verify FAILED — dropping"); return
    inner = json.loads(unseal(_KEY, b).decode())             # {hidden, seq, hidden_dim}
    payload = {"id": sid, "first": False, "last": _ring_last(r), "hidden": inner["hidden"], "seq": inner["seq"]}
    cache = None
    if m.get("cached"):                                      # KV-over-peer: run this stage's cached shard_forward
        cache = {"session": m.get("session"), "pos": int(m.get("pos", 0))}
        payload.update(cached=True, session=cache["session"], pos=cache["pos"])
    if m.get("return_logits"):
        payload["return_logits"] = True
    try:
        out = await _run_stage(payload)
    except Exception as e:
        out = {"ok": False, "error": str(e)}
    await ring_send(sid, m["seq"], m["epoch"], out, m.get("return_logits"), cache)


async def _peer_handle(ws, *_):
    async for raw in ws:
        try:
            m = json.loads(raw); t = m.get("t")
            if t == "peer_ping":                             # reachability probe → signed pong
                await ws.send(json.dumps({"t": "peer_pong", "nonce": m["nonce"],
                                          "sig": b64e(_sk.sign(m["nonce"].encode()))}))
            elif t == "act":
                await _handle_act(m)
            elif t == "moe_dispatch":                        # MoE peer: an expert dispatch from the backbone → run + reply
                await _handle_moe_dispatch(ws, m)
        except Exception as e:
            print(f"[torch-worker] peer frame error: {e}")


async def _start_peer_listener():
    global _PEER_URL, _PEER_SERVER
    port = int(os.environ.get("MOREGPU_PEER_PORT", "0") or "0")
    _PEER_SERVER = await websockets.serve(_peer_handle, "0.0.0.0", port, max_size=None)
    socks = getattr(_PEER_SERVER, "sockets", None) or getattr(getattr(_PEER_SERVER, "server", None), "sockets", None)
    bound = socks[0].getsockname()[1]
    _PEER_URL = f"ws://{_lan_ip()}:{bound}/peer"
    print(f"[torch-worker] peer transport ON — listening {_PEER_URL}")


async def _handle_ring_wire(m):
    """Coordinator wires this stage: stash successor endpoint + ring token/epoch, probe the successor, ack back
    with reachability so the coordinator can pick `direct` vs `relay` for this edge."""
    sid = m["sid"]; r = RING.setdefault(sid, {})
    if r.get("conn") is not None:
        try:
            await r["conn"].close()
        except Exception:
            pass
    r.update(epoch=m["epoch"], token=m["token"], succ=m.get("succ"), conn=None, relay=False)
    reachable = True
    if m.get("succ"):
        reachable = await _peer_probe(m["succ"]["url"], sid)
    await _ctrl_send({"t": "ring_ack", "reqId": m.get("reqId"), "sid": sid, "reachable": reachable})


async def _handle_ring_mode(m):
    """Coordinator's final verdict for this stage's OUTGOING edge (sent after the probe / on an edge fault): relay=True
    → BRIDGE the hop through the coordinator (mixed pipe); relay=False → keep the direct peer hop. Arrives strictly
    after the ring_wire ack, so it never races the wire handler."""
    sid = m["sid"]; r = RING.setdefault(sid, {})
    if m.get("epoch") == r.get("epoch"):
        r["relay"] = bool(m.get("relay"))


async def _handle_ring_pred(m):
    sid = m["sid"]; r = RING.setdefault(sid, {})
    r["pred_pub"] = b64d(m["pred_pub"]); r["epoch"] = m.get("epoch", r.get("epoch"))


async def _handle_inject(m):
    """Stage 0 only: embed input_ids + run stage-0 blocks (cached iff the frame carries a KV session), then hand off
    to the successor (direct or bridged). The coordinator is now off the per-token data path until the tail delivers
    `complete`."""
    sid = m.get("sid"); r = RING.get(sid)
    if not r or m.get("epoch") != r.get("epoch"):
        await _ctrl_send({"t": "complete", "sid": sid, "seq": m.get("seq"), "epoch": m.get("epoch"),
                          "ok": False, "error": "stage 0 not wired / epoch mismatch"})
        return
    payload = {"id": sid, "first": True, "last": _ring_last(r), "input_ids": m["input_ids"]}
    cache = None
    if m.get("cached"):                                      # KV-over-peer: prefill (pos 0) or a 1-token decode step
        cache = {"session": m.get("session"), "pos": int(m.get("pos", 0))}
        payload.update(cached=True, session=cache["session"], pos=cache["pos"])
    if m.get("return_logits"):
        payload["return_logits"] = True
    try:
        out = await _run_stage(payload)
    except Exception as e:
        await _ctrl_send({"t": "complete", "sid": sid, "seq": m["seq"], "epoch": m["epoch"],
                          "ok": False, "error": str(e)})
        return
    await ring_send(sid, m["seq"], m["epoch"], out, m.get("return_logits"), cache)


# ── PEER TRANSPORT (MoE all-to-all): the BACKBONE drives the whole forward and dispatches each layer's routed experts
# DIRECTLY to their holder workers over the peer WS (sealed + Ed25519-signed), combining locally — so the coordinator
# relays ZERO expert activations (it only injects + collects the final logits). Any peer failure → the backbone
# reports `moe_complete ok:False` and the coordinator falls back to the unchanged relayed moePipe.
MOE_WIRE: dict = {}     # sid -> BACKBONE side: {epoch, token, holders:[{id,url,pub,experts}], conns:{id:ws}}
MOE_PEER: dict = {}     # sid -> HOLDER side:   {epoch, token, backbone_pub:bytes}


async def _run_moe(fn, payload):
    return await _LOOP.run_in_executor(TORCH_POOL, _paced, fn, (payload,), _CEIL[0], True)


async def _moe_close_conns(mw):
    for c in list(mw.get("conns", {}).values()):
        try:
            await c.close()
        except Exception:
            pass
    mw["conns"] = {}


async def _handle_moe_wire(m):
    """BACKBONE: stash every holder's peer endpoint + probe reachability, ack so the coordinator picks peer vs relay."""
    sid = m["sid"]
    old = MOE_WIRE.get(sid)
    if old:
        await _moe_close_conns(old)
    MOE_WIRE[sid] = {"epoch": m["epoch"], "token": m["token"], "holders": m["holders"], "conns": {}}
    reachable = True
    for h in m["holders"]:
        if not await _peer_probe(h["url"], sid):
            reachable = False; break
    await _ctrl_send({"t": "moe_wire_ack", "reqId": m.get("reqId"), "sid": sid, "reachable": reachable})


async def _handle_moe_wire_holder(m):
    """HOLDER: stash the backbone's pubkey + ring token/epoch so it can origin-verify inbound dispatch frames."""
    sid = m["sid"]
    MOE_PEER[sid] = {"epoch": m["epoch"], "token": m["token"], "backbone_pub": b64d(m["backbone_pub"])}


async def _moe_dispatch_to_holder(sid, L, epoch, token, seq, k, moe_in, topi, topw, experts, h, mw):
    """BACKBONE→HOLDER over the peer WS: seal+sign the routed-FFN input + this holder's routed experts, await the
    signed partial back, origin-verify it, and return the partial hidden. One persistent conn per holder."""
    payload = {"layer": L, "seq": seq, "k": k, "hidden": moe_in, "topk_i": topi, "topk_w": topw, "experts": experts}
    sealed = seal(_KEY, json.dumps(payload).encode())
    sig = b64e(_sk.sign(f'{sid}|{L}|{epoch}|{sealed["iv"]}|{sealed["ct"]}'.encode()))
    frame = {"t": "moe_dispatch", "sid": sid, "layer": L, "epoch": epoch, "token": token, "sealed": sealed, "sig": sig}
    conn = mw["conns"].get(h["id"])
    if conn is None:
        conn = await websockets.connect(h["url"], max_size=None); mw["conns"][h["id"]] = conn
    await conn.send(json.dumps(frame))
    resp = json.loads(await conn.recv())
    if resp.get("t") != "moe_partial" or int(resp.get("layer", -1)) != L or resp.get("epoch") != epoch:
        raise RuntimeError(f"bad moe_partial from holder {h['id']}")
    b = resp["sealed"]; msg = f'{sid}|{L}|{epoch}|{b["iv"]}|{b["ct"]}'.encode()
    Ed25519PublicKey.from_public_bytes(b64d(h["pub"])).verify(b64d(resp["sig"]), msg)  # origin-auth the holder
    return json.loads(unseal(_KEY, b).decode())["partial"]


async def _handle_moe_inject(m):
    """BACKBONE: run one whole MoE forward — embed → per-layer (route locally → DISPATCH routed experts to holders
    over peer, concurrently → COMBINE locally) → head — then deliver argmax/logits to the coordinator. Mirrors the
    coordinator-relayed moePipe numerically; partials are summed in holder order (== ascending expert range)."""
    sid = m["sid"]; epoch = m["epoch"]; seq_no = m["seq"]; mw = MOE_WIRE.get(sid)
    if not mw or mw.get("epoch") != epoch:
        await _ctrl_send({"t": "moe_complete", "sid": sid, "seq": seq_no, "epoch": epoch, "ok": False,
                          "error": "backbone not wired / epoch mismatch"}); return
    try:
        bb = MOE_BB.get(sid)
        if bb is None:
            raise RuntimeError(f"moe backbone {sid} not loaded")
        emb = await _run_moe(moe_embed, {"id": sid, "input_ids": m["input_ids"]})
        hidden = emb["hidden"]; S = emb["seq"]; token = mw["token"]
        for L in range(bb["n_layer"]):
            rt = await _run_moe(moe_route, {"id": sid, "layer": L, "hidden": hidden, "seq": S})
            moe_in = rt["moe_in"]; topi = rt["topk_i"]; topw = rt["topk_w"]; k = rt["k"]
            used = set(topi)
            tasks = []
            for h in mw["holders"]:                          # holder order == ascending expert range → matches moe_apply's sum order
                ru = [e for e in h["experts"] if e in used]
                if not ru:
                    continue
                tasks.append(_moe_dispatch_to_holder(sid, L, epoch, token, S, k, moe_in, topi, topw, ru, h, mw))
            partials = list(await asyncio.gather(*tasks)) if tasks else []   # gather preserves task (holder) order
            ap = await _run_moe(moe_apply, {"id": sid, "layer": L, "attn_hidden": rt["attn_hidden"], "partials": partials, "seq": S})
            hidden = ap["hidden"]
        hd = await _run_moe(moe_head, {"id": sid, "hidden": hidden, "seq": S, "return_logits": m.get("return_logits")})
        frame = {"t": "moe_complete", "sid": sid, "seq": seq_no, "epoch": epoch, "ok": True, "argmax": hd.get("argmax")}
        if hd.get("logits") is not None:
            frame["logits"] = hd["logits"]
        await _ctrl_send(frame)
    except Exception as e:                                   # a holder down / verify fail → coordinator falls back to relay
        await _moe_close_conns(mw)
        await _ctrl_send({"t": "moe_complete", "sid": sid, "seq": seq_no, "epoch": epoch, "ok": False, "error": str(e)})


async def _handle_moe_dispatch(ws, m):
    """HOLDER: an inbound dispatch from the backbone over the peer WS — epoch/token fence → Ed25519 origin-verify →
    unseal → run expert_forward for this holder's routed experts → seal+sign the partial back on the same conn."""
    sid = m.get("sid"); mp = MOE_PEER.get(sid); L = int(m.get("layer", -1)); epoch = m.get("epoch")
    if not mp or epoch != mp.get("epoch") or m.get("token") != mp.get("token"):
        return                                               # stale epoch / wrong token → drop
    b = m["sealed"]; msg = f'{sid}|{L}|{epoch}|{b["iv"]}|{b["ct"]}'.encode()
    try:
        Ed25519PublicKey.from_public_bytes(mp["backbone_pub"]).verify(b64d(m["sig"]), msg)
    except Exception:
        print("[torch-worker] moe_dispatch: Ed25519 verify FAILED — dropping"); return
    payload = json.loads(unseal(_KEY, b).decode()); payload["id"] = sid
    out = await _run_moe(expert_forward, payload)
    sealed = seal(_KEY, json.dumps({"partial": out["partial"]}).encode())
    sig = b64e(_sk.sign(f'{sid}|{L}|{epoch}|{sealed["iv"]}|{sealed["ct"]}'.encode()))
    await ws.send(json.dumps({"t": "moe_partial", "sid": sid, "layer": L, "epoch": epoch, "sealed": sealed, "sig": sig}))


async def run():
    global _LOOP, _KEY
    loop = asyncio.get_event_loop()
    _LOOP = loop
    if PEER_TRANSPORT:
        await _start_peer_listener()
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
                node = {"id": NAME, "backend": NODE_BACKEND, "label": BACKEND, "os": platform.system().lower()}
                if PEER_TRANSPORT and _PEER_URL:  # advertise the peer endpoint so the coordinator can wire a direct ring
                    node["peer"] = {"url": _PEER_URL, "pub": PUBKEY_B64}
                await ws.send(json.dumps({"t": "register", "joinToken": A.token, "pubkey": PUBKEY_B64, "node": node}))
                key = None
                # websockets requires writes to be serialized, and handlers now run concurrently (below), so
                # every send goes through this lock.
                send_lock = asyncio.Lock()
                inflight_tasks: set = set()
                async def ws_send(obj):
                    async with send_lock:
                        await ws.send(json.dumps(obj))
                _CTRL_SEND[0] = ws_send  # peer handlers deliver complete/edge_fault over the live control WS
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
                        out = await loop.run_in_executor(TORCH_POOL, _paced, compute_shard, (req,), ceil)  # off the event loop, throttled to the duty ceiling
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
                        res = await loop.run_in_executor(TORCH_POOL, _paced, dispatch, (mm["op"], payload), ceil, mm["op"] in _PACED_OPS)
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
                                _KEY = key; _CEIL[0] = ceil  # peer handlers seal/unseal with the same tenant key
                                # fresh session: the coordinator forgets our resident state on disconnect, so a
                                # reconnecting worker starts clean too (no stale models/weights pinned in VRAM).
                                # KEEP in-progress PUSH staging, though: a stage that dropped mid-stream can then
                                # RESUME (the coordinator re-streams only the missing bytes). A genuinely new
                                # coordinator wipes it anyway via push_begin (resume=false) before its first chunk.
                                resident.clear(); MODELS.clear(); SHARDS.clear(); SHARD_TOKS.clear(); SHARD_KV.clear(); MOE_BB.clear(); MOE_EXPERTS.clear(); TRAIN.update(model=None, opt=None, step=0, trainable={}); _empty_cache()
                                RING.clear(); MOE_WIRE.clear(); MOE_PEER.clear()  # a fresh coordinator session re-wires the ring/mesh after (re)load
                                print(f"[torch-worker] joined pool on {DEV} · duty ceiling {int(ceil*100)}%")
                            elif t == "control":
                                if "pause" in m: paused = bool(m["pause"]); pause_reason = "admin" if paused else ""
                                if "ceil" in m and m["ceil"] is not None: ceil = float(m["ceil"]); _CEIL[0] = ceil
                                print(f"[torch-worker] admin control → {'PAUSED' if paused else 'active'} · ceiling {int(ceil*100)}%")
                            elif t == "ring_wire":  # PEER: coordinator hands this stage its successor endpoint + probes reachability
                                spawn(_handle_ring_wire(m))
                            elif t == "ring_pred":  # PEER: coordinator hands this stage its predecessor's pubkey (origin auth)
                                spawn(_handle_ring_pred(m))
                            elif t == "ring_mode":  # PEER: coordinator's final edge verdict — direct hop vs bridge through it
                                spawn(_handle_ring_mode(m))
                            elif t == "inject":     # PEER: coordinator kicks off a token at stage 0
                                spawn(_handle_inject(m))
                            elif t == "bridge_in":  # PEER (mixed pipe): a relay-edge activation routed in via the coordinator
                                spawn(_handle_act(m))
                            elif t == "moe_wire":   # PEER (MoE): backbone learns holder endpoints + probes them
                                spawn(_handle_moe_wire(m))
                            elif t == "moe_wire_holder":  # PEER (MoE): holder learns the backbone's pubkey (origin auth)
                                spawn(_handle_moe_wire_holder(m))
                            elif t == "moe_inject":  # PEER (MoE): coordinator kicks off a whole forward at the backbone
                                spawn(_handle_moe_inject(m))
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
