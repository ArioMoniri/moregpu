"""Initial weights for training tasks: a segment/classify ``encoder`` and a ``finetune_model`` spec ``source``.

Two kinds of source are accepted:

* a **path** (a JEPA encoder export directory, optionally ``file://``): confined to ``MOREGPU_OUTPUT_DIR`` ∪
  ``MOREGPU_MODEL_ROOTS`` (:func:`moregpu_worker.paths.export_source`);
* ``pushed://<id>``: a blob the coordinator streamed with ``/data/push`` (``blob_begin/chunk/end``) into this worker's
  :class:`~moregpu_worker.data.blobs.BlobStore`. Only a fully received blob whose size and sha256 were verified at
  ``blob_end`` resolves. A ``sha256`` is **required** and must equal the blob's.

Both get the same checks before any tensor is read:

1. size ≤ ``MOREGPU_MODEL_MAX_BYTES`` (default 20 GiB), the cap on every model artefact;
2. the file is hashed and compared with the pinned ``sha256`` (a pushed blob is re-hashed, so a staged file changed on
   disk after ``blob_end`` is caught) — :class:`IntegrityError` on a mismatch;
3. **safetensors only**: the 8-byte header length and the JSON header are checked by hand, so a pickle or a
   ``torch.save`` archive is refused (:class:`RefusedFormat`) before anything parses it. Nothing here calls
   ``torch.load``.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path

from .. import paths
from .errors import IntegrityError, RefusedFormat, RefusedSource
from .fetch import _pushed, model_max_bytes, sha256_file

_SHA = re.compile(r"^[0-9a-f]{64}$")
_MAX_HEADER = 100 << 20          # safetensors' own limit on the JSON header
ENCODER_CONFIG_KEY = "moregpu.encoder_config"   # safetensors metadata written by JEPA exports


def _check_size(p: Path, what: str) -> int:
    size = p.stat().st_size
    cap = model_max_bytes()
    if size > cap:
        raise RefusedSource(f"{what}: {size} bytes exceeds the model size cap {cap} (MOREGPU_MODEL_MAX_BYTES)")
    return size


def check_safetensors(path) -> None:
    """Refuse (:class:`RefusedFormat`) anything that is not a well-formed safetensors file, without parsing it with
    anything but ``int.from_bytes`` and ``json``: a pickle / ``torch.save`` zip is never opened."""
    p = Path(path)
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(8)
        if head[:2] == b"PK" or head[:1] == b"\x80":
            raise RefusedFormat(f"{p.name} is a pickle / torch.save archive; pushed and task-init weights must be "
                                f"safetensors (pickles are refused)")
        if len(head) < 8:
            raise RefusedFormat(f"{p.name} is not a safetensors file (only {size} bytes)")
        n = int.from_bytes(head, "little")
        if n < 2 or n > _MAX_HEADER or n > size - 8:
            raise RefusedFormat(f"{p.name} is not a safetensors file (header length {n} for a {size}-byte file)")
        raw = f.read(n)
    try:
        hdr = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise RefusedFormat(f"{p.name} is not a safetensors file (the header is not JSON)") from None
    if not isinstance(hdr, dict):
        raise RefusedFormat(f"{p.name} is not a safetensors file (the header is not a JSON object)")


def resolve_pushed(uri: str, sha256: str | None, blobs=None) -> Path:
    """``pushed://<id>`` → the staged file, after the size cap, a re-hash against ``sha256`` and the safetensors check."""
    if not sha256:
        raise RefusedSource(f"{uri}: pushed:// weights need a pinned sha256")
    if not isinstance(sha256, str) or not _SHA.match(sha256):
        raise RefusedSource(f"{uri}: sha256 must be 64 lowercase hex characters")
    p = _pushed(uri[len("pushed://"):], sha256, blobs)       # RefusedSource (unknown/incomplete) / IntegrityError
    _check_size(p, uri)
    got = sha256_file(p)
    if got != sha256:
        raise IntegrityError(f"{uri}: staged blob hashes to {got}, not the pinned {sha256}; refusing to load")
    check_safetensors(p)
    return p


def _read(p: Path) -> tuple[dict, dict]:
    from safetensors import safe_open
    try:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys()}, dict(f.metadata() or {})
    except Exception as e:  # malformed tensor table (offsets, dtypes) behind a valid-looking header
        raise RefusedFormat(f"{p.name} is not a valid safetensors file ({e})") from None


def load_encoder(enc: dict, blobs=None) -> tuple[dict, dict, dict]:
    """Resolve ``encoder: {init: "export", path | source, sha256?, config?}`` → ``(state_dict, vit_config, info)``.

    ``path`` / ``source`` is a JEPA export directory (confined) or ``pushed://<id>`` (a single ``encoder.safetensors``
    blob; ``sha256`` required). The ViT config is ``config`` if given, else the export's ``encoder_config.json`` or,
    for a blob, its ``moregpu.encoder_config`` safetensors metadata (JEPA exports write it)."""
    src = enc.get("source") or enc.get("path")
    if not isinstance(src, str) or not src:
        raise ValueError("encoder init 'export' needs a path (an export dir) or pushed://<id>")
    if src.startswith("pushed://"):
        p = resolve_pushed(src, enc.get("sha256"), blobs)
        sd, meta = _read(p)
        cfg = enc.get("config")
        if cfg is None and ENCODER_CONFIG_KEY in meta:
            cfg = json.loads(meta[ENCODER_CONFIG_KEY])
        if cfg is None:
            raise ValueError(f"{src}: no encoder config — pass encoder.config, or push a JEPA export's "
                             f"encoder.safetensors (it carries '{ENCODER_CONFIG_KEY}' metadata)")
        info = {"source": "pushed", "sha256": enc["sha256"]}
    else:
        if "://" in src:
            if not src.startswith("file://"):
                raise RefusedSource(f"encoder source {src!r}: use an export path or pushed://<id>")
            src = urllib.parse.unquote(urllib.parse.urlparse(src).path)
        d = Path(paths.export_source(src))                 # MOREGPU_OUTPUT_DIR ∪ MOREGPU_MODEL_ROOTS
        cfg_path, w = d / "encoder_config.json", d / "encoder.safetensors"
        if not cfg_path.exists() or not w.exists():
            raise FileNotFoundError(f"no JEPA encoder export at {d}")
        _check_size(w, str(w))
        got = sha256_file(w)
        if enc.get("sha256") and got != enc["sha256"]:
            raise IntegrityError(f"{w}: sha256 {got} != the pinned {enc['sha256']}; refusing to load")
        check_safetensors(w)
        sd, _ = _read(w)
        cfg = enc.get("config") or json.loads(cfg_path.read_text())
        info = {"source": "file", "sha256": got}
    cfg = dict(cfg)
    cfg["img_size"] = list(cfg["img_size"])
    return sd, cfg, info


def blobs_of(data_plane):
    """The BlobStore behind a task's data plane (``None`` → the process-wide default store)."""
    return getattr(data_plane, "blobs", None) if data_plane is not None else None


__all__ = ["check_safetensors", "resolve_pushed", "load_encoder", "blobs_of", "ENCODER_CONFIG_KEY"]
