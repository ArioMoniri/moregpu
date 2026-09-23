"""Task registry (ADR-0105): built-ins + admin-pinned entry-point plugins (group `moregpu.train_tasks`).

A plugin loads only if its distribution name + version + wheel sha256 (from pip's direct_url.json archive hash) is in
the worker's allowlist file (MOREGPU_PLUGIN_ALLOWLIST). The coordinator can only NAME a task; code never travels."""
from __future__ import annotations

import importlib
import json
import os
from importlib import metadata
from pathlib import Path

BUILTINS = {
    "toy_linear": "moregpu_worker.train.tasks.toy:ToyLinearTask",
    "llm_lora": "moregpu_worker.train.tasks.llm_lora:LlmLoraTask",
    "ijepa_2d": "moregpu_worker.train.tasks.jepa:IJepa2DTask",
    "jepa_2p5d": "moregpu_worker.train.tasks.jepa:Jepa2p5DTask",
    "jepa_3d": "moregpu_worker.train.tasks.jepa:Jepa3DTask",
    "classify": "moregpu_worker.train.tasks.vision:ClassifyTask",
    "segment": "moregpu_worker.train.tasks.vision:SegmentTask",
}
GROUP = "moregpu.train_tasks"


def _load(path: str):
    mod, _, attr = path.partition(":")
    return getattr(importlib.import_module(mod), attr)


ENV = "MOREGPU_PLUGIN_ALLOWLIST"


def _allowlist(path: str | os.PathLike | None, env: str = ENV) -> list[dict]:
    p = path or os.environ.get(env)
    if not p or not Path(p).exists():
        return []
    return json.loads(Path(p).read_text())


def _wheel_sha(dist) -> str | None:
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    try:
        return json.loads(raw).get("archive_info", {}).get("hashes", {}).get("sha256") or \
            (json.loads(raw).get("archive_info", {}).get("hash", "").partition("sha256=")[2] or None)
    except ValueError:
        return None


def discover_plugins(eps=None, allowlist_path=None, group: str = GROUP, env: str = ENV) -> tuple[dict, dict]:
    """Returns ({name: loader}, {name: refusal reason}). `group`/`env` let other plugin kinds (e.g. vision models,
    group `moregpu.models`, env MOREGPU_MODEL_PLUGIN_ALLOWLIST) reuse the same pinning rule with their own allowlist."""
    allow = _allowlist(allowlist_path, env)
    if eps is None:
        eps = metadata.entry_points(group=group)
    found, refused = {}, {}
    for ep in eps:
        d = ep.dist
        name, ver, sha = (d.metadata["Name"] if d else None), (d.version if d else None), (_wheel_sha(d) if d else None)
        match = [a for a in allow if a["dist"] == name]
        if not match:
            refused[ep.name] = f"distribution {name!r} is not in the plugin allowlist"
        elif not any(a["version"] == ver for a in match):
            refused[ep.name] = f"{name} {ver} is not the pinned version"
        elif not sha or not any(a["wheel_sha256"] == sha and a["version"] == ver for a in match):
            refused[ep.name] = f"{name} {ver}: wheel sha256 missing or not pinned"
        else:
            found[ep.name] = ep.load
    return found, refused


def available() -> list[str]:
    plugins, _ = discover_plugins()
    return sorted(set(BUILTINS) | set(plugins))


def create(name: str):
    if name in BUILTINS:
        return _load(BUILTINS[name])()
    plugins, refused = discover_plugins()
    if name in plugins:
        return plugins[name]()()
    if name in refused:
        raise KeyError(f"task plugin {name!r} refused: {refused[name]}")
    raise KeyError(f"unknown training task {name!r}; available: {available()}")
