"""Admin-installed model plugins (ADR-0113 priority 2): entry-point group `moregpu.models`.

A plugin is usable only if its distribution name + version + wheel sha256 (pip's direct_url.json archive hash) are
pinned in the worker's allowlist JSON at MOREGPU_MODEL_PLUGIN_ALLOWLIST (separate from the training-task allowlist).
The coordinator can only NAME an installed plugin; code never travels over the wire. The entry point must resolve to a
callable `build(**arch.kwargs) -> torch.nn.Module`; weights, if the spec has a source, are loaded strictly as a
state_dict/safetensors by the state-dict adapter.
"""
from __future__ import annotations

from importlib import metadata

from ...train import registry
from ..errors import RefusedPlugin, UnknownArch

GROUP = "moregpu.models"
ENV = "MOREGPU_MODEL_PLUGIN_ALLOWLIST"


def _entry_points():
    return list(metadata.entry_points(group=GROUP))


def discover(eps=None, allowlist_path=None) -> tuple[dict, dict]:
    """({name: loader}, {name: refusal reason}) for installed `moregpu.models` entry points."""
    return registry.discover_plugins(_entry_points() if eps is None else eps, allowlist_path=allowlist_path,
                                     group=GROUP, env=ENV)


def get(name: str):
    """Return (build callable, {dist, version, wheel_sha256}) or raise RefusedPlugin / UnknownArch."""
    eps = _entry_points()
    found, refused = discover(eps)
    if name in found:
        d = next(ep for ep in eps if ep.name == name).dist
        info = {"dist": d.metadata["Name"], "version": d.version, "wheel_sha256": registry._wheel_sha(d)}
        return found[name](), info
    if name in refused:
        raise RefusedPlugin(f"model plugin {name!r} refused: {refused[name]} (pin it in {ENV}; see docs/MODELS.md)")
    raise UnknownArch(f"no installed `{GROUP}` plugin named {name!r}; installed: {sorted(found) or 'none'}")
