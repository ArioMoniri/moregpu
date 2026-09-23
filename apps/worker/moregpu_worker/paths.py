"""Filesystem confinement for everything the coordinator can name by path (exports, export reads).

* Writes (``TrainTask.export``, ``task_export``) are confined to ``MOREGPU_OUTPUT_DIR`` (default ``./moregpu-out``
  relative to the worker's working directory). A relative path resolves inside it; an absolute path must *realpath*
  inside it (``..`` and symlinks are resolved first, so ``../x`` and a symlink pointing outside are refused).
* Reads of exports (``vision_infer_load {export}``, a segment/classify ``encoder: {init: export, path}``) are confined
  to ``MOREGPU_OUTPUT_DIR`` ∪ ``MOREGPU_MODEL_ROOTS`` (``os.pathsep``-separated); relative paths resolve inside
  ``MOREGPU_OUTPUT_DIR``.

Both roots are read from the environment at call time.
"""
from __future__ import annotations

import os


class ConfinementError(PermissionError):
    """The path resolves outside the allowed roots."""


def output_root() -> str:
    """realpath of ``MOREGPU_OUTPUT_DIR`` (default ``<cwd>/moregpu-out``)."""
    return os.path.realpath(os.environ.get("MOREGPU_OUTPUT_DIR") or os.path.join(os.getcwd(), "moregpu-out"))


def model_roots() -> list[str]:
    return [os.path.realpath(r) for r in os.environ.get("MOREGPU_MODEL_ROOTS", "").split(os.pathsep) if r.strip()]


def export_read_roots() -> list[str]:
    return [output_root(), *model_roots()]


def _inside(real: str, root: str) -> bool:
    return real == root or os.path.commonpath([real, root]) == root


def confine(path, roots) -> str:
    """Return the realpath of ``path`` if it lies inside one of ``roots`` (a relative path resolves inside
    ``roots[0]``), else raise :class:`ConfinementError`."""
    path = os.fspath(path) if not isinstance(path, str) else path
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ConfinementError(f"bad path {path!r}")
    roots = [os.path.realpath(os.fspath(r)) for r in roots if r]
    if not roots:
        raise ConfinementError("no allowed roots configured")
    real = os.path.realpath(path if os.path.isabs(path) else os.path.join(roots[0], path))
    if not any(_inside(real, r) for r in roots):
        raise ConfinementError(f"{path!r} resolves to {real!r}, outside the allowed roots {roots}")
    return real


def export_dir(path) -> str:
    """Confine an export *destination* to MOREGPU_OUTPUT_DIR and create it."""
    try:
        real = confine(path, [output_root()])
    except ConfinementError as e:
        raise ConfinementError(f"export path not allowed: {e} (exports are confined to MOREGPU_OUTPUT_DIR)") from None
    os.makedirs(real, exist_ok=True)
    return real


def export_source(path) -> str:
    """Confine an export *read* to MOREGPU_OUTPUT_DIR ∪ MOREGPU_MODEL_ROOTS."""
    try:
        return confine(path, export_read_roots())
    except ConfinementError as e:
        raise ConfinementError(f"export read not allowed: {e} "
                               f"(reads are confined to MOREGPU_OUTPUT_DIR and MOREGPU_MODEL_ROOTS)") from None
