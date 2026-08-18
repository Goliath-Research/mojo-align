"""MOJO_ALIGN_* env contract with one-release METHYLGRAPHER_MOJO_* dual-read."""

from __future__ import annotations

import os
import sys

INSTALL_PREFIX = "/opt/mojo-align"
_LEGACY_PREFIX = "/opt/methylgrapher-mojo"
_NEW = "MOJO_ALIGN_"
_OLD = "METHYLGRAPHER_MOJO_"
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    print(msg, file=sys.stderr)


def getenv(suffix: str, default: str = "") -> str:
    """Read ``MOJO_ALIGN_<suffix>``, then legacy ``METHYLGRAPHER_MOJO_<suffix>``."""
    new_key = _NEW + suffix
    old_key = _OLD + suffix
    value = os.environ.get(new_key, "").strip()
    if value:
        return value
    legacy = os.environ.get(old_key, "").strip()
    if legacy:
        _warn_once(
            old_key,
            f"warning: {old_key} is deprecated; use {new_key}",
        )
        return legacy
    return default


def install_prefix() -> str:
    """In-container / staged install root (``/opt/mojo-align``)."""
    if os.path.isdir(INSTALL_PREFIX):
        return INSTALL_PREFIX
    if os.path.isdir(_LEGACY_PREFIX):
        _warn_once(
            _LEGACY_PREFIX,
            f"warning: {_LEGACY_PREFIX} is deprecated; use {INSTALL_PREFIX}",
        )
        return _LEGACY_PREFIX
    return INSTALL_PREFIX


def python_search_paths() -> list[str]:
    cwd = os.getcwd()
    root = getenv("ROOT")
    prefix = install_prefix()
    out: list[str] = []
    for path in (
        cwd,
        os.path.join(cwd, "methylgrapher"),
        os.path.join(cwd, "giraffe", "python"),
        os.path.join(cwd, "fq2bam-meth", "python"),
        os.path.join(cwd, "gpu-common", "python"),
        root,
        os.path.join(root, "methylgrapher") if root else "",
        os.path.join(root, "giraffe", "python") if root else "",
        os.path.join(root, "fq2bam-meth", "python") if root else "",
        os.path.join(root, "gpu-common", "python") if root else "",
        prefix,
        os.path.join(prefix, "methylgrapher"),
        os.path.join(prefix, "giraffe", "python"),
        os.path.join(prefix, "fq2bam-meth", "python"),
        os.path.join(prefix, "gpu-common", "python"),
        os.path.join(prefix, "scripts"),
    ):
        if path and path not in out:
            out.append(path)
    return out


def ensure_sys_path() -> None:
    """Prepend cwd / ``MOJO_ALIGN_ROOT`` / ``/opt/mojo-align`` onto ``sys.path``."""
    for path in reversed(python_search_paths()):
        if path not in sys.path:
            sys.path.insert(0, path)
