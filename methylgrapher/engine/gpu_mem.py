"""Shim — replaces ``engine.gpu_mem`` with ``gpu-common/python/gpu_mem.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('gpu_mem', 'gpu-common/python/gpu_mem.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
