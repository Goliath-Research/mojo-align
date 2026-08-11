"""Shim — replaces ``engine.min_mmap_bridge`` with ``giraffe/python/min_mmap_bridge.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('min_mmap_bridge', 'giraffe/python/min_mmap_bridge.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
