"""Shim — replaces ``engine.gpu_h2d`` with ``giraffe/python/gpu_h2d.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('gpu_h2d', 'giraffe/python/gpu_h2d.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
