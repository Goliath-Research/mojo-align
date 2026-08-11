"""Shim — replaces ``engine.quartet_map`` with ``giraffe/python/quartet_map.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('quartet_map', 'giraffe/python/quartet_map.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
