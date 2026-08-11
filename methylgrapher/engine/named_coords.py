"""Shim — replaces ``engine.named_coords`` with ``giraffe/python/named_coords.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('named_coords', 'giraffe/python/named_coords.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
