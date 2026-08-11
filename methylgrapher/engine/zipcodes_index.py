"""Shim — replaces ``engine.zipcodes_index`` with ``giraffe/python/zipcodes_index.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('zipcodes_index', 'giraffe/python/zipcodes_index.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
