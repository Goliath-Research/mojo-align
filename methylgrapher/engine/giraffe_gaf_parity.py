"""Shim — replaces ``engine.giraffe_gaf_parity`` with ``giraffe/python/giraffe_gaf_parity.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('giraffe_gaf_parity', 'giraffe/python/giraffe_gaf_parity.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
