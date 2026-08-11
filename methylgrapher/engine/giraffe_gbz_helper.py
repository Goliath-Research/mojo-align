"""Shim — replaces ``engine.giraffe_gbz_helper`` with ``giraffe/python/giraffe_gbz_helper.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('giraffe_gbz_helper', 'giraffe/python/giraffe_gbz_helper.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
