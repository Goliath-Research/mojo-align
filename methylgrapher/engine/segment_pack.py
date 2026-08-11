"""Shim — replaces ``engine.segment_pack`` with ``giraffe/python/segment_pack.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('segment_pack', 'giraffe/python/segment_pack.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
