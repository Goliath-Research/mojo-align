"""Shim — replaces ``engine.grch38_offsets`` with ``giraffe/python/grch38_offsets.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('grch38_offsets', 'giraffe/python/grch38_offsets.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
