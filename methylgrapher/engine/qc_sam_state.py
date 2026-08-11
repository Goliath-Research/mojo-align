"""Shim — replaces ``engine.qc_sam_state`` with ``giraffe/python/qc_sam_state.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('qc_sam_state', 'giraffe/python/qc_sam_state.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
