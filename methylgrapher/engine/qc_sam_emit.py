"""Shim — replaces ``engine.qc_sam_emit`` with ``giraffe/python/qc_sam_emit.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('qc_sam_emit', 'giraffe/python/qc_sam_emit.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
