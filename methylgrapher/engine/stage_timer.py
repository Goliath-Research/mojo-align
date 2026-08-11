"""Shim — replaces ``engine.stage_timer`` with ``giraffe/python/stage_timer.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('stage_timer', 'giraffe/python/stage_timer.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
