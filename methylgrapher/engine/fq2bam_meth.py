"""Shim — replaces ``engine.fq2bam_meth`` with ``fq2bam-meth/python/fq2bam_meth.py`` in sys.modules."""

from __future__ import annotations

import sys

from ._pkg_shim import load_sibling as _load_sibling

_impl = _load_sibling('fq2bam_meth', 'fq2bam-meth/python/fq2bam_meth.py')
# Identity swap so monkeypatch / ``is`` checks hit the real implementation.
sys.modules[__name__] = _impl
