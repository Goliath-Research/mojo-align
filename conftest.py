"""Pytest path bootstrap for the mojo-align package layout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (
    ROOT / "methylgrapher",
    ROOT / "giraffe" / "python",
    ROOT / "giraffe" / "scripts",
    ROOT / "fq2bam-meth" / "python",
    ROOT / "gpu-common" / "python",
    ROOT,
):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
