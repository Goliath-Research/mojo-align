#!/usr/bin/env python3
"""CLI: build dense-v1 Mojo linear k-mer pack beside a C2T FASTA."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fq2bam-meth" / "python"))

from mojo_linear_pack import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
