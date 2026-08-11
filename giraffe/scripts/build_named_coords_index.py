#!/usr/bin/env python3
"""Build GBZ→GFA named-coordinates index for Mojo Giraffe GAF emit/repair.

Example::

    vg gbwt -Z $GBZ --translation /tmp/c2t.trans.tsv
    python scripts/build_named_coords_index.py \\
      --translation /tmp/c2t.trans.tsv \\
      --pack /work/cache/mojo_segments/hprc-d9-bs.wl.C2T.giraffe.gbz.mojo_segments \\
      --out /work/cache/mojo_segments/hprc-d9-bs.wl.gbz_to_gfa.named_coords
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # mojo-align
sys.path.insert(0, str(ROOT / "methylgrapher"))
sys.path.insert(0, str(ROOT))

from engine.named_coords import build_named_coords_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--translation",
        required=True,
        type=Path,
        help="vg gbwt --translation TSV (T seg n1,n2,...)",
    )
    ap.add_argument(
        "--pack",
        required=True,
        type=Path,
        help="dense mojo_segments pack (for chopped-node lengths)",
    )
    ap.add_argument("--out", required=True, type=Path, help="output index directory")
    args = ap.parse_args()
    build_named_coords_index(args.translation, args.pack, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
