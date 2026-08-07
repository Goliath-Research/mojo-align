#!/usr/bin/env python3
"""Build dense Mojo segment pack from GBZ (required for production) or GFA.

Production d9-bs: build **per strand** from each Giraffe GBZ via ``vg convert``
(companion ``wl.gfa`` node set ≠ C2T/G2A GBZ node set — do not symlink C2T↔G2A).

Usage:
  # Preferred production path:
  python scripts/build_mojo_segment_pack.py --from-gbz \\
    --gbz /work/genomes/.../hprc-d9-bs.wl.C2T.giraffe.gbz \\
    --out /work/cache/mojo_segments/hprc-d9-bs.wl.C2T.giraffe.gbz.mojo_segments

  # Fixture / toy GFA:
  python scripts/build_mojo_segment_pack.py \\
    --gfa tests/data/giraffe_fixture/gbz_toy/toy.wl.gfa \\
    --gbz .../toy.wl.C2T.giraffe.gbz --out /tmp/toy.mojo_segments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.segment_pack import (  # noqa: E402
    build_dense_pack_from_gfa,
    build_dense_pack_from_gbz,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfa", default="", help="GFA with S-lines (fixtures only)")
    ap.add_argument("--gbz", default="", help="Source GBZ path")
    ap.add_argument(
        "--from-gbz",
        action="store_true",
        help="Build pack by streaming vg convert from --gbz (production)",
    )
    ap.add_argument("--out", required=True, help="Output .mojo_segments directory")
    ap.add_argument("--vg", default="", help="vg binary (or set VG_PATH)")
    args = ap.parse_args()
    if args.from_gbz:
        if not args.gbz:
            ap.error("--from-gbz requires --gbz")
        build_dense_pack_from_gbz(args.gbz, args.out, vg_path=args.vg or None)
        return 0
    if not args.gfa:
        ap.error("provide --gfa or --from-gbz --gbz")
    build_dense_pack_from_gfa(args.gfa, args.out, source_gbz=args.gbz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
