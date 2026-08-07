#!/usr/bin/env python3
"""Build Mojo segment cache from a PrepareGenome Giraffe GBZ.

Usage:
  python scripts/build_mojo_gbz_cache.py \\
    --gbz /work/genomes/.../hprc-d9-bs.wl.C2T.giraffe.gbz

Creates ``{gbz}.mojo_segments/segments.jsonl`` for mmap-friendly reload.
Requires ``vg`` on PATH (or VG_PATH).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.giraffe_gbz_helper import ensure_segment_cache, probe_gbz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbz", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print(probe_gbz(args.gbz))
    d = ensure_segment_cache(args.gbz, force=args.force)
    print(f"cache_ready={d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
