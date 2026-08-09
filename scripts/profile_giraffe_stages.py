#!/usr/bin/env python3
"""Profile quartet_map stage timings (seed/locate/extend/GAF) on a GBZ fixture.

Usage:
  python3 scripts/profile_giraffe_stages.py \\
    --gbz tests/data/giraffe_fixture/gbz_toy/toy.giraffe.gbz \\
    --fq1 tests/data/giraffe_fixture/R1.fastq \\
    --fq2 tests/data/giraffe_fixture/R2.fastq \\
    --min tests/data/giraffe_fixture/gbz_toy/toy.shortread.withzip.min \\
    --out /tmp/giraffe_stage_profile.json

Emits STAGE_TIMER JSON (also written to --out) for docs/BENCHMARK_GIRAFFE.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    fix = root / "tests/data/giraffe_fixture"
    ap.add_argument(
        "--gbz",
        default=str(fix / "gbz_toy/toy.giraffe.gbz"),
    )
    ap.add_argument("--fq1", default=str(fix / "R1.fastq"))
    ap.add_argument("--fq2", default=str(fix / "R2.fastq"))
    ap.add_argument(
        "--min",
        default=str(fix / "gbz_toy/toy.shortread.withzip.min"),
    )
    ap.add_argument("--dist", default=str(fix / "gbz_toy/toy.dist"))
    ap.add_argument("--zipcodes", default="")
    ap.add_argument("--device", default=os.environ.get("METHYLGRAPHER_GIRAFFE_DEVICE", "cpu"))
    ap.add_argument("--out", default="/tmp/giraffe_stage_profile.json")
    ap.add_argument("--gaf", default="/tmp/giraffe_stage_profile.gaf")
    args = ap.parse_args()

    sys.path.insert(0, str(root))
    os.environ["METHYLGRAPHER_PROFILE_STAGES"] = "1"
    os.environ["METHYLGRAPHER_PROFILE_JSON"] = args.out
    os.environ.setdefault("METHYLGRAPHER_MOJO_READ_BATCH", "256")

    from engine.quartet_map import map_fastq_to_gaf

    t0 = time.perf_counter()
    n = map_fastq_to_gaf(
        gbz=args.gbz,
        fq1=args.fq1,
        out_gaf=args.gaf,
        fq2=args.fq2,
        dist=args.dist if Path(args.dist).is_file() else "",
        min_path=args.min if Path(args.min).is_file() else "",
        zipcodes=args.zipcodes,
        k=5,
        device=args.device,
    )
    wall = time.perf_counter() - t0
    report = {"wall_s": round(wall, 6), "n_records": n, "device": args.device}
    if Path(args.out).is_file():
        stages = json.loads(Path(args.out).read_text())
        report["stages"] = stages
    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
