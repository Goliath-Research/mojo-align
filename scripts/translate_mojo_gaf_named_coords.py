#!/usr/bin/env python3
"""Translate Mojo Giraffe GAF paths from GBZ node ids to GFA named-coordinates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.named_coords import (  # noqa: E402
    NamedCoordsIndex,
    default_index_dir,
    index_ready,
    translate_gaf_file,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gaf", type=Path, help="input GAF (Mojo GBZ node ids)")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output GAF (default: in-place via temp sibling)",
    )
    ap.add_argument(
        "--index",
        type=Path,
        default=None,
        help="named_coords index dir (default: $METHYLGRAPHER_MOJO_SEGMENTS_CACHE/...)",
    )
    args = ap.parse_args()
    index_dir = args.index or default_index_dir()
    if not index_ready(index_dir):
        print(f"ERROR: named_coords index not ready: {index_dir}", file=sys.stderr)
        return 2

    in_gaf = args.gaf
    inplace = args.output is None
    out_gaf = args.output
    if inplace:
        tmp = tempfile.NamedTemporaryFile(
            prefix=in_gaf.name + ".",
            suffix=".named.gaf",
            dir=str(in_gaf.parent),
            delete=False,
        )
        out_gaf = Path(tmp.name)
        tmp.close()

    with NamedCoordsIndex(index_dir) as idx:
        n = translate_gaf_file(in_gaf, out_gaf, idx)

    stamp = {
        "format": "gbz-to-gfa-v1",
        "n_lines": n,
        "index": str(index_dir.resolve()),
        "source_gaf": str(in_gaf.resolve()),
    }
    if inplace:
        shutil.move(str(out_gaf), str(in_gaf))
        stamp_path = Path(str(in_gaf) + ".named_coords.json")
    else:
        stamp_path = Path(str(out_gaf) + ".named_coords.json")
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    print(f"translated {n} lines → {in_gaf if inplace else out_gaf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
