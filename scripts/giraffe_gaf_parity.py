#!/usr/bin/env python3
"""CLI wrapper around engine.giraffe_gaf_parity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.giraffe_gaf_parity import compare, parse_gaf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mojo", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--require-extra-tags", action="store_true")
    args = ap.parse_args()
    return compare(
        parse_gaf(args.mojo),
        parse_gaf(args.golden),
        require_extra=args.require_extra_tags,
    )


if __name__ == "__main__":
    sys.exit(main())
