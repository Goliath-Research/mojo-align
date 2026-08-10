#!/usr/bin/env python3
"""Repair MojoGiraffe GAF rows for methylGrapher MethylCall.

Mojo emit historically omitted AS:i: / bq:Z: and wrote short cs:Z::matched
while claiming full-query coordinates (qstart=0,qend=qlen). MethylCall
requires AS:i:, full-span cs for those coordinates, and bq:Z: qualities.

Streams stdin → stdout (or -i in-place via temp sibling).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _repair_line(line: str) -> str:
    raw = line.rstrip("\n")
    if not raw or raw.startswith("#"):
        return line if line.endswith("\n") else line + "\n"
    parts = raw.split("\t")
    if len(parts) < 12:
        return line if line.endswith("\n") else line + "\n"
    try:
        qlen = int(parts[1])
    except ValueError:
        return line if line.endswith("\n") else line + "\n"

    tags = parts[12:]
    have_as = False
    have_bq = False
    new_tags: list[str] = []
    for t in tags:
        if t.startswith("cs:Z:"):
            new_tags.append(f"cs:Z::{qlen}")
        elif t.startswith("AS:i:"):
            have_as = True
            new_tags.append(t)
        elif t.startswith("bq:Z:"):
            have_bq = True
            new_tags.append(t)
        else:
            new_tags.append(t)
    # Ensure a cs tag exists even if Mojo omitted it.
    if not any(t.startswith("cs:Z:") for t in new_tags):
        new_tags.insert(0, f"cs:Z::{qlen}")
    if not have_as:
        # Column 9 is matches; Mojo full-span emit uses qlen there.
        try:
            matches = int(parts[9])
        except (ValueError, IndexError):
            matches = qlen
        new_tags.append(f"AS:i:{matches}")
    if not have_bq and qlen > 0:
        # Synthetic high quality (Phred+33 'I' = 40) — Mojo path had no FASTQ quals.
        new_tags.append("bq:Z:" + ("I" * qlen))
    parts = parts[:12] + new_tags
    return "\t".join(parts) + "\n"


def repair_stream(fin, fout) -> int:
    n = 0
    for line in fin:
        fout.write(_repair_line(line))
        n += 1
        if n % 2_000_000 == 0:
            print(f"repaired {n} lines…", file=sys.stderr, flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gaf", type=Path, help="alignment.gaf path")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path (default: <gaf>.methylcall_fixed)",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="atomically replace input (keeps <gaf>.pre_methylcall_fix.bak)",
    )
    args = ap.parse_args()
    src: Path = args.gaf
    if not src.is_file():
        print(f"missing {src}", file=sys.stderr)
        return 2
    out = args.output or src.with_suffix(src.suffix + ".methylcall_fixed")
    if args.replace:
        fd, tmp_name = tempfile.mkstemp(
            prefix=src.name + ".", suffix=".tmp", dir=str(src.parent)
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with src.open("r", encoding="utf-8", errors="replace") as fin, tmp.open(
                "w", encoding="utf-8"
            ) as fout:
                n = repair_stream(fin, fout)
            bak = Path(str(src) + ".pre_methylcall_fix.bak")
            if not bak.exists():
                src.rename(bak)
            else:
                src.unlink()
            tmp.rename(src)
            print(f"replaced {src} ({n} lines); backup {bak}", file=sys.stderr)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
    else:
        with src.open("r", encoding="utf-8", errors="replace") as fin, out.open(
            "w", encoding="utf-8"
        ) as fout:
            n = repair_stream(fin, fout)
        print(f"wrote {out} ({n} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
