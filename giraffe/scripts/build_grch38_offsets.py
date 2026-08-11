#!/usr/bin/env python3
"""Build dense GRCh38 segment→(chrom,start,len) offsets from wl.gfa.

Output directory layout (grch38-dense-v1):

  meta.json      format + max_id + chrom_count
  chroms.tsv     chrom\\tlength (SQ lengths; 0 if unknown)
  records.bin    (max_id+1) × 16-byte LE records:
                   u32 chrom_idx (0xFFFFFFFF = unset)
                   u32 length
                   u64 start_0based

Numeric segment IDs only (HPRC d9-bs). Used by MojoGiraffe QC SAM emit.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RECORD = struct.Struct("<IIQ")  # chrom_idx, length, start
UNSET = 0xFFFFFFFF
SEG_RE = re.compile(r"([><])([^><]+)")


def _is_grch38_sample(name: str) -> bool:
    """True when a PanSN sample name denotes the GRCh38 linear reference."""
    head = name.strip().split("#", 1)[0].upper()
    return head == "HG38" or head.startswith("GRCH38")


def _normalize_chrom(name: str) -> Optional[str]:
    raw = str(name).strip()
    if not raw:
        return None
    if "#" in raw:
        if not _is_grch38_sample(raw):
            return None
        raw = raw.split("#")[-1]
    if raw.lower().startswith("chr"):
        raw = raw[3:]
    if raw.upper() == "M":
        return "MT"
    if raw in {"X", "Y", "MT"} or raw.isdigit():
        return raw
    return None


def _fasta_lengths(fa: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not fa.is_file():
        return out
    chrom: Optional[str] = None
    n = 0
    with fa.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if chrom is not None:
                    out[chrom] = n
                name = line[1:].split()[0]
                chrom = _normalize_chrom(name) or name
                if chrom.lower().startswith("chr"):
                    chrom = chrom[3:]
                n = 0
            else:
                n += len(line.strip())
        if chrom is not None:
            out[chrom] = n
    return out


def build(
    gfa: Path,
    out_dir: Path,
    *,
    ref_fasta: Optional[Path] = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_len: Dict[str, int] = {}
    # First pass: lengths + max numeric id + GRCh38 walk offsets (same as runner).
    offsets: Dict[str, Tuple[str, int, int]] = {}  # id -> chrom, start, length
    max_id = 0
    n_s = n_w = 0
    with gfa.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line[0] not in "SWP":
                continue
            if line[0] == "S":
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 3:
                    continue
                seg_id, seq = parts[1], parts[2]
                if seq == "*":
                    length = 0
                    for tag in parts[3:]:
                        if tag.startswith("LN:i:"):
                            length = int(tag[5:])
                            break
                else:
                    length = len(seq)
                seg_len[seg_id] = length
                n_s += 1
                if seg_id.isdigit():
                    max_id = max(max_id, int(seg_id))
                continue
            parts = line.rstrip("\r\n").split("\t")
            path_seq = ""
            chrom: Optional[str] = None
            path_start = 0
            if line[0] == "W" and len(parts) >= 7:
                if not _is_grch38_sample(parts[1]):
                    continue
                chrom = _normalize_chrom(parts[3])
                if chrom is None:
                    continue
                try:
                    path_start = int(parts[4])
                except ValueError:
                    path_start = 0
                path_seq = parts[6]
                n_w += 1
            elif line[0] == "P" and len(parts) >= 3:
                chrom = _normalize_chrom(parts[1])
                if chrom is None:
                    continue
                path_seq = "".join(
                    (">" if tok.endswith("+") else "<") + tok[:-1]
                    for tok in parts[2].split(",")
                    if tok
                )
            else:
                continue
            cursor = path_start
            for m in SEG_RE.finditer(path_seq):
                _orient, seg_id = m.group(1), m.group(2)
                length = int(seg_len.get(seg_id, 0))
                if seg_id not in offsets and chrom is not None:
                    offsets[seg_id] = (chrom, cursor, length)
                cursor += length

    if max_id <= 0:
        raise SystemExit(f"no numeric S-line ids in {gfa}")

    chrom_order: List[str] = []
    chrom_index: Dict[str, int] = {}
    for _sid, (chrom, _st, _ln) in offsets.items():
        if chrom not in chrom_index:
            chrom_index[chrom] = len(chrom_order)
            chrom_order.append(chrom)
    # Stable canonical order for human autosomes + XY/MT when present.
    prefer = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
    ordered = [c for c in prefer if c in chrom_index] + [
        c for c in chrom_order if c not in prefer
    ]
    chrom_index = {c: i for i, c in enumerate(ordered)}
    chrom_order = ordered

    fa_lens = _fasta_lengths(ref_fasta) if ref_fasta else {}
    chroms_path = out_dir / "chroms.tsv"
    with chroms_path.open("w", encoding="utf-8") as fh:
        for c in chrom_order:
            fh.write(f"{c}\t{int(fa_lens.get(c, 0))}\n")

    n_rec = max_id + 1
    rec_path = out_dir / "records.bin"
    # Sparse fill via bytearray would need ~1GiB; write mmap-style file.
    import numpy as np

    chrom_idx = np.full(n_rec, UNSET, dtype=np.uint32)
    lengths = np.zeros(n_rec, dtype=np.uint32)
    starts = np.zeros(n_rec, dtype=np.uint64)
    n_set = 0
    for sid, (chrom, start, length) in offsets.items():
        if not sid.isdigit():
            continue
        i = int(sid)
        if i < 0 or i > max_id:
            continue
        chrom_idx[i] = np.uint32(chrom_index[chrom])
        lengths[i] = np.uint32(length)
        starts[i] = np.uint64(start)
        n_set += 1

    # Interleave into records.bin
    out = np.empty(n_rec, dtype=[("c", "<u4"), ("l", "<u4"), ("s", "<u8")])
    out["c"] = chrom_idx
    out["l"] = lengths
    out["s"] = starts
    out.tofile(rec_path)

    meta = {
        "format": "grch38-dense-v1",
        "max_id": max_id,
        "record_bytes": 16,
        "chrom_count": len(chrom_order),
        "n_set": n_set,
        "n_s_lines": n_s,
        "n_w_lines": n_w,
        "source_gfa": str(gfa.resolve()),
        "ref_fasta": str(ref_fasta.resolve()) if ref_fasta else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-gfa", type=Path, required=True)
    ap.add_argument("-out_dir", type=Path, required=True)
    ap.add_argument("-ref", type=Path, default=None, help="linear FASTA for @SQ lengths")
    args = ap.parse_args()
    build(args.gfa, args.out_dir, ref_fasta=args.ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
