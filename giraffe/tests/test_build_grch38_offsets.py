"""Unit test for scripts/build_grch38_offsets.py (dense v1)."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # mojo-align repo root
PKG = Path(__file__).resolve().parents[1]  # giraffe/
sys.path.insert(0, str(PKG / "scripts"))

from build_grch38_offsets import UNSET, build  # noqa: E402


def test_build_grch38_offsets_dense(tmp_path: Path) -> None:
    gfa = tmp_path / "toy.wl.gfa"
    gfa.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.1",
                "S\t11\tACGTCGGA",
                "S\t12\tTTCGAA",
                "W\tGRCh38\t0\tchr1\t1000\t1014\t>11>12",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "offsets"
    meta = build(gfa, out, ref_fasta=None)
    assert meta["format"] == "grch38-dense-v1"
    assert meta["max_id"] == 12
    assert meta["n_set"] == 2
    chroms = (out / "chroms.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert chroms[0].startswith("1\t")
    rec = (out / "records.bin").read_bytes()
    assert len(rec) == 13 * 16
    # seg 11 → chrom 0, len 8, start 1000
    c, ln, st = struct.unpack_from("<IIQ", rec, 11 * 16)
    assert c == 0 and ln == 8 and st == 1000
    c12, ln12, st12 = struct.unpack_from("<IIQ", rec, 12 * 16)
    assert c12 == 0 and ln12 == 6 and st12 == 1008
    # unset id 1
    c1, _, _ = struct.unpack_from("<IIQ", rec, 1 * 16)
    assert c1 == UNSET
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["n_set"] == 2
