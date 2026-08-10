"""Tests for GBZ→GFA named-coordinates translation."""
from __future__ import annotations

import struct
from pathlib import Path

from engine.named_coords import (
    NamedCoordsIndex,
    build_named_coords_index,
    translate_gaf_line,
    translate_path,
)


def _write_pack(tmp: Path, lengths: list[int]) -> Path:
    pack = tmp / "pack"
    pack.mkdir()
    off = 0
    with (pack / "offsets.bin").open("wb") as fh, (pack / "ids.txt").open("w") as ids, (
        pack / "sequences.bin"
    ).open("wb") as seq:
        for i, ln in enumerate(lengths, start=1):
            fh.write(struct.pack("<QQ", off, ln))
            ids.write(f"{i}\n")
            seq.write(b"A" * ln)
            off += ln
    (pack / "meta.json").write_text(
        '{"format":"dense-v1","n_segments":%d,"n_bp":%d,"source":"test"}\n'
        % (len(lengths), off),
        encoding="utf-8",
    )
    return pack


def test_build_and_translate_collapse(tmp_path: Path):
    # GFA seg1 = nodes 1,2 (len 10+20); seg2 = node 3 (len 5)
    pack = _write_pack(tmp_path, [10, 20, 5])
    trans = tmp_path / "t.tsv"
    trans.write_text("T\t1\t1,2\nT\t2\t3\n", encoding="utf-8")
    idx_dir = build_named_coords_index(trans, pack, tmp_path / "idx")
    with NamedCoordsIndex(idx_dir) as idx:
        assert idx.lookup(1) == (1, 0)
        assert idx.lookup(2) == (1, 10)
        assert idx.lookup(3) == (2, 0)
        path, pstart = translate_path(">2", idx)
        assert path == ">1"
        assert pstart == 10
        path2, pstart2 = translate_path(">1>2", idx)
        assert path2 == ">1"
        assert pstart2 == 0
        path3, pstart3 = translate_path(">2>3", idx)
        assert path3 == ">1>2"
        assert pstart3 == 10

        line = (
            "r1\t15\t0\t15\t+\t>2\t15\t0\t15\t15\t15\t60\tcs:Z::15\n"
        )
        out = translate_gaf_line(line, idx).rstrip("\n").split("\t")
        assert out[5] == ">1"
        assert out[7] == "10"
        assert out[8] == "25"
