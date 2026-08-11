"""QC SAM emit must set PE flags from Mojo ri:i tags."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine import grch38_offsets as off


@pytest.fixture()
def offsets_root(tmp_path: Path) -> Path:
    """Minimal dense table: segment 1 → chr1:0 length 1000."""
    root = tmp_path / "offsets"
    root.mkdir()
    (root / "meta.json").write_text(
        '{"format":"grch38-dense-v1","max_id":1}', encoding="utf-8"
    )
    (root / "chroms.tsv").write_text("chr1\t1000\n", encoding="utf-8")
    # record: cidx=0, length=1000, start0=0
    import struct

    (root / "records.bin").write_bytes(
        struct.pack("<IIQ", 0xFFFFFFFF, 0, 0) + struct.pack("<IIQ", 0, 1000, 0)
    )
    return root


def test_append_hits_sets_read_and_proper_pair_flags(tmp_path: Path, offsets_root: Path) -> None:
    sam = tmp_path / "qc.sam"
    fh = off.open_sam(str(sam), str(offsets_root))
    try:
        n = off.append_hits(
            fh,
            [
                ("q1", ">1", 60, "ri:i:1\tos:Z:ACGTACGT\trc:Z:CT"),
                ("q1", ">1", 60, "ri:i:2\tos:Z:TGCATGCA\trc:Z:GA"),
            ],
        )
    finally:
        off.close_sam(fh)
    assert n == 2
    body = [
        ln for ln in sam.read_text(encoding="utf-8").splitlines() if not ln.startswith("@")
    ]
    assert len(body) == 2
    f1 = int(body[0].split("\t")[1])
    f2 = int(body[1].split("\t")[1])
    assert f1 & 0x1  # paired
    assert f1 & 0x40  # read1
    assert f1 & 0x2  # proper pair (mate in batch)
    assert f2 & 0x1
    assert f2 & 0x80  # read2
    assert f2 & 0x2
    # Mate coords filled
    assert body[0].split("\t")[6] == "="
    assert int(body[0].split("\t")[7]) > 0
