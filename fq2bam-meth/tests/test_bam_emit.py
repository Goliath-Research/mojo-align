"""BAM emit: BGZF framing + samtools-readable records."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from bam_emit import BamWriter, _pack_seq, _reg2bin


def test_pack_seq_and_bin():
    assert _pack_seq(b"ACGT") == bytes([0x12, 0x48])  # A=1 C=2 G=4 T=8
    assert _reg2bin(0, 1) == ((1 << 15) - 1) // 7


@pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools required")
def test_bam_writer_roundtrip(tmp_path: Path):
    bam = tmp_path / "t.bam"
    w = BamWriter(str(bam), ["chr1"], [1000], rg_id="mojo1", level=1)
    w.write_batch(
        qnames=["r1", "r1"],
        flags=[99, 147],
        tids=[0, 0],
        pos0s=[10, 50],
        mapqs=[60, 60],
        sls=[0, 0],
        srs=[0, 0],
        seqs=["ACGTACGTAC", "TGCATGCATG"],
        quals=["IIIIIIIIII", "IIIIIIIIII"],
        next_tids=[0, 0],
        next_pos0s=[50, 10],
        tlens=[50, -50],
        nms=[0, 1],
    )
    w.close()
    view = subprocess.run(
        ["samtools", "view", "-h", str(bam)],
        check=True,
        capture_output=True,
        text=True,
    )
    text = view.stdout
    assert "@SQ\tSN:chr1\tLN:1000" in text
    assert "@RG\tID:mojo1" in text
    lines = [ln for ln in text.splitlines() if not ln.startswith("@")]
    assert len(lines) == 2
    p = lines[0].split("\t")
    assert p[0] == "r1"
    assert p[1] == "99"
    assert p[2] == "chr1"
    assert p[3] == "11"  # 1-based
    assert p[5] == "10M"
    assert p[9] == "ACGTACGTAC"
    assert "RG:Z:mojo1" in lines[0]
    assert "NM:i:0" in lines[0]
    fs = subprocess.run(
        ["samtools", "flagstat", str(bam)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "2 + 0 mapped" in fs.stdout
