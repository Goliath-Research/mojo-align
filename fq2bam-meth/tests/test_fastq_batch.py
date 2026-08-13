"""Bulk FASTQ reader used by the FM mapper."""

from __future__ import annotations

import gzip
from pathlib import Path

from fastq_batch import FastqPairReader

FIX = Path(__file__).resolve().parent / "data" / "fq2bam_fixture"


def test_fastq_pair_reader_c2t_g2a():
    r = FastqPairReader(str(FIX / "R1.fastq"), str(FIX / "R2.fastq"), "C2T", "G2A")
    b = r.read_batch(8)
    r.close()
    assert b.n1 >= 1
    assert b.names1[0] == "read1"
    # fixture R1 is AT-only so C2T is unchanged
    assert b.seq1[0] == b.orig1[0]
    assert len(b.names2) == b.n1
    eof = FastqPairReader(str(FIX / "R1.fastq"), str(FIX / "R2.fastq"))
    _ = eof.read_batch(1000)
    empty = eof.read_batch(8)
    eof.close()
    assert empty.n1 == 0


def test_fastq_reader_gzip(tmp_path: Path):
    src = FIX / "R1.fastq"
    gz = tmp_path / "r1.fq.gz"
    with src.open("rb") as fin, gzip.open(gz, "wb") as fout:
        fout.write(fin.read())
    r = FastqPairReader(str(gz), "", "C2T", "")
    b = r.read_batch(8)
    r.close()
    assert b.n1 >= 1
    assert b.names1[0] == "read1"
    assert b.names2 == []
