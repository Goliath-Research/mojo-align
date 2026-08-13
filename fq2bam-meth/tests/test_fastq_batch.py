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


def test_pack_align_and_export_ptrs():
    import ctypes

    r = FastqPairReader(str(FIX / "R1.fastq"), str(FIX / "R2.fastq"), "C2T", "G2A")
    b = r.read_batch(8)
    r.close()
    assert b.n1 >= 1
    assert isinstance(b.seq1[0], bytearray)
    max_len = max(b.max_len, 8)
    n_seq = b.n1 * 2
    dest = (ctypes.c_uint8 * (n_seq * max_len))()
    b.pack_align(ctypes.addressof(dest), max_len, 1)
    slen = len(b.seq1[0])
    assert bytes(dest[0:slen]) == bytes(b.seq1[0])
    if slen < max_len:
        assert dest[slen] == 78
    else:
        dest2 = (ctypes.c_uint8 * (n_seq * (max_len + 4)))()
        b.pack_align(ctypes.addressof(dest2), max_len + 4, 1)
        assert dest2[slen] == 78
    name_a = (ctypes.c_uint64 * b.n1)()
    name_n = (ctypes.c_uint32 * b.n1)()
    orig_a = (ctypes.c_uint64 * b.n1)()
    orig_n = (ctypes.c_uint32 * b.n1)()
    qual_a = (ctypes.c_uint64 * b.n1)()
    qual_n = (ctypes.c_uint32 * b.n1)()
    n2a = (ctypes.c_uint64 * b.n1)()
    n2n = (ctypes.c_uint32 * b.n1)()
    o2a = (ctypes.c_uint64 * b.n1)()
    o2n = (ctypes.c_uint32 * b.n1)()
    q2a = (ctypes.c_uint64 * b.n1)()
    q2n = (ctypes.c_uint32 * b.n1)()
    b.export_ptrs(
        1,
        ctypes.addressof(name_a),
        ctypes.addressof(name_n),
        ctypes.addressof(orig_a),
        ctypes.addressof(orig_n),
        ctypes.addressof(qual_a),
        ctypes.addressof(qual_n),
        ctypes.addressof(n2a),
        ctypes.addressof(n2n),
        ctypes.addressof(o2a),
        ctypes.addressof(o2n),
        ctypes.addressof(q2a),
        ctypes.addressof(q2n),
    )
    assert name_n[0] == len(b.names1[0])
    assert orig_n[0] == len(b.orig1[0])
    raw = (ctypes.c_char * orig_n[0]).from_address(orig_a[0])
    assert raw.raw == bytes(b.orig1[0])
