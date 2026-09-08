"""XM/XG encoder parity vs sequence+XG cytosine calls."""

from __future__ import annotations

from meth_tags import (
    encode_bismark_xm,
    meth_call_from_seq_xg,
    meth_call_from_xm,
    xg_from_flag,
)


def test_xg_from_flag():
    assert xg_from_flag(99) == "CT"  # read1
    assert xg_from_flag(147) == "GA"  # read2


def test_xm_matches_sequence_xg_on_cpg_window():
    ref = "ACGTACGTAC"
    read = "ACGTATGTAC"  # unmeth C at index 5 (ref C, read T)
    cigar = [(0, 10)]
    xm = encode_bismark_xm(ref, read, cigar, "CT")
    assert len(xm) == 10
    for i, (r, b, xc) in enumerate(zip(ref, read, xm)):
        from_xm = meth_call_from_xm(xc)
        from_seq = meth_call_from_seq_xg(r, b, "CT")
        assert from_xm == from_seq, f"pos {i}: XM={xc!r} seq={from_seq} xm={from_xm}"
    assert xm[1] == "Z"  # ref C, read C
    assert xm[5] == "z"  # ref C, read T
