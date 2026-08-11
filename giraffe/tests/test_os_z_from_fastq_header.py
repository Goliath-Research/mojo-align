"""MethylGrapher FASTQ header → GAF ``os:Z`` / ``rc:Z`` contract.

Converted FASTQ bodies (C→T / G→A) must not be emitted as ``os:Z``.
MethylCall scores met from the original bisulfite bases in ``os:Z``.
"""

from __future__ import annotations

from pathlib import Path

from engine.mcall import alignment_path_parse, alignment_to_methylation
from engine.quartet_map import (
    _parse_mg_fastq_header,
    _pe_extra_tags,
    _iter_fastq,
)


def test_parse_c2t_header_keeps_original_with_c() -> None:
    original = "ACGTACGTAC"
    converted = "ATGTATGTAT"  # C→T body
    rec = _parse_mg_fastq_header(f"read1_C2T_0_{original}", converted)
    assert rec.name == "read1"
    assert rec.seq == converted
    assert rec.original_seq == original
    assert "C" in rec.original_seq
    assert "C" not in rec.seq
    assert rec.conversion == "C2T"
    tags = _pe_extra_tags(1, rec, "GA")
    assert f"os:Z:{original}" in tags
    assert "rc:Z:CT" in tags
    assert f"os:Z:{converted}" not in tags


def test_parse_g2a_header_keeps_original_with_g() -> None:
    original = "GATCGA"
    converted = "AATCAA"  # G→A body
    rec = _parse_mg_fastq_header(f"mate2_G2A_1_{original}", converted)
    assert rec.name == "mate2"
    assert rec.original_seq == original
    assert rec.conversion == "G2A"
    tags = _pe_extra_tags(2, rec, "CT")
    assert f"os:Z:{original}" in tags
    assert "rc:Z:GA" in tags


def test_plain_fastq_falls_back_to_body() -> None:
    rec = _parse_mg_fastq_header("plainread", "ACGT")
    assert rec.name == "plainread"
    assert rec.original_seq == "ACGT"
    assert rec.conversion == ""
    tags = _pe_extra_tags(1, rec, "CT")
    assert "os:Z:ACGT" in tags
    assert "rc:Z:CT" in tags


def test_iter_fastq_mg_header(tmp_path: Path) -> None:
    fq = tmp_path / "c2t.fq"
    orig = "ACGCAC"
    body = "ATGTAT"
    fq.write_text(f"@q1_C2T_0_{orig}\n{body}\n+\n!!!!!!\n")
    rows = list(_iter_fastq(str(fq)))
    assert len(rows) == 1
    assert rows[0].original_seq == orig
    assert rows[0].seq == body


def test_converted_os_z_yields_unmet_original_yields_met() -> None:
    """Regression: os:Z=converted body → all unmet; original with C → met."""
    seqs = {"1": "CGTACGTACG"}
    converted = "TGTATGTATG"
    original = "CGTACGTACG"

    def _gaf_row(os_seq: str) -> list:
        # GAF cols 0–11 + tags; path pre-parsed like get_best_alignment_* does.
        return [
            "r",
            10,
            0,
            10,
            "+",
            alignment_path_parse(">1"),
            10,
            0,
            10,
            10,
            10,
            60,
            "cs:Z::10",
            f"os:Z:{os_seq}",
            "rc:Z:CT",
            "bq:Z:IIIIIIIIII",
        ]

    bad, _ = alignment_to_methylation([_gaf_row(converted)], seqs, cg_only=True)
    good, _ = alignment_to_methylation([_gaf_row(original)], seqs, cg_only=True)
    assert bad, "expected CG calls from converted os"
    assert all(row[4] == 0 for row in bad), (
        "converted os:Z must score every cytosine unmet"
    )
    assert good, "expected CG calls from original os"
    assert any(row[4] == 1 for row in good), (
        "original os:Z with C at CG must yield met=1"
    )
