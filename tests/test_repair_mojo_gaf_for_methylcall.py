"""Unit tests for Mojo GAF → MethylCall repair."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from repair_mojo_gaf_for_methylcall import _repair_line  # noqa: E402


def test_repair_adds_as_bq_and_full_cs():
    line = (
        "read1\t151\t0\t151\t+\t>1\t151\t0\t151\t151\t151\t20\t"
        "cs:Z::32\tri:i:1\tos:Z:" + ("A" * 151) + "\trc:Z:CT\n"
    )
    out = _repair_line(line).rstrip("\n").split("\t")
    tags = out[12:]
    assert "cs:Z::151" in tags
    assert "AS:i:151" in tags
    assert any(t.startswith("bq:Z:") and len(t) == 5 + 151 for t in tags)
    assert "rc:Z:CT" in tags


def test_repair_preserves_existing_as_bq():
    line = (
        "read1\t10\t0\t10\t+\t>1\t10\t0\t10\t10\t10\t60\t"
        "cs:Z::10\tAS:i:99\tbq:Z:" + ("I" * 10) + "\n"
    )
    tags = _repair_line(line).rstrip("\n").split("\t")[12:]
    assert "AS:i:99" in tags
    assert tags.count("AS:i:99") == 1
