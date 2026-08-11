"""Golden GAF parity helper tests (no Mojo runtime required)."""

from __future__ import annotations

from pathlib import Path

from engine.giraffe_gaf_parity import compare, parse_gaf

ROOT = Path(__file__).resolve().parents[2]  # mojo-align repo root
PKG = Path(__file__).resolve().parents[1]  # giraffe/
GOLDEN = PKG / "tests/data/giraffe_fixture/golden.gaf"


def test_golden_parses():
    rows = parse_gaf(str(GOLDEN))
    assert len(rows) == 6
    assert ("readA", "ri:i:1") in rows
    assert rows[("readA", "ri:i:1")]["path"] == ">1"


def test_self_parity():
    rows = parse_gaf(str(GOLDEN))
    assert compare(rows, rows, require_extra=True) == 0
