"""GBZ helper + toy fixture tests (needs vg for convert; skip if absent)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from engine.giraffe_gbz_helper import (
    map_gbz_fastq_to_gaf,
    resolve_gbz_quartet,
    stream_gbz_to_segments,
)
from engine.giraffe_gaf_parity import compare, parse_gaf

ROOT = Path(__file__).resolve().parents[1]
GBZ_TOY = ROOT / "tests/data/giraffe_fixture/gbz_toy"
GOLDEN = ROOT / "tests/data/giraffe_fixture/golden.gaf"
R1 = ROOT / "tests/data/giraffe_fixture/R1.fastq"
R2 = ROOT / "tests/data/giraffe_fixture/R2.fastq"

pytestmark = pytest.mark.skipif(
    shutil.which("vg") is None and not Path("/usr/local/bin/vg").exists(),
    reason="vg not on PATH",
)


def _vg() -> str:
    return shutil.which("vg") or "/usr/local/bin/vg"


def test_resolve_quartet_toy():
    q = resolve_gbz_quartet(str(GBZ_TOY / "toy.wl.C2T"))
    assert q is not None
    assert Path(q["gbz"]).is_file()
    assert Path(q["dist"]).is_file()
    assert Path(q["min"]).is_file()


def test_stream_and_map_parity(tmp_path):
    gbz = GBZ_TOY / "toy.giraffe.gbz"
    if not gbz.is_file():
        pytest.skip("gbz toy missing — run vg autoindex fixture build")
    segs = stream_gbz_to_segments(str(gbz), vg_path=_vg())
    assert len(segs) >= 3
    out = tmp_path / "mojo_gbz.gaf"
    n = map_gbz_fastq_to_gaf(
        gbz=str(gbz),
        fq1=str(R1),
        fq2=str(R2),
        out_gaf=str(out),
        k=5,
    )
    assert n == 6
    assert compare(parse_gaf(str(out)), parse_gaf(str(GOLDEN)), require_extra=True) == 0
