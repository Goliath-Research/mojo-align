"""Quartet map parity on toy GBZ fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.giraffe_gaf_parity import compare, parse_gaf
from engine.quartet_map import map_fastq_to_gaf
from engine.segment_pack import build_dense_pack_from_gfa

ROOT = Path(__file__).resolve().parents[2]  # mojo-align repo root
PKG = Path(__file__).resolve().parents[1]  # giraffe/
GBZ_TOY = PKG / "tests/data/giraffe_fixture/gbz_toy"
GOLDEN = PKG / "tests/data/giraffe_fixture/golden.gaf"
R1 = PKG / "tests/data/giraffe_fixture/R1.fastq"
R2 = PKG / "tests/data/giraffe_fixture/R2.fastq"


def test_toy_quartet_map_parity(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("MOJO_ALIGN_SEGMENTS_CACHE", str(cache_root))
    gbz = GBZ_TOY / "toy.wl.C2T.giraffe.gbz"
    if not gbz.is_file():
        pytest.skip("gbz toy missing")
    # Prebuild dense pack from companion GFA
    out = cache_root / (gbz.name + ".mojo_segments")
    build_dense_pack_from_gfa(str(GBZ_TOY / "toy.wl.gfa"), out, source_gbz=str(gbz))
    gaf = tmp_path / "out.gaf"
    n = map_fastq_to_gaf(
        gbz=str(gbz),
        fq1=str(R1),
        fq2=str(R2),
        out_gaf=str(gaf),
        dist=str(GBZ_TOY / "toy.wl.C2T.dist"),
        min_path=str(GBZ_TOY / "toy.wl.C2T.shortread.withzip.min"),
        zipcodes=str(GBZ_TOY / "toy.wl.C2T.shortread.zipcodes"),
        k=5,
    )
    assert n == 6
    assert compare(parse_gaf(str(gaf)), parse_gaf(str(GOLDEN)), require_extra=True) == 0
