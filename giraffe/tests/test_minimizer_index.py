"""MinimizerIndex header + toy/prod probe tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.minimizer_index import MinimizerIndex, probe_minimizer, wang_hash_64

ROOT = Path(__file__).resolve().parents[2]  # mojo-align repo root
PKG = Path(__file__).resolve().parents[1]  # giraffe/
TOY_MIN = PKG / "tests/data/giraffe_fixture/gbz_toy/toy.wl.C2T.shortread.withzip.min"
PROD_MIN = Path(
    "/lambda/nfs/Work/genomes/pangenome/GRCh38/d9-bs/1.70/"
    "hprc-d9-bs.wl.C2T.shortread.withzip.min"
)


def test_wang_hash_stable():
    assert wang_hash_64(0) != 0 or wang_hash_64(1) != wang_hash_64(2)


def test_toy_min_header():
    info = probe_minimizer(TOY_MIN)
    assert info["k"] == 29
    assert info["w"] == 11
    assert info["cell_count"] == 1024
    assert info["n_keys"] == 0


def test_toy_locate_empty():
    with MinimizerIndex(TOY_MIN) as idx:
        assert idx.locate_read("ACGTACGTAC") == []


@pytest.mark.skipif(not PROD_MIN.is_file(), reason="production min not mounted")
def test_prod_min_header():
    info = probe_minimizer(PROD_MIN)
    assert info["k"] == 29
    assert info["n_keys"] > 1000
    assert info["cell_count"] == 1 << 30
