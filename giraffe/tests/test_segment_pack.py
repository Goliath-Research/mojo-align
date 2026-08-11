"""Dense segment pack build/load."""

from __future__ import annotations

from pathlib import Path

from engine.segment_pack import SegmentPack, build_dense_pack_from_gfa, pack_ready

ROOT = Path(__file__).resolve().parents[2]  # mojo-align repo root
PKG = Path(__file__).resolve().parents[1]  # giraffe/
TOY_GFA = PKG / "tests/data/giraffe_fixture/gbz_toy/toy.wl.gfa"


def test_build_and_load_dense(tmp_path):
    out = tmp_path / "toy.mojo_segments"
    build_dense_pack_from_gfa(str(TOY_GFA), out, source_gbz="toy.gbz")
    assert pack_ready(out)
    pack = SegmentPack(out)
    assert len(pack) == 3
    assert pack.get("1") == "ACGTACGTAC"
    assert pack.get("3") == "TTTTAAAACC"
