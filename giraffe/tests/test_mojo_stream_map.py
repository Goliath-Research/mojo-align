"""Mojo stream map smoke + oracle parity on toy GBZ fixture."""

from __future__ import annotations

import os
import subprocess
import tempfile
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
SMOKE = PKG / "scripts/smoke_mojo_stream_map.mojo"


def test_oracle_quartet_map_still_golden(tmp_path, monkeypatch):
    """Python quartet_map remains the parity oracle."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("METHYLGRAPHER_MOJO_SEGMENTS_CACHE", str(cache_root))
    gbz = GBZ_TOY / "toy.wl.C2T.giraffe.gbz"
    if not gbz.is_file():
        pytest.skip("gbz toy missing")
    out = cache_root / (gbz.name + ".mojo_segments")
    build_dense_pack_from_gfa(str(GBZ_TOY / "toy.wl.gfa"), out, source_gbz=str(gbz))
    gaf = tmp_path / "oracle.gaf"
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


def test_mojo_stream_map_smoke():
    """Production path is Mojo stream map (not quartet_map hot loop)."""
    if not SMOKE.is_file():
        pytest.skip("smoke script missing")
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".pixi/bin") + os.pathsep + env.get("PATH", "")
    # Prefer pixi mojo when available.
    cmd = [
        "pixi",
        "run",
        "mojo",
        "-I",
        "gpu-common/src",
        "-I",
        "fq2bam-meth/src",
        "-I",
        "giraffe/src",
        "-I",
        "methylgrapher/src",
        str(SMOKE),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("pixi/mojo unavailable")
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, out[-2000:]
    assert "mojo_stream_map" in out
    assert "PASS" in out
    assert "quartet_map device=" not in out
