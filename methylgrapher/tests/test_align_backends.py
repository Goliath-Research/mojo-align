"""Unit tests for Align map backends (no vg / GPU required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.align_backends import (
    build_vg_giraffe_gaf_cmd,
    companion_gfa_for_index,
    gfa_usable_for_mojo,
    gpu_giraffe_fallback,
    normalize_align_engine,
    resolve_map_command,
)


def test_normalize_align_engine():
    assert normalize_align_engine("cpu_vg") == "cpu_vg"
    assert normalize_align_engine("gpu_giraffe") == "gpu_giraffe"
    assert normalize_align_engine("mojo_giraffe") == "mojo_giraffe"
    with pytest.raises(RuntimeError):
        normalize_align_engine("parabricks_bam")


def test_cpu_vg_command_has_named_coordinates_gaf():
    eng, cmd = resolve_map_command(
        align_engine="cpu_vg",
        vg_path="vg",
        thread=8,
        output_format="gaf",
        index_params="-Z x.gbz -d x.dist -m x.min -z x.zip",
        giraffe_input="-f a.fq -f b.fq",
    )
    assert eng == "cpu_vg"
    assert "--named-coordinates" in cmd
    assert "-o gaf" in cmd


def test_gpu_giraffe_prefers_gbz_quartet(tmp_path, monkeypatch):
    pref = tmp_path / "toy.wl.C2T"
    # touch quartet
    Path(str(pref) + ".giraffe.gbz").write_bytes(b"GBZ")
    Path(str(pref) + ".dist").write_bytes(b"D")
    Path(str(pref) + ".shortread.withzip.min").write_bytes(b"M")
    Path(str(pref) + ".shortread.zipcodes").write_bytes(b"Z")
    # huge GFA would have blocked old path
    gfa = tmp_path / "toy.wl.gfa"
    gfa.write_bytes(b"N" * 1000)
    bin_sh = tmp_path / "methylGrapher"
    bin_sh.write_text("#!/bin/sh\n")
    bin_sh.chmod(0o755)
    monkeypatch.setenv("METHYLGRAPHER_GPU_GIRAFFE_FALLBACK", "mojo")
    monkeypatch.setenv("MOJO_ALIGN_GIRAFFE_BIN", str(bin_sh))
    monkeypatch.setenv("MOJO_ALIGN_GIRAFFE_MAX_GFA_BYTES", "100")
    eng, cmd = resolve_map_command(
        align_engine="gpu_giraffe",
        vg_path="/usr/bin/vg",
        thread=64,
        output_format="gaf",
        index_params="-Z x.gbz -d x.dist",
        giraffe_input="-f a.fq -f b.fq",
        index_prefix=str(pref),
    )
    assert "mojo_gbz" in eng
    assert "MojoGiraffe" in cmd
    assert "-gbz" in cmd
    assert str(pref) + ".giraffe.gbz" in cmd or ".giraffe.gbz" in cmd


def test_gpu_giraffe_fallback_vg(monkeypatch):
    monkeypatch.setenv("METHYLGRAPHER_GPU_GIRAFFE_FALLBACK", "vg")
    eng, cmd = resolve_map_command(
        align_engine="gpu_giraffe",
        vg_path="/usr/bin/vg",
        thread=64,
        output_format="gaf",
        index_params="-Z x.gbz -d x.dist",
        giraffe_input="-f a.fq",
    )
    assert eng.startswith("gpu_giraffe")
    assert "--named-coordinates" in cmd


def test_gpu_giraffe_large_gbz_without_cache_falls_to_vg(tmp_path, monkeypatch):
    """Production GBZ without mojo_segments must not block Align on mid-run convert."""
    pref = tmp_path / "prod.wl.C2T"
    Path(str(pref) + ".giraffe.gbz").write_bytes(b"G" * (65 * 1024 * 1024))
    Path(str(pref) + ".dist").write_bytes(b"D")
    Path(str(pref) + ".shortread.withzip.min").write_bytes(b"M")
    Path(str(pref) + ".shortread.zipcodes").write_bytes(b"Z")
    bin_sh = tmp_path / "methylGrapher"
    bin_sh.write_text("#!/bin/sh\n")
    bin_sh.chmod(0o755)
    monkeypatch.setenv("METHYLGRAPHER_GPU_GIRAFFE_FALLBACK", "mojo")
    monkeypatch.setenv("MOJO_ALIGN_GIRAFFE_BIN", str(bin_sh))
    monkeypatch.setenv("MOJO_ALIGN_GBZ_DIRECT_MAX_BYTES", str(64 * 1024 * 1024))
    eng, cmd = resolve_map_command(
        align_engine="gpu_giraffe",
        vg_path="/usr/bin/vg",
        thread=64,
        output_format="gaf",
        index_params="-Z x.gbz -d x.dist -m x.min -z x.zip",
        giraffe_input="-f a.fq -f b.fq",
        index_prefix=str(pref),
    )
    assert eng == "gpu_giraffe+vg_autoscale"
    assert "vg giraffe" in cmd
    assert "MojoGiraffe" not in cmd
    assert ">&2" not in cmd  # vg path is a plain one-liner


def test_gpu_giraffe_fallback_error(monkeypatch):
    monkeypatch.setenv("METHYLGRAPHER_GPU_GIRAFFE_FALLBACK", "error")
    with pytest.raises(RuntimeError, match="unavailable|error"):
        resolve_map_command(
            align_engine="gpu_giraffe",
            vg_path="vg",
            thread=8,
            output_format="gaf",
            index_params="-Z x.gbz",
            giraffe_input="-f a.fq",
        )


def test_rejects_bam_output():
    with pytest.raises(RuntimeError, match="GAF"):
        build_vg_giraffe_gaf_cmd(
            vg_path="vg",
            thread=1,
            output_format="bam",
            index_params="-Z x",
            giraffe_input="-f a.fq",
        )


def test_env_default_engine(monkeypatch):
    monkeypatch.setenv("METHYLGRAPHER_ALIGN_ENGINE", "gpu_giraffe")
    monkeypatch.setenv("METHYLGRAPHER_GPU_GIRAFFE_FALLBACK", "mojo")
    assert normalize_align_engine(None) == "gpu_giraffe"
    assert gpu_giraffe_fallback() == "mojo"


def test_companion_gfa_and_usable(tmp_path):
    gfa = tmp_path / "hprc.wl.gfa"
    gfa.write_text("S\t1\tACGT\n")
    assert companion_gfa_for_index(str(tmp_path / "hprc.wl.C2T")) == str(gfa)
    assert gfa_usable_for_mojo(str(gfa))
