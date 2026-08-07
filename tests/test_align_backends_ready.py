"""align_backends READY gate keeps production on vg until cutover."""

from __future__ import annotations

import os
from pathlib import Path

from engine.align_backends import resolve_map_command

ROOT = Path(__file__).resolve().parents[1]
GBZ_TOY = ROOT / "tests/data/giraffe_fixture/gbz_toy"


def test_gpu_giraffe_without_ready_autoscale_vg(monkeypatch, tmp_path):
    monkeypatch.delenv("METHYLGRAPHER_MOJO_GIRAFFE_READY", raising=False)
    monkeypatch.setenv("METHYLGRAPHER_GPU_GIRAFFE_FALLBACK", "mojo")
    prefix = str(GBZ_TOY / "toy.wl.C2T")
    eng, cmd = resolve_map_command(
        align_engine="gpu_giraffe",
        vg_path="vg",
        thread=4,
        output_format="gaf",
        index_params=f"-Z {prefix}.giraffe.gbz",
        giraffe_input="-f /tmp/r1.fq -f /tmp/r2.fq",
        index_prefix=prefix,
    )
    assert eng == "gpu_giraffe+vg_autoscale"
    assert "vg giraffe" in cmd


def test_gpu_giraffe_with_ready_selects_mojo(monkeypatch, tmp_path):
    monkeypatch.setenv("METHYLGRAPHER_MOJO_GIRAFFE_READY", "1")
    monkeypatch.setenv("METHYLGRAPHER_GPU_GIRAFFE_FALLBACK", "mojo")
    monkeypatch.setenv("METHYLGRAPHER_MOJO_GBZ_DIRECT_MAX_BYTES", str(10**12))
    # Point mojo bin at a fake executable
    fake = tmp_path / "methylGrapher"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("METHYLGRAPHER_MOJO_GIRAFFE_BIN", str(fake))
    # tiny gbz always "direct" when max_direct huge; segment cache not required
    prefix = str(GBZ_TOY / "toy.wl.C2T")
    eng, cmd = resolve_map_command(
        align_engine="gpu_giraffe",
        vg_path="vg",
        thread=4,
        output_format="gaf",
        index_params=f"-Z {prefix}.giraffe.gbz",
        giraffe_input="-f /tmp/r1.fq -f /tmp/r2.fq",
        index_prefix=prefix,
    )
    assert "mojo_gbz" in eng
    assert "MojoGiraffe" in cmd or "methylGrapher" in cmd
