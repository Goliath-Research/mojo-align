"""Unit tests for compare_linear_align_parity (no Clara / GPU required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "compare_linear_align_parity.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("compare_linear_align_parity", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_spearman_identical():
    mod = _load()
    xs = [1.0, 2.0, 3.0, 10.0]
    assert mod._spearman(xs, xs) == pytest.approx(1.0)


def test_spearman_inverse():
    mod = _load()
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [4.0, 3.0, 2.0, 1.0]
    assert mod._spearman(xs, ys) == pytest.approx(-1.0)


def test_compare_uses_align_path_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load()
    sample_id = "s1"
    clara_dir = tmp_path / "align.linear.parabricks"
    mojo_dir = tmp_path / "align.linear.mojo"
    clara_dir.mkdir()
    mojo_dir.mkdir()
    clara_bam = clara_dir / f"{sample_id}.bam"
    mojo_bam = mojo_dir / f"{sample_id}.bam"
    clara_bam.write_bytes(b"fake")
    mojo_bam.write_bytes(b"fake")

    def fake_flagstat(bam: Path):
        if "parabricks" in str(bam):
            return {
                "total": 100,
                "mapped": 90,
                "primary_mapped": 90,
                "mapped_rate": 0.90,
                "primary_mapped_rate": 0.90,
                "raw": "",
            }
        return {
            "total": 100,
            "mapped": 89,
            "primary_mapped": 89,
            "mapped_rate": 0.89,
            "primary_mapped_rate": 0.89,
            "raw": "",
        }

    monkeypatch.setattr(mod, "_run_flagstat", fake_flagstat)
    monkeypatch.setattr(
        mod,
        "_idxstats_counts",
        lambda bam: {"chr1": 50, "chr2": 40} if "parabricks" in str(bam) else {"chr1": 49, "chr2": 40},
    )

    report = mod.compare(
        sample_dir=tmp_path,
        sample_id=sample_id,
        max_delta=0.02,
        min_idxstats_spearman=0.95,
    )
    assert report["pass"] is True
    assert report["abs_delta_mapped_rate"] == pytest.approx(0.01)
