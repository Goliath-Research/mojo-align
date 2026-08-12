"""Unit + integration tests for MojoFq2bamMeth / linear WGBS Align."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from engine.fq2bam_meth import (
    convert_fasta_c2t,
    convert_fastq,
    resolve_linear_engine,
    resolve_linear_mapper,
    run_mojo_fq2bam_meth,
    write_parabricks_shaped_metrics,
    _parse_flagstat,
    _parse_samtools_stats,
)

FIX = Path(__file__).resolve().parent / "data" / "fq2bam_fixture"


def test_convert_fasta_c2t(tmp_path: Path):
    src = tmp_path / "in.fa"
    dst = tmp_path / "out.fa"
    src.write_text(">chr1\nACGTCacgt\n")
    convert_fasta_c2t(src, dst)
    body = dst.read_text().splitlines()[1]
    assert body == "ATGTTaTgt"  # only C/c → T; other case preserved


def test_convert_fastq_modes(tmp_path: Path):
    src = tmp_path / "r.fq"
    src.write_text("@r\nACGT\n+\nIIII\n")
    c2t = tmp_path / "c2t.fq"
    g2a = tmp_path / "g2a.fq"
    convert_fastq(src, c2t, "C2T")
    convert_fastq(src, g2a, "G2A")
    assert c2t.read_text().splitlines()[1] == "ATGT"
    assert g2a.read_text().splitlines()[1] == "ACAT"


def test_resolve_linear_mapper_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("METHYLGRAPHER_LINEAR_MAPPER", raising=False)
    assert resolve_linear_mapper("auto") == "mojo"
    monkeypatch.setenv("METHYLGRAPHER_LINEAR_MAPPER", "bwa")
    assert resolve_linear_mapper("cpu") == "bwa"


def test_resolve_linear_engine_default_is_parity(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("METHYLGRAPHER_LINEAR_ENGINE", raising=False)
    assert resolve_linear_engine() == "parity"
    monkeypatch.setenv("METHYLGRAPHER_LINEAR_ENGINE", "science")
    assert resolve_linear_engine() == "parity"
    monkeypatch.setenv("METHYLGRAPHER_LINEAR_ENGINE", "speed")
    assert resolve_linear_engine() == "speed"
    monkeypatch.setenv("METHYLGRAPHER_LINEAR_ENGINE", "fast")
    assert resolve_linear_engine() == "speed"
    monkeypatch.setenv("METHYLGRAPHER_LINEAR_ENGINE", "fm")
    assert resolve_linear_engine() == "fm"
    monkeypatch.setenv("METHYLGRAPHER_LINEAR_ENGINE", "bwa-mem")
    assert resolve_linear_engine() == "fm"
    monkeypatch.setenv("METHYLGRAPHER_LINEAR_ENGINE", "vote")
    with pytest.raises(RuntimeError, match="parity.*speed.*fm"):
        resolve_linear_engine()


def test_metrics_marks_placeholders(tmp_path: Path):
    out = tmp_path / "m.json"
    write_parabricks_shaped_metrics(
        sample_id="s",
        out_json=out,
        flagstat={"total": 10, "mapped": 8},
        stats={"insert_size_average": 150.0, "average_quality": 37.0},
        mapper="mojo",
        device="cpu",
    )
    payload = json.loads(out.read_text())
    assert payload["metrics_source"] == "samtools+placeholders"
    assert "placeholder_fields" in payload
    assert payload["insert_size_metrics"]["median_insert_size"] == 150.0
    assert payload["alignment_summary"]["mapped_rate"] == 0.8


def test_parse_flagstat_and_stats():
    fs = _parse_flagstat(
        "100 + 0 in total\n80 + 0 mapped (80.00%)\n100 + 0 paired in sequencing\n"
    )
    assert fs["total"] == 100
    assert fs["mapped"] == 80
    st = _parse_samtools_stats(
        "SN\tinsert size average:\t180.5\nSN\taverage quality:\t36.2\n"
    )
    assert st["insert_size_average"] == 180.5
    assert st["average_quality"] == 36.2


@pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools required")
def test_end_to_end_mojo_mapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("METHYLGRAPHER_LINEAR_MAPPER", "mojo")
    out_bam = tmp_path / "out.bam"
    qc = tmp_path / "qc"
    result = run_mojo_fq2bam_meth(
        fq1=FIX / "R1.fastq",
        fq2=FIX / "R2.fastq",
        reference_fasta=FIX / "ref.fa",
        out_bam=out_bam,
        out_qc_dir=qc,
        sample_id="toy",
        threads=2,
        device="cpu",
        work_dir=tmp_path / "work",
        k=8,
    )
    assert out_bam.is_file()
    assert Path(str(out_bam) + ".bai").is_file() or Path(str(out_bam) + ".csi").is_file()
    metrics = json.loads((qc / "toy.json").read_text())
    assert metrics["engine"] == "mojo_fq2bam_meth"
    assert metrics["mapper"] in {"mojo", "bwa_fallback"}
    assert metrics["alignment_summary"]["mapped_reads"] >= 1
    assert result["bamPath"] == str(out_bam)


@pytest.mark.skipif(
    shutil.which("samtools") is None or shutil.which("bwa") is None,
    reason="bwa+samtools required",
)
def test_end_to_end_bwa_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("METHYLGRAPHER_LINEAR_MAPPER", "bwa")
    out_bam = tmp_path / "out.bam"
    qc = tmp_path / "qc"
    result = run_mojo_fq2bam_meth(
        fq1=FIX / "R1.fastq",
        fq2=FIX / "R2.fastq",
        reference_fasta=FIX / "ref.fa",
        out_bam=out_bam,
        out_qc_dir=qc,
        sample_id="toy",
        threads=2,
        device="cpu",
        work_dir=tmp_path / "work",
        k=8,
    )
    assert out_bam.is_file()
    assert result["mapper"] == "bwa"


def test_ensure_dense_pack_builds_under_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mojo_linear_pack import ensure_dense_pack, pack_is_complete

    monkeypatch.delenv("METHYLGRAPHER_LINEAR_CACHE_BUILD", raising=False)
    c2t = tmp_path / "ref.C2T.fa"
    convert_fasta_c2t(FIX / "ref.fa", c2t)
    cache = tmp_path / "ref.fa.mojo_linear_k8"
    logs: list[str] = []
    out = ensure_dense_pack(c2t_fasta=c2t, cache_dir=cache, k=8, log=logs.append)
    assert out == cache
    assert pack_is_complete(cache)
    assert (cache / "ref.fa").exists()
    assert Path(str(cache) + ".lock").is_file()
    # Second call is a no-op (already complete).
    n_before = len(logs)
    ensure_dense_pack(c2t_fasta=c2t, cache_dir=cache, k=8, log=logs.append)
    assert len(logs) == n_before  # no lock/build messages


def test_ensure_dense_pack_respects_build_disable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from mojo_linear_pack import ensure_dense_pack

    monkeypatch.setenv("METHYLGRAPHER_LINEAR_CACHE_BUILD", "0")
    c2t = tmp_path / "ref.C2T.fa"
    convert_fasta_c2t(FIX / "ref.fa", c2t)
    cache = tmp_path / "missing.mojo_linear_k8"
    with pytest.raises(RuntimeError, match="missing"):
        ensure_dense_pack(c2t_fasta=c2t, cache_dir=cache, k=8)
