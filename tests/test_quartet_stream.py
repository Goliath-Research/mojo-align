"""Streaming FASTQ batches must not require loading the whole file."""

from __future__ import annotations

from pathlib import Path

from engine.quartet_map import _iter_fastq_batches, _read_batch_size


def test_iter_fastq_batches_pairs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("METHYLGRAPHER_MOJO_READ_BATCH", "2")
    assert _read_batch_size() == 2
    r1 = tmp_path / "r1.fq"
    r2 = tmp_path / "r2.fq"
    # 5 pairs
    body1 = "".join(f"@r{i}_1\nACGT\n+\n!!!!\n" for i in range(5))
    body2 = "".join(f"@r{i}_2\nTGCA\n+\n!!!!\n" for i in range(5))
    r1.write_text(body1)
    r2.write_text(body2)
    batches = list(_iter_fastq_batches(str(r1), str(r2)))
    assert [len(b) for b in batches] == [2, 2, 1]
    assert batches[0][0][0][0] == "r0"
    assert batches[-1][0][1][1] == "TGCA"


def test_iter_fastq_batches_mismatch(tmp_path: Path) -> None:
    r1 = tmp_path / "r1.fq"
    r2 = tmp_path / "r2.fq"
    r1.write_text("@a\nACGT\n+\n!!!!\n@b\nACGT\n+\n!!!!\n")
    r2.write_text("@a\nTGCA\n+\n!!!!\n")
    try:
        list(_iter_fastq_batches(str(r1), str(r2), batch_size=10))
        raise AssertionError("expected mismatch error")
    except RuntimeError as exc:
        assert "mismatch" in str(exc).lower()
