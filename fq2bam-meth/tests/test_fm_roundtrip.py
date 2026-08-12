"""Unique k-mer BWT+SA round-trip on fleet bwameth.c2t (Mojo FmIndex)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PREFIX = Path(
    "/work/genomes/linear/GRCh38/ensembl-114/"
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa.bwameth.c2t"
)
REPO = Path(__file__).resolve().parents[2]


def _mojo_bin() -> list[str]:
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "run", "mojo"]
    mojo = shutil.which("mojo")
    if mojo:
        return [mojo]
    raise RuntimeError("mojo / pixi not found")


@pytest.mark.skipif(
    not Path(str(PREFIX) + ".bwt").is_file(),
    reason="fleet bwameth.c2t.bwt not present",
)
def test_fm_unique_kmer_bwt_sa_roundtrip():
    bwt = Path(str(PREFIX) + ".bwt")
    assert bwt.is_file(), f"missing {bwt}"
    cmd = _mojo_bin() + [
        "-I",
        "gpu-common/src",
        "-I",
        "fq2bam-meth/src",
        "-I",
        "giraffe/src",
        "-I",
        "methylgrapher/src",
        "fq2bam-meth/tests/fm_roundtrip.mojo",
    ]
    env = os.environ.copy()
    env.setdefault("MODULAR_NVPTX_COMPILER_PATH", "/usr/bin/ptxas")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise AssertionError(
            f"fm_roundtrip failed rc={proc.returncode}\n{out[-4000:]}"
        )
    assert "OK fm round-trip" in out, out[-2000:]
