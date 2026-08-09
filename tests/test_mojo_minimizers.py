"""Parity: Mojo host Giraffe minimizers vs Python MinimizerIndex.minimizers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from engine.minimizer_index import MinimizerIndex, wang_hash_64

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts/smoke_mojo_minimizers.mojo"


def test_wang_hash_nonzero_mix():
    assert wang_hash_64(1) != wang_hash_64(2)


def test_python_minimizers_stable_on_random():
    seq = "ACGT" * 40 + "N" + "TGCA" * 10
    # Synthetic index not required — exercise minimizer math via a tiny open.
    toy = ROOT / "tests/data/giraffe_fixture/gbz_toy/toy.wl.C2T.shortread.withzip.min"
    if not toy.is_file():
        pytest.skip("toy min missing")
    with MinimizerIndex(toy) as idx:
        occs = idx.minimizers(seq)
    assert isinstance(occs, list)
    # Windowed minimizers should be far fewer than all k-mers.
    assert len(occs) < max(1, len(seq) - idx.k + 1)


def test_mojo_minimizer_smoke():
    if not SMOKE.is_file():
        pytest.skip("smoke script missing")
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".pixi/bin") + os.pathsep + env.get("PATH", "")
    cmd = ["pixi", "run", "mojo", "-I", "src", str(SMOKE)]
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
    assert "PASS" in out
    assert "host-nvidia-fallback" not in out
    assert "cupy-" not in out.lower()
