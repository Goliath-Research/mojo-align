"""GPU/CPU minimizer batch parity for quartet_map seed stage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # mojo-align repo root
PKG = Path(__file__).resolve().parents[1]  # giraffe/
sys.path.insert(0, str(ROOT / "methylgrapher"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "scripts"))

from engine.minimizer_index import MinimizerIndex  # noqa: E402
from giraffe_gpu_minimizer import minimizers_batch_gpu  # noqa: E402

GBZ_TOY_MIN = (
    ROOT
    / "tests/data/giraffe_fixture/gbz_toy/toy.wl.C2T.shortread.withzip.min"
)


@pytest.mark.skipif(not GBZ_TOY_MIN.is_file(), reason="toy minimizer missing")
def test_gpu_minimizer_batch_matches_cpu():
    seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    with MinimizerIndex(GBZ_TOY_MIN) as idx:
        cpu = idx.minimizers(seq)
        gpu_batch, backend = minimizers_batch_gpu(
            [seq], k=int(idx.k), w=int(idx.w), device="cpu"
        )
        assert backend == "cpu"
        assert [(o.key, o.hash, o.offset, o.is_reverse) for o in gpu_batch[0]] == [
            (o.key, o.hash, o.offset, o.is_reverse) for o in cpu
        ]


def test_device_probe_keys():
    from giraffe_gpu_minimizer import device_probe

    p = device_probe()
    assert p["cpu"] is True
    assert "backend" in p
