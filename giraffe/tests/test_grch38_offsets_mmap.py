"""engine.grch38_offsets mmap lookup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # mojo-align repo root
PKG = Path(__file__).resolve().parents[1]  # giraffe/
sys.path.insert(0, str(ROOT / "methylgrapher"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "scripts"))

from build_grch38_offsets import build  # noqa: E402
from engine.grch38_offsets import close_table, lookup, open_table  # noqa: E402


def test_mmap_lookup(tmp_path: Path) -> None:
    gfa = tmp_path / "toy.wl.gfa"
    gfa.write_text(
        "H\tVN:Z:1.1\nS\t11\tACGTCGGA\nW\tGRCh38\t0\tchr1\t1000\t1008\t>11\n",
        encoding="utf-8",
    )
    out = tmp_path / "off"
    build(gfa, out)
    open_table(str(out))
    try:
        assert lookup(11) == ("1", 1000, 8)
        assert lookup(1) is None
    finally:
        close_table()
