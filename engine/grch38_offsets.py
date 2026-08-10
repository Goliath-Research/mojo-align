"""Dense grch38-dense-v1 segment offset table (mmap) for Mojo QC SAM emit."""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import List, Optional, Tuple

RECORD = struct.Struct("<IIQ")
UNSET = 0xFFFFFFFF


class Grch38OffsetsTable:
    __slots__ = ("root", "max_id", "chroms", "chrom_lens", "_mm", "_fh")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != "grch38-dense-v1":
            raise RuntimeError(f"unsupported offsets format: {meta.get('format')}")
        self.max_id = int(meta["max_id"])
        self.chroms: List[str] = []
        self.chrom_lens: List[int] = []
        chroms_path = self.root / "chroms.tsv"
        if chroms_path.is_file():
            for line in chroms_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split("\t")
                self.chroms.append(parts[0])
                self.chrom_lens.append(int(parts[1]) if len(parts) > 1 and parts[1] else 0)
        rec = self.root / "records.bin"
        self._fh = open(rec, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        expect = (self.max_id + 1) * 16
        if len(self._mm) < expect:
            raise RuntimeError(f"records.bin too small: {len(self._mm)} < {expect}")

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass

    def lookup(self, seg_id: int) -> Optional[Tuple[str, int, int]]:
        """Return (chrom, start0, length) or None."""
        if seg_id < 0 or seg_id > self.max_id:
            return None
        cidx, length, start = RECORD.unpack_from(self._mm, seg_id * 16)
        if cidx == UNSET or cidx >= len(self.chroms):
            return None
        return self.chroms[cidx], int(start), int(length)


_TABLE: Optional[Grch38OffsetsTable] = None
_mapped = 0
_skip_no_seq = 0
_skip_no_anchor = 0


def open_table(root: str) -> None:
    global _TABLE, _mapped, _skip_no_seq, _skip_no_anchor
    close_table()
    _TABLE = Grch38OffsetsTable(root)
    _mapped = _skip_no_seq = _skip_no_anchor = 0


def close_table() -> None:
    global _TABLE
    if _TABLE is not None:
        _TABLE.close()
        _TABLE = None


def chroms() -> List[str]:
    return list(_TABLE.chroms) if _TABLE else []


def chrom_lens() -> List[int]:
    return list(_TABLE.chrom_lens) if _TABLE else []


def lookup(seg_id: int) -> Optional[Tuple[str, int, int]]:
    if _TABLE is None:
        return None
    return _TABLE.lookup(int(seg_id))


def bump_mapped() -> int:
    global _mapped
    _mapped += 1
    return _mapped


def bump_skip_no_seq() -> None:
    global _skip_no_seq
    _skip_no_seq += 1


def bump_skip_no_anchor() -> None:
    global _skip_no_anchor
    _skip_no_anchor += 1


def summary() -> str:
    return (
        f"mapped={_mapped} skip_no_seq={_skip_no_seq} "
        f"skip_no_anchor={_skip_no_anchor}"
    )
