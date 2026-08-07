"""SPIZ zipcodes sidecar reader (vg shortread.zipcodes).

Payload words in the minimizer index point into this structure for clustering.
Staged: parse header + expose raw bytes for Mojo ``giraffe_dist`` heuristics.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path
from typing import Optional

_U32 = struct.Struct("<I")
MAGIC_SPIZ = 0x5A495053  # 'SPIZ' little-endian


class ZipcodesIndex:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(path)
        self._fh = self.path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self._mm) < 8:
            raise ValueError(f"zipcodes too small: {path}")
        magic = _U32.unpack_from(self._mm, 0)[0]
        self.version = _U32.unpack_from(self._mm, 4)[0]
        self.magic_ok = magic == MAGIC_SPIZ
        self.nbytes = len(self._mm)

    def close(self) -> None:
        self._mm.close()
        self._fh.close()

    def __enter__(self) -> "ZipcodesIndex":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def cluster_key(self, payload0: int, payload1: int = 0) -> int:
        """Cheap cluster bucket from zip payload words (stand-in for full decode)."""
        return (payload0 ^ (payload1 << 1)) & 0xFFFFFFFF


class DistIndex:
    """Distance index probe (``.dist``) — presence + size for readiness."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(path)
        self.nbytes = self.path.stat().st_size

    def distance_ok(self, a_node: int, b_node: int, cap: int = 200) -> bool:
        """Staged heuristic: same node or nearby ids within cap (until full decode)."""
        if a_node == b_node:
            return True
        return abs(a_node - b_node) <= cap


def probe_zipcodes(path: str) -> dict:
    with ZipcodesIndex(path) as z:
        return {
            "path": path,
            "magic_ok": z.magic_ok,
            "version": z.version,
            "nbytes": z.nbytes,
        }


def probe_dist(path: Optional[str]) -> dict:
    if not path:
        return {"path": "", "exists": False}
    p = Path(path)
    return {"path": path, "exists": p.is_file(), "nbytes": p.stat().st_size if p.is_file() else 0}
