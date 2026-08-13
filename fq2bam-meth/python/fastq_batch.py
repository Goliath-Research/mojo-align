"""Bulk FASTQ batch reader for the FM mapper.

Mojo-per-line ``readline`` across the Python FFI was the full-sample wall
(~270 s). This reads a whole mapper batch in one Python call, using ``pigz
-dc`` when available (parallel gzip) and ``bytes.translate`` for C2T/G2A.
"""

from __future__ import annotations

import gzip
import io
import shutil
import subprocess
from typing import BinaryIO, List, Optional


def _qname(header: bytes) -> str:
    n = header.rstrip(b"\r\n")
    if n.startswith(b"@"):
        n = n[1:]
    sp = n.find(b" ")
    if sp >= 0:
        n = n[:sp]
    sl = n.find(b"/")
    if sl >= 0:
        n = n[:sl]
    return n.decode("ascii", "replace")


def _open_rb(path: str) -> tuple[BinaryIO, Optional[subprocess.Popen[bytes]]]:
    low = path.lower()
    if not (low.endswith(".gz") or low.endswith(".gzip")):
        return open(path, "rb", buffering=8 * 1024 * 1024), None
    pigz = shutil.which("pigz")
    if pigz:
        proc = subprocess.Popen(
            [pigz, "-dc", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=8 * 1024 * 1024,
        )
        assert proc.stdout is not None
        buf = io.BufferedReader(proc.stdout, 8 * 1024 * 1024)
        return buf, proc
    return gzip.open(path, "rb"), None


class FastqBatch:
    __slots__ = ("n1", "names1", "seq1", "orig1", "qual1", "names2", "seq2", "orig2", "qual2")

    def __init__(self) -> None:
        self.n1 = 0
        self.names1: List[str] = []
        self.seq1: List[str] = []
        self.orig1: List[str] = []
        self.qual1: List[str] = []
        self.names2: List[str] = []
        self.seq2: List[str] = []
        self.orig2: List[str] = []
        self.qual2: List[str] = []


class FastqPairReader:
    """Streaming PE/SE FASTQ reader. ``read_batch`` returns a ``FastqBatch``."""

    def __init__(self, path1: str, path2: str = "", bs_r1: str = "", bs_r2: str = "") -> None:
        self._f1, self._p1 = _open_rb(path1)
        self._f2: Optional[BinaryIO] = None
        self._p2: Optional[subprocess.Popen[bytes]] = None
        if path2:
            self._f2, self._p2 = _open_rb(path2)
        self._tr1 = bytes.maketrans(b"Cc", b"Tt") if bs_r1 == "C2T" else (
            bytes.maketrans(b"Gg", b"Aa") if bs_r1 == "G2A" else None
        )
        self._tr2 = bytes.maketrans(b"Gg", b"Aa") if bs_r2 == "G2A" else (
            bytes.maketrans(b"Cc", b"Tt") if bs_r2 == "C2T" else None
        )

    def _one(self, fh: BinaryIO, tr: Optional[bytes]) -> tuple[str, str, str, str] | None:
        n = fh.readline()
        if not n:
            return None
        s = fh.readline()
        _ = fh.readline()
        q = fh.readline()
        raw = s.rstrip(b"\r\n")
        orig = raw.decode("ascii", "replace")
        aln = orig if tr is None else raw.translate(tr).decode("ascii", "replace")
        qual = q.rstrip(b"\r\n").decode("ascii", "replace") or "*"
        return _qname(n), aln, orig, qual

    def read_batch(self, n_pairs: int) -> FastqBatch:
        out = FastqBatch()
        n = max(1, int(n_pairs))
        for _ in range(n):
            rec = self._one(self._f1, self._tr1)
            if rec is None:
                break
            name, aln, orig, qual = rec
            out.names1.append(name)
            out.seq1.append(aln)
            out.orig1.append(orig)
            out.qual1.append(qual)
            if self._f2 is not None:
                rec2 = self._one(self._f2, self._tr2)
                if rec2 is None:
                    raise RuntimeError("paired FASTQ length mismatch (R2 ended early)")
                n2, a2, o2, q2 = rec2
                out.names2.append(n2)
                out.seq2.append(a2)
                out.orig2.append(o2)
                out.qual2.append(q2)
        out.n1 = len(out.names1)
        return out

    def close(self) -> None:
        try:
            self._f1.close()
        except Exception:
            pass
        if self._f2 is not None:
            try:
                self._f2.close()
            except Exception:
                pass
        for proc in (self._p1, self._p2):
            if proc is None:
                continue
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
