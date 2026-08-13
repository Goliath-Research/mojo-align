"""Bulk FASTQ batch reader for the FM mapper.

Keeps SEQ/QUAL as ``bytearray`` (no per-read ``str`` / Mojo ``String`` copies).
``pack_align`` / ``export_ptrs`` write straight into Mojo host buffers.
"""

from __future__ import annotations

import ctypes
import gzip
import io
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
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
    __slots__ = (
        "n1",
        "max_len",
        "names1",
        "seq1",
        "orig1",
        "qual1",
        "names2",
        "seq2",
        "orig2",
        "qual2",
        "_views",
        "_name_bufs",
        "arena",
        "seq1_off",
        "seq1_len",
        "seq2_off",
        "seq2_len",
        "name1_off",
        "name1_len",
        "orig1_off",
        "orig1_len",
        "qual1_off",
        "qual1_len",
        "name2_off",
        "name2_len",
        "orig2_off",
        "orig2_len",
        "qual2_off",
        "qual2_len",
        "_seq_view",
    )

    def __init__(self) -> None:
        self.n1 = 0
        self.max_len = 0
        self.names1: List[str] = []
        self.seq1: List[bytearray] = []
        self.orig1: List[bytearray] = []
        self.qual1: List[bytearray] = []
        self.names2: List[str] = []
        self.seq2: List[bytearray] = []
        self.orig2: List[bytearray] = []
        self.qual2: List[bytearray] = []
        self._views: list = []
        self._name_bufs: List[bytearray] = []
        self.arena = bytearray()
        self.seq1_off: List[int] = []
        self.seq1_len: List[int] = []
        self.seq2_off: List[int] = []
        self.seq2_len: List[int] = []
        self.name1_off: List[int] = []
        self.name1_len: List[int] = []
        self.orig1_off: List[int] = []
        self.orig1_len: List[int] = []
        self.qual1_off: List[int] = []
        self.qual1_len: List[int] = []
        self.name2_off: List[int] = []
        self.name2_len: List[int] = []
        self.orig2_off: List[int] = []
        self.orig2_len: List[int] = []
        self.qual2_off: List[int] = []
        self.qual2_len: List[int] = []
        self._seq_view = None

    def _push(self, data: bytes) -> tuple[int, int]:
        off = len(self.arena)
        self.arena.extend(data)
        return off, len(data)

    def _pin_arena(self) -> int:
        if self._seq_view is not None:
            return ctypes.addressof(self._seq_view)
        if not self.arena:
            return 0
        view = (ctypes.c_char * len(self.arena)).from_buffer(self.arena)
        self._seq_view = view
        self._views = [view]
        self._name_bufs = [self.arena]
        return ctypes.addressof(view)

    def _fill_offs(
        self,
        offs: List[int],
        lens: List[int],
        addr_dest: int,
        len_dest: int,
        base: int,
    ) -> None:
        n = self.n1
        if n <= 0:
            return
        addrs = (ctypes.c_uint64 * n).from_address(addr_dest)
        out_n = (ctypes.c_uint32 * n).from_address(len_dest)
        for i in range(n):
            out_n[i] = lens[i]
            addrs[i] = (base + offs[i]) if lens[i] else 0

    def zero_align(self, dest_addr: int, n_bases: int) -> None:
        if n_bases > 0:
            ctypes.memset(ctypes.c_void_p(dest_addr), 78, n_bases)

    def export_seq_ptrs(self, seq_a_addr: int, lens_addr: int, paired: int) -> None:
        """Pin the SEQ arena once and write R1[+R2] pointer/length tables."""
        n1 = self.n1
        n_seq = n1 * 2 if paired else n1
        if n_seq <= 0:
            return
        base = self._pin_arena()
        addrs = (ctypes.c_uint64 * n_seq).from_address(seq_a_addr)
        lens = (ctypes.c_uint32 * n_seq).from_address(lens_addr)
        for i in range(n1):
            lens[i] = self.seq1_len[i]
            addrs[i] = base + self.seq1_off[i]
        if paired:
            for i in range(n1):
                lens[n1 + i] = self.seq2_len[i]
                addrs[n1 + i] = base + self.seq2_off[i]

    def pack_align(self, dest_addr: int, max_len: int, paired: int) -> None:
        """Write R1 (then R2) converted seqs into a strided host buffer; pad with N."""
        n1 = self.n1
        n_seq = n1 * 2 if paired else n1
        n_bases = n_seq * max_len
        if n_bases > 0:
            ctypes.memset(ctypes.c_void_p(dest_addr), 78, n_bases)
        for i, s in enumerate(self.seq1):
            if s:
                src = (ctypes.c_uint8 * len(s)).from_buffer(s)
                ctypes.memmove(ctypes.c_void_p(dest_addr + i * max_len), src, len(s))
        if paired:
            for i, s in enumerate(self.seq2):
                if s:
                    src = (ctypes.c_uint8 * len(s)).from_buffer(s)
                    ctypes.memmove(
                        ctypes.c_void_p(dest_addr + (n1 + i) * max_len), src, len(s)
                    )

    def fill_lens(self, lens_addr: int, paired: int) -> None:
        n1 = self.n1
        n_seq = n1 * 2 if paired else n1
        if n_seq <= 0:
            return
        arr = (ctypes.c_uint32 * n_seq).from_address(lens_addr)
        for i, s in enumerate(self.seq1):
            arr[i] = len(s)
        if paired:
            for i, s in enumerate(self.seq2):
                arr[n1 + i] = len(s)

    def export_ptrs(
        self,
        paired: int,
        n1_name_a: int,
        n1_name_n: int,
        n1_orig_a: int,
        n1_orig_n: int,
        n1_qual_a: int,
        n1_qual_n: int,
        n2_name_a: int,
        n2_name_n: int,
        n2_orig_a: int,
        n2_orig_n: int,
        n2_qual_a: int,
        n2_qual_n: int,
    ) -> None:
        """Pin the read arena once and write name/orig/qual pointer tables."""
        n1 = self.n1
        if n1 <= 0:
            return
        base = self._pin_arena()
        self._fill_offs(self.name1_off, self.name1_len, n1_name_a, n1_name_n, base)
        self._fill_offs(self.orig1_off, self.orig1_len, n1_orig_a, n1_orig_n, base)
        self._fill_offs(self.qual1_off, self.qual1_len, n1_qual_a, n1_qual_n, base)
        if paired:
            self._fill_offs(self.name2_off, self.name2_len, n2_name_a, n2_name_n, base)
            self._fill_offs(self.orig2_off, self.orig2_len, n2_orig_a, n2_orig_n, base)
            self._fill_offs(self.qual2_off, self.qual2_len, n2_qual_a, n2_qual_n, base)


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
        self._pool = ThreadPoolExecutor(max_workers=1)

    def read_batch_async(self, n_pairs: int) -> Future:
        return self._pool.submit(self.read_batch, n_pairs)

    def _one(
        self, fh: BinaryIO, tr: Optional[bytes]
    ) -> tuple[str, bytearray, bytearray, bytearray] | None:
        n = fh.readline()
        if not n:
            return None
        s = fh.readline()
        _ = fh.readline()
        q = fh.readline()
        raw = s.rstrip(b"\r\n")
        orig = bytearray(raw)
        aln = orig if tr is None else bytearray(raw.translate(tr))
        qual = bytearray(q.rstrip(b"\r\n"))
        return _qname(n), aln, orig, qual

    def read_batch(self, n_pairs: int) -> FastqBatch:
        out = FastqBatch()
        n = max(1, int(n_pairs))
        mx = 0
        for _ in range(n):
            rec = self._one(self._f1, self._tr1)
            if rec is None:
                break
            name, aln, orig, qual = rec
            no, nl = out._push(name.encode("ascii", "replace"))
            out.name1_off.append(no)
            out.name1_len.append(nl)
            oo, ol = out._push(orig)
            out.orig1_off.append(oo)
            out.orig1_len.append(ol)
            qo, ql = out._push(qual)
            out.qual1_off.append(qo)
            out.qual1_len.append(ql)
            off, ln = out._push(aln)
            out.seq1_off.append(off)
            out.seq1_len.append(ln)
            out.names1.append(name)
            out.seq1.append(aln)
            out.orig1.append(orig)
            out.qual1.append(qual)
            if ln > mx:
                mx = ln
            if self._f2 is not None:
                rec2 = self._one(self._f2, self._tr2)
                if rec2 is None:
                    raise RuntimeError("paired FASTQ length mismatch (R2 ended early)")
                n2, a2, o2, q2 = rec2
                no2, nl2 = out._push(n2.encode("ascii", "replace"))
                out.name2_off.append(no2)
                out.name2_len.append(nl2)
                oo2, ol2 = out._push(o2)
                out.orig2_off.append(oo2)
                out.orig2_len.append(ol2)
                qo2, ql2 = out._push(q2)
                out.qual2_off.append(qo2)
                out.qual2_len.append(ql2)
                off2, ln2 = out._push(a2)
                out.seq2_off.append(off2)
                out.seq2_len.append(ln2)
                out.names2.append(n2)
                out.seq2.append(a2)
                out.orig2.append(o2)
                out.qual2.append(q2)
                if ln2 > mx:
                    mx = ln2
        out.n1 = len(out.names1)
        out.max_len = mx
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
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
