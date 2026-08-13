"""BGZF BAM writer used by the Mojo FM mapper (no SAM, no samtools view).

Little-endian BAM spec + BGZF framing. SEQ is original (pre-C2T) bases;
flag 0x10 reverse-complements SEQ and QUAL.
"""

from __future__ import annotations

import ctypes
import os
import struct
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Sequence

_BAM_MAGIC = b"BAM\x01"
_BGZF_EOF = bytes.fromhex(
    "1f8b08040000000000ff0600424302001b0003000000000000000000"
)
_MAX_UNCOMPRESSED = 65280

_SEQ_NT16 = [15] * 256
for _i, _c in enumerate(b"=ACMGRSVTWYHKDBN"):
    _SEQ_NT16[_c] = _i
    _SEQ_NT16[_c + 32] = _i  # lowercase

_RC = bytes.maketrans(b"ACGTacgt", b"TGCAtgca")


def _reg2bin(beg: int, end: int) -> int:
    """SAM spec binning (0-based, half-open)."""
    if end <= beg:
        end = beg + 1
    end -= 1
    if beg >> 14 == end >> 14:
        return ((1 << 15) - 1) // 7 + (beg >> 14)
    if beg >> 17 == end >> 17:
        return ((1 << 12) - 1) // 7 + (beg >> 17)
    if beg >> 20 == end >> 20:
        return ((1 << 9) - 1) // 7 + (beg >> 20)
    if beg >> 23 == end >> 23:
        return ((1 << 6) - 1) // 7 + (beg >> 23)
    if beg >> 26 == end >> 26:
        return ((1 << 3) - 1) // 7 + (beg >> 26)
    return 0


def _pack_seq(seq: bytes) -> bytes:
    n = len(seq)
    out = bytearray((n + 1) // 2)
    i = 0
    while i + 1 < n:
        out[i >> 1] = (_SEQ_NT16[seq[i]] << 4) | _SEQ_NT16[seq[i + 1]]
        i += 2
    if i < n:
        out[i >> 1] = _SEQ_NT16[seq[i]] << 4
    return bytes(out)


def _qual_phred(qual: str, qlen: int) -> bytes:
    if not qual or qual == "*":
        return b"\xff" * qlen
    raw = qual.encode("ascii", "replace")
    if len(raw) != qlen:
        return b"\xff" * qlen
    return bytes((b - 33) & 0xFF for b in raw)


def _rc_seq(seq: str) -> str:
    return seq.encode("ascii", "replace").translate(_RC)[::-1].decode("ascii")


def _bgzf_block(chunk: bytes, level: int) -> bytes:
    """One BGZF member (raw deflate). zlib releases the GIL here."""
    cobj = zlib.compressobj(level, zlib.DEFLATED, -15)
    payload = cobj.compress(chunk) + cobj.flush()
    crc = zlib.crc32(chunk) & 0xFFFFFFFF
    bsize = 25 + len(payload)
    header = struct.pack(
        "<BBBBLBBHBBHH",
        31,
        139,
        8,
        4,
        0,
        0,
        255,
        6,
        66,
        67,
        2,
        bsize,
    )
    return header + payload + struct.pack("<II", crc, len(chunk) & 0xFFFFFFFF)


class _BgzfWriter:
    def __init__(self, path: str, level: int = 1, threads: int = 1) -> None:
        self._fh = open(path, "wb", buffering=8 * 1024 * 1024)
        self._level = max(0, min(level, 9))
        self._buf = bytearray()
        self._off = 0
        self._threads = max(1, int(threads))
        self._pool: ThreadPoolExecutor | None = None
        self._pending: list[Future[bytes]] = []
        if self._threads > 1:
            self._pool = ThreadPoolExecutor(max_workers=self._threads)

    def write(self, data: bytes | bytearray) -> None:
        self._buf.extend(data)
        self._drain_full_blocks()

    def write_from_addr(self, addr: int, n: int) -> None:
        """BGZF ``n`` bytes at ``addr`` with one copy per 65 KiB block."""
        n = int(n)
        if n <= 0:
            return
        off = 0
        tail = len(self._buf) - self._off
        if tail < 0:
            tail = 0
        if tail:
            need = _MAX_UNCOMPRESSED - tail
            if need < 0:
                need = 0
            take = n if n < need else need
            if take:
                window = (ctypes.c_ubyte * take).from_address(int(addr))
                self._buf.extend(memoryview(window))
                off = take
                self._drain_full_blocks()
        while off + _MAX_UNCOMPRESSED <= n:
            take = _MAX_UNCOMPRESSED
            window = (ctypes.c_ubyte * take).from_address(int(addr) + off)
            self._submit(bytes(memoryview(window)))
            off += take
        if off < n:
            take = n - off
            window = (ctypes.c_ubyte * take).from_address(int(addr) + off)
            self._buf.extend(memoryview(window))

    def _drain_full_blocks(self) -> None:
        while len(self._buf) - self._off >= _MAX_UNCOMPRESSED:
            start = self._off
            end = start + _MAX_UNCOMPRESSED
            chunk = bytes(self._buf[start:end])
            self._off = end
            self._submit(chunk)
        if self._off >= 1 << 20:
            del self._buf[: self._off]
            self._off = 0

    def _submit(self, chunk: bytes) -> None:
        if not chunk:
            return
        if self._pool is None:
            self._fh.write(_bgzf_block(chunk, self._level))
            return
        self._pending.append(self._pool.submit(_bgzf_block, chunk, self._level))
        # Keep zlib busy during gather: ~32 in-flight blocks per worker (~2 MiB
        # uncompressed each worker at 65 KiB/block).
        while len(self._pending) >= self._threads * 32:
            self._fh.write(self._pending.pop(0).result())

    def _flush_tail(self) -> None:
        if self._off:
            del self._buf[: self._off]
            self._off = 0
        if self._buf:
            self._submit(bytes(self._buf))
            self._buf.clear()
        while self._pending:
            self._fh.write(self._pending.pop(0).result())

    def close(self) -> None:
        self._flush_tail()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self._fh.write(_BGZF_EOF)
        self._fh.close()


class BamArena:
    """Uncompressed BAM alignment blocks (no BGZF) for GPU sort/permute.

    Each ``append_raw`` stores one mapper batch as a chunk. After mapping,
    Mojo gathers records by ``(chunk_id, local_off)`` into coordinate order.
    """

    def __init__(self) -> None:
        self._chunks: list[bytearray] = []
        self._n = 0
        self._sorted = bytearray()
        self._sorted_hdr = None

    def append_raw(self, addr: int, n: int) -> int:
        """Copy ``n`` bytes at ``addr``; return the new chunk index."""
        if n <= 0:
            self._chunks.append(bytearray())
            return len(self._chunks) - 1
        self._chunks.append(bytearray(ctypes.string_at(int(addr), int(n))))
        self._n += int(n)
        return len(self._chunks) - 1

    def n_chunks(self) -> int:
        return len(self._chunks)

    def nbytes(self) -> int:
        return self._n

    def alloc_sorted(self, n: int) -> int:
        """Pinned-ish host bytearray for parallel gather; returns start address."""
        n = int(n)
        if n <= 0:
            self._sorted = bytearray()
            self._sorted_hdr = None
            return 0
        self._sorted = bytearray(n)
        self._sorted_hdr = (ctypes.c_char * 1).from_buffer(self._sorted)
        return ctypes.addressof(self._sorted_hdr)

    def chunk_addr(self, i: int) -> int:
        c = self._chunks[int(i)]
        if not c:
            return 0
        buf = (ctypes.c_char * len(c)).from_buffer(c)
        return ctypes.addressof(buf)

    def chunk_len(self, i: int) -> int:
        return len(self._chunks[int(i)])


class BamWriter:
    """Stream BAM records to ``path`` (BGZF)."""

    def __init__(
        self,
        path: str,
        sq_names: Sequence[str],
        sq_lens: Sequence[int],
        rg_id: str = "mojo1",
        sm: str = "sample",
        lb: str = "lib1",
        pl: str = "ILLUMINA",
        pg_vn: str = "0.1.0-mojo-fm",
        level: int = 1,
        sort_order: str = "unsorted",
    ) -> None:
        if len(sq_names) != len(sq_lens):
            raise ValueError("sq_names/sq_lens length mismatch")
        so = sort_order if sort_order in {
            "unsorted",
            "coordinate",
            "queryname",
            "unknown",
        } else "unsorted"
        self._rg = rg_id.encode("ascii")
        try:
            threads = int(os.environ.get("METHYLGRAPHER_BAM_THREADS", "32"))
        except ValueError:
            threads = 32
        if threads < 1:
            threads = 1
        self._bgzf = _BgzfWriter(path, level=level, threads=threads)
        hd = [f"@HD\tVN:1.6\tSO:{so}"]
        for name, ln in zip(sq_names, sq_lens):
            hd.append(f"@SQ\tSN:{name}\tLN:{ln}")
        hd.append(f"@RG\tID:{rg_id}\tSM:{sm}\tLB:{lb}\tPL:{pl}\tPU:{rg_id}")
        hd.append(f"@PG\tID:MojoFq2bamMeth\tPN:MojoFq2bamMeth\tVN:{pg_vn}")
        text = ("\n".join(hd) + "\n").encode("ascii")
        blob = bytearray(_BAM_MAGIC)
        blob += struct.pack("<i", len(text))
        blob += text
        blob += struct.pack("<i", len(sq_names))
        for name, ln in zip(sq_names, sq_lens):
            nb = name.encode("ascii") + b"\x00"
            blob += struct.pack("<i", len(nb))
            blob += nb
            blob += struct.pack("<i", int(ln))
        self._bgzf.write(blob)

    def write_batch(
        self,
        qnames: Sequence[str],
        flags: Sequence[int],
        tids: Sequence[int],
        pos0s: Sequence[int],
        mapqs: Sequence[int],
        sls: Sequence[int],
        srs: Sequence[int],
        seqs: Sequence[str],
        quals: Sequence[str],
        next_tids: Sequence[int],
        next_pos0s: Sequence[int],
        tlens: Sequence[int],
        nms: Sequence[int],
    ) -> int:
        """Pack one batch of alignments. Returns mapped count."""
        n = len(qnames)
        mapped = 0
        rec = bytearray(1024)
        for i in range(n):
            seq = seqs[i]
            flag = int(flags[i])
            if flag & 16:
                seq = _rc_seq(seq)
                q = quals[i]
                if q and q != "*":
                    q = q[::-1]
                else:
                    q = quals[i]
            else:
                q = quals[i]
            qlen = len(seq)
            sl = int(sls[i])
            sr = int(srs[i])
            if sl < 0:
                sl = 0
            if sr < 0:
                sr = 0
            if sl + sr >= qlen:
                sl = 0
                sr = 0
            mid = qlen - sl - sr
            cigar: list[int] = []
            if sl:
                cigar.append((sl << 4) | 4)  # S
            if mid > 0:
                cigar.append((mid << 4) | 0)  # M
            if sr:
                cigar.append((sr << 4) | 4)
            tid = int(tids[i])
            pos0 = int(pos0s[i])
            if tid < 0 or (flag & 4):
                tid = -1
                pos0 = -1
                cigar = []
                bin_ = 4680
            else:
                mapped += 1
                end = pos0 + mid
                if end <= pos0:
                    end = pos0 + 1
                bin_ = _reg2bin(pos0, end)
            qn = qnames[i].encode("ascii", "replace") + b"\x00"
            l_qname = len(qn)
            n_cigar = len(cigar)
            seq_packed = _pack_seq(seq.encode("ascii", "replace")) if qlen else b""
            qual_b = _qual_phred(q, qlen)
            # RG:Z + NM:i
            tags = b"RGZ" + self._rg + b"\x00"
            tags += b"NM" + b"C" + struct.pack("<B", min(255, max(0, int(nms[i]))))
            block = 32 + l_qname + 4 * n_cigar + len(seq_packed) + qlen + len(tags)
            mq = max(0, min(255, int(mapqs[i])))
            ntid = int(next_tids[i])
            npos = int(next_pos0s[i])
            if ntid < 0:
                ntid = -1
                npos = -1
            need = 4 + block
            if len(rec) < need:
                rec = bytearray(need + 256)
            struct.pack_into(
                "<iiiIIiiii",
                rec,
                0,
                block,
                tid,
                pos0,
                (bin_ << 16) | (mq << 8) | l_qname,
                (flag << 16) | n_cigar,
                qlen,
                ntid,
                npos,
                int(tlens[i]),
            )
            off = 36
            rec[off : off + l_qname] = qn
            off += l_qname
            if n_cigar:
                struct.pack_into("<" + "I" * n_cigar, rec, off, *cigar)
                off += 4 * n_cigar
            rec[off : off + len(seq_packed)] = seq_packed
            off += len(seq_packed)
            rec[off : off + qlen] = qual_b
            off += qlen
            rec[off : off + len(tags)] = tags
            off += len(tags)
            self._bgzf.write(rec[:off])
        return mapped

    def write_raw(self, addr: int, n: int) -> None:
        """Append ``n`` bytes at ``addr`` (Mojo host buffer) into BGZF."""
        if n <= 0:
            return
        self._bgzf.write_from_addr(int(addr), int(n))

    def close(self) -> None:
        self._bgzf.close()


def looks_bam_path(path: str) -> bool:
    p = path.lower()
    return p.endswith(".bam") and not p.endswith(".sam")
