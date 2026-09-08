"""BGZF BAM writer used by the Mojo FM mapper (no SAM, no samtools view).

Little-endian BAM spec + BGZF framing. SEQ is original (pre-C2T) bases;
flag 0x10 reverse-complements SEQ and QUAL.
"""

from __future__ import annotations

import array
import ctypes
import heapq
import mmap
import os
import struct
import zlib
from pathlib import Path
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

    Set ``spill_dir`` or ``METHYLGRAPHER_BAM_ARENA_DIR`` to mmap chunks on
    SSD (30×-scale). ``ram`` / ``off`` / empty keeps the in-memory path.
    """

    def __init__(self, spill_dir: str | os.PathLike[str] | None = None) -> None:
        raw = (
            str(spill_dir)
            if spill_dir is not None
            else os.environ.get("METHYLGRAPHER_BAM_ARENA_DIR", "")
        )
        self._spill: Path | None = None
        if raw and raw.strip().lower() not in {"", "ram", "0", "false", "off", "none"}:
            self._spill = Path(raw)
            self._spill.mkdir(parents=True, exist_ok=True)
        self._chunks: list[bytearray] = []
        self._maps: list[mmap.mmap | None] = []
        self._headers: list[object | None] = []
        self._lens: list[int] = []
        self._n = 0
        self._sorted = bytearray()
        self._sorted_hdr = None

    def append_raw(self, addr: int, n: int) -> int:
        """Copy ``n`` bytes at ``addr``; return the new chunk index."""
        n = int(n)
        if self._spill is None:
            if n <= 0:
                self._chunks.append(bytearray())
                self._lens.append(0)
                return len(self._chunks) - 1
            self._chunks.append(bytearray(ctypes.string_at(int(addr), n)))
            self._lens.append(n)
            self._n += n
            return len(self._chunks) - 1
        idx = len(self._lens)
        path = self._spill / f"chunk_{idx:06d}.bin"
        if n <= 0:
            path.write_bytes(b"")
            self._maps.append(None)
            self._headers.append(None)
            self._lens.append(0)
            return idx
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, n)
            mm = mmap.mmap(fd, n)
        finally:
            os.close(fd)
        hdr = (ctypes.c_char * n).from_buffer(mm)
        ctypes.memmove(ctypes.addressof(hdr), int(addr), n)
        mm.flush()
        self._maps.append(mm)
        self._headers.append(hdr)
        self._lens.append(n)
        self._n += n
        return idx

    def n_chunks(self) -> int:
        if self._spill is not None:
            return len(self._lens)
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
        i = int(i)
        if self._spill is not None:
            if self._lens[i] <= 0:
                return 0
            return ctypes.addressof(self._headers[i])
        c = self._chunks[i]
        if not c:
            return 0
        buf = (ctypes.c_char * len(c)).from_buffer(c)
        return ctypes.addressof(buf)

    def chunk_len(self, i: int) -> int:
        i = int(i)
        if self._spill is not None:
            return self._lens[i]
        return len(self._chunks[i])

    def clear(self) -> None:
        """Drop chunks so the next tile can reuse this arena."""
        if self._spill is not None:
            self._headers.clear()
            for mm in self._maps:
                if mm is not None:
                    mm.close()
            self._maps.clear()
            for i in range(len(self._lens)):
                p = self._spill / f"chunk_{i:06d}.bin"
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
        self._chunks.clear()
        self._lens.clear()
        self._n = 0
        self._sorted = bytearray()
        self._sorted_hdr = None


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
        xms: Sequence[str] | None = None,
        xgs: Sequence[str] | None = None,
    ) -> int:
        """Pack one batch of alignments. Returns mapped count.

        Optional ``xms`` / ``xgs`` write Bismark-style XM:Z and XG:Z aux tags.
        """
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
            # RG:Z + NM:i (+ optional XM:Z / XG:Z)
            tags = b"RGZ" + self._rg + b"\x00"
            tags += b"NM" + b"C" + struct.pack("<B", min(255, max(0, int(nms[i]))))
            if xgs is not None and i < len(xgs) and xgs[i]:
                tags += b"XGZ" + str(xgs[i]).encode("ascii") + b"\x00"
            if xms is not None and i < len(xms) and xms[i]:
                tags += b"XMZ" + str(xms[i]).encode("ascii") + b"\x00"
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

    def write_bytes(self, data: bytes | bytearray | memoryview) -> None:
        if not data:
            return
        self._bgzf.write(data)

    def close(self) -> None:
        self._bgzf.close()


_RUN_MAGIC = b"MJRN\x01\x00\x00\x00"
_DUP_SENTINEL = (1 << 64) - 1


def _or_dup_flag_bytes(buf: bytearray, rec_off: int = 0) -> None:
    """Set BAM flag 0x400 (duplicate). ``buf[rec_off]`` is block_size."""
    flag_nc = int.from_bytes(buf[rec_off + 16 : rec_off + 20], "little")
    flag = (flag_nc >> 16) | 1024
    nc = flag_nc & 65535
    buf[rec_off + 16 : rec_off + 20] = ((flag << 16) | nc).to_bytes(4, "little")


class _RawRunWriter:
    """Uncompressed concatenated BAM records (one sorted tile)."""

    def __init__(self, path: str) -> None:
        self._fh = open(path, "wb", buffering=8 * 1024 * 1024)

    def write_raw(self, addr: int, n: int) -> None:
        n = int(n)
        if n <= 0:
            return
        self._fh.write(ctypes.string_at(int(addr), n))

    def close(self) -> None:
        self._fh.close()


class _RunKeys:
    __slots__ = ("n", "coord", "dhi", "dlo", "score", "pair", "rec_len", "perm", "dupperm")

    def __init__(self, path: Path) -> None:
        data = path.read_bytes()
        if data[:8] != _RUN_MAGIC:
            raise ValueError(f"bad run keys magic: {path}")
        n = struct.unpack_from("<Q", data, 8)[0]
        off = 16

        def take_q() -> array.array:
            nonlocal off
            a = array.array("Q")
            a.frombytes(data[off : off + n * 8])
            off += n * 8
            return a

        def take_i() -> array.array:
            nonlocal off
            a = array.array("I")
            a.frombytes(data[off : off + n * 4])
            off += n * 4
            return a

        self.n = n
        self.coord = take_q()
        self.dhi = take_q()
        self.dlo = take_q()
        self.score = take_i()
        self.pair = take_i()
        self.rec_len = take_i()
        self.perm = take_i()
        self.dupperm = take_i()


def _dump_u(fh, addr: int, n: int, width: int) -> None:
    fh.write(ctypes.string_at(int(addr), int(n) * width))


def _bam_coord_key(rec: bytes) -> tuple[int, int]:
    tid, pos = struct.unpack_from("<ii", rec, 4)
    if tid < 0:
        return (1 << 31, 0)
    return (tid, pos)


def _read_bam_rec(fh) -> bytes | None:
    hdr = fh.read(4)
    if not hdr or len(hdr) < 4:
        return None
    block = int.from_bytes(hdr, "little", signed=True)
    body = fh.read(block)
    if len(body) != block:
        raise ValueError("truncated uncompressed BAM record")
    return hdr + body


def order_run_perms(
    n: int,
    coord_addr: int,
    dhi_addr: int,
    dlo_addr: int,
    perm_addr: int,
    dupperm_addr: int,
) -> None:
    """Host argsort of one tile. GPU radix left leftover n (not a multiple of
    256) unsorted; these perms are what gather + k-way merge must follow."""
    n = int(n)
    if n <= 0:
        return
    coord = (ctypes.c_uint64 * n).from_address(int(coord_addr))
    dhi = (ctypes.c_uint64 * n).from_address(int(dhi_addr))
    dlo = (ctypes.c_uint64 * n).from_address(int(dlo_addr))
    perm = (ctypes.c_uint32 * n).from_address(int(perm_addr))
    dupperm = (ctypes.c_uint32 * n).from_address(int(dupperm_addr))
    idx = list(range(n))
    idx.sort(key=coord.__getitem__)
    for j, i in enumerate(idx):
        perm[j] = i
    idx.sort(key=lambda i: (dhi[i], dlo[i], i))
    for j, i in enumerate(idx):
        dupperm[j] = i


class BamRunStore:
    """SSD merge-sort runs: GPU-sorted BAM tiles, then Python k-way merge.

    Each run is one GPU-sorted tile (``METHYLGRAPHER_FM_SORT_TILE`` records).
    Host RAM and HBM stay O(tile); the 30× BAM lives on NVMe. Merge uses
    ``heapq`` (Mojo min-heap inside the FM mapper OOMs the kernel compile).
    """

    def __init__(self, spill_dir: str | os.PathLike[str] | None = None) -> None:
        raw = (
            str(spill_dir)
            if spill_dir is not None
            else os.environ.get("METHYLGRAPHER_BAM_ARENA_DIR", "")
        )
        if not raw or raw.strip().lower() in {"", "ram", "0", "false", "off", "none"}:
            raw = os.environ.get("METHYLGRAPHER_BAM_RUN_DIR", "")
        if not raw or raw.strip().lower() in {"", "ram", "0", "false", "off", "none"}:
            raise ValueError(
                "BamRunStore needs METHYLGRAPHER_BAM_ARENA_DIR or an explicit spill_dir"
            )
        self._dir = Path(raw) / "sorted_runs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._n = 0
        self._runs: list[tuple[Path, Path]] = []

    def n_runs(self) -> int:
        return self._n

    def open_raw_writer(self) -> _RawRunWriter:
        bam = self._dir / f"run_{self._n:06d}.bam.bin"
        return _RawRunWriter(str(bam))

    def write_sorted_from_arena(
        self,
        n: int,
        arena: BamArena,
        rec_addr: int,
        len_addr: int,
        perm_addr: int,
    ) -> None:
        """Parse arena chunks (pack order) and write a coord-sorted run BAM.

        Sort key is the packed record's refID/pos, not the Mojo ``h_coord``
        sidecar (leftover tiles can disagree). ``perm`` is updated to that order
        so markdup orig indices still line up.
        """
        n = int(n)
        _ = (rec_addr, len_addr)
        perm = (ctypes.c_uint32 * n).from_address(int(perm_addr))
        n_chunks = int(arena.n_chunks())
        addrs = [int(arena.chunk_addr(cid)) for cid in range(n_chunks)]
        # (tid_key, pos, orig, cid, off, rlen) — do not copy record bodies.
        entries: list[tuple[int, int, int, int, int, int]] = []
        orig = 0
        for cid in range(n_chunks):
            clen = int(arena.chunk_len(cid))
            if clen <= 0:
                continue
            addr = addrs[cid]
            off = 0
            while off + 4 <= clen:
                hdr = ctypes.string_at(addr + off, 12)
                block, tid, pos = struct.unpack_from("<iii", hdr, 0)
                rlen = 4 + block
                if rlen < 36 or off + rlen > clen:
                    raise ValueError("corrupt uncompressed BAM chunk")
                tidk = tid if tid >= 0 else (1 << 31)
                entries.append((tidk, pos, orig, cid, off, rlen))
                orig += 1
                off += rlen
        if orig != n:
            raise ValueError(f"arena records {orig} != tile n {n}")
        entries.sort()
        bam = self._dir / f"run_{self._n:06d}.bam.bin"
        with open(bam, "wb", buffering=8 * 1024 * 1024) as fh:
            for j, (_k, _p, orig_i, cid, loc, rlen) in enumerate(entries):
                perm[j] = orig_i
                fh.write(ctypes.string_at(addrs[cid] + loc, rlen))

    def write_keys(
        self,
        n: int,
        coord_addr: int,
        dhi_addr: int,
        dlo_addr: int,
        score_addr: int,
        pair_addr: int,
        len_addr: int,
        perm_addr: int,
        dupperm_addr: int,
    ) -> int:
        """Columnar keys for run ``n`` records; call after ``open_raw_writer`` close."""
        n = int(n)
        bam = self._dir / f"run_{self._n:06d}.bam.bin"
        keys = self._dir / f"run_{self._n:06d}.keys"
        with open(keys, "wb") as fh:
            fh.write(_RUN_MAGIC)
            fh.write(struct.pack("<Q", n))
            _dump_u(fh, coord_addr, n, 8)
            _dump_u(fh, dhi_addr, n, 8)
            _dump_u(fh, dlo_addr, n, 8)
            _dump_u(fh, score_addr, n, 4)
            _dump_u(fh, pair_addr, n, 4)
            _dump_u(fh, len_addr, n, 4)
            _dump_u(fh, perm_addr, n, 4)
            _dump_u(fh, dupperm_addr, n, 4)
        self._runs.append((bam, keys))
        idx = self._n
        self._n += 1
        return idx

    def merge_into(self, writer: BamWriter, do_markdup: bool = True) -> int:
        """K-way merge coord-sorted runs into ``writer``. Returns dups marked."""
        if not self._runs:
            return 0
        loaded = [(_RunKeys(k), bam) for bam, k in self._runs]
        dups_marked = 0
        flags: list[array.array] = [
            array.array("B", bytes(keys.n)) for keys, _ in loaded
        ]
        if do_markdup:
            dups_marked = _markdup_runs(loaded, flags)
        heap: list[tuple[tuple[int, int], int, int]] = []
        fhs: list[object] = []
        heads: list[bytes | None] = []
        for rid, (_keys, bam) in enumerate(loaded):
            fh = open(bam, "rb", buffering=8 * 1024 * 1024)
            fhs.append(fh)
            rec = _read_bam_rec(fh)
            heads.append(rec)
            if rec is not None:
                heapq.heappush(heap, (_bam_coord_key(rec), rid, 0))
        while heap:
            _key, rid, j = heapq.heappop(heap)
            rec = heads[rid]
            if rec is None:
                continue
            orig = loaded[rid][0].perm[j]
            buf = bytearray(rec)
            if flags[rid][orig]:
                _or_dup_flag_bytes(buf)
            writer.write_bytes(buf)
            nxt = _read_bam_rec(fhs[rid])
            heads[rid] = nxt
            j += 1
            if nxt is not None:
                heapq.heappush(heap, (_bam_coord_key(nxt), rid, j))
        for fh in fhs:
            fh.close()
        return dups_marked


def _markdup_runs(
    loaded: Sequence[tuple[_RunKeys, Path]],
    flags: list[array.array],
) -> int:
    """Same Picard-style rule as ``gpu_sort_markdup`` over k run heads."""
    heap: list[tuple[int, int, int, int, int]] = []
    for rid, (keys, _bam) in enumerate(loaded):
        if keys.n:
            orig = keys.dupperm[0]
            heapq.heappush(heap, (keys.dhi[orig], keys.dlo[orig], orig, rid, 0))
    marked = 0
    pending: list[tuple[int, int]] = []  # (rid, orig)
    cur_hi = 0
    cur_lo = 0

    def flush_group() -> None:
        nonlocal marked
        if not pending:
            return
        rid0, orig0 = pending[0]
        k0 = loaded[rid0][0]
        if k0.dhi[orig0] == _DUP_SENTINEL:
            pending.clear()
            return
        best_pair = k0.pair[orig0]
        best_score = k0.score[orig0]
        for rid, orig in pending:
            k = loaded[rid][0]
            sc = k.score[orig]
            pid = k.pair[orig]
            if sc > best_score or (sc == best_score and pid < best_pair):
                best_score = sc
                best_pair = pid
        for rid, orig in pending:
            if loaded[rid][0].pair[orig] != best_pair:
                flags[rid][orig] = 1
                marked += 1
        pending.clear()

    while heap:
        dhi, dlo, orig, rid, i = heapq.heappop(heap)
        if pending and (dhi != cur_hi or dlo != cur_lo):
            flush_group()
        if not pending:
            cur_hi = dhi
            cur_lo = dlo
        pending.append((rid, orig))
        i += 1
        keys = loaded[rid][0]
        if i < keys.n:
            norig = keys.dupperm[i]
            heapq.heappush(heap, (keys.dhi[norig], keys.dlo[norig], norig, rid, i))
    flush_group()
    return marked


def looks_bam_path(path: str) -> bool:
    p = path.lower()
    return p.endswith(".bam") and not p.endswith(".sam")
