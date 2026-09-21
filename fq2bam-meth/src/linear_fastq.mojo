# Bulk FASTQ ingest for the FM BAM path.
#
# pigz (or gzip -dc) decompresses in C; Mojo reads the pipe with libc `read`
# into an 8 MiB buffer, splits 4-line records, C2T/G2A-converts, and writes
# a byte arena. No Python `readline` / `str` / `export_ptrs` on this path.

from std.collections import List
from std.ffi import external_call
from std.memory import UnsafePointer, unsafe_memmove
from std.python import Python, PythonObject


comptime FQ_PIPE_BUF = 8 * 1024 * 1024
comptime FQ_ARENA_MIN = 4 * 1024 * 1024


def _fq_u8(addr: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)


def _fq_u32(addr: Int) -> UnsafePointer[UInt32, MutAnyOrigin]:
    return UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=addr)


def _fq_u64(addr: Int) -> UnsafePointer[UInt64, MutAnyOrigin]:
    return UnsafePointer[UInt64, MutAnyOrigin](unsafe_from_address=addr)


def _bs_mode(mode: String) -> Int:
    if mode == "C2T":
        return 1
    if mode == "G2A":
        return 2
    return 0


def _cvt(c: UInt8, mode: Int) -> UInt8:
    if mode == 1:
        if c == 67 or c == 99:
            return UInt8(84)
        return c
    if mode == 2:
        if c == 71 or c == 103:
            return UInt8(65)
        return c
    return c


def _looks_gz(path: String) -> Bool:
    var low = path.lower()
    return low.endswith(".gz") or low.endswith(".gzip")


struct FastqPipe(Copyable, Movable):
    var fd: Int32
    var has_proc: Bool
    var proc: PythonObject
    var buf: List[UInt8]
    var pos: Int
    var end: Int
    var eof: Bool
    var convert: Int

    def __init__(out self):
        self.fd = Int32(-1)
        self.has_proc = False
        self.proc = Python.none()
        self.buf = List[UInt8]()
        self.pos = 0
        self.end = 0
        self.eof = False
        self.convert = 0


struct FastqArena(Copyable, Movable):
    var data: List[UInt8]
    var used: Int
    var n1: Int
    var max_len: Int
    var name1_off: List[Int]
    var name1_len: List[Int]
    var orig1_off: List[Int]
    var orig1_len: List[Int]
    var qual1_off: List[Int]
    var qual1_len: List[Int]
    var seq1_off: List[Int]
    var seq1_len: List[Int]
    var name2_off: List[Int]
    var name2_len: List[Int]
    var orig2_off: List[Int]
    var orig2_len: List[Int]
    var qual2_off: List[Int]
    var qual2_len: List[Int]
    var seq2_off: List[Int]
    var seq2_len: List[Int]

    def __init__(out self):
        self.data = List[UInt8]()
        self.used = 0
        self.n1 = 0
        self.max_len = 0
        self.name1_off = List[Int]()
        self.name1_len = List[Int]()
        self.orig1_off = List[Int]()
        self.orig1_len = List[Int]()
        self.qual1_off = List[Int]()
        self.qual1_len = List[Int]()
        self.seq1_off = List[Int]()
        self.seq1_len = List[Int]()
        self.name2_off = List[Int]()
        self.name2_len = List[Int]()
        self.orig2_off = List[Int]()
        self.orig2_len = List[Int]()
        self.qual2_off = List[Int]()
        self.qual2_len = List[Int]()
        self.seq2_off = List[Int]()
        self.seq2_len = List[Int]()


struct FastqPairStream(Copyable, Movable):
    var r1: FastqPipe
    var r2: FastqPipe
    var paired: Bool

    def __init__(out self):
        self.r1 = FastqPipe()
        self.r2 = FastqPipe()
        self.paired = False


def _open_pipe(path: String, convert: Int) raises -> FastqPipe:
    var p = FastqPipe()
    p.convert = convert
    p.buf.resize(FQ_PIPE_BUF, UInt8(0))
    var os_mod = Python.import_module("os")
    if not _looks_gz(path):
        p.fd = Int32(Int(py=os_mod.open(path, os_mod.O_RDONLY)))
        return p^
    var shutil = Python.import_module("shutil")
    var sp = Python.import_module("subprocess")
    var decomp = shutil.which("pigz")
    var cmd = Python.list()
    if decomp is not None:
        cmd.append(decomp)
        cmd.append("-dc")
        cmd.append(path)
    else:
        var gz = shutil.which("gzip")
        if gz is None:
            raise Error("pigz or gzip required for gzip FASTQ: " + path)
        cmd.append(gz)
        cmd.append("-dc")
        cmd.append(path)
    var proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    p.fd = Int32(Int(py=os_mod.dup(proc.stdout.fileno())))
    p.has_proc = True
    p.proc = proc
    return p^


def _close_pipe(mut p: FastqPipe) raises:
    var os_mod = Python.import_module("os")
    if Int(p.fd) >= 0:
        try:
            _ = os_mod.close(p.fd)
        except:
            pass
        p.fd = Int32(-1)
    if p.has_proc:
        try:
            p.proc.kill()
            _ = p.proc.wait()
        except:
            pass
        p.has_proc = False
        p.proc = Python.none()


def _pipe_refill(mut p: FastqPipe) raises:
    if p.eof:
        return
    var cap = len(p.buf)
    if cap < FQ_PIPE_BUF:
        p.buf.resize(FQ_PIPE_BUF, UInt8(0))
        cap = len(p.buf)
    var leftover = p.end - p.pos
    if leftover < 0:
        leftover = 0
    if p.pos > 0 and leftover > 0:
        unsafe_memmove(
            dest=_fq_u8(Int(p.buf.unsafe_ptr())),
            src=_fq_u8(Int(p.buf.unsafe_ptr()) + p.pos),
            count=leftover,
        )
        p.pos = 0
        p.end = leftover
    elif p.pos >= p.end:
        p.pos = 0
        p.end = 0
        leftover = 0
    var space = cap - p.end
    if space < 1:
        var neu = cap * 2
        p.buf.resize(neu, UInt8(0))
        cap = neu
        space = cap - p.end
    var n = external_call["read", Int](
        p.fd, _fq_u8(Int(p.buf.unsafe_ptr()) + p.end), UInt(space)
    )
    if n < 0:
        n = external_call["read", Int](
            p.fd, _fq_u8(Int(p.buf.unsafe_ptr()) + p.end), UInt(space)
        )
    if n < 0:
        raise Error("FASTQ read failed")
    if n == 0:
        # Distinguish clean EOF from pigz/gzip death (CRC/OOM/kill).
        if p.has_proc:
            var rc_obj = p.proc.poll()
            if rc_obj is not None:
                var rc = Int(py=rc_obj)
                if rc != 0:
                    raise Error(
                        "FASTQ decompress failed rc="
                        + String(rc)
                        + " (pigz/gzip exited before EOF)"
                    )
        p.eof = True
        return
    p.end = p.end + n


def _pipe_line(mut p: FastqPipe, mut off: Int, mut n: Int) raises -> Bool:
    while True:
        var i = p.pos
        while i < p.end:
            if p.buf[i] == 10:
                var s = p.pos
                var e = i
                if e > s and p.buf[e - 1] == 13:
                    e -= 1
                p.pos = i + 1
                off = s
                n = e - s
                return True
            i += 1
        if p.eof:
            if p.pos < p.end:
                var s2 = p.pos
                var e2 = p.end
                if e2 > s2 and p.buf[e2 - 1] == 13:
                    e2 -= 1
                p.pos = p.end
                off = s2
                n = e2 - s2
                return n > 0
            return False
        var before = p.end - p.pos
        _pipe_refill(p)
        if p.eof and (p.end - p.pos) == before:
            if p.pos < p.end:
                var s3 = p.pos
                var e3 = p.end
                if e3 > s3 and p.buf[e3 - 1] == 13:
                    e3 -= 1
                p.pos = p.end
                off = s3
                n = e3 - s3
                return n > 0
            return False


def _arena_need(mut a: FastqArena, n: Int):
    var need = a.used + n
    var cap = len(a.data)
    if need <= cap:
        return
    if cap < FQ_ARENA_MIN:
        cap = FQ_ARENA_MIN
    while cap < need:
        cap = cap * 2
    a.data.resize(cap, UInt8(0))


def _arena_push_raw(
    mut a: FastqArena, src_addr: Int, n: Int, convert: Int
) raises -> Int:
    _arena_need(a, n)
    var off = a.used
    if n > 0:
        if convert == 0:
            unsafe_memmove(
                dest=_fq_u8(Int(a.data.unsafe_ptr()) + off),
                src=_fq_u8(src_addr),
                count=n,
            )
        else:
            var dst = _fq_u8(Int(a.data.unsafe_ptr()) + off)
            var src = _fq_u8(src_addr)
            var i = 0
            while i < n:
                dst[i] = _cvt(src[i], convert)
                i += 1
    a.used = a.used + n
    return off


def _qname_span(buf_addr: Int, n: Int, mut start: Int, mut ln: Int):
    var s = 0
    var e = n
    if e > 0 and Int(_fq_u8(buf_addr)[0]) == 64:
        s = 1
    var i = s
    while i < e:
        var c = Int(_fq_u8(buf_addr)[i])
        if c == 32 or c == 47:
            e = i
            break
        i += 1
    var out_n = e - s
    if out_n < 0:
        out_n = 0
    start = s
    ln = out_n


def _ensure_tables(mut a: FastqArena, n_pairs: Int, paired: Bool):
    if len(a.name1_off) < n_pairs:
        a.name1_off.resize(n_pairs, 0)
        a.name1_len.resize(n_pairs, 0)
        a.orig1_off.resize(n_pairs, 0)
        a.orig1_len.resize(n_pairs, 0)
        a.qual1_off.resize(n_pairs, 0)
        a.qual1_len.resize(n_pairs, 0)
        a.seq1_off.resize(n_pairs, 0)
        a.seq1_len.resize(n_pairs, 0)
    if paired and len(a.name2_off) < n_pairs:
        a.name2_off.resize(n_pairs, 0)
        a.name2_len.resize(n_pairs, 0)
        a.orig2_off.resize(n_pairs, 0)
        a.orig2_len.resize(n_pairs, 0)
        a.qual2_off.resize(n_pairs, 0)
        a.qual2_len.resize(n_pairs, 0)
        a.seq2_off.resize(n_pairs, 0)
        a.seq2_len.resize(n_pairs, 0)


def fq_reserve(mut a: FastqArena, n_pairs: Int, paired: Bool):
    var cap = n_pairs * 2 * 512
    if paired:
        cap = n_pairs * 4 * 512
    if cap < FQ_ARENA_MIN:
        cap = FQ_ARENA_MIN
    if len(a.data) < cap:
        a.data.resize(cap, UInt8(0))
    _ensure_tables(a, n_pairs, paired)


def fq_reset(mut a: FastqArena):
    a.used = 0
    a.n1 = 0
    a.max_len = 0


def _store_rec(
    mut a: FastqArena,
    mut pipe: FastqPipe,
    idx: Int,
    mate2: Bool,
) raises -> Bool:
    var hoff = 0
    var hlen = 0
    if not _pipe_line(pipe, hoff, hlen):
        return False
    var soff = 0
    var slen = 0
    var poff = 0
    var plen = 0
    var qoff = 0
    var qlen = 0
    if not _pipe_line(pipe, soff, slen):
        raise Error("truncated FASTQ (missing SEQ)")
    if not _pipe_line(pipe, poff, plen):
        raise Error("truncated FASTQ (missing +)")
    if not _pipe_line(pipe, qoff, qlen):
        raise Error("truncated FASTQ (missing QUAL)")
    _ = poff
    _ = plen
    var buf_addr = Int(pipe.buf.unsafe_ptr())
    var qs = 0
    var qn = 0
    _qname_span(buf_addr + hoff, hlen, qs, qn)
    var name_off = _arena_push_raw(a, buf_addr + hoff + qs, qn, 0)
    var orig_off = _arena_push_raw(a, buf_addr + soff, slen, 0)
    var seq_off = _arena_push_raw(a, buf_addr + soff, slen, pipe.convert)
    var qual_off = _arena_push_raw(a, buf_addr + qoff, qlen, 0)
    if mate2:
        a.name2_off[idx] = name_off
        a.name2_len[idx] = qn
        a.orig2_off[idx] = orig_off
        a.orig2_len[idx] = slen
        a.seq2_off[idx] = seq_off
        a.seq2_len[idx] = slen
        a.qual2_off[idx] = qual_off
        a.qual2_len[idx] = qlen
    else:
        a.name1_off[idx] = name_off
        a.name1_len[idx] = qn
        a.orig1_off[idx] = orig_off
        a.orig1_len[idx] = slen
        a.seq1_off[idx] = seq_off
        a.seq1_len[idx] = slen
        a.qual1_off[idx] = qual_off
        a.qual1_len[idx] = qlen
    if slen > a.max_len:
        a.max_len = slen
    return True


def fq_open(
    fq1: String, fq2: String, bs_r1: String, bs_r2: String
) raises -> FastqPairStream:
    var s = FastqPairStream()
    s.r1 = _open_pipe(fq1, _bs_mode(bs_r1))
    s.paired = fq2.byte_length() > 0
    if s.paired:
        s.r2 = _open_pipe(fq2, _bs_mode(bs_r2))
    return s^


def fq_close(mut s: FastqPairStream) raises:
    _close_pipe(s.r1)
    if s.paired:
        _close_pipe(s.r2)


def fq_read_batch(mut s: FastqPairStream, mut a: FastqArena, n_pairs: Int) raises:
    fq_reset(a)
    var n = n_pairs
    if n < 1:
        n = 1
    _ensure_tables(a, n, s.paired)
    var i = 0
    while i < n:
        if not _store_rec(a, s.r1, i, False):
            break
        if s.paired:
            if not _store_rec(a, s.r2, i, True):
                raise Error("paired FASTQ length mismatch (R2 ended early)")
        i += 1
    a.n1 = i


def fq_pack_bases(
    mut a: FastqArena,
    paired: Bool,
    dest_addr: Int,
    max_len: Int,
    lens_addr: Int,
):
    var n1 = a.n1
    var n_seq = n1
    if paired:
        n_seq = n1 * 2
    var n_bases = n_seq * max_len
    if n_bases > 0:
        _ = external_call["memset", UnsafePointer[UInt8, MutAnyOrigin]](
            _fq_u8(dest_addr), Int32(78), UInt(n_bases)
        )
    var lens = _fq_u32(lens_addr)
    var base = Int(a.data.unsafe_ptr())
    var i = 0
    while i < n1:
        var ln = a.seq1_len[i]
        lens[i] = UInt32(ln)
        if ln > 0:
            unsafe_memmove(
                dest=_fq_u8(dest_addr + i * max_len),
                src=_fq_u8(base + a.seq1_off[i]),
                count=ln,
            )
        i += 1
    if paired:
        i = 0
        while i < n1:
            var ln2 = a.seq2_len[i]
            lens[n1 + i] = UInt32(ln2)
            if ln2 > 0:
                unsafe_memmove(
                    dest=_fq_u8(dest_addr + (n1 + i) * max_len),
                    src=_fq_u8(base + a.seq2_off[i]),
                    count=ln2,
                )
            i += 1


def _fill_offs(
    base: Int,
    offs: List[Int],
    lens: List[Int],
    n: Int,
    addr_dest: Int,
    len_dest: Int,
):
    if n <= 0:
        return
    var addrs = _fq_u64(addr_dest)
    var out_n = _fq_u32(len_dest)
    var i = 0
    while i < n:
        var ln = lens[i]
        out_n[i] = UInt32(ln)
        if ln > 0:
            addrs[i] = UInt64(base + offs[i])
        else:
            addrs[i] = UInt64(0)
        i += 1


def fq_export_ptrs(
    mut a: FastqArena,
    paired: Bool,
    n1_name_a: Int,
    n1_name_n: Int,
    n1_orig_a: Int,
    n1_orig_n: Int,
    n1_qual_a: Int,
    n1_qual_n: Int,
    n2_name_a: Int,
    n2_name_n: Int,
    n2_orig_a: Int,
    n2_orig_n: Int,
    n2_qual_a: Int,
    n2_qual_n: Int,
):
    var n1 = a.n1
    if n1 <= 0:
        return
    var base = Int(a.data.unsafe_ptr())
    _fill_offs(base, a.name1_off, a.name1_len, n1, n1_name_a, n1_name_n)
    _fill_offs(base, a.orig1_off, a.orig1_len, n1, n1_orig_a, n1_orig_n)
    _fill_offs(base, a.qual1_off, a.qual1_len, n1, n1_qual_a, n1_qual_n)
    if paired:
        _fill_offs(base, a.name2_off, a.name2_len, n1, n2_name_a, n2_name_n)
        _fill_offs(base, a.orig2_off, a.orig2_len, n1, n2_orig_a, n2_orig_n)
        _fill_offs(base, a.qual2_off, a.qual2_len, n1, n2_qual_a, n2_qual_n)
