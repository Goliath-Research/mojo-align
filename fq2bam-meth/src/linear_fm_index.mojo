# BWA 0.7.x FM-index loader (mmap .bwt / .sa / .pac / .ann).
# Layout matches bwt_restore_bwt / bwt_restore_sa / bns_restore_core.
# Used by the FM / BWA-MEM-style GPU engine (linear_gpu_fm.mojo).

from std.collections import List
from std.memory import UnsafePointer
from std.python import Python, PythonObject

from mojo_align_env import ensure_python_path


struct FmContig(Copyable, Movable):
    var name: String
    var length: Int
    var offset: Int

    def __init__(out self, name: String, length: Int, offset: Int):
        self.name = name
        self.length = length
        self.offset = offset


struct FmMatch(Copyable, Movable):
    """Exact FM backward-search result (k..l inclusive, occ hits)."""

    var k: UInt64
    var l: UInt64
    var occ: Int

    def __init__(out self, k: UInt64 = 0, l: UInt64 = 0, occ: Int = 0):
        self.k = k
        self.l = l
        self.occ = occ


struct FmDepos(Copyable, Movable):
    """Forward pac coordinate + reverse flag (bns_depos)."""

    var pos: Int
    var is_rev: Int

    def __init__(out self, pos: Int = 0, is_rev: Int = 0):
        self.pos = pos
        self.is_rev = is_rev


struct FmIndex(Copyable, Movable):
    """Memory-mapped BWA FM-index for a bwameth.c2t prefix."""

    var prefix: String
    var primary: UInt64
    var seq_len: UInt64
    var l_pac: UInt64
    var sa_intv: Int
    var n_sa: Int
    var bwt_size: Int
    var L2: InlineArray[UInt64, 5]
    var bwt_addr: Int
    var sa_addr: Int
    var pac_addr: Int
    var pac_size: Int
    var contigs: List[FmContig]
    var _keep_bwt: PythonObject
    var _keep_sa: PythonObject
    var _keep_pac: PythonObject

    def __init__(out self):
        self.prefix = String("")
        self.primary = 0
        self.seq_len = 0
        self.l_pac = 0
        self.sa_intv = 32
        self.n_sa = 0
        self.bwt_size = 0
        self.L2 = InlineArray[UInt64, 5](fill=UInt64(0))
        self.bwt_addr = 0
        self.sa_addr = 0
        self.pac_addr = 0
        self.pac_size = 0
        self.contigs = List[FmContig]()
        self._keep_bwt = Python.none()
        self._keep_sa = Python.none()
        self._keep_pac = Python.none()

    def _bootstrap_sys_path(self) raises:
        ensure_python_path()

    def load(mut self, prefix: String) raises:
        """Load prefix.bwt/.sa/.pac/.ann (Clara / bwa 0.7 layout)."""
        self._bootstrap_sys_path()
        var os_mod = Python.import_module("os")
        var bridge = Python.import_module("min_mmap_bridge")
        var bwt_path = prefix + ".bwt"
        var sa_path = prefix + ".sa"
        var pac_path = prefix + ".pac"
        var ann_path = prefix + ".ann"
        if not Bool(os_mod.path.isfile(bwt_path)):
            raise Error("FmIndex missing " + bwt_path)
        if not Bool(os_mod.path.isfile(sa_path)):
            raise Error("FmIndex missing " + sa_path)
        if not Bool(os_mod.path.isfile(pac_path)):
            raise Error("FmIndex missing " + pac_path)
        if not Bool(os_mod.path.isfile(ann_path)):
            raise Error("FmIndex missing " + ann_path)

        # open_min_mmap → (addr, size, None, fd); keep fd alive via _keep_*.
        var opened_b = bridge.open_min_mmap(bwt_path)
        var opened_s = bridge.open_min_mmap(sa_path)
        var opened_p = bridge.open_min_mmap(pac_path)
        self._keep_bwt = Python.tuple(opened_b[2], opened_b[3])
        self._keep_sa = Python.tuple(opened_s[2], opened_s[3])
        self._keep_pac = Python.tuple(opened_p[2], opened_p[3])
        self.bwt_addr = Int(py=opened_b[0])
        self.sa_addr = Int(py=opened_s[0])
        self.pac_addr = Int(py=opened_p[0])
        self.pac_size = Int(py=opened_p[1])
        var bwt_bytes = Int(py=opened_b[1])
        # header: primary + L2[1..4] = 5 * uint64 = 40 bytes
        self.bwt_size = (bwt_bytes - 40) // 4
        self.primary = self._u64(self.bwt_addr, 0)
        self.L2[0] = 0
        self.L2[1] = self._u64(self.bwt_addr, 1)
        self.L2[2] = self._u64(self.bwt_addr, 2)
        self.L2[3] = self._u64(self.bwt_addr, 3)
        self.L2[4] = self._u64(self.bwt_addr, 4)
        self.seq_len = self.L2[4]
        # SA header: primary, L2[1..4], sa_intv, seq_len, then sa[1..]
        var sa_primary = self._u64(self.sa_addr, 0)
        if sa_primary != self.primary:
            raise Error("FmIndex SA/BWT primary mismatch")
        self.sa_intv = Int(self._u64(self.sa_addr, 5))
        var sa_seq = self._u64(self.sa_addr, 6)
        if sa_seq != self.seq_len:
            raise Error("FmIndex SA/BWT seq_len mismatch")
        self.n_sa = Int((self.seq_len + UInt64(self.sa_intv)) // UInt64(self.sa_intv))

        # .ann
        self.contigs = List[FmContig]()
        var builtins = Python.import_module("builtins")
        var fh = builtins.open(ann_path, "r")
        var hdr = String(fh.readline()).split()
        if len(hdr) < 2:
            raise Error("FmIndex bad .ann header")
        self.l_pac = UInt64(Int(hdr[0]))
        var n_seqs = Int(hdr[1])
        var off = 0
        var i = 0
        while i < n_seqs:
            var l1 = String(fh.readline()).split()
            var l2 = String(fh.readline()).split()
            if len(l1) < 2 or len(l2) < 2:
                raise Error("FmIndex truncated .ann")
            var name = String(l1[1])
            var ln = Int(l2[1])  # contig length from .ann line 2
            self.contigs.append(FmContig(name, ln, off))
            off += ln
            i += 1
        fh.close()
        if UInt64(off) != self.l_pac:
            raise Error("FmIndex .ann offsets != l_pac")
        self.prefix = prefix
        print(
            "MojoLinear FM-index loaded seq_len=",
            Int(self.seq_len),
            " l_pac=",
            Int(self.l_pac),
            " bwt_words=",
            self.bwt_size,
            " n_sa=",
            self.n_sa,
            " contigs=",
            len(self.contigs),
            " prefix=",
            prefix,
        )

    def _u64(self, base_addr: Int, idx: Int) -> UInt64:
        var p = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=base_addr + idx * 8
        )
        var v: UInt64 = 0
        var b = 0
        while b < 8:
            v |= UInt64(p[b]) << (UInt64(b) * 8)
            b += 1
        return v

    def _bwt_u32(self, idx: Int) -> UInt32:
        # BWT payload starts after 40-byte header.
        var p = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.bwt_addr + 40 + idx * 4
        )
        var v: UInt32 = 0
        var b = 0
        while b < 4:
            v |= UInt32(p[b]) << (UInt32(b) * 8)
            b += 1
        return v

    def _sa_sample(self, idx: Int) -> UInt64:
        # sa[0] = -1 (not stored); file has sa[1..] at word offset 7
        if idx <= 0:
            return UInt64(0xFFFFFFFFFFFFFFFF)
        return self._u64(self.sa_addr, 7 + (idx - 1))

    def pac_base(self, pos: Int) -> Int:
        if pos < 0 or UInt64(pos) >= self.l_pac:
            return 4
        var byte = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.pac_addr + (pos >> 2)
        )[0]
        return Int((byte >> UInt8(((~pos) & 3) << 1)) & 3)

    @staticmethod
    def _occ_aux(y: UInt64, c: Int) -> Int:
        var yy = y
        var a: UInt64
        var b: UInt64
        if (c & 2) != 0:
            a = yy
        else:
            a = ~yy
        if (c & 1) != 0:
            b = yy
        else:
            b = ~yy
        yy = (a >> 1) & b & UInt64(0x5555555555555555)
        yy = (yy & UInt64(0x3333333333333333)) + (
            (yy >> 2) & UInt64(0x3333333333333333)
        )
        yy = ((yy + (yy >> 4)) & UInt64(0x0F0F0F0F0F0F0F0F)) * UInt64(
            0x0101010101010101
        )
        return Int(yy >> 56)

    def b0(self, k: UInt64) -> Int:
        var kk = Int(k)
        var idx = ((kk >> 7) << 4) + 8 + (((kk) & 0x7F) >> 4)
        var w = self._bwt_u32(idx)
        return Int((w >> UInt32(((~kk) & 15) << 1)) & 3)

    def occ(self, k: Int, c: Int) raises -> UInt64:
        if k < 0:
            return 0
        if UInt64(k) == self.seq_len:
            return self.L2[c + 1] - self.L2[c]
        var kk = UInt64(k)
        if kk >= self.primary:
            kk -= 1
        var base = Int((kk >> 7) << 4)
        # little-endian uint64 occ counts
        var n = UInt64(self._bwt_u32(base + c * 2)) | (
            UInt64(self._bwt_u32(base + c * 2 + 1)) << 32
        )
        var p = base + 8
        var end = p + Int(
            (((kk >> 5) - ((kk & UInt64(0xFFFFFFFFFFFFFF80)) >> 5)) << 1)
        )
        while p < end:
            var y = (UInt64(self._bwt_u32(p)) << 32) | UInt64(
                self._bwt_u32(p + 1)
            )
            n += UInt64(self._occ_aux(y, c))
            p += 2
        var bits = Int(((~Int(kk)) & 31) << 1)
        var mask: UInt64 = UInt64(0xFFFFFFFFFFFFFFFF)
        if bits > 0 and bits < 64:
            mask = ~((UInt64(1) << UInt64(bits)) - 1)
        var y2 = (
            (UInt64(self._bwt_u32(p)) << 32) | UInt64(self._bwt_u32(p + 1))
        ) & mask
        n += UInt64(self._occ_aux(y2, c))
        if c == 0:
            n -= UInt64((~Int(kk)) & 31)
        return n

    def inv_psi(self, k: UInt64) raises -> UInt64:
        if k == self.primary:
            return 0
        var x = k
        if k > self.primary:
            x -= 1
        var c = self.b0(x)
        return self.L2[c] + self.occ(Int(k), c)

    def sa_at(self, k: UInt64) raises -> UInt64:
        var kk = k
        var sa: UInt64 = 0
        var mask = UInt64(self.sa_intv - 1)
        while (kk & mask) != 0:
            sa += 1
            kk = self.inv_psi(kk)
        return sa + self._sa_sample(Int(kk // UInt64(self.sa_intv)))

    def match_exact(
        self, codes: UnsafePointer[UInt8, MutAnyOrigin], length: Int
    ) raises -> FmMatch:
        """Backward search. Returns FmMatch with occ=0 if no match."""
        var k: UInt64 = 0
        var l = self.seq_len
        var i = length - 1
        while i >= 0:
            var c = Int(codes[i])
            if c > 3:
                return FmMatch()
            var ok = self.occ(Int(k) - 1, c)
            var ol = self.occ(Int(l), c)
            k = self.L2[c] + ok + 1
            l = self.L2[c] + ol
            if k > l:
                return FmMatch()
            i -= 1
        return FmMatch(k, l, Int(l - k + 1))

    def depos(self, pos: UInt64) raises -> FmDepos:
        """Return forward pac coordinate + reverse flag (bns_depos)."""
        if pos >= self.l_pac:
            var rp = self.l_pac * 2 - 1 - pos
            return FmDepos(Int(rp), 1)
        return FmDepos(Int(pos), 0)

    def rid_at(self, pos_f: Int) raises -> Int:
        """Binary search contig for forward pac coordinate."""
        var lo = 0
        var hi = len(self.contigs)
        while lo < hi:
            var mid = (lo + hi) // 2
            var c = self.contigs[mid]
            if pos_f < c.offset:
                hi = mid
            elif pos_f >= c.offset + c.length:
                lo = mid + 1
            else:
                return mid
        return -1

    def contig_name(self, rid: Int) raises -> String:
        return self.contigs[rid].name

    def contig_length(self, rid: Int) raises -> Int:
        return self.contigs[rid].length

    def contig_offset(self, rid: Int) raises -> Int:
        return self.contigs[rid].offset

    def contig_count(self) -> Int:
        return len(self.contigs)


def fm_prefix_from_ref(ref_fa: String) raises -> String:
    """Resolve BWA prefix: ref.bwameth.c2t or ref itself if it already is."""
    var os_mod = Python.import_module("os")
    if Bool(os_mod.path.isfile(ref_fa + ".bwt")):
        return ref_fa
    var bw = ref_fa + ".bwameth.c2t"
    if Bool(os_mod.path.isfile(bw + ".bwt")):
        return bw
    raise Error("No BWA FM-index (.bwt) beside " + ref_fa)
