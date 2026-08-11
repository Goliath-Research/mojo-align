# Linear C2T reference index for MojoFq2bamMeth (short-read WGBS).
#
# Fleet cache: dense-v1 binary pack (kmers.bin / offsets.bin / postings.bin),
# mmap'd like Giraffe mojo_segments. In-memory Dict build remains for tiny
# fixtures when no pack is present.

from std.collections import Dict, List
from std.memory import UnsafePointer
from std.python import Python, PythonObject

from utility import open_text_read, open_text_write


struct LinearContig(Copyable, Movable):
    var name: String
    var seq: String

    def __init__(out self, name: String, seq: String):
        self.name = name
        self.seq = seq


def _u64_le(addr: Int, idx: Int) -> UInt64:
    var base = addr + idx * 8
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=base)
    var v: UInt64 = 0
    var b = 0
    while b < 8:
        v |= UInt64(p[b]) << (UInt64(b) * 8)
        b += 1
    return v


def _u32_le(addr: Int, idx: Int) -> UInt32:
    var base = addr + idx * 4
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=base)
    var v: UInt32 = 0
    var b = 0
    while b < 4:
        v |= UInt32(p[b]) << (UInt32(b) * 8)
        b += 1
    return v


def _encode_kmer_u64(mer: String, k: Int) raises -> UInt64:
    if mer.byte_length() != k:
        raise Error("kmer length mismatch")
    var v: UInt64 = 0
    var i = 0
    while i < k:
        var ch = String(mer[byte = i : i + 1]).upper()
        var bits: UInt64 = 0
        if ch == "A":
            bits = 0
        elif ch == "C":
            bits = 1
        elif ch == "G":
            bits = 2
        elif ch == "T":
            bits = 3
        else:
            # Ambiguous — return sentinel larger than any valid k-mer.
            return UInt64(0xFFFFFFFFFFFFFFFF)
        v = (v << 2) | bits
        i += 1
    return v


struct LinearIndex(Copyable, Movable):
    """FASTA + dense-v1 mmap postings (or in-memory Dict for toys)."""

    var contigs: List[LinearContig]
    var postings: Dict[String, String]
    var hit_table: List[String]
    var k: Int
    var cache_dir: String
    var n_postings: Int
    var dense: Bool
    var n_keys: Int
    var kmers_addr: Int
    var kmers_size: Int
    var offsets_addr: Int
    var offsets_size: Int
    var postings_addr: Int
    var postings_size: Int
    var contig_names: List[String]
    var _keep_kmers: PythonObject
    var _keep_offsets: PythonObject
    var _keep_postings: PythonObject

    def __init__(out self, k: Int = 15):
        self.contigs = List[LinearContig]()
        self.postings = Dict[String, String]()
        self.hit_table = List[String]()
        self.k = k
        self.cache_dir = String("")
        self.n_postings = 0
        self.dense = False
        self.n_keys = 0
        self.kmers_addr = 0
        self.kmers_size = 0
        self.offsets_addr = 0
        self.offsets_size = 0
        self.postings_addr = 0
        self.postings_size = 0
        self.contig_names = List[String]()
        self._keep_kmers = Python.none()
        self._keep_offsets = Python.none()
        self._keep_postings = Python.none()

    def load_fasta(mut self, fasta_path: String) raises:
        # Bulk-read via Python (Mojo readline on multi-GB refs is glacial).
        print("MojoLinear loading FASTA ", fasta_path)
        var builtins = Python.import_module("builtins")
        var fh = builtins.open(fasta_path, "r")
        var text = String(fh.read())
        fh.close()
        self.contigs = List[LinearContig]()
        var name = String("")
        var seq = String("")
        var lines = text.split("\n")
        var li = 0
        var n_lines = len(lines)
        while li < n_lines:
            var line = String(lines[li])
            li += 1
            if line.byte_length() == 0:
                continue
            if line.startswith(">"):
                if name.byte_length() > 0:
                    self.contigs.append(LinearContig(name, seq))
                name = String(line[byte = 1 : line.byte_length()])
                var sp = name.split(" ")
                if len(sp) > 0:
                    name = String(sp[0])
                var tab = name.split("\t")
                if len(tab) > 0:
                    name = String(tab[0])
                seq = String("")
            else:
                seq += line.upper()
        if name.byte_length() > 0:
            self.contigs.append(LinearContig(name, seq))
        print(
            "MojoLinear FASTA contigs=",
            len(self.contigs),
            " bases=",
            self.total_bases(),
        )

    def _add_posting(mut self, mer: String, loc: String) raises:
        if mer in self.postings:
            self.postings[mer] = self.postings[mer] + "," + loc
        else:
            self.postings[mer] = loc
        self.n_postings += 1

    def build_kmer_index(mut self) raises:
        self.postings = Dict[String, String]()
        self.hit_table = List[String]()
        self.n_postings = 0
        self.dense = False
        for c in self.contigs:
            var n = c.seq.byte_length()
            if n < self.k:
                continue
            var i = 0
            while i <= n - self.k:
                var mer = String(c.seq[byte = i : i + self.k])
                var loc = c.name + ":" + String(i)
                self._add_posting(mer, loc)
                i += 1

    def _bootstrap_sys_path(self) raises:
        var os_mod = Python.import_module("os")
        var sys_mod = Python.import_module("sys")
        sys_mod.path.insert(0, String(os_mod.getcwd()))
        sys_mod.path.insert(0, "/home/ubuntu/mojo-align/fq2bam-meth/python")
        sys_mod.path.insert(0, "/home/ubuntu/mojo-align/giraffe/python")
        sys_mod.path.insert(0, "/home/ubuntu/mojo-align/methylgrapher")
        sys_mod.path.insert(0, "/opt/methylgrapher-mojo")

    def _load_dense_pack(mut self, cache_dir: String) raises -> Bool:
        var os_mod = Python.import_module("os")
        var meta_path = cache_dir + "/meta.json"
        var kmers_path = cache_dir + "/kmers.bin"
        var offsets_path = cache_dir + "/offsets.bin"
        var postings_path = cache_dir + "/postings.bin"
        if (
            not Bool(os_mod.path.isfile(meta_path))
            or not Bool(os_mod.path.isfile(kmers_path))
            or not Bool(os_mod.path.isfile(offsets_path))
            or not Bool(os_mod.path.isfile(postings_path))
        ):
            return False

        self._bootstrap_sys_path()
        var json_mod = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var fh = builtins.open(meta_path, "r")
        var meta = json_mod.load(fh)
        fh.close()
        var fmt = String(meta["format"])
        if fmt != "dense-v1":
            return False
        self.k = Int(py=meta["k"])
        self.n_keys = Int(py=meta["n_keys"])
        self.n_postings = Int(py=meta["n_postings"])
        self.contig_names = List[String]()
        var names = meta["contigs"]
        var ni = 0
        var n_names = Int(names.__len__())
        while ni < n_names:
            self.contig_names.append(String(names[ni]))
            ni += 1

        var bridge = Python.import_module("min_mmap_bridge")
        var opened_k = bridge.open_min_mmap(kmers_path)
        var opened_o = bridge.open_min_mmap(offsets_path)
        var opened_p = bridge.open_min_mmap(postings_path)
        self.kmers_addr = Int(py=opened_k[0])
        self.kmers_size = Int(py=opened_k[1])
        self._keep_kmers = Python.tuple(opened_k[2], opened_k[3])
        self.offsets_addr = Int(py=opened_o[0])
        self.offsets_size = Int(py=opened_o[1])
        self._keep_offsets = Python.tuple(opened_o[2], opened_o[3])
        self.postings_addr = Int(py=opened_p[0])
        self.postings_size = Int(py=opened_p[1])
        self._keep_postings = Python.tuple(opened_p[2], opened_p[3])
        if self.kmers_addr == 0 or self.offsets_addr == 0 or self.postings_addr == 0:
            raise Error("dense linear pack mmap failed: " + cache_dir)
        self.dense = True
        self.cache_dir = cache_dir
        print(
            "MojoLinear dense-v1 mmap n_keys=",
            self.n_keys,
            " n_postings=",
            self.n_postings,
            " root=",
            cache_dir,
        )
        return True

    def save_cache(self, cache_dir: String) raises:
        """Build dense-v1 pack via Python (fleet format). Requires contigs loaded."""
        self._bootstrap_sys_path()
        var os_mod = Python.import_module("os")
        _ = os_mod.makedirs(cache_dir, exist_ok=True)
        # Write a temporary FASTA from in-memory contigs, then pack.
        var tmp_fa = cache_dir + "/ref.fa"
        var fa = open_text_write(tmp_fa)
        for c in self.contigs:
            fa.write(">" + c.name + "\n")
            fa.write(c.seq + "\n")
        fa.close()
        var pack = Python.import_module("mojo_linear_pack")
        _ = pack.build_dense_pack(tmp_fa, cache_dir, self.k)
        print("MojoLinear saved dense-v1 → ", cache_dir)

    def load_cache(mut self, cache_dir: String) raises -> Bool:
        if self._load_dense_pack(cache_dir):
            # Sequences for extend/verify (ensure_mojo_linear_index.sh links ref.fa).
            var os_mod = Python.import_module("os")
            var ref_fa = cache_dir + "/ref.fa"
            if Bool(os_mod.path.isfile(ref_fa)):
                self.load_fasta(ref_fa)
            else:
                raise Error(
                    "dense pack missing ref.fa (C2T FASTA link) under " + cache_dir
                )
            return True

        # Legacy hits.tsv (toys / old caches)
        var os_mod2 = Python.import_module("os")
        var path = cache_dir + "/hits.tsv"
        if not Bool(os_mod2.path.isfile(path)):
            return False
        self.cache_dir = cache_dir
        self.dense = False
        self.load_fasta(cache_dir + "/ref.fa")
        self.postings = Dict[String, String]()
        self.hit_table = List[String]()
        self.n_postings = 0
        var fh = open_text_read(path)
        while True:
            var line_obj = fh.readline()
            var line = String(line_obj)
            if line.byte_length() == 0:
                break
            while line.byte_length() > 0:
                var last = String(line[byte = line.byte_length() - 1 : line.byte_length()])
                if last == "\n" or last == "\r":
                    line = String(line[byte = 0 : line.byte_length() - 1])
                else:
                    break
            if line.byte_length() == 0:
                continue
            var parts = line.split("\t")
            if len(parts) < 2:
                continue
            var mer = String(parts[0])
            var locs = String(parts[1])
            if mer in self.postings:
                self.postings[mer] = self.postings[mer] + "," + locs
            else:
                self.postings[mer] = locs
            self.n_postings += 1
        fh.close()
        return True

    def _dense_lookup(self, key: UInt64) raises -> List[String]:
        var hits = List[String]()
        if self.n_keys <= 0 or key == UInt64(0xFFFFFFFFFFFFFFFF):
            return hits^
        var lo = 0
        var hi = self.n_keys
        while lo < hi:
            var mid = (lo + hi) // 2
            var mk = _u64_le(self.kmers_addr, mid)
            if mk < key:
                lo = mid + 1
            else:
                hi = mid
        if lo >= self.n_keys:
            return hits^
        if _u64_le(self.kmers_addr, lo) != key:
            return hits^
        var start = Int(_u64_le(self.offsets_addr, lo))
        var end = Int(_u64_le(self.offsets_addr, lo + 1))
        var i = start
        while i < end:
            var cid = Int(_u32_le(self.postings_addr, i * 2))
            var pos = Int(_u32_le(self.postings_addr, i * 2 + 1))
            var cname = String("?")
            if cid >= 0 and cid < len(self.contig_names):
                cname = self.contig_names[cid]
            elif cid >= 0 and cid < len(self.contigs):
                cname = self.contigs[cid].name
            hits.append(cname + ":" + String(pos))
            i += 1
        return hits^

    def lookup_kmer(self, mer: String) raises -> List[String]:
        if self.dense:
            var key = _encode_kmer_u64(mer, self.k)
            return self._dense_lookup(key)

        var hits = List[String]()
        if mer not in self.postings:
            return hits^
        var locs = self.postings[mer]
        var parts = locs.split(",")
        for p in parts:
            if String(p).byte_length() > 0:
                hits.append(String(p))
        return hits^

    def contig_count(self) -> Int:
        return len(self.contigs)

    def total_bases(self) -> Int:
        var n = 0
        for c in self.contigs:
            n += c.seq.byte_length()
        return n
