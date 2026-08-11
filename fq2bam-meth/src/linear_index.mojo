# Linear C2T reference index for MojoFq2bamMeth (short-read WGBS).
#
# Device-friendly hash postings: Dict[kmer → "contig:off,..."] — no full
# hit_table linear scan on lookup.

from std.collections import Dict, List
from std.python import Python

from utility import open_text_read, open_text_write


struct LinearContig(Copyable, Movable):
    var name: String
    var seq: String

    def __init__(out self, name: String, seq: String):
        self.name = name
        self.seq = seq


struct LinearIndex(Copyable, Movable):
    """FASTA + hashed k-mer postings (Dict)."""

    var contigs: List[LinearContig]
    var postings: Dict[String, String]
    var hit_table: List[String]  # legacy cache rows; optional
    var k: Int
    var cache_dir: String
    var n_postings: Int

    def __init__(out self, k: Int = 15):
        self.contigs = List[LinearContig]()
        self.postings = Dict[String, String]()
        self.hit_table = List[String]()
        self.k = k
        self.cache_dir = String("")
        self.n_postings = 0

    def load_fasta(mut self, fasta_path: String) raises:
        self.contigs = List[LinearContig]()
        var fh = open_text_read(fasta_path)
        var name = String("")
        var seq = String("")
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
        fh.close()

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

    def save_cache(self, cache_dir: String) raises:
        var os_mod = Python.import_module("os")
        _ = os_mod.makedirs(cache_dir, exist_ok=True)
        var meta = open_text_write(cache_dir + "/meta.txt")
        meta.write("k=" + String(self.k) + "\n")
        meta.write("contigs=" + String(len(self.contigs)) + "\n")
        meta.write("format=hash-v1\n")
        meta.close()
        var hits = open_text_write(cache_dir + "/hits.tsv")
        for mer in self.postings:
            hits.write(mer + "\t" + self.postings[mer] + "\n")
        hits.close()
        var fa = open_text_write(cache_dir + "/ref.fa")
        for c in self.contigs:
            fa.write(">" + c.name + "\n")
            fa.write(c.seq + "\n")
        fa.close()

    def load_cache(mut self, cache_dir: String) raises -> Bool:
        var os_mod = Python.import_module("os")
        var path = cache_dir + "/hits.tsv"
        if not Bool(os_mod.path.isfile(path)):
            return False
        self.cache_dir = cache_dir
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
            # Legacy rows were "kmer\tcontig:offset" (single); hash-v1 may be
            # "kmer\tcontig:o1,contig:o2".
            if mer in self.postings:
                self.postings[mer] = self.postings[mer] + "," + locs
            else:
                self.postings[mer] = locs
            self.n_postings += 1
        fh.close()
        return True

    def lookup_kmer(self, mer: String) raises -> List[String]:
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
