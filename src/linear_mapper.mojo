# End-to-end Mojo linear WGBS mapper (C2T FASTA + converted FASTQ → SAM).

from std.collections import List
from std.python import Python, PythonObject
from std.sys import argv as sys_argv, exit

from linear_extend import extend_read, hit_to_sam_line, pair_hits
from linear_gpu_kernels import seed_kmers_portable
from linear_index import LinearIndex
from utility import get_kv_value, open_text_write


struct FastqRead(Copyable, Movable):
    var name: String
    var seq: String
    var qual: String

    def __init__(out self, name: String, seq: String, qual: String = "*"):
        self.name = name
        self.seq = seq
        self.qual = qual


def _strip_nl(mut s: String):
    while s.byte_length() > 0:
        var last = String(s[byte = s.byte_length() - 1 : s.byte_length()])
        if last == "\n" or last == "\r":
            s = String(s[byte = 0 : s.byte_length() - 1])
        else:
            break


def _parse_fastq(path: String) raises -> List[FastqRead]:
    var builtins = Python.import_module("builtins")
    var gzip = Python.import_module("gzip")
    var low = path.lower()
    var fh: PythonObject
    if low.endswith(".gz") or low.endswith(".gzip"):
        fh = gzip.open(path, "rt")
    else:
        fh = builtins.open(path, "r")
    var rows = List[FastqRead]()
    while True:
        var n = String(fh.readline())
        if n.byte_length() == 0:
            break
        var s = String(fh.readline())
        _ = String(fh.readline())
        var q = String(fh.readline())
        _strip_nl(n)
        _strip_nl(s)
        _strip_nl(q)
        if n.startswith("@"):
            n = String(n[byte = 1 : n.byte_length()])
        var bare = n
        var parts = n.split(" ")
        if len(parts) > 0:
            bare = String(parts[0])
        var slash = bare.split("/")
        if len(slash) > 0:
            bare = String(slash[0])
        if q.byte_length() == 0:
            q = String("*")
        rows.append(FastqRead(bare, s, q))
    fh.close()
    return rows^


def _write_sam_header(fh: PythonObject, index: LinearIndex) raises:
    fh.write("@HD\tVN:1.6\tSO:unsorted\n")
    for c in index.contigs:
        fh.write(
            "@SQ\tSN:"
            + c.name
            + "\tLN:"
            + String(c.seq.byte_length())
            + "\n"
        )
    fh.write("@PG\tID:MojoFq2bamMeth\tPN:MojoFq2bamMeth\tVN:0.1.0-mojo\n")


def map_fastq_to_sam(
    ref_fasta: String,
    fq1: String,
    out_sam: String,
    device: String = "auto",
    k: Int = 15,
    fq2: String = "",
    cache_dir: String = "",
) raises -> Int:
    var index = LinearIndex(k)
    var loaded = False
    if cache_dir.byte_length() > 0:
        loaded = index.load_cache(cache_dir)
    if not loaded:
        index.load_fasta(ref_fasta)
        index.build_kmer_index()
        if cache_dir.byte_length() > 0:
            index.save_cache(cache_dir)

    var reads = _parse_fastq(fq1)
    var seqs = List[String]()
    for r in reads:
        seqs.append(r.seq)
    # Portable GPU seed touch (nvidia:sm_90 / amdgpu:gfx942); extend uses index.
    _ = seed_kmers_portable(device, seqs, k)
    print(
        "MojoLinear index contigs=",
        index.contig_count(),
        " bases=",
        index.total_bases(),
        " hits=",
        len(index.hit_table),
        " reads=",
        len(reads),
    )

    var fh = open_text_write(out_sam)
    _write_sam_header(fh, index)

    var n_mapped = 0
    if fq2.byte_length() == 0:
        for r in reads:
            var h = extend_read(index, r.name, r.seq)
            h.qual = r.qual
            if h.contig != "*":
                n_mapped += 1
            fh.write(hit_to_sam_line(h) + "\n")
    else:
        var mates = _parse_fastq(fq2)
        var n = len(reads)
        if len(mates) < n:
            n = len(mates)
        var i = 0
        while i < n:
            var r1 = reads[i].copy()
            var r2 = mates[i].copy()
            var h1 = extend_read(index, r1.name, r1.seq)
            var h2 = extend_read(index, r2.name, r2.seq)
            h1.qual = r1.qual
            h2.qual = r2.qual
            var paired = pair_hits(h1, h2)
            var a = paired.r1.copy()
            var b = paired.r2.copy()
            if a.contig != "*":
                n_mapped += 1
            if b.contig != "*":
                n_mapped += 1
            fh.write(hit_to_sam_line(a) + "\n")
            fh.write(hit_to_sam_line(b) + "\n")
            i += 1

    fh.close()
    print("wrote SAM → ", out_sam, " mapped_records=", n_mapped)
    return n_mapped


def run_mojo_linear_cli(args: List[String]) raises -> Int:
    """CLI: -ref <fa> -fq1 <fq> -out_sam <path> [-fq2] [-device] [-k] [-cache_dir]."""
    var ref_fa = get_kv_value(args, "ref", "")
    var fq1 = get_kv_value(args, "fq1", "")
    var fq2 = get_kv_value(args, "fq2", "")
    var out_sam = get_kv_value(args, "out_sam", "aligned.sam")
    var device = get_kv_value(args, "device", "auto")
    var k = Int(get_kv_value(args, "k", "15"))
    var cache_dir = get_kv_value(args, "cache_dir", "")
    if ref_fa.byte_length() == 0 or fq1.byte_length() == 0:
        raise Error("MojoLinearMap requires -ref and -fq1")
    _ = map_fastq_to_sam(ref_fa, fq1, out_sam, device, k, fq2, cache_dir)
    return 0


def main() raises:
    var raw = sys_argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    exit(run_mojo_linear_cli(args))
