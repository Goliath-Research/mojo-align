# End-to-end Mojo linear WGBS mapper (C2T FASTA + converted FASTQ → SAM).
#
# Streaming batches — never materializes full production FASTQ as Mojo rows.
# GPU seeds feed extend (not discarded warmup).

from std.collections import List
from std.python import Python, PythonObject
from std.sys import argv as sys_argv, exit

from linear_extend import extend_read, extend_read_with_seeds, hit_to_sam_line, pair_hits
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


def _bs_convert(seq: String, mode: String) raises -> String:
    """Fused bisulfite convert (C2T / G2A) for streaming path."""
    if mode.byte_length() == 0 or mode == "none":
        return seq
    var out = String("")
    var i = 0
    var n = seq.byte_length()
    while i < n:
        var ch = String(seq[byte = i : i + 1])
        var u = ch.upper()
        if mode == "C2T" and u == "C":
            out += "T"
        elif mode == "G2A" and u == "G":
            out += "A"
        else:
            out += ch
        i += 1
    return out^


def _open_fastq(path: String) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var gzip = Python.import_module("gzip")
    var low = path.lower()
    if low.endswith(".gz") or low.endswith(".gzip"):
        return gzip.open(path, "rt")
    return builtins.open(path, "r")


def _read_one(fh: PythonObject) raises -> FastqRead:
    var n = String(fh.readline())
    if n.byte_length() == 0:
        return FastqRead("", "", "")
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
    return FastqRead(bare, s, q)


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


def _batch_size() raises -> Int:
    # Larger default helps beat Clara wall on GH200; override via env.
    var os_mod = Python.import_module("os")
    var raw = String(os_mod.environ.get("METHYLGRAPHER_LINEAR_READ_BATCH", "16384"))
    var n = Int(raw)
    if n < 1:
        return 16384
    return n


def map_fastq_to_sam(
    ref_fasta: String,
    fq1: String,
    out_sam: String,
    device: String = "auto",
    k: Int = 15,
    fq2: String = "",
    cache_dir: String = "",
    bs_r1: String = "",
    bs_r2: String = "",
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

    print(
        "MojoLinear index contigs=",
        index.contig_count(),
        " bases=",
        index.total_bases(),
        " postings=",
        index.n_postings,
        " device=",
        device,
    )

    var fh = open_text_write(out_sam)
    _write_sam_header(fh, index)

    var fh1 = _open_fastq(fq1)
    var paired = fq2.byte_length() > 0
    var fh2 = fh1
    if paired:
        fh2 = _open_fastq(fq2)

    var n_mapped = 0
    var n_reads = 0
    var bs = _batch_size()

    while True:
        var batch1 = List[FastqRead]()
        var batch2 = List[FastqRead]()
        var i = 0
        while i < bs:
            var r1 = _read_one(fh1)
            if r1.name.byte_length() == 0:
                break
            if bs_r1.byte_length() > 0:
                r1.seq = _bs_convert(r1.seq, bs_r1)
            batch1.append(r1^)
            if paired:
                var r2 = _read_one(fh2)
                if r2.name.byte_length() == 0:
                    raise Error("paired FASTQ length mismatch (R2 ended early)")
                if bs_r2.byte_length() > 0:
                    r2.seq = _bs_convert(r2.seq, bs_r2)
                batch2.append(r2^)
            i += 1
        if len(batch1) == 0:
            break

        var seqs = List[String]()
        for r in batch1:
            seqs.append(r.seq)
        if paired:
            for r in batch2:
                seqs.append(r.seq)
        # GPU / portable seeds — used for extend (not discarded).
        var seed_batch = seed_kmers_portable(device, seqs, k)

        var j = 0
        while j < len(batch1):
            var a = batch1[j].copy()
            var seeds_a = seed_batch[j].copy()
            var h1 = extend_read_with_seeds(index, a.name, a.seq, seeds_a)
            h1.qual = a.qual
            if not paired:
                if h1.contig != "*":
                    n_mapped += 1
                fh.write(hit_to_sam_line(h1) + "\n")
                n_reads += 1
            else:
                var b = batch2[j].copy()
                var seed_idx = len(batch1) + j
                var seeds_b = seed_batch[seed_idx].copy()
                var h2 = extend_read_with_seeds(index, b.name, b.seq, seeds_b)
                h2.qual = b.qual
                var paired_hits = pair_hits(h1, h2)
                var p1 = paired_hits.r1.copy()
                var p2 = paired_hits.r2.copy()
                if p1.contig != "*":
                    n_mapped += 1
                if p2.contig != "*":
                    n_mapped += 1
                fh.write(hit_to_sam_line(p1) + "\n")
                fh.write(hit_to_sam_line(p2) + "\n")
                n_reads += 2
            j += 1

    fh1.close()
    if paired:
        fh2.close()
    fh.close()
    print("wrote SAM → ", out_sam, " mapped_records=", n_mapped, " reads=", n_reads)
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
    var bs_r1 = get_kv_value(args, "bs_r1", "")
    var bs_r2 = get_kv_value(args, "bs_r2", "")
    if ref_fa.byte_length() == 0 or fq1.byte_length() == 0:
        raise Error("MojoLinearMap requires -ref and -fq1")
    _ = map_fastq_to_sam(
        ref_fa, fq1, out_sam, device, k, fq2, cache_dir, bs_r1, bs_r2
    )
    return 0


def main() raises:
    var raw = sys_argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    exit(run_mojo_linear_cli(args))
