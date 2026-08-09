# End-to-end Mojo Giraffe mapper (GFA or GBZ index → GAF).

from std.collections import List
from std.python import Python

from giraffe_device import extract_kmers_batch, require_device_or_raise
from giraffe_extend import extend_exact
from giraffe_hit import AlignmentHit
from giraffe_gaf_emit import write_gaf
from giraffe_gbz import map_gbz_via_helper
from giraffe_gpu_kernels import kernel_target_label
from giraffe_index import GraphIndex
from utility import get_kv_value


struct FastqRead(Copyable, Movable):
    var name: String
    var seq: String

    def __init__(out self, name: String, seq: String):
        self.name = name
        self.seq = seq


def _strip_nl(mut s: String):
    while s.byte_length() > 0:
        var last = String(s[byte = s.byte_length() - 1 : s.byte_length()])
        if last == "\n" or last == "\r":
            s = String(s[byte = 0 : s.byte_length() - 1])
        else:
            break


def _parse_fastq(path: String) raises -> List[FastqRead]:
    var builtins = Python.import_module("builtins")
    var fh = builtins.open(path, "r")
    var rows = List[FastqRead]()
    while True:
        var n = String(fh.readline())
        if n.byte_length() == 0:
            break
        var s = String(fh.readline())
        _ = String(fh.readline())
        _ = String(fh.readline())
        _strip_nl(n)
        _strip_nl(s)
        if n.startswith("@"):
            n = String(n[byte = 1 : n.byte_length()])
        var bare = n
        var parts = n.split("_")
        if len(parts) > 0:
            bare = String(parts[0])
        rows.append(FastqRead(bare, s))
    fh.close()
    return rows^


def _first_hit(
    index: GraphIndex, name: String, seq: String
) raises -> AlignmentHit:
    var hits = extend_exact(index, name, seq)
    if len(hits) == 0:
        return AlignmentHit(name, "*", seq.byte_length(), 0, "cs:Z:*")
    return hits[0].copy()


def map_fastq_to_gaf(
    gfa_path: String,
    fq_path: String,
    out_gaf: String,
    device: String = "auto",
    k: Int = 5,
    fq2_path: String = "",
) raises -> Int:
    var index = GraphIndex(k, k)
    index.load_gfa(gfa_path)
    index.build_minimizer_index()
    var dev = require_device_or_raise(device)
    print(
        "Mojo Giraffe device=",
        dev,
        " target=",
        kernel_target_label(dev),
        " segments=",
        index.segment_count(),
    )

    var reads = _parse_fastq(fq_path)
    var seqs = List[String]()
    for r in reads:
        seqs.append(r.seq)
    var seeded = extract_kmers_batch(dev, seqs, k)
    print("seeded reads=", len(seeded))

    var all_hits = List[AlignmentHit]()
    if fq2_path.byte_length() == 0:
        for r in reads:
            var hits = extend_exact(index, r.name, r.seq)
            for h in hits:
                all_hits.append(h.copy())
    else:
        var mates = _parse_fastq(fq2_path)
        var n = len(reads)
        if len(mates) < n:
            n = len(mates)
        var i = 0
        while i < n:
            var r1 = reads[i].copy()
            var r2 = mates[i].copy()
            var h1 = _first_hit(index, r1.name, r1.seq)
            var h2 = _first_hit(index, r2.name, r2.seq)
            h1.extra_tags = "ri:i:1\tos:Z:" + r1.seq + "\trc:Z:CT"
            h2.extra_tags = "ri:i:2\tos:Z:" + r2.seq + "\trc:Z:GA"
            all_hits.append(h1.copy())
            all_hits.append(h2.copy())
            i += 1

    write_gaf(out_gaf, all_hits)
    print("wrote ", len(all_hits), " GAF alignments → ", out_gaf)
    return len(all_hits)


def map_gbz_fastq_to_gaf(
    gbz_path: String,
    fq_path: String,
    out_gaf: String,
    device: String = "auto",
    k: Int = 5,
    fq2_path: String = "",
    dist_path: String = "",
    min_path: String = "",
    zip_path: String = "",
) raises -> Int:
    """GBZ-native map: quartet min/zip/dist + dense pack (Mojo-orchestrated)."""
    return map_gbz_via_helper(
        gbz_path,
        fq_path,
        out_gaf,
        fq2_path,
        dist_path,
        min_path,
        zip_path,
        k,
        device,
    )


def run_mojo_giraffe_cli(args: List[String]) raises -> Int:
    """CLI: MojoGiraffe (-gfa|-gbz) -fq1 … -out_gaf … [-fq2] [-dist] [-min] [-zipcodes]."""
    var gfa = get_kv_value(args, "gfa", "")
    var gbz = get_kv_value(args, "gbz", "")
    var fq1 = get_kv_value(args, "fq1", "")
    var fq2 = get_kv_value(args, "fq2", "")
    var out_gaf = get_kv_value(args, "out_gaf", "alignment.gaf")
    var device = get_kv_value(args, "device", "auto")
    var k = Int(get_kv_value(args, "k", "5"))
    var dist = get_kv_value(args, "dist", "")
    var min_path = get_kv_value(args, "min", "")
    var zipcodes = get_kv_value(args, "zipcodes", "")
    if fq1.byte_length() == 0:
        raise Error("MojoGiraffe requires -fq1")
    if gbz.byte_length() > 0:
        _ = map_gbz_fastq_to_gaf(
            gbz, fq1, out_gaf, device, k, fq2, dist, min_path, zipcodes
        )
        return 0
    if gfa.byte_length() == 0:
        raise Error("MojoGiraffe requires -gfa or -gbz")
    _ = map_fastq_to_gaf(gfa, fq1, out_gaf, device, k, fq2)
    return 0
