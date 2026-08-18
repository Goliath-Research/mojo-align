# End-to-end Mojo Giraffe mapper (GFA or GBZ index -> GAF).

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
    """Converted FASTQ record; ``original_seq`` feeds MethylCall ``os:Z``."""

    var name: String
    var seq: String
    var original_seq: String
    var conversion: String

    def __init__(
        out self,
        name: String,
        seq: String,
        original_seq: String,
        conversion: String,
    ):
        self.name = name
        self.seq = seq
        self.original_seq = original_seq
        self.conversion = conversion


def _strip_nl(mut s: String):
    while s.byte_length() > 0:
        var last = String(s[byte = s.byte_length() - 1 : s.byte_length()])
        if last == "\n" or last == "\r":
            s = String(s[byte = 0 : s.byte_length() - 1])
        else:
            break


def _rc_from_conversion(conversion: String, fallback: String) -> String:
    if conversion == "C2T":
        return "CT"
    if conversion == "G2A":
        return "GA"
    return fallback


def _pe_extra_tags(
    ri: Int, original_seq: String, conversion: String, fallback_rc: String
) -> String:
    var os = original_seq
    var rc = _rc_from_conversion(conversion, fallback_rc)
    return "ri:i:" + String(ri) + "\tos:Z:" + os + "\trc:Z:" + rc


def _parse_mg_fastq_fields(n: String, s: String) -> FastqRead:
    var bare = n
    var original = s
    var conversion = String("")
    var parts = n.split("_")
    if len(parts) >= 4:
        var conv = String(parts[1])
        if conv == "C2T" or conv == "G2A":
            bare = String(parts[0])
            conversion = conv
            original = String(parts[3])
            var pi = 4
            while pi < len(parts):
                original = original + "_" + String(parts[pi])
                pi += 1
        else:
            bare = String(parts[0])
    elif len(parts) > 0:
        bare = String(parts[0])
    var sp = bare.split(" ")
    if len(sp) > 0:
        bare = String(sp[0])
    return FastqRead(bare, s, original, conversion)


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
        rows.append(_parse_mg_fastq_fields(n, s))
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
            h1.extra_tags = _pe_extra_tags(1, r1.original_seq, r1.conversion, "CT")
            h2.extra_tags = _pe_extra_tags(2, r2.original_seq, r2.conversion, "GA")
            all_hits.append(h1.copy())
            all_hits.append(h2.copy())
            i += 1

    write_gaf(out_gaf, all_hits)
    print("wrote ", len(all_hits), " GAF alignments -> ", out_gaf)
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
    """CLI: MojoGiraffe (-gfa|-gbz) -fq1 … (-out_gaf|-out_sam) …

    ``-out_sam`` enables QC linear SAM emit (sets MOJO_ALIGN_EMIT=sam)
    and requires ``-segment_offsets <dir>`` (grch38-dense-v1 from
    scripts/build_grch38_offsets.py).
    """
    var os_mod = Python.import_module("os")
    var gfa = get_kv_value(args, "gfa", "")
    var gbz = get_kv_value(args, "gbz", "")
    var fq1 = get_kv_value(args, "fq1", "")
    var fq2 = get_kv_value(args, "fq2", "")
    var out_gaf = get_kv_value(args, "out_gaf", "")
    var out_sam = get_kv_value(args, "out_sam", "")
    var segment_offsets = get_kv_value(args, "segment_offsets", "")
    var device = get_kv_value(args, "device", "auto")
    var k = Int(get_kv_value(args, "k", "5"))
    var dist = get_kv_value(args, "dist", "")
    var min_path = get_kv_value(args, "min", "")
    var zipcodes = get_kv_value(args, "zipcodes", "")
    if fq1.byte_length() == 0:
        raise Error("MojoGiraffe requires -fq1")
    var out_path = out_gaf
    if out_sam.byte_length() > 0:
        if segment_offsets.byte_length() == 0:
            raise Error("MojoGiraffe -out_sam requires -segment_offsets")
        os_mod.environ["MOJO_ALIGN_EMIT"] = "sam"
        os_mod.environ["MOJO_ALIGN_SEGMENT_OFFSETS"] = segment_offsets
        out_path = out_sam
        print(
            "MojoGiraffe QC SAM emit offsets=",
            segment_offsets,
            " out=",
            out_path,
            flush=True,
        )
    elif out_path.byte_length() == 0:
        out_path = "alignment.gaf"
    if gbz.byte_length() > 0:
        _ = map_gbz_fastq_to_gaf(
            gbz, fq1, out_path, device, k, fq2, dist, min_path, zipcodes
        )
        return 0
    if gfa.byte_length() == 0:
        raise Error("MojoGiraffe requires -gfa or -gbz")
    _ = map_fastq_to_gaf(gfa, fq1, out_path, device, k, fq2)
    return 0
