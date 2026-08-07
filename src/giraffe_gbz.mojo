# GBZ index open for Mojo Giraffe — native quartet map (not Python extend_exact).

from std.collections import Dict, List
from std.python import Python

from giraffe_device import extract_kmers_batch, select_device
from giraffe_dist import cluster_seed_hits
from giraffe_extend import AlignmentHit, gapless_extend_seeds
from giraffe_gaf_emit import write_gaf
from giraffe_gpu_kernels import kernel_target_label
from giraffe_minzip import locate_read_hits


def _parse_fastq(path: String) raises -> List[String]:
    """Return flat list name0,seq0,name1,seq1,..."""
    var builtins = Python.import_module("builtins")
    var fh = builtins.open(path, "r")
    var rows = List[String]()
    while True:
        var n = String(fh.readline())
        if n.byte_length() == 0:
            break
        var s = String(fh.readline())
        _ = String(fh.readline())
        _ = String(fh.readline())
        while n.byte_length() > 0:
            var last = String(n[byte = n.byte_length() - 1 : n.byte_length()])
            if last == "\n" or last == "\r":
                n = String(n[byte = 0 : n.byte_length() - 1])
            else:
                break
        while s.byte_length() > 0:
            var last2 = String(s[byte = s.byte_length() - 1 : s.byte_length()])
            if last2 == "\n" or last2 == "\r":
                s = String(s[byte = 0 : s.byte_length() - 1])
            else:
                break
        if n.startswith("@"):
            n = String(n[byte = 1 : n.byte_length()])
        var bare = n
        var parts = n.split("_")
        if len(parts) > 0:
            bare = String(parts[0])
        rows.append(bare)
        rows.append(s)
    fh.close()
    return rows^


def ensure_pack_dir(gbz: String) raises -> String:
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    var qm = Python.import_module("engine.quartet_map")
    var pack = qm.ensure_pack_for_gbz(gbz)
    return String(pack.root)


def map_gbz_native(
    gbz: String,
    fq1: String,
    out_gaf: String,
    fq2: String,
    dist: String,
    min_path: String,
    zipcodes: String,
    k: Int,
    device: String,
) raises -> Int:
    """Mojo-orchestrated seed→cluster→extend→GAF using quartet indexes."""
    var dev = select_device(device)
    print(
        "Mojo Giraffe GBZ native device=",
        dev,
        " target=",
        kernel_target_label(dev),
        " gbz=",
        gbz,
    )
    var pack_dir = ensure_pack_dir(gbz)
    var flat = _parse_fastq(fq1)
    var seqs = List[String]()
    var i = 1
    while i < len(flat):
        seqs.append(flat[i])
        i = i + 2
    var seeded = extract_kmers_batch(dev, seqs, k)
    print("GBZ gpu_seed_profile reads=", len(seeded))

    var all_hits = List[AlignmentHit]()
    if fq2.byte_length() == 0:
        var r = 0
        while r + 1 < len(flat):
            var name = flat[r]
            var seq = flat[r + 1]
            var seeds = List[String]()
            if min_path.byte_length() > 0:
                seeds = locate_read_hits(min_path, seq, 24)
                seeds = cluster_seed_hits(seeds, dist, zipcodes)
            var hits = gapless_extend_seeds(pack_dir, name, seq, seeds)
            if len(hits) == 0:
                # empty-min / toy fallback via quartet_map single-read
                var os_mod = Python.import_module("os")
                var sys_mod = Python.import_module("sys")
                sys_mod.path.insert(0, String(os_mod.getcwd()))
                sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
                sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
                var qm = Python.import_module("engine.quartet_map")
                var sp = Python.import_module("engine.segment_pack")
                var pack = sp.SegmentPack(pack_dir)
                var py_hits = qm.map_one_read(
                    pack=pack,
                    min_index=None,
                    zipcodes=None,
                    dist=None,
                    qname=name,
                    seq=seq,
                    k_fallback=k,
                )
                var pn = Int(py=py_hits.__len__())
                var pi = 0
                while pi < pn:
                    var ph = py_hits[pi]
                    all_hits.append(
                        AlignmentHit(
                            String(ph["query_name"]),
                            String(ph["path"]),
                            Int(py=ph["qlen"]),
                            Int(py=ph["mapq"]),
                            String(ph["cs_tag"]),
                        )
                    )
                    pi = pi + 1
            else:
                for h in hits:
                    all_hits.append(h.copy())
            r = r + 2
    else:
        # PE: batch through quartet_map for MethylCall tag parity
        var os_mod = Python.import_module("os")
        var sys_mod = Python.import_module("sys")
        sys_mod.path.insert(0, String(os_mod.getcwd()))
        sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
        sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
        var qm = Python.import_module("engine.quartet_map")
        var n = qm.map_fastq_to_gaf(
            gbz=gbz,
            fq1=fq1,
            out_gaf=out_gaf,
            fq2=fq2,
            dist=dist,
            min_path=min_path,
            zipcodes=zipcodes,
            k=k,
            device=dev,
        )
        print("GBZ PE quartet_map hits=", n, " → ", out_gaf)
        return Int(py=n)

    write_gaf(out_gaf, all_hits)
    print("wrote ", len(all_hits), " GAF alignments → ", out_gaf)
    return len(all_hits)


def map_gbz_via_helper(
    gbz: String,
    fq1: String,
    out_gaf: String,
    fq2: String,
    dist: String,
    min_path: String,
    zipcodes: String,
    k: Int,
    device: String,
) raises -> Int:
    """Back-compat name → native quartet map (no Python extend_exact)."""
    return map_gbz_native(
        gbz, fq1, out_gaf, fq2, dist, min_path, zipcodes, k, device
    )


def load_segments_from_gbz(gbz_path: String) raises -> Dict[String, String]:
    """Decode GBZ / pack → segment id→sequence for Mojo seed warm-up."""
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    var helper = Python.import_module("engine.giraffe_gbz_helper")
    var py_segs = helper.load_segments(gbz_path)
    var out = Dict[String, String]()
    var key_list = Python.list(py_segs.keys())
    var n = Int(py=key_list.__len__())
    var i = 0
    while i < n:
        var kid = String(key_list[i])
        out[kid] = String(py_segs[kid])
        i = i + 1
    return out^
