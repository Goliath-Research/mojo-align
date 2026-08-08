# GBZ index open for Mojo Giraffe — native quartet map (not Python extend_exact).

from std.collections import Dict, List
from std.python import Python

from giraffe_device import extract_kmers_batch, require_device_or_raise, select_device
from giraffe_dist import cluster_seed_hits
from giraffe_extend import AlignmentHit, gapless_extend_seeds
from giraffe_gaf_emit import write_gaf
from giraffe_gpu_kernels import kernel_target_label, probe_device_context
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
    """Mojo-orchestrated seed→cluster→extend→GAF using quartet indexes.

    Production WGBS is paired-end and Buffy-scale (~100s GB FASTQ). Never call
    ``_parse_fastq`` on those inputs — it materializes the whole file as Mojo
    ``String`` rows and OOM-kills the container (exit 137). PE and SE both go
    through streaming ``engine.quartet_map.map_fastq_to_gaf``.
    """
    var dev = require_device_or_raise(device)
    var backend = probe_device_context(dev)
    print(
        "Mojo Giraffe GBZ native device=",
        dev,
        " target=",
        kernel_target_label(dev),
        " backend=",
        backend,
        " gbz=",
        gbz,
    )
    # Warm DeviceContext + pack/hash kernels so Align is visibly on GPU before
    # the streaming Python quartet map (minimizer locate + extend).
    if dev != "cpu":
        var warm = List[String]()
        warm.append("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA")
        warm.append("TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA")
        _ = extract_kmers_batch(dev, warm, k)

    # Ensure dense segment pack exists (may build once); do not preload reads.
    _ = ensure_pack_dir(gbz)

    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    os_mod.environ["METHYLGRAPHER_GIRAFFE_DEVICE"] = dev
    os_mod.environ["METHYLGRAPHER_ALIGN_DEVICE"] = dev
    os_mod.environ["METHYLGRAPHER_LAST_GPU_BACKEND"] = backend
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
    if fq2.byte_length() == 0:
        print("GBZ SE quartet_map hits=", n, " → ", out_gaf)
    else:
        print("GBZ PE quartet_map hits=", n, " → ", out_gaf)
    return Int(py=n)


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
