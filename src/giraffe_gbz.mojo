# GBZ index open for Mojo Giraffe — native stream map (not Python quartet_map).

from std.collections import Dict
from std.python import Python

from giraffe_device import require_device_or_raise
from giraffe_gpu_kernels import kernel_target_label
from giraffe_stream_map import map_fastq_stream_to_gaf


def ensure_pack_dir(gbz: String) raises -> String:
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    # Pack ensure stays in Python (one-time build / resolve); map loop is Mojo.
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
    """Mojo stream map: GPU seed -> locate -> cluster -> gapless -> GAF.

    Production WGBS is paired-end and Buffy-scale (~100s GB FASTQ). Streaming
    batches never materialize the whole FASTQ as Mojo ``String`` rows.
    """
    var dev = require_device_or_raise(device)
    # Do NOT probe/warm DeviceContext here: each throwaway context can retain
    # HBM until process exit and races the resident-index preflight below.
    # map_fastq_stream_to_gaf owns the single production DeviceContext session.
    print(
        "Mojo Giraffe GBZ native device=",
        dev,
        " target=",
        kernel_target_label(dev),
        " gbz=",
        gbz,
    )

    var pack_dir = ensure_pack_dir(gbz)
    var n = map_fastq_stream_to_gaf(
        pack_dir,
        fq1,
        out_gaf,
        fq2,
        dist,
        min_path,
        zipcodes,
        k,
        dev,
    )
    if fq2.byte_length() == 0:
        print("GBZ SE mojo_stream_map hits=", n, " -> ", out_gaf)
    else:
        print("GBZ PE mojo_stream_map hits=", n, " -> ", out_gaf)
    return n


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
    """Back-compat name -> native Mojo stream map."""
    return map_gbz_native(
        gbz, fq1, out_gaf, fq2, dist, min_path, zipcodes, k, device
    )


def load_segments_from_gbz(gbz_path: String) raises -> Dict[String, String]:
    """Decode GBZ / pack -> segment id→sequence for Mojo seed warm-up."""
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
