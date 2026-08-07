# GBZ index open for Mojo Giraffe (via staged Python helper).

from std.collections import Dict, List
from std.python import Python


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
    """Full GBZ→GAF map through staged helper (Mojo CLI entry)."""
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    var helper = Python.import_module("engine.giraffe_gbz_helper")
    var n = helper.map_gbz_fastq_to_gaf(
        gbz=gbz,
        fq1=fq1,
        out_gaf=out_gaf,
        fq2=fq2,
        dist=dist,
        min_path=min_path,
        zipcodes=zipcodes,
        k=k,
        device=device,
    )
    return Int(py=n)


def load_segments_from_gbz(gbz_path: String) raises -> Dict[String, String]:
    """Decode GBZ → segment id→sequence for Mojo seed warm-up."""
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
