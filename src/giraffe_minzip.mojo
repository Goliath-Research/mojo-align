# Minimizer + zipcode locate for GBZ-native Mojo Giraffe.
#
# Production: mmap vg ``.shortread.withzip.min`` via engine.minimizer_index
# (Q1Q1 / v11). Fixture rebuild-from-segments path removed.

from std.collections import Dict, List
from std.python import Python, PythonObject


def probe_min_index(min_path: String) raises -> String:
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    var mod = Python.import_module("engine.minimizer_index")
    var info = mod.probe_minimizer(min_path)
    return String(info)


def locate_read_hits(min_path: String, seq: String, hit_cap: Int = 24) raises -> List[String]:
    """Return ``node_id:orient:offset`` strings from minimizer locate."""
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    var mod = Python.import_module("engine.minimizer_index")
    var idx = mod.MinimizerIndex(min_path)
    var hits = idx.locate_read(seq, hit_cap=hit_cap)
    var out = List[String]()
    var n = Int(py=hits.__len__())
    var i = 0
    while i < n:
        var h = hits[i]
        var orient = String("0")
        if Bool(h.is_rev):
            orient = "1"
        out.append(
            String(h.node_id) + ":" + orient + ":" + String(h.offset)
        )
        i = i + 1
    idx.close()
    return out^


def locate_batch_hits(
    min_path: String, seqs: List[String], hit_cap: Int = 24
) raises -> List[List[String]]:
    """In-process batch locate into Mojo buffers (one mmap open)."""
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    var mod = Python.import_module("engine.minimizer_index")
    var idx = mod.MinimizerIndex(min_path)
    var out = List[List[String]]()
    for seq in seqs:
        var hits = idx.locate_read(seq, hit_cap=hit_cap)
        var row = List[String]()
        var n = Int(py=hits.__len__())
        var i = 0
        while i < n:
            var h = hits[i]
            var orient = String("0")
            if Bool(h.is_rev):
                orient = "1"
            row.append(
                String(h.node_id) + ":" + orient + ":" + String(h.offset)
            )
            i = i + 1
        out.append(row^)
    idx.close()
    return out^


def build_min_index_from_segments(
    segments: Dict[String, String], k: Int
) raises -> List[String]:
    """Removed production path — kept stub raising to catch regressions."""
    raise Error(
        "build_min_index_from_segments removed; use .shortread.withzip.min via locate_read_hits"
    )
