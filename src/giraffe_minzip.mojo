# Minimizer + zipcode locate for GBZ-native Mojo Giraffe.
#
# Production science locate:
#   1) Mojo Giraffe (k,w) minimizers — DeviceContext pack/hash on NVIDIA/AMD
#      (``giraffe_minimizer``), never CuPy / ``giraffe_gpu_minimizer``
#   2) vg ``.shortread.withzip.min`` HT probe via thin Python mmap helper
#      (``MinimizerIndex.locate_key_batches``) — one call per FASTQ batch

from std.collections import Dict, List
from std.python import Python, PythonObject

from giraffe_minimizer import minimizers_batch


def _engine_paths() raises:
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    sys_mod.path.insert(0, String(os_mod.getcwd()) + "/scripts")
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo/scripts")


def probe_min_index(min_path: String) raises -> String:
    _engine_paths()
    var mod = Python.import_module("engine.minimizer_index")
    var info = mod.probe_minimizer(min_path)
    return String(info)


def _open_min_index(min_path: String) raises -> PythonObject:
    _engine_paths()
    var mod = Python.import_module("engine.minimizer_index")
    return mod.MinimizerIndex(min_path)


def locate_read_hits(min_path: String, seq: String, hit_cap: Int = 24) raises -> List[String]:
    """Return ``node_id:orient:offset`` strings from Mojo minimizers + HT locate."""
    var seqs = List[String]()
    seqs.append(seq)
    var batch = locate_batch_hits_native("cpu", min_path, seqs, hit_cap)
    if len(batch) == 0:
        return List[String]()
    return batch[0].copy()


def locate_batch_hits(
    min_path: String, seqs: List[String], hit_cap: Int = 24
) raises -> List[List[String]]:
    """Host Mojo minimizers + HT locate (no DeviceContext)."""
    return locate_batch_hits_native("cpu", min_path, seqs, hit_cap)


def locate_batch_hits_native(
    device: String,
    min_path: String,
    seqs: List[String],
    hit_cap: Int = 24,
) raises -> List[List[String]]:
    """Production locate: Mojo DeviceContext/host minimizers → HT key batches."""
    if min_path.byte_length() == 0 or len(seqs) == 0:
        var empty = List[List[String]]()
        for _s in seqs:
            empty.append(List[String]())
        return empty^

    var idx = _open_min_index(min_path)
    var k = Int(py=idx.k)
    var w = Int(py=idx.w)
    var result = minimizers_batch(device, seqs, k, w)
    var backend = result.backend.copy()

    # Pack keys → one Python locate_key_batches call.
    var py_keys = Python.list()
    for occs in result.occs:
        var row = Python.list()
        for occ in occs:
            row.append(Int(occ.key))
        py_keys.append(row)

    var located = idx.locate_key_batches(py_keys, hit_cap=hit_cap)
    idx.close()

    var out = List[List[String]]()
    var n = Int(py=located.__len__())
    var i = 0
    while i < n:
        var hits = located[i]
        var row_out = List[String]()
        var m = Int(py=hits.__len__())
        var j = 0
        while j < m:
            row_out.append(String(hits[j]))
            j += 1
        out.append(row_out^)
        i += 1

    var os_mod = Python.import_module("os")
    os_mod.environ["METHYLGRAPHER_LAST_SEED_BACKEND"] = backend + "+mojo_stream"
    print("mojo_stream seed_backend=", backend, "+mojo_stream")
    return out^


def build_min_index_from_segments(
    segments: Dict[String, String], k: Int
) raises -> List[String]:
    """Removed production path — kept stub raising to catch regressions."""
    raise Error(
        "build_min_index_from_segments removed; use .shortread.withzip.min via locate_read_hits"
    )
