# Linear SAM emit for MojoGiraffe QC (os:Z + GRCh38 segment offsets).
#
# Streams SAM text; never builds a qname→sequence dictionary. Sequence comes
# from AlignmentHit.extra_tags ``os:Z:`` (set at map time from the FASTQ batch).
# Formatting + buffered writes live in engine.grch38_offsets (already overlay-
# mounted) so sisters pick up the fast path without a runner restart.

from std.collections import List
from std.python import Python, PythonObject

from giraffe_hit import AlignmentHit


def _off_mod() raises -> PythonObject:
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    return Python.import_module("engine.grch38_offsets")


def ensure_qc_offsets(root: String) raises:
    _ = root


def close_qc_offsets() raises:
    pass


def open_sam_write(path: String, offsets_root: String) raises -> PythonObject:
    return _off_mod().open_sam(path, offsets_root)


def append_sam_hits(fh: PythonObject, hits: List[AlignmentHit]) raises -> Int:
    var builtins = Python.import_module("builtins")
    var rows = builtins.list()
    for h in hits:
        var row = builtins.list()
        row.append(h.query_name)
        row.append(h.path)
        row.append(h.mapq)
        row.append(h.extra_tags)
        rows.append(row)
    return Int(py=_off_mod().append_hits(fh, rows))


def sam_emit_summary() raises -> String:
    return String(_off_mod().summary())


def close_sam_write(fh: PythonObject) raises:
    _off_mod().close_sam(fh)
