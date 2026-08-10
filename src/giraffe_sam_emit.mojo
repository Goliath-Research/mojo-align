# Linear SAM emit for MojoGiraffe QC (os:Z + GRCh38 segment offsets).
#
# Streams SAM text; never builds a qname→sequence dictionary. Sequence comes
# from AlignmentHit.extra_tags ``os:Z:`` (set at map time from the FASTQ batch).
# Dense offset table is mmap'd once via engine.grch38_offsets (no per-read dict).

from std.collections import List
from std.python import Python, PythonObject

from giraffe_hit import AlignmentHit


struct LinAnchor(Copyable, Movable):
    var chrom: String
    var pos1: Int
    var ok: Bool

    def __init__(out self, chrom: String, pos1: Int, ok: Bool):
        self.chrom = chrom
        self.pos1 = pos1
        self.ok = ok


def _offsets_mod() raises -> PythonObject:
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    return Python.import_module("engine.grch38_offsets")


def ensure_qc_offsets(root: String) raises:
    _offsets_mod().open_table(root)
    print("grch38_offsets open root=", root, flush=True)


def close_qc_offsets() raises:
    _offsets_mod().close_table()


def _tag_value(extras: String, prefix: String) raises -> String:
    if extras.byte_length() == 0:
        return String("")
    var idx = -1
    if extras.startswith(prefix):
        idx = 0
    else:
        var needle = "\t" + prefix
        var at = extras.find(needle)
        if at >= 0:
            idx = at + 1
    if idx < 0:
        return String("")
    var start = idx + prefix.byte_length()
    var rest = String(extras[byte = start : extras.byte_length()])
    var tab = rest.find("\t")
    if tab < 0:
        return rest
    return String(rest[byte = 0:tab])


def _parse_int_sid(sid: String) raises -> Int:
    try:
        return Int(sid)
    except:
        return -1


def project_hit_linear(hit: AlignmentHit) raises -> LinAnchor:
    """First path segment present in the GRCh38 offset table."""
    var path = hit.path
    if path == "*" or path.byte_length() == 0:
        return LinAnchor("", 0, False)
    var off = _offsets_mod()
    var i = 0
    var n = path.byte_length()
    while i < n:
        var ch = String(path[byte = i : i + 1])
        if ch != ">" and ch != "<":
            i += 1
            continue
        var j = i + 1
        while j < n:
            var c2 = String(path[byte = j : j + 1])
            if c2 == ">" or c2 == "<":
                break
            j += 1
        var sid = String(path[byte = i + 1 : j])
        var info = off.lookup(_parse_int_sid(sid))
        if info is not None:
            var chrom = String(info[0])
            var start0 = Int(py=info[1])
            return LinAnchor(chrom, start0 + 1, True)
        i = j
    return LinAnchor("", 0, False)


def format_sam_line(hit: AlignmentHit) raises -> String:
    """Primary linear SAM line; empty when unmapped / no anchor / no os:Z."""
    var off = _offsets_mod()
    if hit.path == "*" or hit.path.byte_length() == 0:
        return String("")
    var seq = _tag_value(hit.extra_tags, "os:Z:")
    if seq.byte_length() == 0:
        off.bump_skip_no_seq()
        return String("")
    var proj = project_hit_linear(hit)
    if not proj.ok:
        off.bump_skip_no_anchor()
        return String("")
    var qlen = seq.byte_length()
    var cigar = String(qlen) + "M"
    var qual = String("")
    var qi = 0
    while qi < qlen:
        qual = qual + "I"
        qi += 1
    return (
        hit.query_name
        + "\t0\t"
        + proj.chrom
        + "\t"
        + String(proj.pos1)
        + "\t"
        + String(hit.mapq)
        + "\t"
        + cigar
        + "\t*\t0\t0\t"
        + seq
        + "\t"
        + qual
    )


def write_sam_header(fh: PythonObject) raises:
    var off = _offsets_mod()
    fh.write("@HD\tVN:1.6\tSO:unsorted\n")
    var chroms = off.chroms()
    var lens = off.chrom_lens()
    var n = Int(py=chroms.__len__())
    var i = 0
    while i < n:
        var sn = String(chroms[i])
        var ln = 1
        if i < Int(py=lens.__len__()):
            var v = Int(py=lens[i])
            if v > 0:
                ln = v
        fh.write("@SQ\tSN:" + sn + "\tLN:" + String(ln) + "\n")
        i += 1


def open_sam_write(path: String, offsets_root: String) raises -> PythonObject:
    ensure_qc_offsets(offsets_root)
    var pathlib = Python.import_module("pathlib")
    var builtins = Python.import_module("builtins")
    if not path.startswith("/dev/"):
        var parent = pathlib.Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
    var fh = builtins.open(path, "w")
    write_sam_header(fh)
    return fh


def append_sam_hits(fh: PythonObject, hits: List[AlignmentHit]) raises -> Int:
    var off = _offsets_mod()
    var n = 0
    for h in hits:
        var line = format_sam_line(h)
        if line.byte_length() == 0:
            continue
        fh.write(line + "\n")
        n += 1
        var mapped = Int(py=off.bump_mapped())
        if mapped % 1000000 == 0:
            print("mojo_qc_sam progress ", off.summary(), flush=True)
    return n


def sam_emit_summary() raises -> String:
    return String(_offsets_mod().summary())
