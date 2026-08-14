# GAF emitter (named-coordinates path column) for Mojo Giraffe.
# When METHYLGRAPHER_MOJO_EMIT=sam, streams linear SAM instead (QC BAM path).
#
# Production GAF: one buffered write per batch (not per-hit fh.write). When
# METHYLGRAPHER_NAMED_COORDS_INDEX (or fleet default) is ready, paths are
# translated GBZ-node → GFA-segment at emit time so Align can skip the
# post-map full-GAF Python rewrite.

from std.collections import List
from std.python import Python, PythonObject

from giraffe_hit import AlignmentHit
from giraffe_sam_emit import (
    append_sam_hits,
    close_sam_write,
    open_sam_write,
    sam_emit_summary,
)
from utility import write_text_file


def _emit_mode_sam() raises -> Bool:
    var os_mod = Python.import_module("os")
    var mode = String(os_mod.environ.get("METHYLGRAPHER_MOJO_EMIT", "")).lower()
    return mode == "sam" or mode == "qc_sam"


def _segment_offsets_root() raises -> String:
    var os_mod = Python.import_module("os")
    return String(os_mod.environ.get("METHYLGRAPHER_MOJO_SEGMENT_OFFSETS", ""))


def _bq_phred40(qlen: Int) raises -> String:
    """Synthetic Phred+33 'I' (Q40) string — Mojo path often has no FASTQ quals."""
    var out = String("")
    var i = 0
    while i < qlen:
        out = out + "I"
        i += 1
    return out


def _extras_have_prefix(extras: String, prefix: String) -> Bool:
    # Tag sniff without splitting: require start-of-string or tab before prefix.
    if extras.byte_length() == 0:
        return False
    if extras.startswith(prefix):
        return True
    return extras.find("\t" + prefix) >= 0


def _try_open_named_coords() raises -> PythonObject:
    """Open NamedCoordsIndex if fleet index is present; else Python.none()."""
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/mojo-align/methylgrapher")
    sys_mod.path.insert(0, "/home/ubuntu/mojo-align")
    try:
        var nc = Python.import_module("engine.named_coords")
        var idx_dir = nc.default_index_dir()
        if not Bool(nc.index_ready(idx_dir)):
            return Python.none()
        print("mojo_gaf named_coords emit-time index=", idx_dir, flush=True)
        return nc.NamedCoordsIndex(idx_dir)
    except e:
        print("mojo_gaf named_coords emit-time skipped: ", e, flush=True)
        return Python.none()


def format_gaf_line(hit: AlignmentHit) raises -> String:
    """GAF row compatible with methylGrapher MethylCall (cs / AS / os / rc / bq)."""
    return format_gaf_line_named(hit, Python.none())


def format_gaf_line_named(
    hit: AlignmentHit, named_idx: PythonObject
) raises -> String:
    var qlen_i = hit.qlen
    var qlen = String(qlen_i)
    # Align cs with the full-span coordinates we emit below.
    var cs = hit.cs_tag
    if hit.path != "*" and qlen_i > 0:
        cs = "cs:Z::" + qlen

    var path = hit.path
    var pstart = 0
    var pend = qlen_i
    if named_idx is not None and path != "*" and path.byte_length() > 0:
        try:
            var nc = Python.import_module("engine.named_coords")
            var tup = nc.translate_path(path, named_idx)
            path = String(tup[0])
            pstart = Int(py=tup[1])
            pend = pstart + qlen_i
        except e:
            path = hit.path
            pstart = 0
            pend = qlen_i

    # qname qlen qstart qend strand path plen pstart pend matches alnblen mapq tags
    var plen_s = qlen
    if pend > qlen_i:
        plen_s = String(pend)
    var line = (
        hit.query_name
        + "\t"
        + qlen
        + "\t0\t"
        + qlen
        + "\t+\t"
        + path
        + "\t"
        + plen_s
        + "\t"
        + String(pstart)
        + "\t"
        + String(pend)
        + "\t"
        + qlen
        + "\t"
        + qlen
        + "\t"
        + String(hit.mapq)
        + "\t"
        + cs
    )
    var extras = hit.extra_tags
    if extras.byte_length() > 0:
        line = line + "\t" + extras
    if not _extras_have_prefix(extras, "AS:i:") and line.find("\tAS:i:") < 0:
        line = line + "\tAS:i:" + qlen
    if qlen_i > 0 and not _extras_have_prefix(extras, "bq:Z:") and line.find(
        "\tbq:Z:"
    ) < 0:
        line = line + "\tbq:Z:" + _bq_phred40(qlen_i)
    return line


def write_gaf(path: String, hits: List[AlignmentHit]) raises:
    var body = String("")
    for h in hits:
        body = body + format_gaf_line(h) + "\n"
    write_text_file(path, body)


def open_gaf_write(path: String) raises -> PythonObject:
    """Open GAF (or QC SAM) for streaming append — never buffer whole file.

    Returns a Python dict with keys ``fh``, ``named``, ``named_lines`` so
    ``append_gaf_hits`` / ``close_emit`` can batch-write and stamp named coords.
    """
    if _emit_mode_sam():
        var root = _segment_offsets_root()
        if root.byte_length() == 0:
            raise Error(
                "METHYLGRAPHER_MOJO_EMIT=sam requires METHYLGRAPHER_MOJO_SEGMENT_OFFSETS"
            )
        print("mojo_emit mode=sam offsets=", root, " out=", path, flush=True)
        return open_sam_write(path, root)
    var pathlib = Python.import_module("pathlib")
    var builtins = Python.import_module("builtins")
    # /dev/fd/N (legacy pipe path) must not mkdir parents.
    if not path.startswith("/dev/"):
        var parent = pathlib.Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
    # Large buffer: one fwrite syscall per batch, not per hit.
    var fh = builtins.open(path, "w", buffering=8 * 1024 * 1024)
    var named = _try_open_named_coords()
    var wrap = Python.dict()
    wrap["fh"] = fh
    wrap["named"] = named
    wrap["named_lines"] = 0
    wrap["path"] = path
    wrap["gaf"] = True
    return wrap


def append_gaf_hits(fh: PythonObject, hits: List[AlignmentHit]) raises -> Int:
    """Write mapped hits (skip path=*); return lines written.

    SAM mode projects to linear chrom/pos via GRCh38 offset table + os:Z.
    GAF mode buffers the whole batch into one write.
    """
    # Legacy bare file handle (tests / older callers).
    var is_wrap = False
    try:
        _ = fh["gaf"]
        is_wrap = True
    except e:
        is_wrap = False
    if not is_wrap:
        if _emit_mode_sam():
            return append_sam_hits(fh, hits)
        var n0 = 0
        var body0 = String("")
        for h in hits:
            if h.path == "*" or h.path.byte_length() == 0:
                continue
            body0 = body0 + format_gaf_line(h) + "\n"
            n0 += 1
        if n0 > 0:
            fh.write(body0)
        return n0

    if _emit_mode_sam():
        return append_sam_hits(fh, hits)

    var out_fh = fh["fh"]
    var named_idx = fh["named"]
    var n = 0
    var body = String("")
    for h in hits:
        if h.path == "*" or h.path.byte_length() == 0:
            continue
        body = body + format_gaf_line_named(h, named_idx) + "\n"
        n += 1
    if n > 0:
        out_fh.write(body)
        if named_idx is not None:
            fh["named_lines"] = Int(py=fh["named_lines"]) + n
    return n


def flush_emit(fh: PythonObject) raises:
    """Flush GAF wrap or bare file handle (no-op for SAM buffer path)."""
    if _emit_mode_sam():
        return
    try:
        _ = fh["gaf"]
        fh["fh"].flush()
    except e:
        try:
            fh.flush()
        except e2:
            pass


def close_emit(fh: PythonObject) raises:
    """Flush/close GAF or QC SAM stream (SAM path keeps a write buffer)."""
    if _emit_mode_sam():
        close_sam_write(fh)
        return
    var is_wrap = False
    try:
        _ = fh["gaf"]
        is_wrap = True
    except e:
        is_wrap = False
    if not is_wrap:
        fh.close()
        return
    var out_fh = fh["fh"]
    out_fh.flush()
    out_fh.close()
    var named_idx = fh["named"]
    if named_idx is not None:
        try:
            named_idx.close()
        except e:
            pass
        var n_lines = Int(py=fh["named_lines"])
        if n_lines > 0:
            var json = Python.import_module("json")
            var pathlib = Python.import_module("pathlib")
            var path = String(fh["path"])
            var stamp = pathlib.Path(path + ".named_coords.json")
            var payload = Python.dict()
            payload["format"] = "gbz-to-gfa-v1"
            payload["n_lines"] = n_lines
            payload["emit_time"] = True
            payload["source_gaf"] = path
            stamp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print("mojo_gaf named_coords emit-time stamped lines=", n_lines, flush=True)


def emit_footer_log() raises:
    """Optional end-of-stream summary (SAM QC path)."""
    if _emit_mode_sam():
        print("mojo_qc_sam done ", sam_emit_summary(), flush=True)
