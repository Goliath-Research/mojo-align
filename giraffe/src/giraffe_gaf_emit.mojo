# GAF emitter (named-coordinates path column) for Mojo Giraffe.
# When MOJO_ALIGN_EMIT=sam, streams linear SAM instead (QC BAM path).
#
# Production GAF: Mojo byte buffer + libc write (no Python per-hit I/O).
# Named-coords via mmap (giraffe_named_coords.mojo). Do not flush every batch —
# OS page cache + close_emit is enough (same as CPU stream path).

from std.collections import List
from std.python import Python, PythonObject

from giraffe_hit import AlignmentHit
from giraffe_named_coords import (
    NamedCoordsMojo,
    named_coords_close,
    named_coords_from_wrap,
    named_coords_store_wrap,
    named_coords_translate_path,
    named_coords_try_open,
)
from giraffe_sam_emit import (
    append_sam_hits,
    close_sam_write,
    open_sam_write,
    sam_emit_summary,
)
from mojo_align_env import getenv_align
from utility import write_text_file


def _emit_mode_sam() raises -> Bool:
    var mode = getenv_align("EMIT", "").lower()
    return mode == "sam" or mode == "qc_sam"


def _segment_offsets_root() raises -> String:
    return getenv_align("SEGMENT_OFFSETS", "")


def _extras_have_prefix(extras: String, prefix: String) -> Bool:
    if extras.byte_length() == 0:
        return False
    if extras.startswith(prefix):
        return True
    return extras.find("\t" + prefix) >= 0


def _append_str(mut buf: List[UInt8], s: String):
    var n = s.byte_length()
    if n <= 0:
        return
    var base = len(buf)
    buf.resize(base + n, UInt8(0))
    var i = 0
    while i < n:
        buf[base + i] = UInt8(ord(s[byte = i : i + 1]))
        i += 1


def _append_bq_phred40(mut buf: List[UInt8], qlen: Int):
    """Synthetic Phred+33 'I' (Q40) — append bytes, no quadratic String."""
    _append_str(buf, "bq:Z:")
    var i = 0
    while i < qlen:
        buf.append(UInt8(73))
        i += 1


def _append_gaf_line(
    mut buf: List[UInt8], hit: AlignmentHit, named: NamedCoordsMojo
) raises:
    var qlen_i = hit.qlen
    var qlen = String(qlen_i)
    var cs = hit.cs_tag
    if hit.path != "*" and qlen_i > 0:
        cs = "cs:Z::" + qlen

    var path = hit.path
    var pstart = 0
    var pend = qlen_i
    if named.alive and path != "*" and path.byte_length() > 0:
        try:
            var new_path = String("")
            var ps = 0
            named_coords_translate_path(path, named, new_path, ps)
            path = new_path
            pstart = ps
            pend = pstart + qlen_i
        except e:
            path = hit.path
            pstart = 0
            pend = qlen_i

    var plen_s = qlen
    if pend > qlen_i:
        plen_s = String(pend)

    _append_str(buf, hit.query_name)
    buf.append(UInt8(9))
    _append_str(buf, qlen)
    _append_str(buf, "\t0\t")
    _append_str(buf, qlen)
    _append_str(buf, "\t+\t")
    _append_str(buf, path)
    buf.append(UInt8(9))
    _append_str(buf, plen_s)
    buf.append(UInt8(9))
    _append_str(buf, String(pstart))
    buf.append(UInt8(9))
    _append_str(buf, String(pend))
    buf.append(UInt8(9))
    _append_str(buf, qlen)
    buf.append(UInt8(9))
    _append_str(buf, qlen)
    buf.append(UInt8(9))
    _append_str(buf, String(hit.mapq))
    buf.append(UInt8(9))
    _append_str(buf, cs)

    var extras = hit.extra_tags
    if extras.byte_length() > 0:
        buf.append(UInt8(9))
        _append_str(buf, extras)
    if not _extras_have_prefix(extras, "AS:i:"):
        _append_str(buf, "\tAS:i:")
        _append_str(buf, qlen)
    if qlen_i > 0 and not _extras_have_prefix(extras, "bq:Z:"):
        buf.append(UInt8(9))
        _append_bq_phred40(buf, qlen_i)
    buf.append(UInt8(10))


def format_gaf_line(hit: AlignmentHit) raises -> String:
    """GAF row compatible with methylGrapher MethylCall (cs / AS / os / rc / bq)."""
    var buf = List[UInt8]()
    _append_gaf_line(buf, hit, NamedCoordsMojo())
    # Drop trailing newline for String return (legacy callers).
    var n = len(buf)
    if n > 0 and buf[n - 1] == 10:
        n -= 1
    var out = String("")
    var i = 0
    while i < n:
        out = out + String(chr(Int(buf[i])))
        i += 1
    return out


def format_gaf_line_named(
    hit: AlignmentHit, named_idx: PythonObject
) raises -> String:
    """Legacy Python NamedCoordsIndex path (tests). Prefer Mojo mmap emit."""
    _ = named_idx
    return format_gaf_line(hit)


def write_gaf(path: String, hits: List[AlignmentHit]) raises:
    var buf = List[UInt8]()
    for h in hits:
        _append_gaf_line(buf, h, NamedCoordsMojo())
    var body = String("")
    var i = 0
    while i < len(buf):
        body = body + String(chr(Int(buf[i])))
        i += 1
    write_text_file(path, body)


def _native_write_all(fd: Int, buf: List[UInt8]) raises:
    var n = len(buf)
    if n <= 0:
        return
    # Prefer os.write(fd, memoryview) over libc write — Mojo std.ffi already
    # binds write with a conflicting signature on some toolchains.
    var os_mod = Python.import_module("os")
    var ctypes = Python.import_module("ctypes")
    var off = 0
    while off < n:
        var chunk = n - off
        var mv = ctypes.string_at(Int(buf.unsafe_ptr()) + off, chunk)
        var w = Int(py=os_mod.write(fd, mv))
        if w <= 0:
            raise Error("GAF write failed")
        off += w


def open_gaf_write(path: String) raises -> PythonObject:
    """Open GAF (or QC SAM) for streaming append — never buffer whole file.

    GAF wrap keys: ``fd``, ``named_*``, ``named_lines``, ``path``, ``gaf``.
    """
    if _emit_mode_sam():
        var root = _segment_offsets_root()
        if root.byte_length() == 0:
            raise Error(
                "MOJO_ALIGN_EMIT=sam requires MOJO_ALIGN_SEGMENT_OFFSETS"
            )
        print("mojo_emit mode=sam offsets=", root, " out=", path, flush=True)
        return open_sam_write(path, root)
    var pathlib = Python.import_module("pathlib")
    var os_mod = Python.import_module("os")
    if not path.startswith("/dev/"):
        var parent = pathlib.Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
    var fd = Int(
        py=os_mod.open(
            path, os_mod.O_WRONLY | os_mod.O_CREAT | os_mod.O_TRUNC, 0o644
        )
    )
    var named = named_coords_try_open()
    var wrap = Python.dict()
    wrap["fd"] = fd
    wrap["named"] = Python.none()
    wrap["named_lines"] = 0
    wrap["path"] = path
    wrap["gaf"] = True
    named_coords_store_wrap(wrap, named)
    return wrap


def append_gaf_hits(fh: PythonObject, hits: List[AlignmentHit]) raises -> Int:
    """Write mapped hits (skip path=*); return lines written."""
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

    var fd = Int(py=fh["fd"])
    var named = named_coords_from_wrap(fh)
    var n = 0
    var buf = List[UInt8]()
    # ~256 bytes/line × 16k hits ≈ 4 MiB; grow as needed.
    buf.reserve(4 * 1024 * 1024)
    for h in hits:
        if h.path == "*" or h.path.byte_length() == 0:
            continue
        _append_gaf_line(buf, h, named)
        n += 1
    if n > 0:
        _native_write_all(fd, buf)
        if named.alive:
            fh["named_lines"] = Int(py=fh["named_lines"]) + n
    return n


def flush_emit(fh: PythonObject) raises:
    """No-op for production GAF (buffered writes; flush on close only)."""
    _ = fh


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
    var os_mod = Python.import_module("os")
    var fd = Int(py=fh["fd"])
    try:
        _ = os_mod.fsync(fd)
    except e:
        pass
    _ = os_mod.close(fd)
    fh["fd"] = -1
    var named = named_coords_from_wrap(fh)
    if named.alive:
        named_coords_close(named)
        named_coords_store_wrap(fh, NamedCoordsMojo())
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
