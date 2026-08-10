# GAF emitter (named-coordinates path column) for Mojo Giraffe.
# When METHYLGRAPHER_MOJO_EMIT=sam, streams linear SAM instead (QC BAM path).

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


def format_gaf_line(hit: AlignmentHit) raises -> String:
    """GAF row compatible with methylGrapher MethylCall (cs / AS / os / rc / bq).

    Coordinates are always full-query span. ``cs:Z`` must match that span —
    short ``cs:Z::matched`` from gapless/GPU scoring breaks MethylCall path math.
    ``AS:i`` and ``bq:Z`` are required by ``engine.mcall`` / native MethylCall.
    """
    var qlen_i = hit.qlen
    var qlen = String(qlen_i)
    # Align cs with the full-span coordinates we emit below.
    var cs = hit.cs_tag
    if hit.path != "*" and qlen_i > 0:
        cs = "cs:Z::" + qlen

    # qname qlen qstart qend strand path plen pstart pend matches alnblen mapq tags
    var line = (
        hit.query_name
        + "\t"
        + qlen
        + "\t0\t"
        + qlen
        + "\t+\t"
        + hit.path
        + "\t"
        + qlen
        + "\t0\t"
        + qlen
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
    """Open GAF (or QC SAM) for streaming append — never buffer whole file."""
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
    return builtins.open(path, "w")


def append_gaf_hits(fh: PythonObject, hits: List[AlignmentHit]) raises -> Int:
    """Write mapped hits (skip path=*); return lines written.

    SAM mode projects to linear chrom/pos via GRCh38 offset table + os:Z.
    """
    if _emit_mode_sam():
        return append_sam_hits(fh, hits)
    var n = 0
    for h in hits:
        if h.path == "*" or h.path.byte_length() == 0:
            continue
        fh.write(format_gaf_line(h) + "\n")
        n += 1
    return n


def close_emit(fh: PythonObject) raises:
    """Flush/close GAF or QC SAM stream (SAM path keeps a write buffer)."""
    if _emit_mode_sam():
        close_sam_write(fh)
    else:
        fh.close()


def emit_footer_log() raises:
    """Optional end-of-stream summary (SAM QC path)."""
    if _emit_mode_sam():
        print("mojo_qc_sam done ", sam_emit_summary(), flush=True)
