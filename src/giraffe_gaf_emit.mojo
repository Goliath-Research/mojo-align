# GAF emitter (named-coordinates path column) for Mojo Giraffe.

from std.collections import List
from std.python import Python, PythonObject

from giraffe_hit import AlignmentHit
from utility import write_text_file


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
    """Open GAF for streaming append (Buffy-scale; never buffer whole file)."""
    var pathlib = Python.import_module("pathlib")
    var builtins = Python.import_module("builtins")
    # /dev/fd/N (legacy pipe path) must not mkdir parents.
    if not path.startswith("/dev/"):
        var parent = pathlib.Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
    return builtins.open(path, "w")


def append_gaf_hits(fh: PythonObject, hits: List[AlignmentHit]) raises -> Int:
    """Write mapped hits (skip path=*); return lines written."""
    var n = 0
    for h in hits:
        if h.path == "*" or h.path.byte_length() == 0:
            continue
        fh.write(format_gaf_line(h) + "\n")
        n += 1
    return n
