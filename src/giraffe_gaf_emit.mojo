# GAF emitter (named-coordinates path column) for Mojo Giraffe.

from std.collections import List
from std.python import Python, PythonObject

from giraffe_hit import AlignmentHit
from utility import write_text_file


def format_gaf_line(hit: AlignmentHit) raises -> String:
    """Minimal GAF row compatible with MethylCall path / cs consumption."""
    var qlen = String(hit.qlen)
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
        + hit.cs_tag
    )
    if hit.extra_tags.byte_length() > 0:
        line = line + "\t" + hit.extra_tags
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
