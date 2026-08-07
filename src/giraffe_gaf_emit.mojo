# GAF emitter (named-coordinates path column) for Mojo Giraffe.

from std.collections import List

from giraffe_extend import AlignmentHit
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
