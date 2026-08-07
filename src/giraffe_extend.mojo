# Seed-and-extend on GFA graphs (fixture-scale exact / near-exact matches).

from std.collections import Dict, List

from giraffe_index import GraphIndex
from giraffe_seed import seed_hits


struct AlignmentHit(Copyable, Movable):
    var query_name: String
    var path: String
    var qlen: Int
    var mapq: Int
    var cs_tag: String
    var extra_tags: String

    def __init__(
        out self,
        query_name: String,
        path: String,
        qlen: Int,
        mapq: Int,
        cs_tag: String,
        extra_tags: String = "",
    ):
        self.query_name = query_name
        self.path = path
        self.qlen = qlen
        self.mapq = mapq
        self.cs_tag = cs_tag
        self.extra_tags = extra_tags


def extend_exact(index: GraphIndex, query_name: String, seq: String) raises -> List[AlignmentHit]:
    var out = List[AlignmentHit]()
    var qlen = seq.byte_length()

    var ids = index.segment_ids()
    for seg_id in ids:
        var s = index.segments[seg_id].copy()
        if s == seq:
            out.append(
                AlignmentHit(
                    query_name,
                    ">" + seg_id,
                    qlen,
                    60,
                    "cs:Z::" + String(qlen),
                )
            )
    if len(out) > 0:
        return out^

    var counts = Dict[String, Int]()
    var hits = seed_hits(index, seq)
    for h in hits:
        var parts = h.split(":")
        if len(parts) < 1:
            continue
        var seg = String(parts[0])
        if seg in counts:
            counts[seg] = counts[seg] + 1
        else:
            counts[seg] = 1

    var best_seg = String("")
    var best_n = 0
    var keys = List[String]()
    for seg in counts:
        keys.append(seg)
    for seg in keys:
        var n = counts[seg]
        if n > best_n:
            best_n = n
            best_seg = seg.copy()
    if best_n > 0 and best_seg.byte_length() > 0:
        var mq = 20
        if best_n >= 2:
            mq = 40
        out.append(
            AlignmentHit(
                query_name,
                ">" + best_seg,
                qlen,
                mq,
                "cs:Z::" + String(qlen),
            )
        )
    return out^
