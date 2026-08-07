# Seed-and-extend on GFA graphs and dense segment packs.

from std.collections import Dict, List
from std.python import Python

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
    """Fixture GFA path (exact / short k-mer majority)."""
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


def gapless_extend_seeds(
    pack_dir: String,
    query_name: String,
    seq: String,
    seeds: List[String],
) raises -> List[AlignmentHit]:
    """Gapless extend from dense pack using ``node:orient:offset`` seeds."""
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    var sp = Python.import_module("engine.segment_pack")
    var qm = Python.import_module("engine.quartet_map")
    var pack = sp.SegmentPack(pack_dir)
    var out = List[AlignmentHit]()
    var qlen = seq.byte_length()
    for seed in seeds:
        var parts = seed.split(":")
        if len(parts) < 3:
            continue
        var node = Int(String(parts[0]))
        var is_rev = String(parts[1]) == "1"
        var offset = Int(String(parts[2]))
        var ext = qm._gapless_extend(pack, node, offset, is_rev, seq)
        if ext is None:
            continue
        var path = String(ext[0])
        var mq = Int(py=ext[1])
        var cs = String(ext[2])
        out.append(AlignmentHit(query_name, path, qlen, mq, cs))
        if len(out) >= 2:
            break
    return out^
