# Distance-index surface for GBZ-native Mojo Giraffe.
#
# Production file: `{prefix}.dist` (vg Giraffe distance index).
# Staged path: cluster seed hits by segment majority (fixture-scale stand-in
# until a native `.dist` decoder lands). Mojo API kept stable for GPU/CPU.

from std.collections import Dict, List


struct SeedCluster(Copyable, Movable):
    var seg_id: String
    var hit_count: Int

    def __init__(out self, seg_id: String, hit_count: Int):
        self.seg_id = seg_id
        self.hit_count = hit_count


def cluster_hits_by_segment(hits: List[String]) raises -> List[SeedCluster]:
    var counts = Dict[String, Int]()
    for h in hits:
        var parts = h.split(":")
        if len(parts) < 1:
            continue
        var seg = String(parts[0])
        if seg in counts:
            counts[seg] = counts[seg] + 1
        else:
            counts[seg] = 1
    var keys = List[String]()
    for seg in counts:
        keys.append(seg)
    var out = List[SeedCluster]()
    for seg in keys:
        out.append(SeedCluster(seg, counts[seg]))
    return out^


def best_cluster(clusters: List[SeedCluster]) raises -> SeedCluster:
    var best = SeedCluster("", 0)
    for c in clusters:
        if c.hit_count > best.hit_count:
            best = c.copy()
    return best^
