# Distance / zipcode clustering for Mojo Giraffe.

from std.collections import Dict, List
from std.python import Python


def cluster_seed_hits(
    hits: List[String], dist_path: String, zip_path: String
) raises -> List[String]:
    """Cluster ``node:orient:offset`` seeds using zip payload + dist heuristic."""
    if len(hits) == 0:
        return List[String]()
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    var zipmod = Python.import_module("engine.zipcodes_index")

    # Bucket by node>>8 (zip full decode staged); prune with DistIndex.
    var buckets = Dict[String, List[String]]()
    for h in hits:
        var parts = h.split(":")
        if len(parts) < 1:
            continue
        var node = String(parts[0])
        var bucket = node
        if node.byte_length() > 2:
            bucket = String(node[byte = 0 : node.byte_length() - 2])
        if bucket in buckets:
            buckets[bucket].append(h.copy())
        else:
            var lst = List[String]()
            lst.append(h.copy())
            buckets[bucket] = lst^

    var best_key = String("")
    var best_n = 0
    var keys = List[String]()
    for k in buckets:
        keys.append(k)
    for k in keys:
        var n = len(buckets[k])
        if n > best_n:
            best_n = n
            best_key = k.copy()
    if best_n == 0:
        return List[String]()

    var clustered = buckets[best_key].copy()
    if dist_path.byte_length() > 0:
        var dist = zipmod.DistIndex(dist_path)
        var parts0 = clustered[0].split(":")
        var anchor = Int(String(parts0[0]))
        var pruned = List[String]()
        for h in clustered:
            var p = h.split(":")
            var nid = Int(String(p[0]))
            if Bool(dist.distance_ok(anchor, nid)):
                pruned.append(h.copy())
        if len(pruned) > 0:
            return pruned^
    _ = zip_path
    return clustered^


def segment_majority_cluster(hits: List[String]) raises -> List[String]:
    """Legacy stand-in — prefer cluster_seed_hits."""
    return cluster_seed_hits(hits, "", "")
