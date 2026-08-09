# Distance / zipcode clustering for Mojo Giraffe (no Python on hot path).
#
# Staged science: bucket by node>>8, keep densest bucket, prune with
# abs(node − anchor) ≤ 200 (same heuristic DistIndex.distance_ok used).
# Full SPIZ / .dist decode comes later; zip_path is reserved.

from std.collections import Dict, List


comptime DIST_CAP = 200


def cluster_seed_hits(
    hits: List[String], dist_path: String, zip_path: String
) raises -> List[String]:
    """Cluster ``node:orient:offset`` seeds (Mojo-native; no DistIndex import)."""
    _ = zip_path
    if len(hits) == 0:
        return List[String]()

    var buckets = Dict[Int, List[String]]()
    for h in hits:
        var parts = h.split(":")
        if len(parts) < 1:
            continue
        var nid = Int(String(parts[0]))
        var bucket = nid >> 8
        if bucket in buckets:
            buckets[bucket].append(h.copy())
        else:
            var lst = List[String]()
            lst.append(h.copy())
            buckets[bucket] = lst^

    var best_key = 0
    var best_n = 0
    var have_best = False
    var keys = List[Int]()
    for k in buckets:
        keys.append(k)
    for k in keys:
        var n = len(buckets[k])
        if (not have_best) or n > best_n:
            best_n = n
            best_key = k
            have_best = True
    if best_n == 0 or not have_best:
        return List[String]()

    var clustered = buckets[best_key].copy()
    # Prune when a dist sidecar path is provided (heuristic until full decode).
    if dist_path.byte_length() == 0:
        return clustered^

    var parts0 = clustered[0].split(":")
    var anchor = Int(String(parts0[0]))
    var pruned = List[String]()
    for h in clustered:
        var p = h.split(":")
        var nid2 = Int(String(p[0]))
        var d = nid2 - anchor
        if d < 0:
            d = -d
        if d <= DIST_CAP:
            pruned.append(h.copy())
    if len(pruned) > 0:
        return pruned^
    return clustered^


def segment_majority_cluster(hits: List[String]) raises -> List[String]:
    """Legacy stand-in — prefer cluster_seed_hits."""
    return cluster_seed_hits(hits, "", "")
