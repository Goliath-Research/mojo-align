# Minimizer + zipcode locate surface for GBZ-native Mojo Giraffe.
#
# Production vg indexes: `{prefix}.shortread.withzip.min` + `.shortread.zipcodes`.
# Staged path: build an in-memory k-mer posting list from GBZ-decoded segments
# (same science seed surface as giraffe_seed; GPU via giraffe_device).

from std.collections import Dict, List

from giraffe_seed import extract_kmers


def build_min_index_from_segments(
    segments: Dict[String, String], k: Int
) raises -> List[String]:
    """Flattened hit table rows: ``kmer\\tseg:offset``."""
    var hit_table = List[String]()
    var ids = List[String]()
    for seg_id in segments:
        ids.append(seg_id)
    for seg_id in ids:
        var seq = segments[seg_id].copy()
        var n = seq.byte_length()
        if n < k:
            continue
        var i = 0
        while i <= n - k:
            var mer = String(seq[byte = i : i + k])
            hit_table.append(mer + "\t" + seg_id + ":" + String(i))
            i += 1
    return hit_table^


def lookup_min_hits(hit_table: List[String], mer: String) raises -> List[String]:
    var hits = List[String]()
    for row in hit_table:
        var parts = row.split("\t")
        if len(parts) >= 2 and String(parts[0]) == mer:
            hits.append(String(parts[1]))
    return hits^


def seed_seq(hit_table: List[String], seq: String, k: Int) raises -> List[String]:
    var out = List[String]()
    var mers = extract_kmers(seq, k)
    for mer in mers:
        var found = lookup_min_hits(hit_table, mer)
        for h in found:
            out.append(h.copy())
    return out^
