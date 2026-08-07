# Minimizer / k-mer extraction for Mojo Giraffe.

from std.collections import List

from giraffe_index import GraphIndex


def extract_kmers(seq: String, k: Int) raises -> List[String]:
    var out = List[String]()
    var n = seq.byte_length()
    if n < k:
        return out^
    var i = 0
    while i <= n - k:
        out.append(String(seq[byte = i : i + k]))
        i += 1
    return out^


def seed_hits(index: GraphIndex, seq: String) raises -> List[String]:
    var hits = List[String]()
    var mers = extract_kmers(seq, index.k)
    for mer in mers:
        var found = index.lookup_kmer(mer)
        for h in found:
            hits.append(h.copy())
    return hits^
