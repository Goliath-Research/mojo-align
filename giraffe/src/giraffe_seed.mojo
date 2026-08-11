# Minimizer / k-mer extraction for Mojo Giraffe.

from std.collections import List

from giraffe_index import GraphIndex
from gpu_kmer import extract_kmers


def seed_hits(index: GraphIndex, seq: String) raises -> List[String]:
    var hits = List[String]()
    var mers = extract_kmers(seq, index.k)
    for mer in mers:
        var found = index.lookup_kmer(mer)
        for h in found:
            hits.append(h.copy())
    return hits^
