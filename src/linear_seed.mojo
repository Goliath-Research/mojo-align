# K-mer extraction for linear MojoFq2bamMeth seeds.

from std.collections import List

from linear_index import LinearIndex


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


def seed_hits(index: LinearIndex, seq: String) raises -> List[String]:
    """Return `contig:offset` postings for every k-mer in ``seq``."""
    var hits = List[String]()
    var mers = extract_kmers(seq, index.k)
    for mer in mers:
        var found = index.lookup_kmer(mer)
        for h in found:
            hits.append(h.copy())
    return hits^
