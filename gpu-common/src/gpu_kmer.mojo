# Portable CPU k-mer extract for gpu-common (shared by linear + giraffe seeds).

from std.collections import List


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
