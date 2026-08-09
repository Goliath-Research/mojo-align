# Native Mojo gapless (+ multi-node heuristic) extend over dense packs.

from std.collections import List

from giraffe_hit import AlignmentHit
from giraffe_pack import DensePack, reverse_complement_dna

def _upper_dna(seq: String) raises -> String:
    var out = String("")
    var i = 0
    var n = seq.byte_length()
    while i < n:
        var ch = String(seq[byte = i : i + 1])
        if ch == "a":
            out += "A"
        elif ch == "c":
            out += "C"
        elif ch == "g":
            out += "G"
        elif ch == "t":
            out += "T"
        else:
            out += ch
        i += 1
    return out^



def _find_substr(hay: String, needle: String) raises -> Int:
    var n = needle.byte_length()
    var m = hay.byte_length()
    if n == 0 or n > m:
        return -1
    var i = 0
    while i <= m - n:
        if String(hay[byte = i : i + n]) == needle:
            return i
        i += 1
    return -1


def _gapless_one(
    pack: DensePack,
    node_id: Int,
    offset: Int,
    is_rev: Bool,
    query: String,
) raises -> AlignmentHit:
    """Port of engine.quartet_map._gapless_extend — Mojo hot path."""
    var q = _upper_dna(query)
    var ref_raw = pack.get(node_id)
    var ref_seq = _upper_dna(ref_raw)
    var empty = AlignmentHit("", "*", 0, 0, "cs:Z:*")
    if ref_seq.byte_length() == 0:
        return empty^

    var ref_aln = ref_seq
    if is_rev:
        ref_aln = reverse_complement_dna(ref_seq)

    var starts = List[Int]()
    if is_rev:
        var s0 = ref_aln.byte_length() - offset - q.byte_length()
        if s0 < 0:
            s0 = 0
        starts.append(s0)
        var s1 = offset - q.byte_length() + 1
        if s1 < 0:
            s1 = 0
        starts.append(s1)
        starts.append(0)
    else:
        starts.append(offset)
        var s2 = offset - q.byte_length() + 29
        if s2 < 0:
            s2 = 0
        starts.append(s2)
        starts.append(0)

    var best_score = -1
    var best_path = String("")
    var best_mq = 0
    var best_cs = String("cs:Z:*")

    for start in starts:
        if start < 0 or start >= ref_aln.byte_length():
            continue
        var matched = 0
        var mism = 0
        var qi = 0
        var ri = start
        while qi < q.byte_length() and ri < ref_aln.byte_length():
            var qb = String(q[byte = qi : qi + 1])
            var rb = String(ref_aln[byte = ri : ri + 1])
            if qb == rb:
                matched += 1
            else:
                mism += 1
                var mism_cap = q.byte_length() // 20
                if mism_cap < 2:
                    mism_cap = 2
                if mism > mism_cap:
                    break
            qi += 1
            ri += 1
        var min_need = 29
        if ref_aln.byte_length() < min_need:
            min_need = ref_aln.byte_length()
        if matched < min_need:
            continue
        var score = matched - 2 * mism
        if score > best_score:
            best_score = score
            best_path = ">" + String(node_id)
            if matched >= (q.byte_length() * 9) // 10:
                if mism == 0:
                    best_mq = 60
                    best_cs = "cs:Z::" + String(q.byte_length())
                else:
                    best_mq = 40
                    best_cs = "cs:Z::" + String(matched)
            else:
                best_mq = 20
                best_cs = "cs:Z::" + String(matched)

    if best_score >= 0:
        return AlignmentHit(
            "", best_path, q.byte_length(), best_mq, best_cs
        )^

    var cand = ref_seq
    var pos = _find_substr(q, cand)
    if pos < 0:
        cand = reverse_complement_dna(ref_seq)
        pos = _find_substr(q, cand)
    if pos >= 0 and cand.byte_length() >= 21:
        var mq2 = 20
        if cand.byte_length() >= 29:
            mq2 = 40
        return AlignmentHit(
            "",
            ">" + String(node_id),
            q.byte_length(),
            mq2,
            "cs:Z::" + String(cand.byte_length()),
        )^

    if ref_seq.byte_length() < q.byte_length():
        var neighbors = List[Int]()
        neighbors.append(node_id + 1)
        neighbors.append(node_id - 1)
        neighbors.append(node_id + 2)
        var qi2 = 0
        var path = ">" + String(node_id)
        for nxt2 in neighbors:
            if nxt2 == node_id or nxt2 < 0:
                continue
            var nref2_raw = pack.get(nxt2)
            var nref2 = _upper_dna(nref2_raw)
            if nref2.byte_length() == 0:
                continue
            if is_rev:
                nref2 = reverse_complement_dna(nref2)
            var rem2 = q.byte_length() - qi2
            if rem2 <= 0:
                break
            var m2 = 0
            while m2 < rem2 and m2 < nref2.byte_length():
                if String(q[byte = qi2 + m2 : qi2 + m2 + 1]) != String(
                    nref2[byte = m2 : m2 + 1]
                ):
                    break
                m2 += 1
            if m2 < 15:
                continue
            path = path + ">" + String(nxt2)
            qi2 += m2
            if qi2 >= (q.byte_length() * 9) // 10:
                var mq3 = 35
                if qi2 >= q.byte_length():
                    mq3 = 50
                return AlignmentHit(
                    "", path, q.byte_length(), mq3, "cs:Z::" + String(qi2)
                )^
    return empty^


def gapless_extend_native(
    pack_dir: String,
    query_name: String,
    seq: String,
    seeds: List[String],
) raises -> List[AlignmentHit]:
    """Native Mojo gapless extend; seeds are ``node:orient:offset``."""
    var pack = DensePack(pack_dir)
    var out = List[AlignmentHit]()
    var qlen = seq.byte_length()
    for seed in seeds:
        var parts = seed.split(":")
        if len(parts) < 3:
            continue
        var node = Int(String(parts[0]))
        var is_rev = String(parts[1]) == "1"
        var offset = Int(String(parts[2]))
        var hit = _gapless_one(pack, node, offset, is_rev, seq)
        if hit.path == "*" or hit.path.byte_length() == 0:
            continue
        out.append(
            AlignmentHit(query_name, hit.path, qlen, hit.mapq, hit.cs_tag)
        )
        if len(out) >= 2:
            break
    return out^
