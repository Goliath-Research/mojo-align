# Seed-and-extend + SAM emit for linear MojoFq2bamMeth.

from std.collections import Dict, List

from linear_index import LinearIndex
from linear_seed import seed_hits
from utility import reverse_complement


struct LinearHit(Copyable, Movable):
    var query_name: String
    var flag: Int
    var contig: String
    var pos: Int  # 1-based
    var mapq: Int
    var cigar: String
    var seq: String
    var qual: String
    var rnext: String
    var pnext: Int
    var tlen: Int

    def __init__(
        out self,
        query_name: String,
        flag: Int,
        contig: String,
        pos: Int,
        mapq: Int,
        cigar: String,
        seq: String,
        qual: String = "*",
        rnext: String = "*",
        pnext: Int = 0,
        tlen: Int = 0,
    ):
        self.query_name = query_name
        self.flag = flag
        self.contig = contig
        self.pos = pos
        self.mapq = mapq
        self.cigar = cigar
        self.seq = seq
        self.qual = qual
        self.rnext = rnext
        self.pnext = pnext
        self.tlen = tlen


def _find_substr(hay: String, needle: String) raises -> Int:
    """Return 0-based start of needle in hay, or -1."""
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


def _vote_and_verify(
    index: LinearIndex,
    name: String,
    seq: String,
    hits: List[String],
) raises -> LinearHit:
    """K-mer majority vote → gapless verify."""
    var qlen = seq.byte_length()
    if len(hits) == 0:
        return LinearHit(name, 4, "*", 0, 0, "*", seq, "*")^

    var counts = Dict[String, Int]()
    for h in hits:
        var parts = h.split(":")
        if len(parts) < 2:
            continue
        var key = String(parts[0]) + ":" + String(parts[1])
        if key in counts:
            counts[key] = counts[key] + 1
        else:
            counts[key] = 1

    var best_key = String("")
    var best_n = 0
    var keys = List[String]()
    for key in counts:
        keys.append(key)
    for key in keys:
        var n = counts[key]
        if n > best_n:
            best_n = n
            best_key = key.copy()

    if best_n <= 0 or best_key.byte_length() == 0:
        return LinearHit(name, 4, "*", 0, 0, "*", seq, "*")^

    var bp = best_key.split(":")
    if len(bp) < 2:
        return LinearHit(name, 4, "*", 0, 0, "*", seq, "*")^
    var contig = String(bp[0])
    var seed_off = Int(String(bp[1]))
    for c in index.contigs:
        if c.name != contig:
            continue
        if seed_off < 0 or seed_off + qlen > c.seq.byte_length():
            continue
        var window = String(c.seq[byte = seed_off : seed_off + qlen])
        if window == seq:
            var mq = 20
            if best_n >= 3:
                mq = 40
            if best_n >= 5:
                mq = 60
            return LinearHit(
                name, 0, contig, seed_off + 1, mq, String(qlen) + "M", seq, "*"
            )
    return LinearHit(name, 4, "*", 0, 0, "*", seq, "*")^


def extend_read_with_seeds(
    index: LinearIndex,
    name: String,
    seq: String,
    seed_kmers: List[String],
) raises -> LinearHit:
    """Map using GPU/host seed k-mers (wired into hash locate + extend).

    Seed/hash path first (production GRCh38). Exact full-contig scan is only
    a tiny-index / fixture shortcut — never O(genome×read) on Buffy refs.
    """
    var qlen = seq.byte_length()

    # Fast path: GPU/host k-mers → postings → gapless verify.
    var hits = List[String]()
    if len(seed_kmers) > 0:
        for mer in seed_kmers:
            var found = index.lookup_kmer(mer)
            for h in found:
                hits.append(h.copy())
    else:
        hits = seed_hits(index, seq)
    var voted = _vote_and_verify(index, name, seq, hits)
    if voted.contig != "*":
        return voted^

    # Fixture / tiny-index only: exact substring (cap total bases).
    var total = index.total_bases()
    if total > 0 and total <= 2_000_000:
        for c in index.contigs:
            var off = _find_substr(c.seq, seq)
            if off >= 0:
                return LinearHit(
                    name, 0, c.name, off + 1, 60, String(qlen) + "M", seq, "*"
                )
        var rc = reverse_complement(seq)
        for c in index.contigs:
            var off2 = _find_substr(c.seq, rc)
            if off2 >= 0:
                return LinearHit(
                    name, 16, c.name, off2 + 1, 60, String(qlen) + "M", seq, "*"
                )
    return LinearHit(name, 4, "*", 0, 0, "*", seq, "*")^


def extend_read(index: LinearIndex, name: String, seq: String) raises -> LinearHit:
    """Map one read: exact substring, else k-mer vote + gapless verify."""
    var empty = List[String]()
    return extend_read_with_seeds(index, name, seq, empty)


struct PairedHits(Copyable, Movable):
    var r1: LinearHit
    var r2: LinearHit

    def __init__(out self, r1: LinearHit, r2: LinearHit):
        self.r1 = r1.copy()
        self.r2 = r2.copy()


def pair_hits(h1: LinearHit, h2: LinearHit) raises -> PairedHits:
    """Set PE flags / mate fields when both map to the same contig."""
    var a = h1.copy()
    var b = h2.copy()
    a.flag = a.flag | 1 | 64  # paired + read1
    b.flag = b.flag | 1 | 128  # paired + read2
    if a.contig != "*" and b.contig != "*" and a.contig == b.contig:
        a.flag = a.flag | 2
        b.flag = b.flag | 2
        a.rnext = "="
        b.rnext = "="
        a.pnext = b.pos
        b.pnext = a.pos
        var tlen = b.pos - a.pos
        if tlen < 0:
            tlen = -tlen
        tlen = tlen + b.seq.byte_length()
        a.tlen = tlen
        b.tlen = -tlen
    else:
        # SAM 0x8 = next segment in the template unmapped (set on the mate).
        if a.contig == "*":
            b.flag = b.flag | 8
        if b.contig == "*":
            a.flag = a.flag | 8
        if a.contig != "*":
            b.rnext = a.contig
            b.pnext = a.pos
        if b.contig != "*":
            a.rnext = b.contig
            a.pnext = b.pos
    return PairedHits(a^, b^)


def hit_to_sam_line(h: LinearHit) -> String:
    return (
        h.query_name
        + "\t"
        + String(h.flag)
        + "\t"
        + h.contig
        + "\t"
        + String(h.pos)
        + "\t"
        + String(h.mapq)
        + "\t"
        + h.cigar
        + "\t"
        + h.rnext
        + "\t"
        + String(h.pnext)
        + "\t"
        + String(h.tlen)
        + "\t"
        + h.seq
        + "\t"
        + h.qual
    )
