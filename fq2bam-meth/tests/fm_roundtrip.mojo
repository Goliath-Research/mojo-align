# Unique k-mer BWT+SA round-trip on fleet bwameth.c2t FM-index.

from linear_fm_index import FmIndex, fm_prefix_from_ref


def main() raises:
    var ref_fa = String("/work/genomes/linear/GRCh38/ensembl-114/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    var prefix = fm_prefix_from_ref(ref_fa)
    var index = FmIndex()
    index.load(prefix)

    var rid = -1
    var ci = 0
    while ci < index.contig_count():
        if index.contig_name(ci) == "f1":
            rid = ci
            break
        ci += 1
    if rid < 0:
        raise Error("contig f1 not found in .ann")
    var pos0 = index.contig_offset(rid) + 1000000
    comptime KLEN = 32
    var codes = Array[UInt8, KLEN](fill=UInt8(0))
    var i = 0
    while i < KLEN:
        var b = index.pac_base(pos0 + i)
        if b > 3:
            raise Error("N in pac")
        codes[i] = UInt8(b)
        i += 1

    var m = index.match_exact(codes.unsafe_ptr(), KLEN)
    if m.occ != 1:
        raise Error("expected unique match occ=1 got " + String(m.occ))
    var sa_pos = index.sa_at(m.k)
    var dep = index.depos(sa_pos)
    if dep.is_rev != 0:
        raise Error("expected forward depos")
    if dep.pos != pos0:
        raise Error(
            "SA round-trip mismatch: want "
            + String(pos0)
            + " got "
            + String(dep.pos)
        )
    print(
        "OK fm round-trip pos=",
        pos0,
        " sa=",
        Int(sa_pos),
        " occ=",
        m.occ,
        " prefix=",
        prefix,
    )
