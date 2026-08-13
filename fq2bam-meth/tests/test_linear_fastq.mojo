# Unit test for Mojo bulk FASTQ (plain + pigz, C2T/G2A, qname strip).

from std.collections import List
from std.python import Python
from std.sys import exit

from linear_fastq import (
    FastqArena,
    fq_close,
    fq_export_ptrs,
    fq_open,
    fq_pack_bases,
    fq_read_batch,
    fq_reserve,
)


def main() raises:
    var builtins = Python.import_module("builtins")
    var gzip = Python.import_module("gzip")
    var os_mod = Python.import_module("os")
    var d = String("/tmp/mojo_fq_test")
    try:
        _ = os_mod.makedirs(d)
    except:
        pass
    var r1 = d + "/r1.fq"
    var r2 = d + "/r2.fq"
    var r1gz = d + "/r1.fq.gz"
    var f1 = builtins.open(r1, "w")
    f1.write("@read1 extra/1\nACGTACGT\n+\nIIIIIIII\n@read2\nCCCC\n+\nJJJJ\n")
    f1.close()
    var f2 = builtins.open(r2, "w")
    f2.write("@read1 extra/2\nGGTAGGTA\n+\nKKKKKKKK\n@read2\nGGGG\n+\nLLLL\n")
    f2.close()
    var gzf = gzip.open(r1gz, "wb")
    var inf = builtins.open(r1, "rb")
    gzf.write(inf.read())
    inf.close()
    gzf.close()

    var s = fq_open(r1, r2, String("C2T"), String("G2A"))
    var a = FastqArena()
    fq_reserve(a, 8, True)
    fq_read_batch(s, a, 8)
    fq_close(s)
    if a.n1 != 2:
        raise Error("expected 2 pairs got " + String(a.n1))
    if a.max_len != 8:
        raise Error("max_len want 8 got " + String(a.max_len))
    var base = Int(a.data.unsafe_ptr())
    # name1[0] == read1
    if a.name1_len[0] != 5:
        raise Error("name1 len")
    if Int(a.data[a.name1_off[0]]) != 114:
        raise Error("name1[0] not 'r'")
    # orig1[0] ACGTACGT, seq C2T -> ATGTATGT
    if a.orig1_len[0] != 8:
        raise Error("orig1 len")
    if Int(a.data[a.orig1_off[0] + 1]) != 67:
        raise Error("orig C")
    if Int(a.data[a.seq1_off[0] + 1]) != 84:
        raise Error("C2T seq T")
    # orig2 GGTAGGTA -> G2A AATAAATA
    if Int(a.data[a.orig2_off[0]]) != 71:
        raise Error("orig2 G")
    if Int(a.data[a.seq2_off[0]]) != 65:
        raise Error("G2A seq A")
    # second read CCCC -> TTTT
    if Int(a.data[a.seq1_off[1]]) != 84:
        raise Error("read2 C2T")
    if a.seq1_len[1] != 4:
        raise Error("read2 len")

    var dest = List[UInt8](length=32, fill=UInt8(0))
    var lens = List[UInt32](length=4, fill=UInt32(0))
    fq_pack_bases(a, True, Int(dest.unsafe_ptr()), 8, Int(lens.unsafe_ptr()))
    if Int(lens[0]) != 8:
        raise Error("lens0")
    if Int(dest[1]) != 84:
        raise Error("packed T")
    if Int(dest[12]) != 78:
        raise Error("pad N for read2 stride")
    if Int(dest[16]) != 65:
        raise Error("packed R2 A")

    var na = List[UInt64](length=2, fill=UInt64(0))
    var nn = List[UInt32](length=2, fill=UInt32(0))
    var oa = List[UInt64](length=2, fill=UInt64(0))
    var on = List[UInt32](length=2, fill=UInt32(0))
    var qa = List[UInt64](length=2, fill=UInt64(0))
    var qn = List[UInt32](length=2, fill=UInt32(0))
    var n2a = List[UInt64](length=2, fill=UInt64(0))
    var n2n = List[UInt32](length=2, fill=UInt32(0))
    var o2a = List[UInt64](length=2, fill=UInt64(0))
    var o2n = List[UInt32](length=2, fill=UInt32(0))
    var q2a = List[UInt64](length=2, fill=UInt64(0))
    var q2n = List[UInt32](length=2, fill=UInt32(0))
    fq_export_ptrs(
        a,
        True,
        Int(na.unsafe_ptr()),
        Int(nn.unsafe_ptr()),
        Int(oa.unsafe_ptr()),
        Int(on.unsafe_ptr()),
        Int(qa.unsafe_ptr()),
        Int(qn.unsafe_ptr()),
        Int(n2a.unsafe_ptr()),
        Int(n2n.unsafe_ptr()),
        Int(o2a.unsafe_ptr()),
        Int(o2n.unsafe_ptr()),
        Int(q2a.unsafe_ptr()),
        Int(q2n.unsafe_ptr()),
    )
    if Int(nn[0]) != 5:
        raise Error("export name len")
    if oa[0] != UInt64(base + a.orig1_off[0]):
        raise Error("export orig ptr")

    var sg = fq_open(r1gz, String(""), String("C2T"), String(""))
    var ag = FastqArena()
    fq_read_batch(sg, ag, 8)
    fq_close(sg)
    if ag.n1 != 2:
        raise Error("gzip n1")
    if Int(ag.data[ag.seq1_off[0] + 1]) != 84:
        raise Error("gzip C2T")

    var eof = fq_open(r1, String(""), String(""), String(""))
    var ae = FastqArena()
    fq_read_batch(eof, ae, 1000)
    fq_read_batch(eof, ae, 8)
    fq_close(eof)
    if ae.n1 != 0:
        raise Error("eof batch should be empty")

    print("OK linear_fastq n1=", a.n1, " max_len=", a.max_len)
    exit(0)
