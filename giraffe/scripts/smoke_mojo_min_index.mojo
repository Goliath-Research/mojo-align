from std.collections import List
from std.sys import exit
from std.python import Python

from giraffe_min_index import MojoMinIndex
from giraffe_minimizer import MinimizerOcc, minimizers_of_seq


def main() raises:
    var toy = "/home/ubuntu/methylGrapher-mojo/tests/data/giraffe_fixture/gbz_toy/toy.wl.C2T.shortread.withzip.min"
    var idx = MojoMinIndex(toy)
    print("toy k=", idx.k, " w=", idx.w, " cells=", idx.cell_count)
    idx.close()

    var prod = "/lambda/nfs/Work/genomes/pangenome/GRCh38/d9-bs/1.70/hprc-d9-bs.wl.C2T.shortread.withzip.min"
    var os_mod = Python.import_module("os")
    if not Bool(os_mod.path.isfile(prod)):
        print("PASS (no prod min)")
        exit(0)

    var pidx = MojoMinIndex(prod)
    print("prod k=", pidx.k, " cells=", pidx.cell_count)
    # Parity: pick first unique cell key from HT via Python oracle, Mojo find_offset.
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
    var minmod = Python.import_module("engine.minimizer_index")
    var py = minmod.MinimizerIndex(prod)
    # Scan a few cells for a real key
    var found_key = UInt64(0)
    var i = 0
    while i < 10000:
        var words = py._cell_words(i * Int(py=py.cell_size))
        var ck = Int(py=words[0])
        var bare = ck & 0x7FFFFFFFFFFFFFFF
        if bare != 0x7FFFFFFFFFFFFFFF and (ck & (1 << 63)) == 0:
            found_key = UInt64(bare)
            break
        i += 1
    if found_key == 0:
        print("WARN: no unique key in first 10k cells")
    else:
        var off = pidx.find_offset(found_key)
        var py_off = py.find_offset(Int(found_key))
        print("key=", Int(found_key), " mojo_off=", off, " py_off=", py_off)
        if py_off is None:
            print("FAIL: python miss")
            exit(1)
        if off != Int(py=py_off):
            print("FAIL: offset mismatch")
            exit(1)
        var hits = pidx.hits_at(off, 24)
        var py_hits = py.find(Int(found_key))
        print("mojo_hits=", len(hits), " py_hits=", Int(py=py_hits.__len__()))
        if len(hits) < 1:
            print("FAIL: mojo no hits")
            exit(1)
    py.close()
    pidx.close()
    print("PASS")
    exit(0)
