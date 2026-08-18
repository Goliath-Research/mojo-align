# Smoke: Mojo Giraffe minimizers match Python keys; DeviceContext path when GPU.

from std.collections import List
from std.sys import exit
from std.python import Python

from giraffe_minimizer import minimizers_batch, minimizers_of_seq
from mojo_align_env import ensure_python_path, giraffe_fixture_root


def main() raises:
    var os_mod = Python.import_module("os")
    os_mod.environ["METHYLGRAPHER_GPU_REQUIRE"] = "0"
    var seq = String("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT")
    var k = 29
    var w = 11
    var host = minimizers_of_seq(seq, k, w)
    print("mojo_host_minimizers n=", len(host))
    if len(host) < 1:
        print("FAIL: expected host minimizers")
        exit(1)

    ensure_python_path()
    var minmod = Python.import_module("engine.minimizer_index")
    var toy = giraffe_fixture_root() + "/gbz_toy/toy.wl.C2T.shortread.withzip.min"
    var idx = minmod.MinimizerIndex(toy)
    var py = idx.minimizers(seq)
    idx.close()
    var n_py = Int(py=py.__len__())
    if n_py != len(host):
        print("FAIL: count mismatch mojo=", len(host), " py=", n_py)
        exit(1)
    var i = 0
    while i < len(host):
        var pk = Int(py=py[i].key)
        var mk = Int(host[i].key)
        if pk != mk:
            print("FAIL: key mismatch i=", i, " mojo=", mk, " py=", pk)
            exit(1)
        i += 1
    print("parity_keys_ok n=", len(host))

    var seqs = List[String]()
    seqs.append(seq)
    seqs.append(seq)
    var batch = minimizers_batch("cpu", seqs, k, w)
    print("batch_backend=", batch.backend, " n0=", len(batch.occs[0]))
    if batch.backend.find("mojo_min") < 0:
        print("FAIL: unexpected backend")
        exit(1)
    if batch.backend.find("cupy") >= 0 or batch.backend.find("host-nvidia") >= 0:
        print("FAIL: CuPy/host-nvidia path must not appear")
        exit(1)
    print("PASS")
    exit(0)
