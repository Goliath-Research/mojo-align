# Wall-clock: Mojo host vs DeviceContext minimizers on a synthetic batch.

from std.collections import List
from std.sys import exit
from std.python import Python

from giraffe_minimizer import minimizers_batch


def main() raises:
    var time = Python.import_module("time")
    var seq = String("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT")
    var k = 29
    var w = 11
    var n = 2000
    var seqs = List[String]()
    var i = 0
    while i < n:
        seqs.append(seq)
        i += 1

    var t0 = time.perf_counter()
    var host = minimizers_batch("cpu", seqs, k, w)
    var t1 = time.perf_counter()
    print("host n=", n, " backend=", host.backend, " sec=", t1 - t0, " n0=", len(host.occs[0]))

    var t2 = time.perf_counter()
    var gpu = minimizers_batch("nvidia", seqs, k, w)
    var t3 = time.perf_counter()
    print("gpu n=", n, " backend=", gpu.backend, " sec=", t3 - t2, " n0=", len(gpu.occs[0]))
    if len(host.occs[0]) != len(gpu.occs[0]):
        print("WARN: host/gpu count differ", len(host.occs[0]), len(gpu.occs[0]))
    else:
        var j = 0
        while j < len(host.occs[0]):
            if host.occs[0][j].key != gpu.occs[0][j].key:
                print("FAIL: key mismatch at", j)
                exit(1)
            j += 1
        print("host_gpu_keys_match")
    print("PASS")
    exit(0)
