# Smoke: Mojo stream map on toy GBZ fixture (no Python quartet_map hot loop).

from std.collections import List
from std.sys import exit
from std.python import Python

from giraffe_gbz import map_gbz_native


def main() raises:
    var root = "/home/ubuntu/methylGrapher-mojo/tests/data/giraffe_fixture"
    var gbz_toy = root + "/gbz_toy"
    var os_mod = Python.import_module("os")
    var tmp = Python.import_module("tempfile")
    var td = String(tmp.mkdtemp(prefix="mojo_stream_"))
    os_mod.environ["METHYLGRAPHER_MOJO_SEGMENTS_CACHE"] = td
    os_mod.environ["METHYLGRAPHER_GPU_REQUIRE"] = "0"
    os_mod.environ["METHYLGRAPHER_PROFILE_STAGES"] = "1"
    os_mod.environ["METHYLGRAPHER_MOJO_READ_BATCH"] = "8"

    var out_gaf = td + "/out.gaf"
    var n = map_gbz_native(
        gbz_toy + "/toy.wl.C2T.giraffe.gbz",
        root + "/R1.fastq",
        out_gaf,
        root + "/R2.fastq",
        gbz_toy + "/toy.wl.C2T.dist",
        gbz_toy + "/toy.wl.C2T.shortread.withzip.min",
        gbz_toy + "/toy.wl.C2T.shortread.zipcodes",
        5,
        "cpu",
    )
    print("smoke_mojo_stream_map records=", n, " gaf=", out_gaf)
    var builtins = Python.import_module("builtins")
    var fh = builtins.open(out_gaf, "r")
    var body = String(fh.read())
    fh.close()
    print("gaf_bytes=", body.byte_length())
    if n < 1:
        print("FAIL: expected mapped records")
        exit(1)
    if body.byte_length() < 10:
        print("FAIL: empty GAF")
        exit(1)
    # Must not have gone through Python quartet_map banner
    var seed = String(os_mod.environ.get("METHYLGRAPHER_LAST_SEED_BACKEND", ""))
    print("seed_backend=", seed)
    if seed.find("mojo_min_mmap") < 0:
        print("FAIL: expected mojo_min_mmap seed backend, got ", seed)
        exit(1)
    if seed.find("host-nvidia-fallback") >= 0 or seed.find("cupy-") >= 0:
        print("FAIL: CuPy/host-nvidia-fallback must not be production seed backend")
        exit(1)
    print("PASS")
    exit(0)
