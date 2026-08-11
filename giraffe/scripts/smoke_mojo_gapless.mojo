# Smoke: native Mojo gapless over toy dense pack.

from std.collections import List
from std.sys import exit

from giraffe_gapless import gapless_extend_native


def main() raises:
    var pack = String("/work/cache/mojo_segments/toy.giraffe.gbz.mojo_segments")
    var seeds = List[String]()
    seeds.append("1:0:0")
    var hits = gapless_extend_native(pack, "readA", "ACGTACGTAC", seeds)
    print("native_gapless_hits=", len(hits))
    if len(hits) > 0:
        print("path=", hits[0].path, " mapq=", hits[0].mapq, " cs=", hits[0].cs_tag)
    exit(0)
