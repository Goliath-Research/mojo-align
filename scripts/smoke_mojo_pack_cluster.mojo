# Smoke: Mojo-native dense pack get + cluster (no Python SegmentPack.get / DistIndex).

from std.collections import List
from std.sys import exit

from giraffe_dist import cluster_seed_hits
from giraffe_pack import DensePack


def main() raises:
    var pack_dir = String("/work/cache/mojo_segments/toy.giraffe.gbz.mojo_segments")
    var pack = DensePack(pack_dir)
    var seq = pack.get(1)
    print("pack_get_len=", seq.byte_length(), " seq=", seq)
    if seq != "ACGTACGTAC":
        print("FAIL pack get mismatch")
        exit(1)

    var hits = List[String]()
    hits.append("1000:0:0")
    hits.append("1001:0:1")
    hits.append("2000:0:0")
    var clustered = cluster_seed_hits(hits, "/tmp/fake.dist", "")
    print("cluster_n=", len(clustered))
    if len(clustered) < 2:
        print("FAIL cluster pruned/kept densest bucket")
        exit(1)
    print("PASS mojo_pack_cluster")
    exit(0)
