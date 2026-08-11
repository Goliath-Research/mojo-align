# Smoke probe: DeviceContext(api=cuda) + pack/hash seed kernels.
from std.collections import List

from gpu_kernels import probe_device_context, seed_kmers_on_device


def main() raises:
    print("probe", probe_device_context("nvidia"))
    var seqs = List[String]()
    seqs.append("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT")
    seqs.append("TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA")
    _ = seed_kmers_on_device("nvidia", seqs, 5)
    print("DeviceContext CUDA seed probe OK")
