# Portable GPU seed surface for MojoFq2bamMeth (NVIDIA + AMD ROCm).
#
# One Mojo source; DeviceContext selects api="cuda" | "hip" | "cpu".
# Shares pack/hash kernels with Mojo Giraffe (giraffe_gpu_kernels).

from std.collections import List

from giraffe_device import extract_kmers_batch, select_device
from giraffe_gpu_kernels import (
    kernel_target_label,
    probe_device_context,
    seed_kmers_on_device,
)
from linear_seed import extract_kmers


def linear_kernel_target(device: String) raises -> String:
    return kernel_target_label(device)


def probe_device_context_linear(device: String) raises -> String:
    return probe_device_context(device)


def seed_kmers_portable(
    device: String, seqs: List[String], k: Int
) raises -> List[List[String]]:
    """Seed k-mers on GPU when DeviceContext works; else host extract."""
    var resolved = select_device(device)
    var target = linear_kernel_target(resolved)
    var backend = probe_device_context(resolved)
    print(
        "MojoLinear device=",
        resolved,
        " target=",
        target,
        " backend=",
        backend,
    )

    if backend.startswith("devicecontext-cuda") or backend.startswith(
        "devicecontext-hip"
    ):
        return seed_kmers_on_device(resolved, seqs, k)

    if resolved == "cpu" or backend.startswith("devicecontext-cpu"):
        var out = List[List[String]]()
        for s in seqs:
            out.append(extract_kmers(s, k))
        return out^

    try:
        return extract_kmers_batch(resolved, seqs, k)
    except e:
        print("linear GPU helper unavailable; CPU seeds: ", e)
        var out2 = List[List[String]]()
        for s in seqs:
            out2.append(extract_kmers(s, k))
        return out2^
