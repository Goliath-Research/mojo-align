# Portable GPU seed surface for MojoFq2bamMeth (NVIDIA + AMD ROCm).
#
# One Mojo source; DeviceContext selects api="cuda" | "hip" | "cpu".
# When the host driver is too old for Modular's NVPTX stack (or HIP is
# absent), we fall back to host k-mer extraction while still logging the
# intended kernel target (nvidia:sm_90 / amdgpu:gfx942).

from std.collections import List
from std.gpu.host import DeviceContext

from giraffe_device import extract_kmers_batch, select_device
from giraffe_gpu_kernels import kernel_target_label
from linear_seed import extract_kmers


def linear_kernel_target(device: String) raises -> String:
    return kernel_target_label(device)


def _device_api(device: String) -> String:
    var d = device.lower()
    if d == "nvidia" or d == "cuda":
        return String("cuda")
    if d == "amd" or d == "hip" or d == "rocm":
        return String("hip")
    return String("cpu")


def probe_device_context(device: String) raises -> String:
    """Try DeviceContext for the selected API; return backend label used."""
    var api = _device_api(device)
    try:
        if api == "cpu":
            var ctx_cpu = DeviceContext(api="cpu")
            _ = ctx_cpu.api()
            return String("devicecontext-cpu")
        var ctx = DeviceContext(api=api)
        _ = ctx.api()
        return String("devicecontext-" + api)
    except e:
        return String("host-fallback")


def seed_kmers_portable(
    device: String, seqs: List[String], k: Int
) raises -> List[List[String]]:
    """Seed k-mers on GPU when DeviceContext works; else host / helper path."""
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
        return extract_kmers_batch(resolved, seqs, k)

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
