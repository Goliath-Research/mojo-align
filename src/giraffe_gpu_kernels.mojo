# Portable GPU seed kernel surface for Mojo Giraffe (GBZ + GFA paths).
#
# Hot loop today: k-mer / minimizer extraction over read batches after GBZ
# segment decode. Mojo 1.0.0b2 (pixi pin) cannot import `gpu.host.DeviceContext`
# here, so acceleration uses:
#   - `giraffe_device.extract_kmers_batch` (CPU / nvidia / amd)
#   - `scripts/giraffe_gpu_minimizer.py` (CuPy when present, else NumPy)
#
# Buffy ≤2h dual-map remains an operator measurement after production caches
# exist for C2T/G2A GBZ (`scripts/build_mojo_gbz_cache.py`).
#
# Target flags when Modular GPU host lands:
#   --target-accelerator=nvidia:sm_90   # GH200 / Hopper
#   --target-accelerator=amdgpu:<arch>  # MI300 / ROCm bakeoff

from std.collections import List

from giraffe_seed import extract_kmers


comptime KERNEL_TARGET_NVIDIA_SM90 = "nvidia:sm_90"
comptime KERNEL_TARGET_AMDGPU = "amdgpu"


def kernel_target_label(device: String) raises -> String:
    var d = device.lower()
    if d == "nvidia" or d == "cuda":
        return String(KERNEL_TARGET_NVIDIA_SM90)
    if d == "amd" or d == "hip":
        return String(KERNEL_TARGET_AMDGPU)
    return String("cpu")


def seed_kmers_portable(seqs: List[String], k: Int) raises -> List[List[String]]:
    """CPU reference implementation of the GPU minimizer seed kernel."""
    var out = List[List[String]]()
    for s in seqs:
        out.append(extract_kmers(s, k))
    return out^
