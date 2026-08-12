# Portable device selection for Align seed kernels (gpu-common).

from std.collections import List
from std.python import Python

from gpu_kernels import (
    kernel_target_label,
    probe_device_context,
    seed_kmers_on_device,
)
from gpu_kmer import extract_kmers


comptime DEVICE_CPU = "cpu"
comptime DEVICE_NVIDIA = "nvidia"
comptime DEVICE_AMD = "amd"


def select_device(requested: String) raises -> String:
    var req = requested.lower()
    if req == "" or req == "auto":
        var os_mod = Python.import_module("os")
        # align_device is the dual-Align contract alias; giraffe_device remains supported.
        var env = String(os_mod.environ.get("METHYLGRAPHER_GIRAFFE_DEVICE", ""))
        if env == "":
            env = String(os_mod.environ.get("METHYLGRAPHER_ALIGN_DEVICE", "auto"))
        req = env.lower()
        if req == "":
            req = String("auto")
    if req == "auto":
        # Prefer a DeviceContext that actually opens (cuda then hip), then
        # fall back to presence of nvidia-smi / rocm-smi. Callers should
        # pass -device auto; do not hardcode nvidia|amd at the CLI.
        var cuda_probe = probe_device_context(String(DEVICE_NVIDIA))
        if cuda_probe.startswith("devicecontext-cuda"):
            return String(DEVICE_NVIDIA)
        var hip_probe = probe_device_context(String(DEVICE_AMD))
        if hip_probe.startswith("devicecontext-hip"):
            return String(DEVICE_AMD)
        var sp = Python.import_module("subprocess")
        var shutil = Python.import_module("shutil")
        if shutil.which("nvidia-smi") is not None:
            var r = sp.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            if String(r.returncode) == "0":
                return String(DEVICE_NVIDIA)
        if shutil.which("rocm-smi") is not None:
            var r2 = sp.run(
                ["rocm-smi", "--showproductname"], capture_output=True, text=True
            )
            if String(r2.returncode) == "0":
                return String(DEVICE_AMD)
        if shutil.which("rocminfo") is not None:
            var r3 = sp.run(["rocminfo"], capture_output=True, text=True)
            if String(r3.returncode) == "0":
                return String(DEVICE_AMD)
        return String(DEVICE_CPU)
    if req == "cuda":
        return String(DEVICE_NVIDIA)
    if req == "hip" or req == "rocm":
        return String(DEVICE_AMD)
    if req == "cpu" or req == "nvidia" or req == "amd":
        return req
    raise Error("unknown giraffe device: " + requested)


def require_device_or_raise(device: String) raises -> String:
    """Resolve device and optionally fail closed when GPU context is required."""
    var resolved = select_device(device)
    var os_mod = Python.import_module("os")
    var require = String(os_mod.environ.get("METHYLGRAPHER_GPU_REQUIRE", "")).lower()
    var backend = probe_device_context(resolved)
    print(
        "MojoGiraffe device=",
        resolved,
        " target=",
        kernel_target_label(resolved),
        " backend=",
        backend,
    )
    if resolved == DEVICE_CPU:
        return resolved
    if backend.startswith("devicecontext-cuda") or backend.startswith(
        "devicecontext-hip"
    ):
        return resolved
    # Driver <580 without ptxas → host-fallback. Fail closed when required or
    # when operator pins nvidia/amd explicitly with REQUIRE=1 (default on for
    # non-cpu pins so silent CPU map cannot masquerade as GPU Giraffe).
    if require == "" or require == "1" or require == "true" or require == "yes":
        if resolved == DEVICE_NVIDIA or resolved == DEVICE_AMD:
            raise Error(
                "align device="
                + resolved
                + " but DeviceContext backend="
                + backend
                + ". Set MODULAR_NVPTX_COMPILER_PATH to system ptxas "
                + "(NVIDIA driver <580) or upgrade driver ≥580; "
                + "or set METHYLGRAPHER_GPU_REQUIRE=0 to allow host fallback."
            )
    print("WARNING: GPU DeviceContext unavailable; continuing on host kernels")
    return resolved


def extract_kmers_batch_cpu(seqs: List[String], k: Int) raises -> List[List[String]]:
    var out = List[List[String]]()
    for s in seqs:
        out.append(extract_kmers(s, k))
    return out^


def extract_kmers_batch(
    device: String, seqs: List[String], k: Int
) raises -> List[List[String]]:
    """Seed k-mers via Mojo DeviceContext only (no CuPy / giraffe_gpu_minimizer)."""
    var resolved = select_device(device)
    if resolved == DEVICE_CPU:
        return extract_kmers_batch_cpu(seqs, k)
    try:
        return seed_kmers_on_device(resolved, seqs, k)
    except e:
        var os_mod = Python.import_module("os")
        var require = String(os_mod.environ.get("METHYLGRAPHER_GPU_REQUIRE", "")).lower()
        if require == "" or require == "1" or require == "true" or require == "yes":
            raise Error(
                "DeviceContext seed failed for device="
                + resolved
                + ": "
                + String(e)
                + " (CuPy fallback disabled on production path)"
            )
        print("DeviceContext seed failed; host k-mers (GPU_REQUIRE off): ", e)
        return extract_kmers_batch_cpu(seqs, k)
