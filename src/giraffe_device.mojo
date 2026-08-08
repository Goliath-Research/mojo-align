# Portable device selection for Mojo Giraffe seed kernels.

from std.collections import List
from std.python import Python

from giraffe_gpu_kernels import (
    kernel_target_label,
    probe_device_context,
    seed_kmers_on_device,
)
from giraffe_seed import extract_kmers


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
        # Probe with shutil.which first — subprocess.run raises FileNotFoundError
        # when the binary is missing (common inside Docker without --gpus).
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
    """Seed k-mers: DeviceContext GPU path first, then CuPy helper, then CPU."""
    var resolved = select_device(device)
    if resolved == DEVICE_CPU:
        return extract_kmers_batch_cpu(seqs, k)

    # Preferred: native Mojo DeviceContext kernels (nvidia:sm_90 / amdgpu).
    try:
        return seed_kmers_on_device(resolved, seqs, k)
    except e:
        print("DeviceContext seed failed; trying Python GPU helper: ", e)

    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    var candidates = List[String]()
    candidates.append(String(os_mod.getcwd()) + "/scripts")
    candidates.append("/opt/methylgrapher-mojo/scripts")
    candidates.append("/home/ubuntu/methylGrapher-mojo/scripts")
    for c in candidates:
        sys_mod.path.insert(0, c)
    try:
        var helper = Python.import_module("giraffe_gpu_minimizer")
        var py_seqs = Python.list()
        for s in seqs:
            py_seqs.append(s)
        var py_out = helper.extract_kmers_batch(py_seqs, k, resolved)
        var out = List[List[String]]()
        var n = Int(py=py_out.__len__())
        var i = 0
        while i < n:
            var row = py_out[i]
            var mers = List[String]()
            var m = Int(py=row.__len__())
            var j = 0
            while j < m:
                mers.append(String(row[j]))
                j += 1
            out.append(mers^)
            i += 1
        return out^
    except e2:
        print("GPU minimizer helper unavailable; falling back to CPU: ", e2)
        return extract_kmers_batch_cpu(seqs, k)
