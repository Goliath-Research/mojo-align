# Portable device selection for Mojo Giraffe seed kernels.

from std.collections import List
from std.python import Python

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


def extract_kmers_batch_cpu(seqs: List[String], k: Int) raises -> List[List[String]]:
    var out = List[List[String]]()
    for s in seqs:
        out.append(extract_kmers(s, k))
    return out^


def extract_kmers_batch(
    device: String, seqs: List[String], k: Int
) raises -> List[List[String]]:
    if device == DEVICE_CPU:
        return extract_kmers_batch_cpu(seqs, k)

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
        var py_out = helper.extract_kmers_batch(py_seqs, k, device)
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
    except e:
        print("GPU minimizer helper unavailable; falling back to CPU")
        return extract_kmers_batch_cpu(seqs, k)
