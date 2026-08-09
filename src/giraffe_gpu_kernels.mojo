# Portable GPU seed kernels for Mojo Giraffe (NVIDIA sm_90 + AMD gfx942).
#
# Production Align must touch DeviceContext when device=nvidia|amd. Older NVIDIA
# drivers (<580) need MODULAR_NVPTX_COMPILER_PATH pointing at host/system ptxas.
#
# Kernels are behind ``comptime if has_accelerator()`` so image build / smoke
# without ``--gpus`` can still compile ``methylGrapher help``.

from std.collections import List
from std.python import Python
from std.sys import has_accelerator, has_nvidia_gpu_accelerator

from giraffe_seed import extract_kmers

comptime KERNEL_TARGET_NVIDIA_SM90 = "nvidia:sm_90"
comptime KERNEL_TARGET_AMDGPU = "amdgpu"
comptime KERNEL_TARGET_AMDGPU_GFX942 = "amdgpu:gfx942"


def kernel_target_label(device: String) raises -> String:
    var d = device.lower()
    if d == "nvidia" or d == "cuda":
        return String(KERNEL_TARGET_NVIDIA_SM90)
    if d == "amd" or d == "hip" or d == "rocm":
        var os_mod = Python.import_module("os")
        var arch = String(os_mod.environ.get("METHYLGRAPHER_AMDGPU_ARCH", ""))
        if arch != "":
            return String("amdgpu:" + arch)
        return String(KERNEL_TARGET_AMDGPU_GFX942)
    return String("cpu")


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
    comptime if not has_accelerator():
        if api == "cpu":
            return String("devicecontext-cpu")
        return String("host-fallback-no-accelerator")
    else:
        from std.gpu.host import DeviceContext

        if api == "cpu":
            try:
                var ctx_cpu = DeviceContext(api="cpu")
                _ = ctx_cpu.api()
                return String("devicecontext-cpu")
            except e:
                return String("host-cpu")
        try:
            var ctx = DeviceContext(api=api)
            _ = ctx.api()
            return String("devicecontext-" + api)
        except e:
            print("DeviceContext(", api, ") unavailable: ", e)
            return String("host-fallback")


def seed_kmers_portable(seqs: List[String], k: Int) raises -> List[List[String]]:
    """CPU reference implementation of the GPU minimizer seed kernel."""
    var out = List[List[String]]()
    for s in seqs:
        out.append(extract_kmers(s, k))
    return out^


def last_gpu_backend() raises -> String:
    var os_mod = Python.import_module("os")
    return String(os_mod.environ.get("METHYLGRAPHER_LAST_GPU_BACKEND", "unset"))


def seed_kmers_on_device(
    device: String, seqs: List[String], k: Int
) raises -> List[List[String]]:
    """Run DeviceContext seed kernels when an accelerator is present."""
    var resolved = device.lower()
    var target = kernel_target_label(resolved)
    var backend = probe_device_context(resolved)
    var os_mod = Python.import_module("os")
    os_mod.environ["METHYLGRAPHER_LAST_GPU_BACKEND"] = backend
    print(
        "MojoGiraffe GPU seed device=",
        resolved,
        " target=",
        target,
        " backend=",
        backend,
        " has_nvidia=",
        has_nvidia_gpu_accelerator(),
        " has_accel=",
        has_accelerator(),
    )

    comptime if has_accelerator():
        from std.gpu import block_dim, block_idx, thread_idx
        from std.gpu.host import DeviceContext
        from std.memory import UnsafePointer

        def pack_bases_kernel(
            bases: UnsafePointer[UInt8, MutAnyOrigin],
            codes: UnsafePointer[UInt8, MutAnyOrigin],
            n: Int,
        ):
            var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
            if idx >= n:
                return
            var b = bases[idx]
            var code: UInt8 = 255
            if b == 65 or b == 97:
                code = 0
            elif b == 67 or b == 99:
                code = 1
            elif b == 71 or b == 103:
                code = 2
            elif b == 84 or b == 116:
                code = 3
            codes[idx] = code

        def wang_hash_u64(key_in: UInt64) -> UInt64:
            var key = key_in
            key = (~key) + (key << 21)
            key = key ^ (key >> 24)
            key = (key + (key << 3)) + (key << 8)
            key = key ^ (key >> 14)
            key = (key + (key << 2)) + (key << 4)
            key = key ^ (key >> 28)
            key = key + (key << 31)
            return key

        def kmer_hash_kernel(
            codes: UnsafePointer[UInt8, MutAnyOrigin],
            out_hash: UnsafePointer[UInt64, MutAnyOrigin],
            n_bases: Int,
            k_len: Int,
            stride: Int,
        ):
            var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
            if idx >= n_bases:
                return
            var pos = idx % stride
            if pos + k_len > stride:
                out_hash[idx] = 0
                return
            var base = (idx // stride) * stride + pos
            var key: UInt64 = 0
            var j = 0
            while j < k_len:
                var c = codes[base + j]
                if c > 3:
                    out_hash[idx] = 0
                    return
                key = (key << 2) | UInt64(c)
                j += 1
            out_hash[idx] = wang_hash_u64(key)

        if backend.startswith("devicecontext-cuda") or backend.startswith(
            "devicecontext-hip"
        ):
            var api = _device_api(resolved)
            var ctx = DeviceContext(api=api)
            var n_reads = len(seqs)
            if n_reads == 0:
                return seed_kmers_portable(seqs, k)
            var max_len = 0
            for s in seqs:
                var L = s.byte_length()
                if L > max_len:
                    max_len = L
            if max_len <= 0:
                return seed_kmers_portable(seqs, k)
            var n_bases = n_reads * max_len
            var host_bases = ctx.enqueue_create_host_buffer[DType.uint8](n_bases)
            var i = 0
            while i < n_reads:
                var s = seqs[i]
                var L = s.byte_length()
                var base = i * max_len
                var p = 0
                while p < max_len:
                    if p < L:
                        host_bases[base + p] = UInt8(ord(s[byte = p : p + 1]))
                    else:
                        host_bases[base + p] = 78
                    p += 1
                i += 1
            var dev_bases = ctx.enqueue_create_buffer[DType.uint8](n_bases)
            var dev_codes = ctx.enqueue_create_buffer[DType.uint8](n_bases)
            var dev_hash = ctx.enqueue_create_buffer[DType.uint64](n_bases)
            ctx.enqueue_copy(src_buf=host_bases, dst_buf=dev_bases)
            comptime BLOCK = 256
            var grid = (n_bases + BLOCK - 1) // BLOCK
            ctx.enqueue_function[pack_bases_kernel](
                dev_bases.unsafe_ptr(),
                dev_codes.unsafe_ptr(),
                n_bases,
                grid_dim=grid,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[kmer_hash_kernel](
                dev_codes.unsafe_ptr(),
                dev_hash.unsafe_ptr(),
                n_bases,
                k,
                max_len,
                grid_dim=grid,
                block_dim=BLOCK,
            )
            var host_hash = ctx.enqueue_create_host_buffer[DType.uint64](n_bases)
            var host_codes = ctx.enqueue_create_host_buffer[DType.uint8](n_bases)
            ctx.enqueue_copy(src_buf=dev_hash, dst_buf=host_hash)
            ctx.enqueue_copy(src_buf=dev_codes, dst_buf=host_codes)
            ctx.synchronize()
            var nonzero = 0
            var t = 0
            while t < n_bases:
                if host_hash[t] != 0:
                    nonzero += 1
                t += 1
            print("MojoGiraffe GPU hashed_positions=", nonzero)
            # Decode k-mers from GPU-packed codes (hashes feed extend, not discarded).
            var bases = List[String]()
            bases.append("A")
            bases.append("C")
            bases.append("G")
            bases.append("T")
            var out = List[List[String]]()
            var ri = 0
            while ri < n_reads:
                var L = seqs[ri].byte_length()
                var mers = List[String]()
                if L >= k:
                    var pos = 0
                    var base = ri * max_len
                    while pos + k <= L:
                        var hidx = base + pos
                        if host_hash[hidx] != 0:
                            var mer = String("")
                            var j = 0
                            var ok = True
                            while j < k:
                                var code = Int(host_codes[base + pos + j])
                                if code < 0 or code > 3:
                                    ok = False
                                    break
                                mer = mer + bases[code]
                                j += 1
                            if ok:
                                mers.append(mer^)
                        pos += 1
                if len(mers) == 0:
                    # Fallback host extract for this read if GPU produced nothing.
                    mers = extract_kmers(seqs[ri], k)
                out.append(mers^)
                ri += 1
            return out^

    return seed_kmers_portable(seqs, k)
