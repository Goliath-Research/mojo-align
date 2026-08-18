# Native Mojo Giraffe (k,w) minimizers — DeviceContext on NVIDIA/AMD, host Mojo otherwise.
# Replaces Python/CuPy ``giraffe_gpu_minimizer`` on the production stream-map hot path.

from std.collections import List
from std.python import Python
from std.sys import has_accelerator

from giraffe_gpu_kernels import (
    kernel_target_label,
    probe_device_context,
)


struct MinimizerOcc(Copyable, Movable):
    var key: UInt64
    var hash: UInt64
    var offset: Int
    var is_reverse: Bool

    def __init__(
        out self, key: UInt64, hash: UInt64, offset: Int, is_reverse: Bool
    ):
        self.key = key
        self.hash = hash
        self.offset = offset
        self.is_reverse = is_reverse


struct MinimizerBatchResult(Copyable, Movable):
    var occs: List[List[MinimizerOcc]]
    var backend: String

    def __init__(out self, var occs: List[List[MinimizerOcc]], backend: String):
        self.occs = occs^
        self.backend = backend


def wang_hash_64(key_in: UInt64) -> UInt64:
    var key = key_in
    key = (~key) + (key << 21)
    key = key ^ (key >> 24)
    key = (key + (key << 3)) + (key << 8)
    key = key ^ (key >> 14)
    key = (key + (key << 2)) + (key << 4)
    key = key ^ (key >> 28)
    key = key + (key << 31)
    return key


def pack_base(b: UInt8) -> UInt8:
    if b == 65 or b == 97:
        return 0
    if b == 67 or b == 99:
        return 1
    if b == 71 or b == 103:
        return 2
    if b == 84 or b == 116:
        return 3
    return 255


def minimizers_of_seq(seq: String, k: Int, w: Int) raises -> List[MinimizerOcc]:
    """Giraffe-style forward-window minimizers (RC-aware Key64 / wang hash)."""
    var out = List[MinimizerOcc]()
    var n = seq.byte_length()
    var win = k + w - 1
    if n < win or k < 1 or w < 1:
        return out^

    var key_mask = (UInt64(1) << UInt64(2 * k)) - 1
    var next_read_offset = 0
    var last_hash = UInt64(0)
    var last_offset = -1
    var have_last = False

    var window_start = 0
    while window_start <= n - win:
        var best_key = UInt64(0)
        var best_hash = UInt64(0)
        var best_off = 0
        var best_rev = False
        var have_best = False
        var ok = True
        var i = 0
        while i < w:
            var pos = window_start + i
            var fk = UInt64(0)
            var rk = UInt64(0)
            var j = 0
            while j < k:
                var code = pack_base(UInt8(ord(seq[byte = pos + j : pos + j + 1])))
                if code > 3:
                    ok = False
                    break
                fk = ((fk << 2) | UInt64(code)) & key_mask
                j += 1
            if not ok:
                break
            j = k - 1
            while j >= 0:
                var code_r = pack_base(UInt8(ord(seq[byte = pos + j : pos + j + 1])))
                rk = ((rk << 2) | (UInt64(code_r) ^ 3)) & key_mask
                j -= 1
            var fh = wang_hash_64(fk)
            var rh = wang_hash_64(rk)
            var cand_key = fk
            var cand_hash = fh
            var cand_rev = False
            if rh < fh:
                cand_key = rk
                cand_hash = rh
                cand_rev = True
            if (
                not have_best
                or cand_hash < best_hash
                or (cand_hash == best_hash and pos < best_off)
            ):
                best_key = cand_key
                best_hash = cand_hash
                best_off = pos
                best_rev = cand_rev
                have_best = True
            i += 1
        if ok and have_best:
            var emit = False
            if not have_last:
                emit = True
            elif last_hash == best_hash or last_offset < best_off:
                if best_off >= next_read_offset:
                    emit = True
            if emit:
                var off = best_off
                if best_rev:
                    off = best_off + k - 1
                out.append(MinimizerOcc(best_key, best_hash, off, best_rev))
                next_read_offset = best_off + 1
                last_hash = best_hash
                last_offset = best_off
                have_last = True
        window_start += 1
    return out^


def minimizers_batch_host(
    seqs: List[String], k: Int, w: Int
) raises -> List[List[MinimizerOcc]]:
    var out = List[List[MinimizerOcc]]()
    for s in seqs:
        out.append(minimizers_of_seq(s, k, w))
    return out^


def _window_reduce_from_tables(
    n: Int,
    k: Int,
    w: Int,
    keys_f: List[UInt64],
    keys_r: List[UInt64],
    hashes_f: List[UInt64],
    hashes_r: List[UInt64],
    valid: List[Bool],
) raises -> List[MinimizerOcc]:
    """Host window argmin over per-position forward/RC key+hash tables."""
    var out = List[MinimizerOcc]()
    var win = k + w - 1
    if n < win:
        return out^
    var next_read_offset = 0
    var last_hash = UInt64(0)
    var last_offset = -1
    var have_last = False
    var window_start = 0
    while window_start <= n - win:
        var best_key = UInt64(0)
        var best_hash = UInt64(0)
        var best_off = 0
        var best_rev = False
        var have_best = False
        var ok = True
        var i = 0
        while i < w:
            var pos = window_start + i
            if not valid[pos]:
                ok = False
                break
            var use_r = hashes_r[pos] < hashes_f[pos]
            var cand_key = keys_f[pos]
            var cand_hash = hashes_f[pos]
            var cand_rev = False
            if use_r:
                cand_key = keys_r[pos]
                cand_hash = hashes_r[pos]
                cand_rev = True
            if (
                not have_best
                or cand_hash < best_hash
                or (cand_hash == best_hash and pos < best_off)
            ):
                best_key = cand_key
                best_hash = cand_hash
                best_off = pos
                best_rev = cand_rev
                have_best = True
            i += 1
        if ok and have_best:
            var emit = False
            if not have_last:
                emit = True
            elif last_hash == best_hash or last_offset < best_off:
                if best_off >= next_read_offset:
                    emit = True
            if emit:
                var off = best_off
                if best_rev:
                    off = best_off + k - 1
                out.append(MinimizerOcc(best_key, best_hash, off, best_rev))
                next_read_offset = best_off + 1
                last_hash = best_hash
                last_offset = best_off
                have_last = True
        window_start += 1
    return out^


def minimizers_batch_devicecontext(
    device: String, seqs: List[String], k: Int, w: Int
) raises -> MinimizerBatchResult:
    """DeviceContext pack+hash (fwd/RC); Mojo host does Giraffe window reduction."""
    var resolved = device.lower()
    var backend = probe_device_context(resolved)
    var target = kernel_target_label(resolved)
    var os_mod = Python.import_module("os")
    var banner_key = "MOJO_ALIGN_MIN_BANNER"
    if String(os_mod.environ.get(banner_key, "")) != backend:
        os_mod.environ[banner_key] = backend
        print(
            "mojo_min DeviceContext device=",
            resolved,
            " target=",
            target,
            " backend=",
            backend,
            " k=",
            k,
            " w=",
            w,
            flush=True,
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

        def kmer_fwd_rc_kernel(
            codes: UnsafePointer[UInt8, MutAnyOrigin],
            keys_f: UnsafePointer[UInt64, MutAnyOrigin],
            keys_r: UnsafePointer[UInt64, MutAnyOrigin],
            hashes_f: UnsafePointer[UInt64, MutAnyOrigin],
            hashes_r: UnsafePointer[UInt64, MutAnyOrigin],
            valid: UnsafePointer[UInt8, MutAnyOrigin],
            n_bases: Int,
            k_len: Int,
            stride: Int,
        ):
            var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
            if idx >= n_bases:
                return
            var pos = idx % stride
            if pos + k_len > stride:
                keys_f[idx] = 0
                keys_r[idx] = 0
                hashes_f[idx] = 0
                hashes_r[idx] = 0
                valid[idx] = 0
                return
            var base = (idx // stride) * stride + pos
            var fk: UInt64 = 0
            var rk: UInt64 = 0
            var j = 0
            while j < k_len:
                var c = codes[base + j]
                if c > 3:
                    keys_f[idx] = 0
                    keys_r[idx] = 0
                    hashes_f[idx] = 0
                    hashes_r[idx] = 0
                    valid[idx] = 0
                    return
                fk = (fk << 2) | UInt64(c)
                j += 1
            j = 0
            while j < k_len:
                var c2 = codes[base + (k_len - 1 - j)]
                rk = (rk << 2) | (UInt64(c2) ^ 3)
                j += 1
            # Inline wang hash (device kernel cannot call host fn).
            var key = fk
            key = (~key) + (key << 21)
            key = key ^ (key >> 24)
            key = (key + (key << 3)) + (key << 8)
            key = key ^ (key >> 14)
            key = (key + (key << 2)) + (key << 4)
            key = key ^ (key >> 28)
            key = key + (key << 31)
            var hf = key
            key = rk
            key = (~key) + (key << 21)
            key = key ^ (key >> 24)
            key = (key + (key << 3)) + (key << 8)
            key = key ^ (key >> 14)
            key = (key + (key << 2)) + (key << 4)
            key = key ^ (key >> 28)
            key = key + (key << 31)
            keys_f[idx] = fk
            keys_r[idx] = rk
            hashes_f[idx] = hf
            hashes_r[idx] = key
            valid[idx] = 1

        if backend.startswith("devicecontext-cuda") or backend.startswith(
            "devicecontext-hip"
        ):
            var api = String("cuda")
            if resolved == "amd" or resolved == "hip" or resolved == "rocm":
                api = String("hip")
            var ctx = DeviceContext(api=api)
            var n_reads = len(seqs)
            if n_reads == 0:
                return MinimizerBatchResult(
                    List[List[MinimizerOcc]](), backend + "+mojo_min"
                )
            var max_len = 0
            for s in seqs:
                var L = s.byte_length()
                if L > max_len:
                    max_len = L
            if max_len < k:
                return MinimizerBatchResult(
                    minimizers_batch_host(seqs, k, w), backend + "+mojo_min_host"
                )

            var n_bases = n_reads * max_len
            var host_bases = ctx.enqueue_create_host_buffer[DType.uint8](n_bases)
            var ri = 0
            while ri < n_reads:
                var s = seqs[ri]
                var L = s.byte_length()
                var base = ri * max_len
                var p = 0
                while p < max_len:
                    if p < L:
                        host_bases[base + p] = UInt8(ord(s[byte = p : p + 1]))
                    else:
                        host_bases[base + p] = 78
                    p += 1
                ri += 1

            var dev_bases = ctx.enqueue_create_buffer[DType.uint8](n_bases)
            var dev_codes = ctx.enqueue_create_buffer[DType.uint8](n_bases)
            var dev_kf = ctx.enqueue_create_buffer[DType.uint64](n_bases)
            var dev_kr = ctx.enqueue_create_buffer[DType.uint64](n_bases)
            var dev_hf = ctx.enqueue_create_buffer[DType.uint64](n_bases)
            var dev_hr = ctx.enqueue_create_buffer[DType.uint64](n_bases)
            var dev_valid = ctx.enqueue_create_buffer[DType.uint8](n_bases)
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
            ctx.enqueue_function[kmer_fwd_rc_kernel](
                dev_codes.unsafe_ptr(),
                dev_kf.unsafe_ptr(),
                dev_kr.unsafe_ptr(),
                dev_hf.unsafe_ptr(),
                dev_hr.unsafe_ptr(),
                dev_valid.unsafe_ptr(),
                n_bases,
                k,
                max_len,
                grid_dim=grid,
                block_dim=BLOCK,
            )
            var host_kf = ctx.enqueue_create_host_buffer[DType.uint64](n_bases)
            var host_kr = ctx.enqueue_create_host_buffer[DType.uint64](n_bases)
            var host_hf = ctx.enqueue_create_host_buffer[DType.uint64](n_bases)
            var host_hr = ctx.enqueue_create_host_buffer[DType.uint64](n_bases)
            var host_valid = ctx.enqueue_create_host_buffer[DType.uint8](n_bases)
            ctx.enqueue_copy(src_buf=dev_kf, dst_buf=host_kf)
            ctx.enqueue_copy(src_buf=dev_kr, dst_buf=host_kr)
            ctx.enqueue_copy(src_buf=dev_hf, dst_buf=host_hf)
            ctx.enqueue_copy(src_buf=dev_hr, dst_buf=host_hr)
            ctx.enqueue_copy(src_buf=dev_valid, dst_buf=host_valid)
            ctx.synchronize()

            # Window-reduce directly from host buffers (no per-position List copies).
            var out = List[List[MinimizerOcc]]()
            ri = 0
            while ri < n_reads:
                var L = seqs[ri].byte_length()
                var base = ri * max_len
                var win = k + w - 1
                var row = List[MinimizerOcc]()
                if L >= win:
                    var next_read_offset = 0
                    var last_hash = UInt64(0)
                    var last_offset = -1
                    var have_last = False
                    var window_start = 0
                    while window_start <= L - win:
                        var best_key = UInt64(0)
                        var best_hash = UInt64(0)
                        var best_off = 0
                        var best_rev = False
                        var have_best = False
                        var ok = True
                        var wi = 0
                        while wi < w:
                            var pos = window_start + wi
                            var idx = base + pos
                            if host_valid[idx] == 0:
                                ok = False
                                break
                            var use_r = host_hr[idx] < host_hf[idx]
                            var cand_key = host_kf[idx]
                            var cand_hash = host_hf[idx]
                            var cand_rev = False
                            if use_r:
                                cand_key = host_kr[idx]
                                cand_hash = host_hr[idx]
                                cand_rev = True
                            if (
                                not have_best
                                or cand_hash < best_hash
                                or (cand_hash == best_hash and pos < best_off)
                            ):
                                best_key = cand_key
                                best_hash = cand_hash
                                best_off = pos
                                best_rev = cand_rev
                                have_best = True
                            wi += 1
                        if ok and have_best:
                            var emit = False
                            if not have_last:
                                emit = True
                            elif last_hash == best_hash or last_offset < best_off:
                                if best_off >= next_read_offset:
                                    emit = True
                            if emit:
                                var off = best_off
                                if best_rev:
                                    off = best_off + k - 1
                                row.append(
                                    MinimizerOcc(best_key, best_hash, off, best_rev)
                                )
                                next_read_offset = best_off + 1
                                last_hash = best_hash
                                last_offset = best_off
                                have_last = True
                        window_start += 1
                out.append(row^)
                ri += 1
            return MinimizerBatchResult(out^, backend + "+mojo_min_buf")

    return MinimizerBatchResult(minimizers_batch_host(seqs, k, w), "mojo_min_host")


def minimizers_batch(
    device: String, seqs: List[String], k: Int, w: Int
) raises -> MinimizerBatchResult:
    """Production entry: DeviceContext when nvidia/amd, else Mojo host."""
    var resolved = device.lower()
    if (
        resolved == "nvidia"
        or resolved == "cuda"
        or resolved == "amd"
        or resolved == "hip"
        or resolved == "rocm"
    ):
        return minimizers_batch_devicecontext(resolved, seqs, k, w)
    return MinimizerBatchResult(minimizers_batch_host(seqs, k, w), "mojo_min_host")
