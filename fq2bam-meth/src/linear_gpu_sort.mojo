# GPU coordinate sort + Picard-style PE markdup for native FM BAM.
#
# LSD 8-bit radix on 64-bit (or 128-bit) keys. Each GPU thread owns 256
# records so histogram/scatter stays stable without atomics. Markdup is a
# segmented scan: group-start threads walk the equal-key run and flag every
# record whose pair_id is not the best in that run.

from std.sys import has_accelerator
from std.memory import UnsafePointer, memcpy


def gpu_sort_markdup(
    api: String,
    n: Int,
    coord_addr: Int,
    dup_hi_addr: Int,
    dup_lo_addr: Int,
    score_addr: Int,
    pair_addr: Int,
    is_dup_addr: Int,
    perm_addr: Int,
    do_markdup: Bool,
) raises:
    """Sort ``n`` records. Writes original-order ``is_dup`` and coord ``perm``.

    Addresses are host pointers to UInt64 (coord/dup_*) or UInt32
    (score/pair/is_dup/perm) arrays of length ``n``.
    """
    if n <= 0:
        return
    comptime if not has_accelerator():
        raise Error("gpu_sort_markdup requires accelerator build")
    else:
        from std.gpu import block_dim, block_idx, thread_idx
        from std.gpu.host import DeviceContext

        comptime RADIX_R = 256
        comptime BINS = 256
        comptime BLOCK = 256

        def init_idx_kernel(
            idx: UnsafePointer[UInt32, MutAnyOrigin], n0: Int
        ):
            var i = Int(block_idx.x * block_dim.x + thread_idx.x)
            if i < n0:
                idx[i] = UInt32(i)

        def zero_u32_kernel(
            p: UnsafePointer[UInt32, MutAnyOrigin], n0: Int
        ):
            var i = Int(block_idx.x * block_dim.x + thread_idx.x)
            if i < n0:
                p[i] = 0

        def hist_kernel(
            keys: UnsafePointer[UInt64, MutAnyOrigin],
            n0: Int,
            shift: Int,
            n_threads: Int,
            hist: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            var t = Int(block_idx.x * block_dim.x + thread_idx.x)
            if t >= n_threads:
                return
            var base = t * BINS
            var d = 0
            while d < BINS:
                hist[base + d] = 0
                d += 1
            var start = t * RADIX_R
            var end = start + RADIX_R
            if end > n0:
                end = n0
            var i = start
            while i < end:
                var digit = Int((keys[i] >> UInt64(shift)) & 255)
                hist[base + digit] = hist[base + digit] + 1
                i += 1

        def col_sum_kernel(
            hist: UnsafePointer[UInt32, MutAnyOrigin],
            n_threads: Int,
            totals: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            var d = Int(block_idx.x * block_dim.x + thread_idx.x)
            if d >= BINS:
                return
            var s: UInt32 = 0
            var t = 0
            while t < n_threads:
                s = s + hist[t * BINS + d]
                t += 1
            totals[d] = s

        def scan_hist_kernel(
            hist: UnsafePointer[UInt32, MutAnyOrigin],
            n_threads: Int,
            digit_base: UnsafePointer[UInt32, MutAnyOrigin],
            offsets: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            var d = Int(block_idx.x * block_dim.x + thread_idx.x)
            if d >= BINS:
                return
            var running = digit_base[d]
            var t = 0
            while t < n_threads:
                var slot = t * BINS + d
                var c = hist[slot]
                offsets[slot] = running
                running = running + c
                t += 1

        def scatter_kernel(
            keys_in: UnsafePointer[UInt64, MutAnyOrigin],
            idx_in: UnsafePointer[UInt32, MutAnyOrigin],
            keys_out: UnsafePointer[UInt64, MutAnyOrigin],
            idx_out: UnsafePointer[UInt32, MutAnyOrigin],
            n0: Int,
            shift: Int,
            n_threads: Int,
            offsets: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            var t = Int(block_idx.x * block_dim.x + thread_idx.x)
            if t >= n_threads:
                return
            var local = InlineArray[UInt32, BINS](fill=UInt32(0))
            var start = t * RADIX_R
            var end = start + RADIX_R
            if end > n0:
                end = n0
            var i = start
            while i < end:
                var digit = Int((keys_in[i] >> UInt64(shift)) & 255)
                var pos = Int(offsets[t * BINS + digit] + local[digit])
                local[digit] = local[digit] + 1
                keys_out[pos] = keys_in[i]
                idx_out[pos] = idx_in[i]
                i += 1

        def permute_u64_kernel(
            src: UnsafePointer[UInt64, MutAnyOrigin],
            perm: UnsafePointer[UInt32, MutAnyOrigin],
            dst: UnsafePointer[UInt64, MutAnyOrigin],
            n0: Int,
        ):
            var i = Int(block_idx.x * block_dim.x + thread_idx.x)
            if i < n0:
                dst[i] = src[Int(perm[i])]

        def gather_u32_kernel(
            src: UnsafePointer[UInt32, MutAnyOrigin],
            perm: UnsafePointer[UInt32, MutAnyOrigin],
            dst: UnsafePointer[UInt32, MutAnyOrigin],
            n0: Int,
        ):
            var i = Int(block_idx.x * block_dim.x + thread_idx.x)
            if i < n0:
                dst[i] = src[Int(perm[i])]

        def scatter_u32_kernel(
            src: UnsafePointer[UInt32, MutAnyOrigin],
            perm: UnsafePointer[UInt32, MutAnyOrigin],
            dst: UnsafePointer[UInt32, MutAnyOrigin],
            n0: Int,
        ):
            var i = Int(block_idx.x * block_dim.x + thread_idx.x)
            if i < n0:
                dst[Int(perm[i])] = src[i]

        def markdup_kernel(
            key_hi: UnsafePointer[UInt64, MutAnyOrigin],
            key_lo: UnsafePointer[UInt64, MutAnyOrigin],
            scores: UnsafePointer[UInt32, MutAnyOrigin],
            pair_ids: UnsafePointer[UInt32, MutAnyOrigin],
            n0: Int,
            is_dup: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            var i = Int(block_idx.x * block_dim.x + thread_idx.x)
            if i >= n0:
                return
            var sentinel = ~UInt64(0)
            if i > 0:
                if key_hi[i] == key_hi[i - 1] and key_lo[i] == key_lo[i - 1]:
                    return
            if key_hi[i] == sentinel:
                is_dup[i] = 0
                return
            var best_pair = pair_ids[i]
            var best_score = scores[i]
            var j = i
            while j < n0:
                if key_hi[j] != key_hi[i] or key_lo[j] != key_lo[i]:
                    break
                var sc = scores[j]
                var pid = pair_ids[j]
                if sc > best_score or (sc == best_score and pid < best_pair):
                    best_score = sc
                    best_pair = pid
                j += 1
            var k = i
            while k < j:
                if pair_ids[k] != best_pair:
                    is_dup[k] = 1
                else:
                    is_dup[k] = 0
                k += 1

        var ctx = DeviceContext(api=api)
        var n_threads = (n + RADIX_R - 1) // RADIX_R
        var hist_n = n_threads * BINS
        var grid_n = (n + BLOCK - 1) // BLOCK
        var grid_t = (n_threads + BLOCK - 1) // BLOCK

        var h_coord = ctx.enqueue_create_host_buffer[DType.uint64](n)
        var h_dhi = ctx.enqueue_create_host_buffer[DType.uint64](n)
        var h_dlo = ctx.enqueue_create_host_buffer[DType.uint64](n)
        var h_score = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var h_pair = ctx.enqueue_create_host_buffer[DType.uint32](n)
        memcpy(
            dest=h_coord.unsafe_ptr(),
            src=UnsafePointer[UInt64, MutAnyOrigin](
                unsafe_from_address=coord_addr
            ),
            count=n,
        )
        memcpy(
            dest=h_dhi.unsafe_ptr(),
            src=UnsafePointer[UInt64, MutAnyOrigin](
                unsafe_from_address=dup_hi_addr
            ),
            count=n,
        )
        memcpy(
            dest=h_dlo.unsafe_ptr(),
            src=UnsafePointer[UInt64, MutAnyOrigin](
                unsafe_from_address=dup_lo_addr
            ),
            count=n,
        )
        memcpy(
            dest=h_score.unsafe_ptr(),
            src=UnsafePointer[UInt32, MutAnyOrigin](
                unsafe_from_address=score_addr
            ),
            count=n,
        )
        memcpy(
            dest=h_pair.unsafe_ptr(),
            src=UnsafePointer[UInt32, MutAnyOrigin](
                unsafe_from_address=pair_addr
            ),
            count=n,
        )

        var d_key_a = ctx.enqueue_create_buffer[DType.uint64](n)
        var d_key_b = ctx.enqueue_create_buffer[DType.uint64](n)
        var d_idx_a = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_idx_b = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_hist = ctx.enqueue_create_buffer[DType.uint32](hist_n)
        var d_off = ctx.enqueue_create_buffer[DType.uint32](hist_n)
        var d_tot = ctx.enqueue_create_buffer[DType.uint32](BINS)
        var h_tot = ctx.enqueue_create_host_buffer[DType.uint32](BINS)
        var h_base = ctx.enqueue_create_host_buffer[DType.uint32](BINS)

        # --- dup-key sort (LSD lo then hi) + markdup ---
        ctx.enqueue_copy(src_buf=h_dlo, dst_buf=d_key_a)
        ctx.enqueue_function[init_idx_kernel](
            d_idx_a.unsafe_ptr(),
            n,
            grid_dim=grid_n,
            block_dim=BLOCK,
        )
        var p = 0
        while p < 8:
            var shift = p * 8
            var keys_in = d_key_a.unsafe_ptr()
            var keys_out = d_key_b.unsafe_ptr()
            var idx_in = d_idx_a.unsafe_ptr()
            var idx_out = d_idx_b.unsafe_ptr()
            if (p & 1) == 1:
                keys_in = d_key_b.unsafe_ptr()
                keys_out = d_key_a.unsafe_ptr()
                idx_in = d_idx_b.unsafe_ptr()
                idx_out = d_idx_a.unsafe_ptr()
            ctx.enqueue_function[hist_kernel](
                keys_in,
                n,
                shift,
                n_threads,
                d_hist.unsafe_ptr(),
                grid_dim=grid_t,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[col_sum_kernel](
                d_hist.unsafe_ptr(),
                n_threads,
                d_tot.unsafe_ptr(),
                grid_dim=1,
                block_dim=BINS,
            )
            ctx.enqueue_copy(src_buf=d_tot, dst_buf=h_tot)
            ctx.synchronize()
            var run: UInt32 = 0
            var d = 0
            while d < BINS:
                h_base[d] = run
                run = run + h_tot[d]
                d += 1
            ctx.enqueue_copy(src_buf=h_base, dst_buf=d_tot)
            ctx.enqueue_function[scan_hist_kernel](
                d_hist.unsafe_ptr(),
                n_threads,
                d_tot.unsafe_ptr(),
                d_off.unsafe_ptr(),
                grid_dim=1,
                block_dim=BINS,
            )
            ctx.enqueue_function[scatter_kernel](
                keys_in,
                idx_in,
                keys_out,
                idx_out,
                n,
                shift,
                n_threads,
                d_off.unsafe_ptr(),
                grid_dim=grid_t,
                block_dim=BLOCK,
            )
            p += 1
        # After 8 passes last odd → idx in d_idx_a. Permute dup_hi by that perm,
        # then radix-sort hi (stable wrt previous lo order via carrying idx).
        var d_hi_src = ctx.enqueue_create_buffer[DType.uint64](n)
        ctx.enqueue_copy(src_buf=h_dhi, dst_buf=d_hi_src)
        ctx.enqueue_function[permute_u64_kernel](
            d_hi_src.unsafe_ptr(),
            d_idx_a.unsafe_ptr(),
            d_key_a.unsafe_ptr(),
            n,
            grid_dim=grid_n,
            block_dim=BLOCK,
        )
        p = 0
        while p < 8:
            var shift2 = p * 8
            var keys_in2 = d_key_a.unsafe_ptr()
            var keys_out2 = d_key_b.unsafe_ptr()
            var idx_in2 = d_idx_a.unsafe_ptr()
            var idx_out2 = d_idx_b.unsafe_ptr()
            if (p & 1) == 1:
                keys_in2 = d_key_b.unsafe_ptr()
                keys_out2 = d_key_a.unsafe_ptr()
                idx_in2 = d_idx_b.unsafe_ptr()
                idx_out2 = d_idx_a.unsafe_ptr()
            ctx.enqueue_function[hist_kernel](
                keys_in2,
                n,
                shift2,
                n_threads,
                d_hist.unsafe_ptr(),
                grid_dim=grid_t,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[col_sum_kernel](
                d_hist.unsafe_ptr(),
                n_threads,
                d_tot.unsafe_ptr(),
                grid_dim=1,
                block_dim=BINS,
            )
            ctx.enqueue_copy(src_buf=d_tot, dst_buf=h_tot)
            ctx.synchronize()
            var run2: UInt32 = 0
            var d2 = 0
            while d2 < BINS:
                h_base[d2] = run2
                run2 = run2 + h_tot[d2]
                d2 += 1
            ctx.enqueue_copy(src_buf=h_base, dst_buf=d_tot)
            ctx.enqueue_function[scan_hist_kernel](
                d_hist.unsafe_ptr(),
                n_threads,
                d_tot.unsafe_ptr(),
                d_off.unsafe_ptr(),
                grid_dim=1,
                block_dim=BINS,
            )
            ctx.enqueue_function[scatter_kernel](
                keys_in2,
                idx_in2,
                keys_out2,
                idx_out2,
                n,
                shift2,
                n_threads,
                d_off.unsafe_ptr(),
                grid_dim=grid_t,
                block_dim=BLOCK,
            )
            p += 1
        # idx in d_idx_a is original-index perm sorted by (dup_hi, dup_lo).
        var d_score = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_pair = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_score_s = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_pair_s = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_dup_s = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_dup = ctx.enqueue_create_buffer[DType.uint32](n)
        ctx.enqueue_copy(src_buf=h_score, dst_buf=d_score)
        ctx.enqueue_copy(src_buf=h_pair, dst_buf=d_pair)
        ctx.enqueue_function[zero_u32_kernel](
            d_dup.unsafe_ptr(),
            n,
            grid_dim=grid_n,
            block_dim=BLOCK,
        )
        if do_markdup:
            ctx.enqueue_function[gather_u32_kernel](
                d_score.unsafe_ptr(),
                d_idx_a.unsafe_ptr(),
                d_score_s.unsafe_ptr(),
                n,
                grid_dim=grid_n,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[gather_u32_kernel](
                d_pair.unsafe_ptr(),
                d_idx_a.unsafe_ptr(),
                d_pair_s.unsafe_ptr(),
                n,
                grid_dim=grid_n,
                block_dim=BLOCK,
            )
            # Sorted dup_hi is in d_key_a. Need sorted dup_lo too.
            var d_lo_src = ctx.enqueue_create_buffer[DType.uint64](n)
            var d_lo_s = ctx.enqueue_create_buffer[DType.uint64](n)
            ctx.enqueue_copy(src_buf=h_dlo, dst_buf=d_lo_src)
            ctx.enqueue_function[permute_u64_kernel](
                d_lo_src.unsafe_ptr(),
                d_idx_a.unsafe_ptr(),
                d_lo_s.unsafe_ptr(),
                n,
                grid_dim=grid_n,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[zero_u32_kernel](
                d_dup_s.unsafe_ptr(),
                n,
                grid_dim=grid_n,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[markdup_kernel](
                d_key_a.unsafe_ptr(),
                d_lo_s.unsafe_ptr(),
                d_score_s.unsafe_ptr(),
                d_pair_s.unsafe_ptr(),
                n,
                d_dup_s.unsafe_ptr(),
                grid_dim=grid_n,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[scatter_u32_kernel](
                d_dup_s.unsafe_ptr(),
                d_idx_a.unsafe_ptr(),
                d_dup.unsafe_ptr(),
                n,
                grid_dim=grid_n,
                block_dim=BLOCK,
            )

        # --- coordinate sort ---
        ctx.enqueue_copy(src_buf=h_coord, dst_buf=d_key_a)
        ctx.enqueue_function[init_idx_kernel](
            d_idx_a.unsafe_ptr(),
            n,
            grid_dim=grid_n,
            block_dim=BLOCK,
        )
        p = 0
        while p < 8:
            var shift3 = p * 8
            var keys_in3 = d_key_a.unsafe_ptr()
            var keys_out3 = d_key_b.unsafe_ptr()
            var idx_in3 = d_idx_a.unsafe_ptr()
            var idx_out3 = d_idx_b.unsafe_ptr()
            if (p & 1) == 1:
                keys_in3 = d_key_b.unsafe_ptr()
                keys_out3 = d_key_a.unsafe_ptr()
                idx_in3 = d_idx_b.unsafe_ptr()
                idx_out3 = d_idx_a.unsafe_ptr()
            ctx.enqueue_function[hist_kernel](
                keys_in3,
                n,
                shift3,
                n_threads,
                d_hist.unsafe_ptr(),
                grid_dim=grid_t,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[col_sum_kernel](
                d_hist.unsafe_ptr(),
                n_threads,
                d_tot.unsafe_ptr(),
                grid_dim=1,
                block_dim=BINS,
            )
            ctx.enqueue_copy(src_buf=d_tot, dst_buf=h_tot)
            ctx.synchronize()
            var run3: UInt32 = 0
            var d3 = 0
            while d3 < BINS:
                h_base[d3] = run3
                run3 = run3 + h_tot[d3]
                d3 += 1
            ctx.enqueue_copy(src_buf=h_base, dst_buf=d_tot)
            ctx.enqueue_function[scan_hist_kernel](
                d_hist.unsafe_ptr(),
                n_threads,
                d_tot.unsafe_ptr(),
                d_off.unsafe_ptr(),
                grid_dim=1,
                block_dim=BINS,
            )
            ctx.enqueue_function[scatter_kernel](
                keys_in3,
                idx_in3,
                keys_out3,
                idx_out3,
                n,
                shift3,
                n_threads,
                d_off.unsafe_ptr(),
                grid_dim=grid_t,
                block_dim=BLOCK,
            )
            p += 1

        var h_perm = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var h_dup = ctx.enqueue_create_host_buffer[DType.uint32](n)
        ctx.enqueue_copy(src_buf=d_idx_a, dst_buf=h_perm)
        ctx.enqueue_copy(src_buf=d_dup, dst_buf=h_dup)
        ctx.synchronize()
        memcpy(
            dest=UnsafePointer[UInt32, MutAnyOrigin](
                unsafe_from_address=perm_addr
            ),
            src=h_perm.unsafe_ptr(),
            count=n,
        )
        memcpy(
            dest=UnsafePointer[UInt32, MutAnyOrigin](
                unsafe_from_address=is_dup_addr
            ),
            src=h_dup.unsafe_ptr(),
            count=n,
        )
