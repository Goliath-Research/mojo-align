# GPU coordinate sort + Picard-style PE markdup for native FM BAM.
#
# LSD 8-bit radix on 64-bit keys, tiled so device working set is O(tile) not
# O(n). Default tile = n (one shot, same as before). METHYLGRAPHER_FM_SORT_TILE
# forces smaller tiles; host k-way merge reconstructs global order. Markdup is
# a host scan on the merged dup-key order
# (same rule as the old GPU kernel).

from std.collections import List
from std.memory import UnsafePointer, unsafe_memmove
from std.python import Python
from std.sys import has_accelerator


def _sort_tile_size(n: Int) raises -> Int:
    var os_mod = Python.import_module("os")
    var raw = String(os_mod.environ.get("METHYLGRAPHER_FM_SORT_TILE", "0"))
    var t = Int(raw)
    if t < 1 or t >= n:
        return n
    return t


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
    dup_perm_addr: Int,
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
        from max.gpu import block_dim, block_idx, thread_idx
        from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

        comptime RADIX_R = 256
        comptime BINS = 256
        comptime BLOCK = 256

        def init_idx_kernel(
            idx: UnsafePointer[UInt32, MutAnyOrigin], n0: Int
        ):
            var i = Int(block_idx.x * block_dim.x + thread_idx.x)
            if i < n0:
                idx[i] = UInt32(i)

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
                var digit = Int((keys[i] >> shift) & 255)
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
                var digit = Int((keys_in[i] >> shift) & 255)
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

        def radix8(
            ctx: DeviceContext,
            mut ka: DeviceBuffer[DType.uint64],
            mut kb: DeviceBuffer[DType.uint64],
            mut ia: DeviceBuffer[DType.uint32],
            mut ib: DeviceBuffer[DType.uint32],
            mut hist: DeviceBuffer[DType.uint32],
            mut offb: DeviceBuffer[DType.uint32],
            mut tot: DeviceBuffer[DType.uint32],
            mut h_tot: HostBuffer[DType.uint32],
            mut h_base: HostBuffer[DType.uint32],
            n_t: Int,
        ) raises:
            var n_threads = (n_t + RADIX_R - 1) // RADIX_R
            var grid_t = (n_threads + BLOCK - 1) // BLOCK
            var p = 0
            while p < 8:
                var shift = p * 8
                var keys_in = ka.unsafe_ptr()
                var keys_out = kb.unsafe_ptr()
                var idx_in = ia.unsafe_ptr()
                var idx_out = ib.unsafe_ptr()
                if (p & 1) == 1:
                    keys_in = kb.unsafe_ptr()
                    keys_out = ka.unsafe_ptr()
                    idx_in = ib.unsafe_ptr()
                    idx_out = ia.unsafe_ptr()
                ctx.enqueue_function[hist_kernel](
                    keys_in,
                    n_t,
                    shift,
                    n_threads,
                    hist.unsafe_ptr(),
                    grid_dim=grid_t,
                    block_dim=BLOCK,
                )
                ctx.enqueue_function[col_sum_kernel](
                    hist.unsafe_ptr(),
                    n_threads,
                    tot.unsafe_ptr(),
                    grid_dim=1,
                    block_dim=BINS,
                )
                ctx.enqueue_copy(src_buf=tot, dst_buf=h_tot)
                ctx.synchronize()
                var run: UInt32 = 0
                var d = 0
                while d < BINS:
                    h_base[d] = run
                    run = run + h_tot[d]
                    d += 1
                ctx.enqueue_copy(src_buf=h_base, dst_buf=tot)
                ctx.enqueue_function[scan_hist_kernel](
                    hist.unsafe_ptr(),
                    n_threads,
                    tot.unsafe_ptr(),
                    offb.unsafe_ptr(),
                    grid_dim=1,
                    block_dim=BINS,
                )
                ctx.enqueue_function[scatter_kernel](
                    keys_in,
                    idx_in,
                    keys_out,
                    idx_out,
                    n_t,
                    shift,
                    n_threads,
                    offb.unsafe_ptr(),
                    grid_dim=grid_t,
                    block_dim=BLOCK,
                )
                p += 1

        var tile = _sort_tile_size(n)
        var n_tiles = (n + tile - 1) // tile
        print(
            "MojoLinear GPU-fm sort n=",
            n,
            " tile=",
            tile,
            " tiles=",
            n_tiles,
        )
        var ctx = DeviceContext(api=api)
        var h_coord = ctx.enqueue_create_host_buffer[DType.uint64](n)
        var h_dhi = ctx.enqueue_create_host_buffer[DType.uint64](n)
        var h_dlo = ctx.enqueue_create_host_buffer[DType.uint64](n)
        var h_score = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var h_pair = ctx.enqueue_create_host_buffer[DType.uint32](n)
        unsafe_memmove(
            dest=h_coord.unsafe_ptr(),
            src=UnsafePointer[UInt64, MutAnyOrigin](
                unsafe_from_address=coord_addr
            ),
            count=n,
        )
        unsafe_memmove(
            dest=h_dhi.unsafe_ptr(),
            src=UnsafePointer[UInt64, MutAnyOrigin](
                unsafe_from_address=dup_hi_addr
            ),
            count=n,
        )
        unsafe_memmove(
            dest=h_dlo.unsafe_ptr(),
            src=UnsafePointer[UInt64, MutAnyOrigin](
                unsafe_from_address=dup_lo_addr
            ),
            count=n,
        )
        unsafe_memmove(
            dest=h_score.unsafe_ptr(),
            src=UnsafePointer[UInt32, MutAnyOrigin](
                unsafe_from_address=score_addr
            ),
            count=n,
        )
        unsafe_memmove(
            dest=h_pair.unsafe_ptr(),
            src=UnsafePointer[UInt32, MutAnyOrigin](
                unsafe_from_address=pair_addr
            ),
            count=n,
        )

        var hist_n = ((tile + RADIX_R - 1) // RADIX_R) * BINS
        var d_key_a = ctx.enqueue_create_buffer[DType.uint64](tile)
        var d_key_b = ctx.enqueue_create_buffer[DType.uint64](tile)
        var d_idx_a = ctx.enqueue_create_buffer[DType.uint32](tile)
        var d_idx_b = ctx.enqueue_create_buffer[DType.uint32](tile)
        var d_hist = ctx.enqueue_create_buffer[DType.uint32](hist_n)
        var d_off = ctx.enqueue_create_buffer[DType.uint32](hist_n)
        var d_tot = ctx.enqueue_create_buffer[DType.uint32](BINS)
        var d_hi_src = ctx.enqueue_create_buffer[DType.uint64](tile)
        var h_tot = ctx.enqueue_create_host_buffer[DType.uint32](BINS)
        var h_base = ctx.enqueue_create_host_buffer[DType.uint32](BINS)
        var h_tile_key = ctx.enqueue_create_host_buffer[DType.uint64](tile)
        var h_tile_hi = ctx.enqueue_create_host_buffer[DType.uint64](tile)
        var h_tile_idx = ctx.enqueue_create_host_buffer[DType.uint32](tile)
        var run_orig = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var tlen = List[Int](length=n_tiles, fill=0)
        var ti = 0
        while ti < n_tiles:
            var b = ti * tile
            var nt = tile
            if b + nt > n:
                nt = n - b
            tlen[ti] = nt
            ti += 1

        var t = 0
        while t < n_tiles:
            var base = t * tile
            var n_t = tlen[t]
            var grid_n = (n_t + BLOCK - 1) // BLOCK
            unsafe_memmove(
                dest=h_tile_key.unsafe_ptr(),
                src=h_dlo.unsafe_ptr() + base,
                count=n_t,
            )
            ctx.enqueue_copy(src_buf=h_tile_key, dst_buf=d_key_a)
            ctx.enqueue_function[init_idx_kernel](
                d_idx_a.unsafe_ptr(),
                n_t,
                grid_dim=grid_n,
                block_dim=BLOCK,
            )
            radix8(
                ctx,
                d_key_a,
                d_key_b,
                d_idx_a,
                d_idx_b,
                d_hist,
                d_off,
                d_tot,
                h_tot,
                h_base,
                n_t,
            )
            unsafe_memmove(
                dest=h_tile_hi.unsafe_ptr(),
                src=h_dhi.unsafe_ptr() + base,
                count=n_t,
            )
            ctx.enqueue_copy(src_buf=h_tile_hi, dst_buf=d_hi_src)
            ctx.enqueue_function[permute_u64_kernel](
                d_hi_src.unsafe_ptr(),
                d_idx_a.unsafe_ptr(),
                d_key_a.unsafe_ptr(),
                n_t,
                grid_dim=grid_n,
                block_dim=BLOCK,
            )
            radix8(
                ctx,
                d_key_a,
                d_key_b,
                d_idx_a,
                d_idx_b,
                d_hist,
                d_off,
                d_tot,
                h_tot,
                h_base,
                n_t,
            )
            ctx.enqueue_copy(src_buf=d_idx_a, dst_buf=h_tile_idx)
            ctx.synchronize()
            var i = 0
            while i < n_t:
                var loc = Int(h_tile_idx[i])
                run_orig[base + i] = UInt32(base + loc)
                i += 1
            t += 1

        var h_dup = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var zi = 0
        while zi < n:
            h_dup[zi] = 0
            zi += 1
        if do_markdup:
            var merged = ctx.enqueue_create_host_buffer[DType.uint32](n)
            var head = List[Int](length=n_tiles, fill=0)
            var out_i = 0
            while out_i < n:
                var best = -1
                var t2 = 0
                while t2 < n_tiles:
                    if head[t2] < tlen[t2]:
                        var o = Int(run_orig[t2 * tile + head[t2]])
                        if best < 0:
                            best = t2
                        else:
                            var bo = Int(run_orig[best * tile + head[best]])
                            var hi = h_dhi[o]
                            var bhi = h_dhi[bo]
                            var lo = h_dlo[o]
                            var blo = h_dlo[bo]
                            var take = False
                            if hi < bhi:
                                take = True
                            elif hi == bhi:
                                if lo < blo:
                                    take = True
                                elif lo == blo and o < bo:
                                    take = True
                            if take:
                                best = t2
                    t2 += 1
                merged[out_i] = run_orig[best * tile + head[best]]
                head[best] = head[best] + 1
                out_i += 1
            var sentinel = ~UInt64(0)
            var gi = 0
            while gi < n:
                var o0 = Int(merged[gi])
                if h_dhi[o0] == sentinel:
                    h_dup[o0] = 0
                    gi += 1
                    continue
                var gj = gi + 1
                while gj < n:
                    var oj = Int(merged[gj])
                    if h_dhi[oj] != h_dhi[o0] or h_dlo[oj] != h_dlo[o0]:
                        break
                    gj += 1
                var best_pair = h_pair[o0]
                var best_score = h_score[o0]
                var gk = gi
                while gk < gj:
                    var ok = Int(merged[gk])
                    var sc = h_score[ok]
                    var pid = h_pair[ok]
                    if sc > best_score or (sc == best_score and pid < best_pair):
                        best_score = sc
                        best_pair = pid
                    gk += 1
                gk = gi
                while gk < gj:
                    var ok2 = Int(merged[gk])
                    if h_pair[ok2] != best_pair:
                        h_dup[ok2] = 1
                    else:
                        h_dup[ok2] = 0
                    gk += 1
                gi = gj
            if dup_perm_addr != 0:
                var dpi = 0
                var dp = UnsafePointer[UInt32, MutAnyOrigin](
                    unsafe_from_address=dup_perm_addr
                )
                while dpi < n:
                    dp[dpi] = merged[dpi]
                    dpi += 1

        t = 0
        while t < n_tiles:
            var base2 = t * tile
            var n_t2 = tlen[t]
            var grid_n2 = (n_t2 + BLOCK - 1) // BLOCK
            unsafe_memmove(
                dest=h_tile_key.unsafe_ptr(),
                src=h_coord.unsafe_ptr() + base2,
                count=n_t2,
            )
            ctx.enqueue_copy(src_buf=h_tile_key, dst_buf=d_key_a)
            ctx.enqueue_function[init_idx_kernel](
                d_idx_a.unsafe_ptr(),
                n_t2,
                grid_dim=grid_n2,
                block_dim=BLOCK,
            )
            radix8(
                ctx,
                d_key_a,
                d_key_b,
                d_idx_a,
                d_idx_b,
                d_hist,
                d_off,
                d_tot,
                h_tot,
                h_base,
                n_t2,
            )
            ctx.enqueue_copy(src_buf=d_idx_a, dst_buf=h_tile_idx)
            ctx.synchronize()
            var j = 0
            while j < n_t2:
                var loc2 = Int(h_tile_idx[j])
                run_orig[base2 + j] = UInt32(base2 + loc2)
                j += 1
            t += 1

        var h_perm = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var headc = List[Int](length=n_tiles, fill=0)
        var out_c = 0
        while out_c < n:
            var bestc = -1
            var t3 = 0
            while t3 < n_tiles:
                if headc[t3] < tlen[t3]:
                    var oc = Int(run_orig[t3 * tile + headc[t3]])
                    if bestc < 0:
                        bestc = t3
                    else:
                        var boc = Int(run_orig[bestc * tile + headc[bestc]])
                        var ck = h_coord[oc]
                        var bck = h_coord[boc]
                        if ck < bck or (ck == bck and oc < boc):
                            bestc = t3
                t3 += 1
            h_perm[out_c] = run_orig[bestc * tile + headc[bestc]]
            headc[bestc] = headc[bestc] + 1
            out_c += 1

        unsafe_memmove(
            dest=UnsafePointer[UInt32, MutAnyOrigin](
                unsafe_from_address=perm_addr
            ),
            src=h_perm.unsafe_ptr(),
            count=n,
        )
        unsafe_memmove(
            dest=UnsafePointer[UInt32, MutAnyOrigin](
                unsafe_from_address=is_dup_addr
            ),
            src=h_dup.unsafe_ptr(),
            count=n,
        )
        if dup_perm_addr != 0 and not do_markdup:
            var dp2 = UnsafePointer[UInt32, MutAnyOrigin](
                unsafe_from_address=dup_perm_addr
            )
            var ii = 0
            while ii < n:
                dp2[ii] = UInt32(ii)
                ii += 1