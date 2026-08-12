# SPEED ENGINE (opt-in). METHYLGRAPHER_LINEAR_ENGINE=speed
#
# Separate from the frozen parity mapper in linear_gpu_locate.mojo.
# Any algorithm is allowed here. It must not become the default until it
# matches or improves Clara concordance gates AND parity mapped-rate.
#
# v2 strategy: unique-seed consensus + rare-seed plurality, then a heavy
# vote only on still-unmapped reads. First-hit (v1) was fast but dropped
# ~31% of maps by accepting the first NM-budget posting.

from std.collections import List
from std.python import Python, PythonObject
from std.sys import has_accelerator

from gpu_device import select_device
from gpu_kernels import _device_api, kernel_target_label, probe_device_context
from linear_extend import pair_hits, LinearHit
from linear_gpu_locate import (
    FastqRec,
    _env_int,
    _hit_from_gpu,
    _open_fastq,
    _read_pe_batch,
    _require_hbm,
    _sam_with_rg,
    _write_sam_header,
)
from linear_index import LinearIndex
from utility import open_text_write


def map_fastq_dense_gpu_speed(
    mut index: LinearIndex,
    fq1: String,
    out_sam: String,
    device: String,
    fq2: String = "",
    bs_r1: String = "",
    bs_r2: String = "",
) raises -> Int:
    """Speed-engine GPU map: unique-consensus then rare/heavy vote (not parity)."""
    if not index.dense:
        raise Error("map_fastq_dense_gpu_speed requires dense-v1 index")
    if (
        index.kmers_addr == 0
        or index.offsets_addr == 0
        or index.postings_addr == 0
        or index.seq_addr == 0
        or index.contig_off_addr == 0
    ):
        raise Error("map_fastq_dense_gpu_speed: null mmap addresses")

    var resolved = select_device(device)
    var backend = probe_device_context(resolved)
    var target = kernel_target_label(resolved)
    print(
        "MojoLinear GPU-speed device=",
        resolved,
        " target=",
        target,
        " backend=",
        backend,
    )
    if not (
        backend.startswith("devicecontext-cuda")
        or backend.startswith("devicecontext-hip")
    ):
        raise Error(
            "MojoLinear GPU-speed needs DeviceContext cuda/hip, got " + backend
        )

    comptime if not has_accelerator():
        raise Error("map_fastq_dense_gpu_speed requires accelerator build")
    else:
        from std.gpu import block_dim, block_idx, thread_idx
        from std.gpu.host import DeviceContext
        from std.memory import UnsafePointer, memcpy

        def copy_bytes_offset_kernel(
            dst: UnsafePointer[UInt8, MutAnyOrigin],
            src: UnsafePointer[UInt8, MutAnyOrigin],
            dst_off: Int,
            n: Int,
        ):
            var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
            if tid < n:
                dst[dst_off + tid] = src[tid]

        def upload_mmap_to_device(
            ctx: DeviceContext,
            dst: UnsafePointer[UInt8, MutAnyOrigin],
            host_addr: Int,
            nbytes: Int,
            label: String,
        ) raises:
            if nbytes <= 0:
                return
            if host_addr == 0:
                raise Error("upload_mmap_to_device: null host mmap address")
            comptime CHUNK = 64 * 1024 * 1024
            comptime BLOCK = 256
            var off = 0
            var last_log = 0
            while off < nbytes:
                var n = nbytes - off
                if n > CHUNK:
                    n = CHUNK
                var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
                var src = UnsafePointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=host_addr + off
                )
                memcpy(dest=host.unsafe_ptr(), src=src, count=n)
                var stage = ctx.enqueue_create_buffer[DType.uint8](n)
                ctx.enqueue_copy(src_buf=host, dst_buf=stage)
                var grid = (n + BLOCK - 1) // BLOCK
                ctx.enqueue_function[copy_bytes_offset_kernel](
                    dst,
                    stage.unsafe_ptr(),
                    off,
                    n,
                    grid_dim=grid,
                    block_dim=BLOCK,
                )
                ctx.synchronize()
                off += n
                if off - last_log >= 1 << 30 or off == nbytes:
                    print(
                        "MojoLinear GPU-speed upload ",
                        label,
                        " ",
                        off,
                        "/",
                        nbytes,
                    )
                    last_log = off

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

        def encode_strided_kernel(
            codes: UnsafePointer[UInt8, MutAnyOrigin],
            keys: UnsafePointer[UInt64, MutAnyOrigin],
            q_offs: UnsafePointer[UInt32, MutAnyOrigin],
            read_lens: UnsafePointer[UInt32, MutAnyOrigin],
            n_reads: Int,
            max_len: Int,
            k_len: Int,
            seed_stride: Int,
            slots_per_read: Int,
        ):
            var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
            var n_slots = n_reads * slots_per_read
            if idx >= n_slots:
                return
            var rid = idx // slots_per_read
            var slot = idx % slots_per_read
            var q_off = slot * seed_stride
            var L = Int(read_lens[rid])
            keys[idx] = UInt64(0xFFFFFFFFFFFFFFFF)
            q_offs[idx] = UInt32(q_off)
            if q_off + k_len > L:
                return
            var base = rid * max_len + q_off
            var key: UInt64 = 0
            var j = 0
            while j < k_len:
                var c = codes[base + j]
                if c > 3:
                    return
                key = (key << 2) | UInt64(c)
                j += 1
            keys[idx] = key

        def locate_bsearch_kernel(
            keys: UnsafePointer[UInt64, MutAnyOrigin],
            kmers: UnsafePointer[UInt64, MutAnyOrigin],
            offsets: UnsafePointer[UInt64, MutAnyOrigin],
            out_start: UnsafePointer[UInt64, MutAnyOrigin],
            out_end: UnsafePointer[UInt64, MutAnyOrigin],
            n_slots: Int,
            n_table: Int,
            max_occ: Int,
        ):
            var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
            if idx >= n_slots:
                return
            out_start[idx] = 0
            out_end[idx] = 0
            var key = keys[idx]
            if key == UInt64(0xFFFFFFFFFFFFFFFF) or n_table <= 0:
                return
            var lo = 0
            var hi = n_table
            while lo < hi:
                var mid = (lo + hi) // 2
                var mk = kmers[mid]
                if mk < key:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= n_table:
                return
            if kmers[lo] != key:
                return
            var start = offsets[lo]
            var end = offsets[lo + 1]
            var occ = Int(end - start)
            if occ <= 0 or occ > max_occ:
                return
            out_start[idx] = start
            out_end[idx] = end

        def rc_codes_kernel(
            codes: UnsafePointer[UInt8, MutAnyOrigin],
            rc_codes: UnsafePointer[UInt8, MutAnyOrigin],
            read_lens: UnsafePointer[UInt32, MutAnyOrigin],
            n_reads: Int,
            max_len: Int,
        ):
            var rid = Int(block_idx.x * block_dim.x + thread_idx.x)
            if rid >= n_reads:
                return
            var L = Int(read_lens[rid])
            var base = rid * max_len
            var i = 0
            while i < max_len:
                rc_codes[base + i] = 255
                i += 1
            var j = 0
            while j < L:
                var c = codes[base + (L - 1 - j)]
                var r: UInt8 = 255
                if c <= 3:
                    r = c ^ 3
                rc_codes[base + j] = r
                j += 1

        def consensus_extend_kernel(
            occ_start: UnsafePointer[UInt64, MutAnyOrigin],
            occ_end: UnsafePointer[UInt64, MutAnyOrigin],
            q_offs: UnsafePointer[UInt32, MutAnyOrigin],
            postings: UnsafePointer[UInt32, MutAnyOrigin],
            n_postings: Int,
            sequences: UnsafePointer[UInt8, MutAnyOrigin],
            contig_off: UnsafePointer[UInt64, MutAnyOrigin],
            n_contigs: Int,
            codes: UnsafePointer[UInt8, MutAnyOrigin],
            read_lens: UnsafePointer[UInt32, MutAnyOrigin],
            max_len: Int,
            slots_per_read: Int,
            n_reads: Int,
            max_diff: Int,
            max_soft: Int,
            vote_occ: Int,
            pass_mode: Int,
            n_active: Int,
            rid_map: UnsafePointer[Int32, MutAnyOrigin],
            skip_cid: UnsafePointer[Int32, MutAnyOrigin],
            skip_nm: UnsafePointer[UInt32, MutAnyOrigin],
            skip_exact: Int,
            strand_flag: Int32,
            out_cid: UnsafePointer[Int32, MutAnyOrigin],
            out_pos: UnsafePointer[UInt32, MutAnyOrigin],
            out_mapq: UnsafePointer[UInt32, MutAnyOrigin],
            out_flag: UnsafePointer[Int32, MutAnyOrigin],
            out_sl: UnsafePointer[UInt32, MutAnyOrigin],
            out_sr: UnsafePointer[UInt32, MutAnyOrigin],
            out_nm: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            """Unique-seed consensus, then rare plurality + gapless/softclip.

            pass_mode 0: all active reads; accept only plurality or NM=0.
            pass_mode 1: skip already-mapped; accept any NM within budget.
            skip_exact=1: leave reads with skip_cid mapped and skip_nm==0.
            """
            var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
            var rid: Int
            if n_active >= 0:
                if tid >= n_active:
                    return
                rid = Int(rid_map[tid])
            else:
                rid = tid
                if rid >= n_reads:
                    return
            if skip_exact == 1:
                if Int(skip_cid[rid]) >= 0 and Int(skip_nm[rid]) == 0:
                    out_cid[rid] = Int32(-1)
                    return
            if pass_mode == 1 and Int(out_cid[rid]) >= 0:
                return
            out_cid[rid] = Int32(-1)
            out_pos[rid] = 0
            out_mapq[rid] = 0
            out_flag[rid] = Int32(4)
            out_sl[rid] = 0
            out_sr[rid] = 0
            out_nm[rid] = 0
            var qlen = Int(read_lens[rid])
            if qlen <= 0:
                return
            var budget = max_diff
            if budget < 2:
                budget = 2
            if budget > qlen // 2:
                budget = qlen // 2
            var clip_cap = max_soft
            if clip_cap > qlen // 4:
                clip_cap = qlen // 4
            var q_base = rid * max_len
            var vote_cap = vote_occ
            if vote_cap < 1:
                vote_cap = 32

            comptime TOP = 256
            comptime KEEP = 16
            comptime MAX_SLOTS = 64
            var cand_key = InlineArray[UInt64, TOP](fill=UInt64(0xFFFFFFFFFFFFFFFF))
            var cand_n = InlineArray[Int32, TOP](fill=Int32(0))
            var n_cand = 0
            var ord_slot = InlineArray[Int32, MAX_SLOTS](fill=Int32(-1))
            var ord_occ = InlineArray[Int32, MAX_SLOTS](fill=Int32(0))
            var n_ord = 0
            var slot_g = 0
            while slot_g < slots_per_read and n_ord < MAX_SLOTS:
                var sidx_g = rid * slots_per_read + slot_g
                var a_g = Int(occ_start[sidx_g])
                var b_g = Int(occ_end[sidx_g])
                var occ_g = b_g - a_g
                if occ_g > 0 and a_g >= 0 and b_g <= n_postings:
                    var ins = n_ord
                    var t = 0
                    while t < n_ord:
                        if occ_g < Int(ord_occ[t]):
                            ins = t
                            break
                        t += 1
                    var shift = n_ord
                    while shift > ins:
                        ord_slot[shift] = ord_slot[shift - 1]
                        ord_occ[shift] = ord_occ[shift - 1]
                        shift -= 1
                    ord_slot[ins] = Int32(slot_g)
                    ord_occ[ins] = Int32(occ_g)
                    n_ord += 1
                slot_g += 1

            var last_i = -1
            var oi = 0
            var unique_only = True
            while oi < n_ord:
                var occ_v = Int(ord_occ[oi])
                if unique_only and occ_v > 1:
                    var b1u = 0
                    var b2u = 0
                    var zu = 0
                    while zu < n_cand:
                        var zv = Int(cand_n[zu])
                        if zv > b1u:
                            b2u = b1u
                            b1u = zv
                        elif zv > b2u:
                            b2u = zv
                        zu += 1
                    if b1u >= 2 and b1u >= b2u + 1:
                        break
                    unique_only = False
                if occ_v > vote_cap:
                    oi += 1
                    continue
                var slot = Int(ord_slot[oi])
                var sidx = rid * slots_per_read + slot
                var a = Int(occ_start[sidx])
                var b = Int(occ_end[sidx])
                var q_off = Int(q_offs[sidx])
                var pi = a
                while pi < b:
                    var cid = Int(postings[pi * 2])
                    var pos = Int(postings[pi * 2 + 1])
                    if cid >= 0 and cid < n_contigs and pos >= q_off:
                        var start0 = pos - q_off
                        var vk = (UInt64(cid) << 32) | UInt64(start0)
                        var found = -1
                        if last_i >= 0 and cand_key[last_i] == vk:
                            found = last_i
                        else:
                            var ci = 0
                            while ci < n_cand:
                                if cand_key[ci] == vk:
                                    found = ci
                                    break
                                ci += 1
                        if found >= 0:
                            cand_n[found] = cand_n[found] + 1
                            last_i = found
                        elif n_cand < TOP:
                            cand_key[n_cand] = vk
                            cand_n[n_cand] = 1
                            last_i = n_cand
                            n_cand += 1
                        else:
                            var min_i = 0
                            var min_n = cand_n[0]
                            var zj = 1
                            while zj < TOP:
                                if cand_n[zj] < min_n:
                                    min_n = cand_n[zj]
                                    min_i = zj
                                zj += 1
                            if min_n <= 1:
                                cand_key[min_i] = vk
                                cand_n[min_i] = 1
                                last_i = min_i
                    pi += 1
                if n_cand > 0 and (not unique_only) and oi + 1 < n_ord:
                    var b1 = 0
                    var b2 = 0
                    var zi = 0
                    while zi < n_cand:
                        var zv2 = Int(cand_n[zi])
                        if zv2 > b1:
                            b2 = b1
                            b1 = zv2
                        elif zv2 > b2:
                            b2 = zv2
                        zi += 1
                    if b1 >= 4 and b1 >= b2 + 2:
                        break
                oi += 1

            var pick_i = InlineArray[Int32, KEEP](fill=Int32(-1))
            var pick_n = InlineArray[Int32, KEEP](fill=Int32(0))
            var n_pick = 0
            var ci2 = 0
            while ci2 < n_cand:
                var votes = cand_n[ci2]
                var inserted = False
                var p = 0
                while p < n_pick:
                    if votes > pick_n[p]:
                        var sh = n_pick
                        if sh > KEEP - 1:
                            sh = KEEP - 1
                        while sh > p:
                            pick_i[sh] = pick_i[sh - 1]
                            pick_n[sh] = pick_n[sh - 1]
                            sh -= 1
                        pick_i[p] = Int32(ci2)
                        pick_n[p] = votes
                        if n_pick < KEEP:
                            n_pick += 1
                        inserted = True
                        break
                    p += 1
                if not inserted and n_pick < KEEP:
                    pick_i[n_pick] = Int32(ci2)
                    pick_n[n_pick] = votes
                    n_pick += 1
                ci2 += 1

            var best_cost = 9999
            var best_votes: Int32 = 0
            var best_cid = -1
            var best_pos = 0
            var best_sl = 0
            var best_sr = 0
            var pk = 0
            while pk < n_pick:
                var ix = Int(pick_i[pk])
                var bkey = cand_key[ix]
                var votes = pick_n[pk]
                var bcid = Int(bkey >> 32)
                var bstart = Int(bkey & UInt64(0xFFFFFFFF))
                if bcid >= 0 and bcid < n_contigs and bstart >= 0:
                    var off0 = Int(contig_off[bcid])
                    var off1 = Int(contig_off[bcid + 1])
                    var clen = off1 - off0
                    var adj_i = 0
                    while adj_i < 3:
                        var adj = 0
                        if adj_i == 1:
                            adj = -1
                        elif adj_i == 2:
                            adj = 1
                        var btry = bstart + adj
                        if btry >= 0 and btry + qlen <= clen:
                            var ref_base = off0 + btry
                            var nm_all = 0
                            var j = 0
                            while j < qlen and nm_all <= budget:
                                var rc = sequences[ref_base + j]
                                var qc = codes[q_base + j]
                                if rc > 3 or qc > 3 or rc != qc:
                                    nm_all += 1
                                j += 1
                            var sl = 0
                            var sr = 0
                            var nm = nm_all
                            if nm_all > budget and clip_cap > 0:
                                while sl < clip_cap:
                                    var rc0 = sequences[ref_base + sl]
                                    var qc0 = codes[q_base + sl]
                                    if rc0 <= 3 and qc0 <= 3 and rc0 == qc0:
                                        break
                                    sl += 1
                                while sr < clip_cap:
                                    var jr = qlen - 1 - sr
                                    if jr <= sl:
                                        break
                                    var rc1 = sequences[ref_base + jr]
                                    var qc1 = codes[q_base + jr]
                                    if rc1 <= 3 and qc1 <= 3 and rc1 == qc1:
                                        break
                                    sr += 1
                                nm = 0
                                var jm = sl
                                while jm < qlen - sr and nm <= budget:
                                    var rcm = sequences[ref_base + jm]
                                    var qcm = codes[q_base + jm]
                                    if rcm > 3 or qcm > 3 or rcm != qcm:
                                        nm += 1
                                    jm += 1
                            var alen = qlen - sl - sr
                            if nm <= budget and alen >= 32:
                                if (
                                    nm < best_cost
                                    or (nm == best_cost and votes > best_votes)
                                ):
                                    best_cost = nm
                                    best_votes = votes
                                    best_cid = bcid
                                    best_pos = btry + sl
                                    best_sl = sl
                                    best_sr = sr
                            if best_cost == 0:
                                break
                            if adj == 0 and (nm > budget or alen < 32):
                                if btry + qlen + 1 <= clen:
                                    var g = 0
                                    while g <= qlen:
                                        var nm_d = 1
                                        var t = 0
                                        while t < g and nm_d <= budget:
                                            var rcd = sequences[ref_base + t]
                                            var qcd = codes[q_base + t]
                                            if rcd > 3 or qcd > 3 or rcd != qcd:
                                                nm_d += 1
                                            t += 1
                                        while t < qlen and nm_d <= budget:
                                            var rcd2 = sequences[ref_base + t + 1]
                                            var qcd2 = codes[q_base + t]
                                            if (
                                                rcd2 > 3
                                                or qcd2 > 3
                                                or rcd2 != qcd2
                                            ):
                                                nm_d += 1
                                            t += 1
                                        if nm_d <= budget and nm_d < best_cost:
                                            best_cost = nm_d
                                            best_votes = votes
                                            best_cid = bcid
                                            best_pos = btry
                                            best_sl = 0
                                            best_sr = 0
                                        g += 8
                                if qlen > 1 and btry + (qlen - 1) <= clen:
                                    var gi = 0
                                    while gi <= qlen - 1:
                                        var nm_i = 1
                                        var ti = 0
                                        while ti < gi and nm_i <= budget:
                                            var rci = sequences[ref_base + ti]
                                            var qci = codes[q_base + ti]
                                            if rci > 3 or qci > 3 or rci != qci:
                                                nm_i += 1
                                            ti += 1
                                        var tj = gi + 1
                                        var rj = gi
                                        while tj < qlen and nm_i <= budget:
                                            var rci2 = sequences[ref_base + rj]
                                            var qci2 = codes[q_base + tj]
                                            if (
                                                rci2 > 3
                                                or qci2 > 3
                                                or rci2 != qci2
                                            ):
                                                nm_i += 1
                                            tj += 1
                                            rj += 1
                                        if nm_i <= budget and nm_i < best_cost:
                                            best_cost = nm_i
                                            best_votes = votes
                                            best_cid = bcid
                                            best_pos = btry
                                            best_sl = 0
                                            best_sr = 0
                                        gi += 8
                        adj_i += 1
                    if best_cost == 0:
                        break
                pk += 1

            var accept = False
            if best_cid < 0 and pass_mode == 1:
                var n_hi = 0
                var ri = 0
                while ri < n_ord and n_hi < 8 and best_cid < 0:
                    var occ_r = Int(ord_occ[ri])
                    if occ_r > vote_cap and occ_r <= 4096:
                        n_hi += 1
                        var slot_r = Int(ord_slot[ri])
                        var sidx_r = rid * slots_per_read + slot_r
                        var a_r = Int(occ_start[sidx_r])
                        var b_r = Int(occ_end[sidx_r])
                        var q_off_r = Int(q_offs[sidx_r])
                        var pi_r = a_r
                        while pi_r < b_r and best_cid < 0:
                            var cid_r = Int(postings[pi_r * 2])
                            var pos_r = Int(postings[pi_r * 2 + 1])
                            if (
                                cid_r >= 0
                                and cid_r < n_contigs
                                and pos_r >= q_off_r
                            ):
                                var bstart_r = pos_r - q_off_r
                                var off0_r = Int(contig_off[cid_r])
                                var off1_r = Int(contig_off[cid_r + 1])
                                if (
                                    bstart_r >= 0
                                    and bstart_r + qlen <= (off1_r - off0_r)
                                ):
                                    var ref_r = off0_r + bstart_r
                                    var nm_r = 0
                                    var jr = 0
                                    while jr < qlen and nm_r == 0:
                                        var rcr = sequences[ref_r + jr]
                                        var qcr = codes[q_base + jr]
                                        if rcr > 3 or qcr > 3 or rcr != qcr:
                                            nm_r = 1
                                        jr += 1
                                    if nm_r == 0:
                                        best_cost = 0
                                        best_votes = 1
                                        best_cid = cid_r
                                        best_pos = bstart_r
                                        best_sl = 0
                                        best_sr = 0
                            pi_r += 1
                    ri += 1
            if best_cid >= 0:
                if pass_mode == 1:
                    accept = True
                elif Int(best_votes) >= 2:
                    accept = True
                elif best_cost == 0:
                    accept = True
            if accept:
                var mq: UInt32 = 20
                if Int(best_votes) >= 2 and best_cost == 0:
                    mq = 60
                elif best_cost == 0:
                    mq = 40
                out_cid[rid] = Int32(best_cid)
                out_pos[rid] = UInt32(best_pos)
                out_mapq[rid] = mq
                out_flag[rid] = strand_flag
                out_sl[rid] = UInt32(best_sl)
                out_sr[rid] = UInt32(best_sr)
                out_nm[rid] = UInt32(best_cost)

        var api = _device_api(resolved)
        var ctx = DeviceContext(api=api)
        var k_len = index.k
        var n_table = index.n_keys
        var n_off = n_table + 1
        var n_post = index.n_postings
        var n_contigs = len(index.contig_names)
        var kmers_bytes = n_table * 8
        var offsets_bytes = n_off * 8
        var postings_bytes = n_post * 8
        var seq_bytes = index.seq_size
        var coff_bytes = (n_contigs + 1) * 8
        var science = (
            kmers_bytes
            + offsets_bytes
            + postings_bytes
            + seq_bytes
            + seq_bytes
            + coff_bytes
        )
        _require_hbm(science, resolved)
        print(
            "MojoLinear GPU-speed upload science_bytes=",
            science,
            " n_keys=",
            n_table,
            " n_postings=",
            n_post,
        )

        var dev_kmers = ctx.enqueue_create_buffer[DType.uint64](n_table)
        var dev_offsets = ctx.enqueue_create_buffer[DType.uint64](n_off)
        var dev_postings = ctx.enqueue_create_buffer[DType.uint32](n_post * 2)
        var dev_seq = ctx.enqueue_create_buffer[DType.uint8](seq_bytes)
        var dev_coff = ctx.enqueue_create_buffer[DType.uint64](n_contigs + 1)
        var time_mod = Python.import_module("time")
        var t_upload0 = time_mod.perf_counter()
        upload_mmap_to_device(
            ctx,
            dev_kmers.unsafe_ptr().bitcast[UInt8](),
            index.kmers_addr,
            kmers_bytes,
            "kmers",
        )
        upload_mmap_to_device(
            ctx,
            dev_offsets.unsafe_ptr().bitcast[UInt8](),
            index.offsets_addr,
            offsets_bytes,
            "offsets",
        )
        upload_mmap_to_device(
            ctx,
            dev_postings.unsafe_ptr().bitcast[UInt8](),
            index.postings_addr,
            postings_bytes,
            "postings",
        )
        upload_mmap_to_device(
            ctx,
            dev_seq.unsafe_ptr(),
            index.seq_addr,
            seq_bytes,
            "seq",
        )
        upload_mmap_to_device(
            ctx,
            dev_coff.unsafe_ptr().bitcast[UInt8](),
            index.contig_off_addr,
            coff_bytes,
            "coff",
        )
        var dev_ref_codes = ctx.enqueue_create_buffer[DType.uint8](seq_bytes)
        comptime BLOCK = 256
        var grid_pack_ref = (seq_bytes + BLOCK - 1) // BLOCK
        ctx.enqueue_function[pack_bases_kernel](
            dev_seq.unsafe_ptr(),
            dev_ref_codes.unsafe_ptr(),
            seq_bytes,
            grid_dim=grid_pack_ref,
            block_dim=BLOCK,
        )
        ctx.synchronize()
        var t_upload1 = time_mod.perf_counter()
        print("MojoLinear GPU-speed upload_wall_s=", t_upload1 - t_upload0)
        print("MojoLinear GPU-speed index resident ok")

        var batch_size = _env_int("METHYLGRAPHER_LINEAR_READ_BATCH", 16384)
        var seed_stride = _env_int("METHYLGRAPHER_SPEED_SEED_STRIDE", 3)
        var max_occ = _env_int("METHYLGRAPHER_SPEED_MAX_OCC", 4096)
        var vote_fast = _env_int("METHYLGRAPHER_SPEED_VOTE_FAST", 32)
        var vote_occ = _env_int("METHYLGRAPHER_SPEED_VOTE_OCC", 256)
        var max_diff = _env_int("METHYLGRAPHER_SPEED_MAX_DIFF", 8)
        var max_soft = _env_int("METHYLGRAPHER_SPEED_MAX_SOFT", 12)
        var paired = fq2.byte_length() > 0
        print(
            "MojoLinear GPU-speed map start batch=",
            batch_size,
            " stride=",
            seed_stride,
            " max_occ=",
            max_occ,
            " vote_fast=",
            vote_fast,
            " vote_occ=",
            vote_occ,
            " max_diff=",
            max_diff,
            " max_soft=",
            max_soft,
            " paired=",
            paired,
        )

        var dummy_rid = ctx.enqueue_create_buffer[DType.int32](1)
        var dummy_skip_c = ctx.enqueue_create_buffer[DType.int32](1)
        var dummy_skip_n = ctx.enqueue_create_buffer[DType.uint32](1)

        var fh = open_text_write(out_sam)
        _write_sam_header(fh, index)
        var fh1 = _open_fastq(fq1)
        var fh2 = fh1
        if paired:
            fh2 = _open_fastq(fq2)

        var n_mapped = 0
        var n_reads = 0
        var n_batches = 0
        var t_map0 = time_mod.perf_counter()
        var batch1 = List[FastqRec]()
        var batch2 = List[FastqRec]()
        _read_pe_batch(
            fh1, fh2, paired, batch_size, bs_r1, bs_r2, batch1, batch2
        )

        while len(batch1) > 0:
            var seqs = List[String]()
            for r in batch1:
                seqs.append(r.seq)
            if paired:
                for r in batch2:
                    seqs.append(r.seq)
            var n_seq = len(seqs)
            var max_len = 0
            var ri0 = 0
            while ri0 < n_seq:
                var L0 = seqs[ri0].byte_length()
                if L0 > max_len:
                    max_len = L0
                ri0 += 1
            if max_len < k_len:
                max_len = k_len
            var slots_per = 1
            if max_len >= k_len:
                slots_per = ((max_len - k_len) // seed_stride) + 1
            var n_slots = n_seq * slots_per
            var n_bases = n_seq * max_len
            var host_bases = ctx.enqueue_create_host_buffer[DType.uint8](n_bases)
            var host_lens = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var si = 0
            while si < n_seq:
                var s = seqs[si]
                var L = s.byte_length()
                host_lens[si] = UInt32(L)
                var base = si * max_len
                var p = 0
                while p < max_len:
                    if p < L:
                        host_bases[base + p] = UInt8(ord(s[byte = p : p + 1]))
                    else:
                        host_bases[base + p] = 78
                    p += 1
                si += 1

            var dev_bases = ctx.enqueue_create_buffer[DType.uint8](n_bases)
            var dev_codes = ctx.enqueue_create_buffer[DType.uint8](n_bases)
            var dev_rc = ctx.enqueue_create_buffer[DType.uint8](n_bases)
            var dev_lens = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_keys = ctx.enqueue_create_buffer[DType.uint64](n_slots)
            var dev_qoff = ctx.enqueue_create_buffer[DType.uint32](n_slots)
            var dev_start = ctx.enqueue_create_buffer[DType.uint64](n_slots)
            var dev_end = ctx.enqueue_create_buffer[DType.uint64](n_slots)
            var dev_cid_f = ctx.enqueue_create_buffer[DType.int32](n_seq)
            var dev_pos_f = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_mapq_f = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_flag_f = ctx.enqueue_create_buffer[DType.int32](n_seq)
            var dev_sl_f = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_sr_f = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_nm_f = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_cid_r = ctx.enqueue_create_buffer[DType.int32](n_seq)
            var dev_pos_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_mapq_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_flag_r = ctx.enqueue_create_buffer[DType.int32](n_seq)
            var dev_sl_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_sr_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_nm_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            ctx.enqueue_copy(src_buf=host_bases, dst_buf=dev_bases)
            ctx.enqueue_copy(src_buf=host_lens, dst_buf=dev_lens)
            var grid_b = (n_bases + BLOCK - 1) // BLOCK
            var grid_s = (n_slots + BLOCK - 1) // BLOCK
            var grid_r = (n_seq + BLOCK - 1) // BLOCK
            ctx.enqueue_function[pack_bases_kernel](
                dev_bases.unsafe_ptr(),
                dev_codes.unsafe_ptr(),
                n_bases,
                grid_dim=grid_b,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[encode_strided_kernel](
                dev_codes.unsafe_ptr(),
                dev_keys.unsafe_ptr(),
                dev_qoff.unsafe_ptr(),
                dev_lens.unsafe_ptr(),
                n_seq,
                max_len,
                k_len,
                seed_stride,
                slots_per,
                grid_dim=grid_s,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[locate_bsearch_kernel](
                dev_keys.unsafe_ptr(),
                dev_kmers.unsafe_ptr(),
                dev_offsets.unsafe_ptr(),
                dev_start.unsafe_ptr(),
                dev_end.unsafe_ptr(),
                n_slots,
                n_table,
                max_occ,
                grid_dim=grid_s,
                block_dim=BLOCK,
            )
            var host_cid_init = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var zi = 0
            while zi < n_seq:
                host_cid_init[zi] = Int32(-1)
                zi += 1
            ctx.enqueue_copy(src_buf=host_cid_init, dst_buf=dev_cid_f)
            ctx.enqueue_function[consensus_extend_kernel](
                dev_start.unsafe_ptr(),
                dev_end.unsafe_ptr(),
                dev_qoff.unsafe_ptr(),
                dev_postings.unsafe_ptr(),
                n_post,
                dev_ref_codes.unsafe_ptr(),
                dev_coff.unsafe_ptr(),
                n_contigs,
                dev_codes.unsafe_ptr(),
                dev_lens.unsafe_ptr(),
                max_len,
                slots_per,
                n_seq,
                max_diff,
                max_soft,
                vote_fast,
                0,
                -1,
                dummy_rid.unsafe_ptr(),
                dummy_skip_c.unsafe_ptr(),
                dummy_skip_n.unsafe_ptr(),
                0,
                Int32(0),
                dev_cid_f.unsafe_ptr(),
                dev_pos_f.unsafe_ptr(),
                dev_mapq_f.unsafe_ptr(),
                dev_flag_f.unsafe_ptr(),
                dev_sl_f.unsafe_ptr(),
                dev_sr_f.unsafe_ptr(),
                dev_nm_f.unsafe_ptr(),
                grid_dim=grid_r,
                block_dim=BLOCK,
            )
            var host_cid_f0 = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            ctx.enqueue_copy(src_buf=dev_cid_f, dst_buf=host_cid_f0)
            ctx.synchronize()
            var n_unmap_f = 0
            var ui = 0
            while ui < n_seq:
                if Int(host_cid_f0[ui]) < 0:
                    n_unmap_f += 1
                ui += 1
            if n_batches == 0:
                print(
                    "MojoLinear GPU-speed fw_pass0_unmapped=",
                    n_unmap_f,
                    "/",
                    n_seq,
                )
            if n_unmap_f > 0:
                var host_rid_f = ctx.enqueue_create_host_buffer[DType.int32](
                    n_unmap_f
                )
                var wf = 0
                ui = 0
                while ui < n_seq:
                    if Int(host_cid_f0[ui]) < 0:
                        host_rid_f[wf] = Int32(ui)
                        wf += 1
                    ui += 1
                var dev_rid_f = ctx.enqueue_create_buffer[DType.int32](n_unmap_f)
                ctx.enqueue_copy(src_buf=host_rid_f, dst_buf=dev_rid_f)
                var grid_uf = (n_unmap_f + BLOCK - 1) // BLOCK
                ctx.enqueue_function[consensus_extend_kernel](
                    dev_start.unsafe_ptr(),
                    dev_end.unsafe_ptr(),
                    dev_qoff.unsafe_ptr(),
                    dev_postings.unsafe_ptr(),
                    n_post,
                    dev_ref_codes.unsafe_ptr(),
                    dev_coff.unsafe_ptr(),
                    n_contigs,
                    dev_codes.unsafe_ptr(),
                    dev_lens.unsafe_ptr(),
                    max_len,
                    slots_per,
                    n_seq,
                    max_diff,
                    max_soft,
                    vote_occ,
                    1,
                    n_unmap_f,
                    dev_rid_f.unsafe_ptr(),
                    dummy_skip_c.unsafe_ptr(),
                    dummy_skip_n.unsafe_ptr(),
                    0,
                    Int32(0),
                    dev_cid_f.unsafe_ptr(),
                    dev_pos_f.unsafe_ptr(),
                    dev_mapq_f.unsafe_ptr(),
                    dev_flag_f.unsafe_ptr(),
                    dev_sl_f.unsafe_ptr(),
                    dev_sr_f.unsafe_ptr(),
                    dev_nm_f.unsafe_ptr(),
                    grid_dim=grid_uf,
                    block_dim=BLOCK,
                )

            var host_cid_fw = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var host_nm_fw = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            ctx.enqueue_copy(src_buf=dev_cid_f, dst_buf=host_cid_fw)
            ctx.enqueue_copy(src_buf=dev_nm_f, dst_buf=host_nm_fw)
            ctx.synchronize()
            if n_batches == 0:
                var n_unmap_f1 = 0
                ui = 0
                while ui < n_seq:
                    if Int(host_cid_fw[ui]) < 0:
                        n_unmap_f1 += 1
                    ui += 1
                print(
                    "MojoLinear GPU-speed fw_unmapped=",
                    n_unmap_f1,
                    "/",
                    n_seq,
                )

            ctx.enqueue_copy(src_buf=host_cid_init, dst_buf=dev_cid_r)
            ctx.enqueue_function[rc_codes_kernel](
                dev_codes.unsafe_ptr(),
                dev_rc.unsafe_ptr(),
                dev_lens.unsafe_ptr(),
                n_seq,
                max_len,
                grid_dim=grid_r,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[encode_strided_kernel](
                dev_rc.unsafe_ptr(),
                dev_keys.unsafe_ptr(),
                dev_qoff.unsafe_ptr(),
                dev_lens.unsafe_ptr(),
                n_seq,
                max_len,
                k_len,
                seed_stride,
                slots_per,
                grid_dim=grid_s,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[locate_bsearch_kernel](
                dev_keys.unsafe_ptr(),
                dev_kmers.unsafe_ptr(),
                dev_offsets.unsafe_ptr(),
                dev_start.unsafe_ptr(),
                dev_end.unsafe_ptr(),
                n_slots,
                n_table,
                max_occ,
                grid_dim=grid_s,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[consensus_extend_kernel](
                dev_start.unsafe_ptr(),
                dev_end.unsafe_ptr(),
                dev_qoff.unsafe_ptr(),
                dev_postings.unsafe_ptr(),
                n_post,
                dev_ref_codes.unsafe_ptr(),
                dev_coff.unsafe_ptr(),
                n_contigs,
                dev_rc.unsafe_ptr(),
                dev_lens.unsafe_ptr(),
                max_len,
                slots_per,
                n_seq,
                max_diff,
                max_soft,
                vote_fast,
                0,
                -1,
                dummy_rid.unsafe_ptr(),
                dev_cid_f.unsafe_ptr(),
                dev_nm_f.unsafe_ptr(),
                1,
                Int32(16),
                dev_cid_r.unsafe_ptr(),
                dev_pos_r.unsafe_ptr(),
                dev_mapq_r.unsafe_ptr(),
                dev_flag_r.unsafe_ptr(),
                dev_sl_r.unsafe_ptr(),
                dev_sr_r.unsafe_ptr(),
                dev_nm_r.unsafe_ptr(),
                grid_dim=grid_r,
                block_dim=BLOCK,
            )
            var host_cid_r0 = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            ctx.enqueue_copy(src_buf=dev_cid_r, dst_buf=host_cid_r0)
            ctx.synchronize()
            var n_unmap_r = 0
            ui = 0
            while ui < n_seq:
                var exact_fw = Int(host_cid_fw[ui]) >= 0 and Int(host_nm_fw[ui]) == 0
                if Int(host_cid_r0[ui]) < 0 and (not exact_fw):
                    n_unmap_r += 1
                ui += 1
            if n_unmap_r > 0:
                var host_rid_r = ctx.enqueue_create_host_buffer[DType.int32](
                    n_unmap_r
                )
                var wr = 0
                ui = 0
                while ui < n_seq:
                    var exact_fw2 = (
                        Int(host_cid_fw[ui]) >= 0 and Int(host_nm_fw[ui]) == 0
                    )
                    if Int(host_cid_r0[ui]) < 0 and (not exact_fw2):
                        host_rid_r[wr] = Int32(ui)
                        wr += 1
                    ui += 1
                var dev_rid_r = ctx.enqueue_create_buffer[DType.int32](n_unmap_r)
                ctx.enqueue_copy(src_buf=host_rid_r, dst_buf=dev_rid_r)
                var grid_ur = (n_unmap_r + BLOCK - 1) // BLOCK
                ctx.enqueue_function[consensus_extend_kernel](
                    dev_start.unsafe_ptr(),
                    dev_end.unsafe_ptr(),
                    dev_qoff.unsafe_ptr(),
                    dev_postings.unsafe_ptr(),
                    n_post,
                    dev_ref_codes.unsafe_ptr(),
                    dev_coff.unsafe_ptr(),
                    n_contigs,
                    dev_rc.unsafe_ptr(),
                    dev_lens.unsafe_ptr(),
                    max_len,
                    slots_per,
                    n_seq,
                    max_diff,
                    max_soft,
                    vote_occ,
                    1,
                    n_unmap_r,
                    dev_rid_r.unsafe_ptr(),
                    dev_cid_f.unsafe_ptr(),
                    dev_nm_f.unsafe_ptr(),
                    1,
                    Int32(16),
                    dev_cid_r.unsafe_ptr(),
                    dev_pos_r.unsafe_ptr(),
                    dev_mapq_r.unsafe_ptr(),
                    dev_flag_r.unsafe_ptr(),
                    dev_sl_r.unsafe_ptr(),
                    dev_sr_r.unsafe_ptr(),
                    dev_nm_r.unsafe_ptr(),
                    grid_dim=grid_ur,
                    block_dim=BLOCK,
                )

            var host_cid_f = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var host_pos_f = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_mapq_f = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_flag_f = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var host_sl_f = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_sr_f = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_nm_f = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_cid_r = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var host_pos_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_mapq_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_flag_r = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var host_sl_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_sr_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_nm_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            ctx.enqueue_copy(src_buf=dev_cid_f, dst_buf=host_cid_f)
            ctx.enqueue_copy(src_buf=dev_pos_f, dst_buf=host_pos_f)
            ctx.enqueue_copy(src_buf=dev_mapq_f, dst_buf=host_mapq_f)
            ctx.enqueue_copy(src_buf=dev_flag_f, dst_buf=host_flag_f)
            ctx.enqueue_copy(src_buf=dev_sl_f, dst_buf=host_sl_f)
            ctx.enqueue_copy(src_buf=dev_sr_f, dst_buf=host_sr_f)
            ctx.enqueue_copy(src_buf=dev_nm_f, dst_buf=host_nm_f)
            ctx.enqueue_copy(src_buf=dev_cid_r, dst_buf=host_cid_r)
            ctx.enqueue_copy(src_buf=dev_pos_r, dst_buf=host_pos_r)
            ctx.enqueue_copy(src_buf=dev_mapq_r, dst_buf=host_mapq_r)
            ctx.enqueue_copy(src_buf=dev_flag_r, dst_buf=host_flag_r)
            ctx.enqueue_copy(src_buf=dev_sl_r, dst_buf=host_sl_r)
            ctx.enqueue_copy(src_buf=dev_sr_r, dst_buf=host_sr_r)
            ctx.enqueue_copy(src_buf=dev_nm_r, dst_buf=host_nm_r)
            ctx.synchronize()

            var n1 = len(batch1)
            var pj = 0
            while pj < n1:
                var use_rc = False
                var f_cid = Int(host_cid_f[pj])
                var r_cid = Int(host_cid_r[pj])
                if f_cid < 0 and r_cid >= 0:
                    use_rc = True
                elif f_cid >= 0 and r_cid >= 0:
                    var fnm = Int(host_nm_f[pj])
                    var rnm = Int(host_nm_r[pj])
                    if rnm < fnm or (
                        rnm == fnm
                        and Int(host_mapq_r[pj]) > Int(host_mapq_f[pj])
                    ):
                        use_rc = True
                var h1: LinearHit
                if use_rc:
                    h1 = _hit_from_gpu(
                        index,
                        batch1[pj].name,
                        batch1[pj].seq,
                        batch1[pj].original_seq,
                        r_cid,
                        Int(host_pos_r[pj]),
                        Int(host_mapq_r[pj]),
                        Int(host_flag_r[pj]),
                        Int(host_sl_r[pj]),
                        Int(host_sr_r[pj]),
                    )
                else:
                    h1 = _hit_from_gpu(
                        index,
                        batch1[pj].name,
                        batch1[pj].seq,
                        batch1[pj].original_seq,
                        f_cid,
                        Int(host_pos_f[pj]),
                        Int(host_mapq_f[pj]),
                        Int(host_flag_f[pj]),
                        Int(host_sl_f[pj]),
                        Int(host_sr_f[pj]),
                    )
                h1.qual = batch1[pj].qual
                if not paired:
                    if h1.contig != "*":
                        n_mapped += 1
                    fh.write(_sam_with_rg(h1) + "\n")
                    n_reads += 1
                else:
                    var r2i = n1 + pj
                    var use_rc2 = False
                    var f2 = Int(host_cid_f[r2i])
                    var r2c = Int(host_cid_r[r2i])
                    if f2 < 0 and r2c >= 0:
                        use_rc2 = True
                    elif f2 >= 0 and r2c >= 0:
                        var fnm2 = Int(host_nm_f[r2i])
                        var rnm2 = Int(host_nm_r[r2i])
                        if rnm2 < fnm2 or (
                            rnm2 == fnm2
                            and Int(host_mapq_r[r2i]) > Int(host_mapq_f[r2i])
                        ):
                            use_rc2 = True
                    var h2: LinearHit
                    if use_rc2:
                        h2 = _hit_from_gpu(
                            index,
                            batch2[pj].name,
                            batch2[pj].seq,
                            batch2[pj].original_seq,
                            r2c,
                            Int(host_pos_r[r2i]),
                            Int(host_mapq_r[r2i]),
                            Int(host_flag_r[r2i]),
                            Int(host_sl_r[r2i]),
                            Int(host_sr_r[r2i]),
                        )
                    else:
                        h2 = _hit_from_gpu(
                            index,
                            batch2[pj].name,
                            batch2[pj].seq,
                            batch2[pj].original_seq,
                            f2,
                            Int(host_pos_f[r2i]),
                            Int(host_mapq_f[r2i]),
                            Int(host_flag_f[r2i]),
                            Int(host_sl_f[r2i]),
                            Int(host_sr_f[r2i]),
                        )
                    h2.qual = batch2[pj].qual
                    var paired_hits = pair_hits(h1, h2)
                    var p1 = paired_hits.r1.copy()
                    var p2 = paired_hits.r2.copy()
                    if p1.contig != "*":
                        n_mapped += 1
                    if p2.contig != "*":
                        n_mapped += 1
                    fh.write(_sam_with_rg(p1) + "\n")
                    fh.write(_sam_with_rg(p2) + "\n")
                    n_reads += 2
                pj += 1

            n_batches += 1
            if n_batches == 1 or n_batches % 5 == 0:
                print(
                    "MojoLinear GPU-speed progress batches=",
                    n_batches,
                    " gpu_reads≈",
                    n_reads,
                )
            _read_pe_batch(
                fh1, fh2, paired, batch_size, bs_r1, bs_r2, batch1, batch2
            )

        fh1.close()
        if paired:
            fh2.close()
        fh.close()
        var t_map1 = time_mod.perf_counter()
        print(
            "wrote SAM -> ",
            out_sam,
            " mapped_records=",
            n_mapped,
            " reads=",
            n_reads,
            " backend=gpu-speed",
        )
        print(
            "MojoLinear GPU-speed map_wall_s=",
            t_map1 - t_map0,
            " reads=",
            n_reads,
        )
        return n_mapped
