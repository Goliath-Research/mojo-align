# FM / BWA-MEM-style GPU engine (opt-in). METHYLGRAPHER_LINEAR_ENGINE=fm
#
# Uses Clara's bwameth BWA 0.7 FM-index (.bwt/.sa/.pac/.ann), not the dense-v1
# k-mer posting pack. Seed via FM backward search (min_seed_len=19), SA resolve,
# gapless/softclip extend on the forward pac (f*/r* contigs). Default remains
# parity until Clara gates pass.

from std.collections import Dict, List
from std.python import Python, PythonObject
from std.sys import has_accelerator

from gpu_device import select_device
from gpu_kernels import _device_api, kernel_target_label, probe_device_context
from linear_extend import hit_to_sam_line, pair_hits, LinearHit
from linear_fm_index import FmIndex, fm_prefix_from_ref
from linear_gpu_locate import (
    FastqRec,
    _canonical_contig,
    _env_int,
    _env_str,
    _open_fastq,
    _read_pe_batch,
    _require_hbm,
)
from utility import open_text_write, reverse_complement


def _write_sam_header_fm(fh: PythonObject, index: FmIndex) raises:
    fh.write("@HD\tVN:1.6\tSO:unsorted\n")
    var seen = Dict[String, Int]()
    var ci = 0
    while ci < index.contig_count():
        var canon = _canonical_contig(index.contig_name(ci))
        if canon not in seen:
            seen[canon] = 1
            fh.write(
                "@SQ\tSN:"
                + canon
                + "\tLN:"
                + String(index.contig_length(ci))
                + "\n"
            )
        ci += 1
    var rg = _env_str("METHYLGRAPHER_RG_ID", "mojo1")
    var sm = _env_str("METHYLGRAPHER_RG_SM", "sample")
    var lb = _env_str("METHYLGRAPHER_RG_LB", "lib1")
    var pl = _env_str("METHYLGRAPHER_RG_PL", "ILLUMINA")
    fh.write(
        "@RG\tID:"
        + rg
        + "\tSM:"
        + sm
        + "\tLB:"
        + lb
        + "\tPL:"
        + pl
        + "\tPU:"
        + rg
        + "\n"
    )
    fh.write("@PG\tID:MojoFq2bamMeth\tPN:MojoFq2bamMeth\tVN:0.1.0-mojo-fm\n")
    try:
        fh.flush()
    except:
        pass


def _sam_with_rg_fm(h: LinearHit) raises -> String:
    return hit_to_sam_line(h) + "\tRG:Z:" + _env_str("METHYLGRAPHER_RG_ID", "mojo1")


def _hit_from_fm(
    index: FmIndex,
    name: String,
    align_seq: String,
    original_seq: String,
    rid: Int,
    start0: Int,
    mapq: Int,
    flag: Int,
    soft_l: Int,
    soft_r: Int,
) raises -> LinearHit:
    var qlen = align_seq.byte_length()
    var emit = original_seq
    if emit.byte_length() == 0:
        emit = align_seq
    if rid < 0:
        return LinearHit(name, 4, "*", 0, 0, "*", emit, "*")
    var m = qlen - soft_l - soft_r
    if m < 1:
        return LinearHit(name, 4, "*", 0, 0, "*", emit, "*")
    var cigar = String("")
    if soft_l > 0:
        cigar = cigar + String(soft_l) + "S"
    cigar = cigar + String(m) + "M"
    if soft_r > 0:
        cigar = cigar + String(soft_r) + "S"
    if (flag & 16) != 0:
        emit = reverse_complement(emit)
    return LinearHit(
        name,
        flag,
        _canonical_contig(index.contig_name(rid)),
        start0 + 1,
        mapq,
        cigar,
        emit,
        "*",
    )


def map_fastq_fm_gpu(
    ref_fa: String,
    fq1: String,
    out_sam: String,
    device: String,
    fq2: String = "",
    bs_r1: String = "",
    bs_r2: String = "",
) raises -> Int:
    """FM-index GPU map: BWA-MEM-shaped exact seed + extend on .pac."""
    var prefix = fm_prefix_from_ref(ref_fa)
    var index = FmIndex()
    index.load(prefix)

    var resolved = select_device(device)
    var backend = probe_device_context(resolved)
    var target = kernel_target_label(resolved)
    print(
        "MojoLinear GPU-fm device=",
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
            "MojoLinear GPU-fm needs DeviceContext cuda/hip, got " + backend
        )

    comptime if not has_accelerator():
        raise Error("map_fastq_fm_gpu requires accelerator build")
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
                        "MojoLinear GPU-fm upload ",
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

        def fm_seed_extend_kernel(
            bwt: UnsafePointer[UInt32, MutAnyOrigin],
            sa: UnsafePointer[UInt64, MutAnyOrigin],
            pac: UnsafePointer[UInt8, MutAnyOrigin],
            contig_off: UnsafePointer[UInt64, MutAnyOrigin],
            contig_len: UnsafePointer[UInt32, MutAnyOrigin],
            n_contigs: Int,
            primary: UInt64,
            seq_len: UInt64,
            l_pac: UInt64,
            l2_0: UInt64,
            l2_1: UInt64,
            l2_2: UInt64,
            l2_3: UInt64,
            l2_4: UInt64,
            sa_intv: Int,
            n_sa: Int,
            codes: UnsafePointer[UInt8, MutAnyOrigin],
            read_lens: UnsafePointer[UInt32, MutAnyOrigin],
            max_len: Int,
            n_reads: Int,
            seed_len: Int,
            seed_stride: Int,
            max_occ: Int,
            max_diff: Int,
            max_soft: Int,
            max_adj: Int,
            out_rid: UnsafePointer[Int32, MutAnyOrigin],
            out_pos: UnsafePointer[UInt32, MutAnyOrigin],
            out_mapq: UnsafePointer[UInt32, MutAnyOrigin],
            out_flag: UnsafePointer[Int32, MutAnyOrigin],
            out_sl: UnsafePointer[UInt32, MutAnyOrigin],
            out_sr: UnsafePointer[UInt32, MutAnyOrigin],
            out_nm: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            var rid = Int(block_idx.x * block_dim.x + thread_idx.x)
            if rid >= n_reads:
                return
            out_rid[rid] = Int32(-1)
            out_pos[rid] = 0
            out_mapq[rid] = 0
            out_flag[rid] = Int32(4)
            out_sl[rid] = 0
            out_sr[rid] = 0
            out_nm[rid] = 0
            var qlen = Int(read_lens[rid])
            if qlen < seed_len:
                return
            var budget = max_diff
            if budget < 2:
                budget = 2
            if budget > qlen // 2:
                budget = qlen // 2
            var clip_cap = max_soft
            if clip_cap > qlen // 4:
                clip_cap = qlen // 4
            var adj_cap = max_adj
            if adj_cap < 0:
                adj_cap = 0
            if adj_cap > 12:
                adj_cap = 12
            var q_base = rid * max_len
            var L2_0 = l2_0
            var L2_1 = l2_1
            var L2_2 = l2_2
            var L2_3 = l2_3
            var L2_4 = l2_4

            var best_nm = 9999
            var best_rid = -1
            var best_pos = 0
            var best_sl = 0
            var best_sr = 0
            var best_mq: UInt32 = 0
            var best_flag = 0

            # Unique SMEM: left-extend from each right endpoint until occ==1.
            var e = seed_len - 1
            while e < qlen:
                var k: UInt64 = 0
                var l = seq_len
                var s = e
                var uk: UInt64 = 0
                var ul: UInt64 = 0
                var us = -1
                var found = False
                var last_k: UInt64 = 0
                var last_l: UInt64 = 0
                var last_s = -1
                while s >= 0:
                    var cc = Int(codes[q_base + s])
                    if cc > 3:
                        break
                    var ok: UInt64 = 0
                    var ol: UInt64 = 0
                    var pass_i = 0
                    while pass_i < 2:
                        if pass_i == 0 and k == 0:
                            ok = 0
                            pass_i += 1
                            continue
                        var kv: UInt64 = l
                        if pass_i == 0:
                            kv = k - 1
                        var nocc: UInt64 = 0
                        var k_m = Int(kv)
                        if k_m < 0:
                            nocc = 0
                        elif UInt64(k_m) == seq_len:
                            if cc == 0:
                                nocc = L2_1 - L2_0
                            elif cc == 1:
                                nocc = L2_2 - L2_1
                            elif cc == 2:
                                nocc = L2_3 - L2_2
                            else:
                                nocc = L2_4 - L2_3
                        else:
                            var kko = UInt64(k_m)
                            if kko >= primary:
                                kko -= 1
                            var baseo = Int((kko >> 7) << 4)
                            nocc = UInt64(bwt[baseo + cc * 2]) | (
                                UInt64(bwt[baseo + cc * 2 + 1]) << 32
                            )
                            var po = baseo + 8
                            var endo = po + Int(
                                (
                                    (
                                        (kko >> 5)
                                        - ((kko & UInt64(0xFFFFFFFFFFFFFF80)) >> 5)
                                    )
                                    << 1
                                )
                            )
                            while po < endo:
                                var yo = (UInt64(bwt[po]) << 32) | UInt64(
                                    bwt[po + 1]
                                )
                                var ao: UInt64
                                var bo: UInt64
                                if (cc & 2) != 0:
                                    ao = yo
                                else:
                                    ao = ~yo
                                if (cc & 1) != 0:
                                    bo = yo
                                else:
                                    bo = ~yo
                                var yyo = (ao >> 1) & bo & UInt64(
                                    0x5555555555555555
                                )
                                yyo = (yyo & UInt64(0x3333333333333333)) + (
                                    (yyo >> 2) & UInt64(0x3333333333333333)
                                )
                                yyo = (
                                    (yyo + (yyo >> 4))
                                    & UInt64(0x0F0F0F0F0F0F0F0F)
                                ) * UInt64(0x0101010101010101)
                                nocc += yyo >> 56
                                po += 2
                            var bitso = Int(((~Int(kko)) & 31) << 1)
                            var masko: UInt64 = UInt64(0xFFFFFFFFFFFFFFFF)
                            if bitso > 0 and bitso < 64:
                                masko = ~((UInt64(1) << UInt64(bitso)) - 1)
                            var y5 = (
                                (UInt64(bwt[po]) << 32) | UInt64(bwt[po + 1])
                            ) & masko
                            var a5: UInt64
                            var b6: UInt64
                            if (cc & 2) != 0:
                                a5 = y5
                            else:
                                a5 = ~y5
                            if (cc & 1) != 0:
                                b6 = y5
                            else:
                                b6 = ~y5
                            var yy5 = (a5 >> 1) & b6 & UInt64(
                                0x5555555555555555
                            )
                            yy5 = (yy5 & UInt64(0x3333333333333333)) + (
                                (yy5 >> 2) & UInt64(0x3333333333333333)
                            )
                            yy5 = (
                                (yy5 + (yy5 >> 4)) & UInt64(0x0F0F0F0F0F0F0F0F)
                            ) * UInt64(0x0101010101010101)
                            nocc += yy5 >> 56
                            if cc == 0:
                                nocc -= UInt64((~Int(kko)) & 31)

                        if pass_i == 0:
                            if Int(k) == 0:
                                ok = 0
                            else:
                                ok = nocc
                        else:
                            ol = nocc
                        pass_i += 1
                    var Lc: UInt64 = L2_0
                    if cc == 1:
                        Lc = L2_1
                    elif cc == 2:
                        Lc = L2_2
                    elif cc == 3:
                        Lc = L2_3
                    k = Lc + ok + 1
                    l = Lc + ol
                    if k > l:
                        break
                    var occ_n = Int(l - k + 1)
                    var mlen = e - s + 1
                    if occ_n == 1 and mlen >= seed_len:
                        uk = k
                        ul = l
                        us = s
                        found = True
                        break
                    if occ_n >= 1 and occ_n <= 16 and mlen >= seed_len:
                        last_k = k
                        last_l = l
                        last_s = s
                    if occ_n > max_occ and mlen >= seed_len:
                        break
                    s -= 1
                if not found and last_s >= 0:
                    uk = last_k
                    ul = last_l
                    us = last_s
                    found = True
                if found:
                    var hi = uk
                    while hi <= ul:
                        # SA lookup of interval member hi
                        var kk = hi
                        var sa_v: UInt64 = 0
                        var smask = UInt64(sa_intv - 1)
                        var sa_ok = True
                        while (kk & smask) != 0:
                            sa_v += 1
                            if kk == primary:
                                kk = 0
                                break
                            var xpsi = kk
                            if kk > primary:
                                xpsi -= 1
                            var bi = Int(xpsi)
                            var bidx = ((bi >> 7) << 4) + 8 + ((bi & 0x7F) >> 4)
                            var w = bwt[bidx]
                            var cc = Int((w >> UInt32(((~bi) & 15) << 1)) & 3)
                            var kv = kk

                            var nocc: UInt64 = 0
                            var k_m = Int(kv)
                            if k_m < 0:
                                nocc = 0
                            elif UInt64(k_m) == seq_len:
                                if cc == 0:
                                    nocc = L2_1 - L2_0
                                elif cc == 1:
                                    nocc = L2_2 - L2_1
                                elif cc == 2:
                                    nocc = L2_3 - L2_2
                                else:
                                    nocc = L2_4 - L2_3
                            else:
                                var kko = UInt64(k_m)
                                if kko >= primary:
                                    kko -= 1
                                var baseo = Int((kko >> 7) << 4)
                                nocc = UInt64(bwt[baseo + cc * 2]) | (
                                    UInt64(bwt[baseo + cc * 2 + 1]) << 32
                                )
                                var po = baseo + 8
                                var endo = po + Int(
                                    (
                                        (
                                            (kko >> 5)
                                            - ((kko & UInt64(0xFFFFFFFFFFFFFF80)) >> 5)
                                        )
                                        << 1
                                    )
                                )
                                while po < endo:
                                    var yo = (UInt64(bwt[po]) << 32) | UInt64(
                                        bwt[po + 1]
                                    )
                                    var ao: UInt64
                                    var bo: UInt64
                                    if (cc & 2) != 0:
                                        ao = yo
                                    else:
                                        ao = ~yo
                                    if (cc & 1) != 0:
                                        bo = yo
                                    else:
                                        bo = ~yo
                                    var yyo = (ao >> 1) & bo & UInt64(
                                        0x5555555555555555
                                    )
                                    yyo = (yyo & UInt64(0x3333333333333333)) + (
                                        (yyo >> 2) & UInt64(0x3333333333333333)
                                    )
                                    yyo = (
                                        (yyo + (yyo >> 4))
                                        & UInt64(0x0F0F0F0F0F0F0F0F)
                                    ) * UInt64(0x0101010101010101)
                                    nocc += yyo >> 56
                                    po += 2
                                var bitso = Int(((~Int(kko)) & 31) << 1)
                                var masko: UInt64 = UInt64(0xFFFFFFFFFFFFFFFF)
                                if bitso > 0 and bitso < 64:
                                    masko = ~((UInt64(1) << UInt64(bitso)) - 1)
                                var y5 = (
                                    (UInt64(bwt[po]) << 32) | UInt64(bwt[po + 1])
                                ) & masko
                                var a5: UInt64
                                var b6: UInt64
                                if (cc & 2) != 0:
                                    a5 = y5
                                else:
                                    a5 = ~y5
                                if (cc & 1) != 0:
                                    b6 = y5
                                else:
                                    b6 = ~y5
                                var yy5 = (a5 >> 1) & b6 & UInt64(
                                    0x5555555555555555
                                )
                                yy5 = (yy5 & UInt64(0x3333333333333333)) + (
                                    (yy5 >> 2) & UInt64(0x3333333333333333)
                                )
                                yy5 = (
                                    (yy5 + (yy5 >> 4)) & UInt64(0x0F0F0F0F0F0F0F0F)
                                ) * UInt64(0x0101010101010101)
                                nocc += yy5 >> 56
                                if cc == 0:
                                    nocc -= UInt64((~Int(kko)) & 31)

                            var Lc2: UInt64 = L2_0
                            if cc == 1:
                                Lc2 = L2_1
                            elif cc == 2:
                                Lc2 = L2_2
                            elif cc == 3:
                                Lc2 = L2_3
                            kk = Lc2 + nocc
                        var sidx = Int(kk // UInt64(sa_intv))
                        if sidx <= 0 or sidx >= n_sa:
                            sa_ok = False
                        var sa_pos = sa_v
                        if sa_ok:
                            sa_pos = sa_v + sa[sidx - 1]
                        if sa_ok:
                            var is_rev = 0
                            var b0 = 0
                            var plen = e - us + 1
                            if sa_pos < l_pac:
                                b0 = Int(sa_pos) - us
                                is_rev = 0
                            else:
                                var seed_end = sa_pos + UInt64(plen) - 1
                                if seed_end >= l_pac * 2:
                                    sa_ok = False
                                else:
                                    var fp_end = Int(l_pac * 2 - 1 - sa_pos)
                                    var fp_start = fp_end - (plen - 1)
                                    b0 = fp_start - (qlen - 1 - e)
                                    is_rev = 1
                            if sa_ok:
                                var adj = -adj_cap
                                while adj <= adj_cap:
                                    var bstart = b0 + adj
                                    var probe = bstart
                                    if is_rev != 0:
                                        probe = bstart
                                    if probe < 0:
                                        probe = 0
                                    if UInt64(probe) >= l_pac:
                                        probe = Int(l_pac) - 1
                                    var lo = 0
                                    var hi2 = n_contigs
                                    var crid = -1
                                    while lo < hi2:
                                        var mid = (lo + hi2) // 2
                                        var off = Int(contig_off[mid])
                                        var ln = Int(contig_len[mid])
                                        if probe < off:
                                            hi2 = mid
                                        elif probe >= off + ln:
                                            lo = mid + 1
                                        else:
                                            crid = mid
                                            break
                                    if crid >= 0:
                                        var coff = Int(contig_off[crid])
                                        var clen = Int(contig_len[crid])
                                        var local = bstart - coff
                                        var sl0 = 0
                                        var sr0 = 0
                                        if local < 0:
                                            sl0 = -local
                                        if local + qlen > clen:
                                            sr0 = local + qlen - clen
                                        if (
                                            sl0 + sr0 < qlen
                                            and sl0 <= clip_cap
                                            and sr0 <= clip_cap
                                        ):
                                            var nm_all = 0
                                            var j = sl0
                                            while j < qlen - sr0 and nm_all <= budget:
                                                var ppos = bstart + j
                                                var rb = Int(
                                                    (
                                                        pac[ppos >> 2]
                                                        >> UInt8(((~ppos) & 3) << 1)
                                                    )
                                                    & 3
                                                )
                                                var qc = Int(codes[q_base + j])
                                                if is_rev != 0:
                                                    var qcr = Int(
                                                        codes[q_base + (qlen - 1 - j)]
                                                    )
                                                    if qcr <= 3:
                                                        qc = 3 - qcr
                                                    else:
                                                        qc = 4
                                                if rb > 3 or qc > 3 or rb != qc:
                                                    nm_all += 1
                                                j += 1
                                            var sl = sl0
                                            var sr = sr0
                                            var nm = nm_all
                                            if nm_all > budget and clip_cap > 0:
                                                while sl < clip_cap:
                                                    var p0 = bstart + sl
                                                    if p0 < coff or p0 >= coff + clen:
                                                        sl += 1
                                                        continue
                                                    var rb0 = Int(
                                                        (
                                                            pac[p0 >> 2]
                                                            >> UInt8(
                                                                ((~p0) & 3) << 1
                                                            )
                                                        )
                                                        & 3
                                                    )
                                                    var qc0 = Int(codes[q_base + sl])
                                                    if is_rev != 0:
                                                        var q0r = Int(
                                                            codes[
                                                                q_base
                                                                + (qlen - 1 - sl)
                                                            ]
                                                        )
                                                        if q0r <= 3:
                                                            qc0 = 3 - q0r
                                                        else:
                                                            qc0 = 4
                                                    if (
                                                        rb0 <= 3
                                                        and qc0 <= 3
                                                        and rb0 == qc0
                                                    ):
                                                        break
                                                    sl += 1
                                                while sr < clip_cap:
                                                    var jr = qlen - 1 - sr
                                                    if jr <= sl:
                                                        break
                                                    var p1 = bstart + jr
                                                    if p1 < coff or p1 >= coff + clen:
                                                        sr += 1
                                                        continue
                                                    var rb1 = Int(
                                                        (
                                                            pac[p1 >> 2]
                                                            >> UInt8(
                                                                ((~p1) & 3) << 1
                                                            )
                                                        )
                                                        & 3
                                                    )
                                                    var qc1 = Int(codes[q_base + jr])
                                                    if is_rev != 0:
                                                        var q1r = Int(
                                                            codes[
                                                                q_base
                                                                + (qlen - 1 - jr)
                                                            ]
                                                        )
                                                        if q1r <= 3:
                                                            qc1 = 3 - q1r
                                                        else:
                                                            qc1 = 4
                                                    if (
                                                        rb1 <= 3
                                                        and qc1 <= 3
                                                        and rb1 == qc1
                                                    ):
                                                        break
                                                    sr += 1
                                                nm = 0
                                                var jm = sl
                                                while (
                                                    jm < qlen - sr and nm <= budget
                                                ):
                                                    var pm = bstart + jm
                                                    var rbm = Int(
                                                        (
                                                            pac[pm >> 2]
                                                            >> UInt8(
                                                                ((~pm) & 3) << 1
                                                            )
                                                        )
                                                        & 3
                                                    )
                                                    var qcm = Int(codes[q_base + jm])
                                                    if is_rev != 0:
                                                        var qmr = Int(
                                                            codes[
                                                                q_base
                                                                + (qlen - 1 - jm)
                                                            ]
                                                        )
                                                        if qmr <= 3:
                                                            qcm = 3 - qmr
                                                        else:
                                                            qcm = 4
                                                    if (
                                                        rbm > 3
                                                        or qcm > 3
                                                        or rbm != qcm
                                                    ):
                                                        nm += 1
                                                    jm += 1
                                            var alen = qlen - sl - sr
                                            if nm <= budget and alen >= 24:
                                                if nm < best_nm:
                                                    best_nm = nm
                                                    best_rid = crid
                                                    best_pos = local + sl
                                                    best_sl = sl
                                                    best_sr = sr
                                                    var mq: UInt32 = 20
                                                    if nm == 0:
                                                        mq = 60
                                                    best_mq = mq
                                                    best_flag = is_rev * 16
                                                    if best_nm == 0:
                                                        out_rid[rid] = Int32(
                                                            best_rid
                                                        )
                                                        out_pos[rid] = UInt32(
                                                            best_pos
                                                        )
                                                        out_mapq[rid] = best_mq
                                                        out_flag[rid] = Int32(
                                                            best_flag
                                                        )
                                                        out_sl[rid] = UInt32(
                                                            best_sl
                                                        )
                                                        out_sr[rid] = UInt32(
                                                            best_sr
                                                        )
                                                        out_nm[rid] = UInt32(
                                                            best_nm
                                                        )
                                                        return
                                    adj += 1
                        hi += 1
                e += seed_stride

            if best_rid >= 0:
                out_rid[rid] = Int32(best_rid)
                out_pos[rid] = UInt32(best_pos)
                out_mapq[rid] = best_mq
                out_flag[rid] = Int32(best_flag)
                out_sl[rid] = UInt32(best_sl)
                out_sr[rid] = UInt32(best_sr)
                out_nm[rid] = UInt32(best_nm)

        var api = _device_api(resolved)
        var ctx = DeviceContext(api=api)
        var bwt_bytes = index.bwt_size * 4
        var sa_bytes = (index.n_sa - 1) * 8
        var pac_bytes = index.pac_size
        var n_contigs = index.contig_count()
        var science = bwt_bytes + sa_bytes + pac_bytes + n_contigs * 12
        _require_hbm(science, resolved)
        print(
            "MojoLinear GPU-fm upload science_bytes=",
            science,
            " bwt=",
            bwt_bytes,
            " sa=",
            sa_bytes,
            " pac=",
            pac_bytes,
        )

        var time_mod = Python.import_module("time")
        var t_upload0 = time_mod.perf_counter()
        var dev_bwt = ctx.enqueue_create_buffer[DType.uint32](index.bwt_size)
        var dev_sa = ctx.enqueue_create_buffer[DType.uint64](index.n_sa - 1)
        var dev_pac = ctx.enqueue_create_buffer[DType.uint8](pac_bytes)
        var dev_coff = ctx.enqueue_create_buffer[DType.uint64](n_contigs)
        var dev_clen = ctx.enqueue_create_buffer[DType.uint32](n_contigs)
        upload_mmap_to_device(
            ctx,
            dev_bwt.unsafe_ptr().bitcast[UInt8](),
            index.bwt_addr + 40,
            bwt_bytes,
            "bwt",
        )
        upload_mmap_to_device(
            ctx,
            dev_sa.unsafe_ptr().bitcast[UInt8](),
            index.sa_addr + 56,
            sa_bytes,
            "sa",
        )
        upload_mmap_to_device(
            ctx,
            dev_pac.unsafe_ptr(),
            index.pac_addr,
            pac_bytes,
            "pac",
        )
        var host_coff = ctx.enqueue_create_host_buffer[DType.uint64](n_contigs)
        var host_clen = ctx.enqueue_create_host_buffer[DType.uint32](n_contigs)
        var ci = 0
        while ci < n_contigs:
            host_coff[ci] = UInt64(index.contig_offset(ci))
            host_clen[ci] = UInt32(index.contig_length(ci))
            ci += 1
        ctx.enqueue_copy(src_buf=host_coff, dst_buf=dev_coff)
        ctx.enqueue_copy(src_buf=host_clen, dst_buf=dev_clen)
        ctx.synchronize()
        var t_upload1 = time_mod.perf_counter()
        print("MojoLinear GPU-fm upload_wall_s=", t_upload1 - t_upload0)
        print("MojoLinear GPU-fm index resident ok")

        var batch_size = _env_int("METHYLGRAPHER_LINEAR_READ_BATCH", 16384)
        var seed_len = _env_int("METHYLGRAPHER_FM_SEED_LEN", 18)
        # Unique SMEM left-extend; stride is right-endpoint spacing.
        var seed_stride = _env_int("METHYLGRAPHER_FM_SEED_STRIDE", 2)
        var max_occ = _env_int("METHYLGRAPHER_FM_MAX_OCC", 4096)
        var max_diff = _env_int("METHYLGRAPHER_FM_MAX_DIFF", 12)
        var max_soft = _env_int("METHYLGRAPHER_FM_MAX_SOFT", 20)
        var max_adj = _env_int("METHYLGRAPHER_FM_ADJ", 8)
        var paired = fq2.byte_length() > 0
        print(
            "MojoLinear GPU-fm map start batch=",
            batch_size,
            " seed_len=",
            seed_len,
            " stride=",
            seed_stride,
            " max_occ=",
            max_occ,
            " max_diff=",
            max_diff,
            " adj=",
            max_adj,
            " paired=",
            paired,
        )

        var fh = open_text_write(out_sam)
        _write_sam_header_fm(fh, index)
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
        comptime BLOCK = 256

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
            if max_len < seed_len:
                max_len = seed_len
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
            var dev_lens = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_rid = ctx.enqueue_create_buffer[DType.int32](n_seq)
            var dev_pos = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_mapq = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_flag = ctx.enqueue_create_buffer[DType.int32](n_seq)
            var dev_sl = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_sr = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_nm = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            ctx.enqueue_copy(src_buf=host_bases, dst_buf=dev_bases)
            ctx.enqueue_copy(src_buf=host_lens, dst_buf=dev_lens)
            var grid_b = (n_bases + BLOCK - 1) // BLOCK
            var grid_r = (n_seq + BLOCK - 1) // BLOCK
            ctx.enqueue_function[pack_bases_kernel](
                dev_bases.unsafe_ptr(),
                dev_codes.unsafe_ptr(),
                n_bases,
                grid_dim=grid_b,
                block_dim=BLOCK,
            )
            ctx.enqueue_function[fm_seed_extend_kernel](
                dev_bwt.unsafe_ptr(),
                dev_sa.unsafe_ptr(),
                dev_pac.unsafe_ptr(),
                dev_coff.unsafe_ptr(),
                dev_clen.unsafe_ptr(),
                n_contigs,
                index.primary,
                index.seq_len,
                index.l_pac,
                index.L2[0],
                index.L2[1],
                index.L2[2],
                index.L2[3],
                index.L2[4],
                index.sa_intv,
                index.n_sa,
                dev_codes.unsafe_ptr(),
                dev_lens.unsafe_ptr(),
                max_len,
                n_seq,
                seed_len,
                seed_stride,
                max_occ,
                max_diff,
                max_soft,
                max_adj,
                dev_rid.unsafe_ptr(),
                dev_pos.unsafe_ptr(),
                dev_mapq.unsafe_ptr(),
                dev_flag.unsafe_ptr(),
                dev_sl.unsafe_ptr(),
                dev_sr.unsafe_ptr(),
                dev_nm.unsafe_ptr(),
                grid_dim=grid_r,
                block_dim=BLOCK,
            )
            var host_rid = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var host_pos = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_mapq = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_flag = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var host_sl = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_sr = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_nm = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            ctx.enqueue_copy(src_buf=dev_rid, dst_buf=host_rid)
            ctx.enqueue_copy(src_buf=dev_pos, dst_buf=host_pos)
            ctx.enqueue_copy(src_buf=dev_mapq, dst_buf=host_mapq)
            ctx.enqueue_copy(src_buf=dev_flag, dst_buf=host_flag)
            ctx.enqueue_copy(src_buf=dev_sl, dst_buf=host_sl)
            ctx.enqueue_copy(src_buf=dev_sr, dst_buf=host_sr)
            ctx.enqueue_copy(src_buf=dev_nm, dst_buf=host_nm)
            ctx.synchronize()

            var n1 = len(batch1)
            var pj = 0
            while pj < n1:
                var h1 = _hit_from_fm(
                    index,
                    batch1[pj].name,
                    batch1[pj].seq,
                    batch1[pj].original_seq,
                    Int(host_rid[pj]),
                    Int(host_pos[pj]),
                    Int(host_mapq[pj]),
                    Int(host_flag[pj]),
                    Int(host_sl[pj]),
                    Int(host_sr[pj]),
                )
                h1.qual = batch1[pj].qual
                if not paired:
                    if h1.contig != "*":
                        n_mapped += 1
                    fh.write(_sam_with_rg_fm(h1) + "\n")
                    n_reads += 1
                else:
                    var r2i = n1 + pj
                    var h2 = _hit_from_fm(
                        index,
                        batch2[pj].name,
                        batch2[pj].seq,
                        batch2[pj].original_seq,
                        Int(host_rid[r2i]),
                        Int(host_pos[r2i]),
                        Int(host_mapq[r2i]),
                        Int(host_flag[r2i]),
                        Int(host_sl[r2i]),
                        Int(host_sr[r2i]),
                    )
                    h2.qual = batch2[pj].qual
                    var paired_hits = pair_hits(h1, h2)
                    var p1 = paired_hits.r1.copy()
                    var p2 = paired_hits.r2.copy()
                    if p1.contig != "*":
                        n_mapped += 1
                    if p2.contig != "*":
                        n_mapped += 1
                    fh.write(_sam_with_rg_fm(p1) + "\n")
                    fh.write(_sam_with_rg_fm(p2) + "\n")
                    n_reads += 2
                pj += 1

            n_batches += 1
            if n_batches == 1 or n_batches % 5 == 0:
                print(
                    "MojoLinear GPU-fm progress batches=",
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
            " backend=gpu-fm",
        )
        print(
            "MojoLinear GPU-fm map_wall_s=",
            t_map1 - t_map0,
            " reads=",
            n_reads,
        )
        return n_mapped
