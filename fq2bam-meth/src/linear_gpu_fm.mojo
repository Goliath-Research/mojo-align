# FM / BWA-MEM-style GPU engine (opt-in). METHYLGRAPHER_LINEAR_ENGINE=fm
#
# Uses Clara's bwameth BWA 0.7 FM-index (.bwt/.sa/.pac/.ann), not the dense-v1
# k-mer posting pack. Seed via FM backward search (min_seed_len=19), SA resolve,
# gapless/softclip extend on the forward pac (f*/r* contigs). Default remains
# parity until Clara gates pass.

from std.algorithm import parallelize
from std.collections import Dict, List
from std.memory import UnsafePointer, memcpy
from std.python import Python, PythonObject
from std.sys import has_accelerator
from std.time import perf_counter as _tick

from gpu_device import select_device
from gpu_kernels import _device_api, kernel_target_label, probe_device_context
from linear_extend import hit_to_sam_line, pair_hits, LinearHit
from linear_fm_index import FmIndex, fm_prefix_from_ref
from linear_fastq import (
    FastqArena,
    FastqPairStream,
    fq_close,
    fq_export_ptrs,
    fq_open,
    fq_pack_bases,
    fq_read_batch,
    fq_reserve,
)
from linear_gpu_sort import gpu_sort_markdup
from linear_gpu_locate import (
    FastqRec,
    _canonical_contig,
    _env_int,
    _env_str,
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


def _path_is_bam(path: String) -> Bool:
    var n = path.byte_length()
    if n < 4:
        return False
    return String(path[byte = n - 4 : n]).lower() == ".bam"


def _open_bam_writer(
    index: FmIndex, path: String
) raises -> PythonObject:
    """Canonical @SQ list + rid→tid map packed into the writer via sq order."""
    var bam_mod = Python.import_module("bam_emit")
    var sq_names = Python.list()
    var sq_lens = Python.list()
    var seen = Dict[String, Int]()
    var n_sq = 0
    var ci = 0
    while ci < index.contig_count():
        var canon = _canonical_contig(index.contig_name(ci))
        if canon not in seen:
            seen[canon] = n_sq
            sq_names.append(canon)
            sq_lens.append(index.contig_length(ci))
            n_sq += 1
        ci += 1
    var rg = _env_str("METHYLGRAPHER_RG_ID", "mojo1")
    var sm = _env_str("METHYLGRAPHER_RG_SM", "sample")
    var lb = _env_str("METHYLGRAPHER_RG_LB", "lib1")
    var pl = _env_str("METHYLGRAPHER_RG_PL", "ILLUMINA")
    var level = _env_int("METHYLGRAPHER_BAM_LEVEL", 1)
    return bam_mod.BamWriter(
        path,
        sq_names,
        sq_lens,
        rg,
        sm,
        lb,
        pl,
        "0.1.0-mojo-fm",
        level,
        "coordinate",
    )


def _open_fq_reader(
    fq1: String, fq2: String, bs_r1: String, bs_r2: String
) raises -> PythonObject:
    var mod = Python.import_module("fastq_batch")
    return mod.FastqPairReader(fq1, fq2, bs_r1, bs_r2)


def _py_batch_to_recs(
    b: PythonObject,
    paired: Bool,
    mut batch1: List[FastqRec],
    mut batch2: List[FastqRec],
) raises -> Int:
    batch1 = List[FastqRec]()
    batch2 = List[FastqRec]()
    var n1 = Int(py=b.n1)
    var i = 0
    while i < n1:
        batch1.append(
            FastqRec(
                String(b.names1[i]),
                String(b.seq1[i].decode()),
                String(b.qual1[i].decode()),
                String(b.orig1[i].decode()),
            )
        )
        if paired:
            batch2.append(
                FastqRec(
                    String(b.names2[i]),
                    String(b.seq2[i].decode()),
                    String(b.qual2[i].decode()),
                    String(b.orig2[i].decode()),
                )
            )
        i += 1
    return n1


def _open_bam_arena() raises -> PythonObject:
    var bam_mod = Python.import_module("bam_emit")
    return bam_mod.BamArena()


def _open_bam_run_store() raises -> PythonObject:
    var bam_mod = Python.import_module("bam_emit")
    return bam_mod.BamRunStore()


def _gather_perm_to_writer(
    writer: PythonObject,
    n: Int,
    bam_arena: PythonObject,
    perm_addr: Int,
    rec_addr: Int,
    len_addr: Int,
    dup_addr: Int,
    bam_blob_addr: Int,
    bam_cap: Int,
    apply_dup: Bool,
) raises:
    var h_perm = _u32_at(perm_addr)
    var h_rec = _u64_at(rec_addr)
    var h_len = _u32_at(len_addr)
    var h_dup = _u32_at(dup_addr)
    var n_chunks = Int(py=bam_arena.n_chunks())
    var chunk_addrs = List[Int]()
    var cii = 0
    while cii < n_chunks:
        chunk_addrs.append(Int(py=bam_arena.chunk_addr(cii)))
        cii += 1
    var gout = 0
    var gi = 0
    var bp_out = _u8_at(bam_blob_addr)
    while gi < n:
        var orig = Int(h_perm[gi])
        var packed = h_rec[orig]
        var cid = Int(packed >> 32)
        var loc = Int(packed & 4294967295)
        var rlen = Int(h_len[orig])
        if gout + rlen > bam_cap:
            writer.write_raw(bam_blob_addr, gout)
            gout = 0
        if rlen > bam_cap:
            raise Error("BAM record larger than batch buffer")
        var src = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=chunk_addrs[cid] + loc
        )
        var dst = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=bam_blob_addr + gout
        )
        memcpy(dest=dst, src=src, count=rlen)
        if apply_dup and Int(h_dup[orig]) != 0:
            _or_dup_flag(bp_out, gout)
        gout += rlen
        gi += 1
    if gout > 0:
        writer.write_raw(bam_blob_addr, gout)


def _flush_sorted_bam_run(
    api: String,
    n_tile: Int,
    do_markdup: Bool,
    bam_arena: PythonObject,
    run_store: PythonObject,
    coord_addr: Int,
    dhi_addr: Int,
    dlo_addr: Int,
    score_addr: Int,
    pair_addr: Int,
    dup_addr: Int,
    perm_addr: Int,
    dupperm_addr: Int,
    rec_addr: Int,
    len_addr: Int,
    bam_blob_addr: Int,
    bam_cap: Int,
) raises:
    if n_tile <= 0:
        return
    gpu_sort_markdup(
        api,
        n_tile,
        coord_addr,
        dhi_addr,
        dlo_addr,
        score_addr,
        pair_addr,
        dup_addr,
        perm_addr,
        do_markdup,
        dupperm_addr,
    )
    var bam_mod = Python.import_module("bam_emit")
    bam_mod.order_run_perms(
        n_tile,
        coord_addr,
        dhi_addr,
        dlo_addr,
        perm_addr,
        dupperm_addr,
    )
    _ = run_store.write_sorted_from_arena(
        n_tile,
        bam_arena,
        rec_addr,
        len_addr,
        perm_addr,
    )
    _ = run_store.write_keys(
        n_tile,
        coord_addr,
        dhi_addr,
        dlo_addr,
        score_addr,
        pair_addr,
        len_addr,
        perm_addr,
        dupperm_addr,
    )
    bam_arena.clear()
    print("MojoLinear GPU-fm flushed sorted BAM run records=", n_tile)


def _maybe_flush_tile(
    use_runs: Bool,
    paired: Bool,
    n1: Int,
    mut n_reads: Int,
    mut n_total: Int,
    mut tile_chunks: Int,
    meta_cap: Int,
    api: String,
    do_markdup: Bool,
    bam_arena: PythonObject,
    run_store: PythonObject,
    coord_addr: Int,
    dhi_addr: Int,
    dlo_addr: Int,
    score_addr: Int,
    pair_addr: Int,
    dup_addr: Int,
    perm_addr: Int,
    dupperm_addr: Int,
    rec_addr: Int,
    len_addr: Int,
    bam_blob_addr: Int,
    bam_cap: Int,
) raises:
    if not use_runs or n1 <= 0:
        return
    var need = n1
    if paired:
        need = n1 * 2
    if n_reads + need <= meta_cap:
        return
    _flush_sorted_bam_run(
        api,
        n_reads,
        do_markdup,
        bam_arena,
        run_store,
        coord_addr,
        dhi_addr,
        dlo_addr,
        score_addr,
        pair_addr,
        dup_addr,
        perm_addr,
        dupperm_addr,
        rec_addr,
        len_addr,
        bam_blob_addr,
        bam_cap,
    )
    n_total += n_reads
    n_reads = 0
    tile_chunks = 0


def _u8_at(addr: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)


def _i32_at(addr: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr)


def _u32_at(addr: Int) -> UnsafePointer[UInt32, MutAnyOrigin]:
    return UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=addr)


def _u64_at(addr: Int) -> UnsafePointer[UInt64, MutAnyOrigin]:
    return UnsafePointer[UInt64, MutAnyOrigin](unsafe_from_address=addr)


def _copy_bytes(dst: Int, src: Int, nbytes: Int):
    if nbytes <= 0:
        return
    memcpy(dest=_u8_at(dst), src=_u8_at(src), count=nbytes)


def _unclipped5(flag: Int, pos0: Int, sl: Int, sr: Int, qlen: Int) -> Int:
    if (flag & 4) != 0 or qlen <= 0:
        return 0
    var slv = sl
    var srv = sr
    if slv < 0:
        slv = 0
    if srv < 0:
        srv = 0
    if slv + srv >= qlen:
        slv = 0
        srv = 0
    var mid = qlen - slv - srv
    if (flag & 16) != 0:
        return pos0 + mid + slv - 1
    return pos0 - slv


def _coord_key(tid: Int, pos0: Int, orig: Int) -> UInt64:
    if tid < 0:
        return (UInt64(4294967295) << 32) | (UInt64(orig) & 4294967295)
    var p = pos0
    if p < 0:
        p = 0
    return (UInt64(tid) << 32) | UInt64(p)


def _pack_end(tid: Int, u5: Int, strand: Int) -> UInt64:
    var t = 0
    if tid >= 0:
        t = tid + 1
    var u = u5 + 1073741824
    if u < 0:
        u = 0
    return (UInt64(t) << 48) | (UInt64(strand & 1) << 47) | (UInt64(u) & 140737488355327)


def _or_dup_flag(p: UnsafePointer[UInt8, MutAnyOrigin], rec_off: Int):
    var b0 = Int(p[rec_off + 16])
    var b1 = Int(p[rec_off + 17])
    var b2 = Int(p[rec_off + 18])
    var b3 = Int(p[rec_off + 19])
    var flag_nc = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    var flag = (flag_nc >> 16) | 1024
    var nc = flag_nc & 65535
    var v = (flag << 16) | nc
    p[rec_off + 16] = UInt8(v & 255)
    p[rec_off + 17] = UInt8((v >> 8) & 255)
    p[rec_off + 18] = UInt8((v >> 16) & 255)
    p[rec_off + 19] = UInt8((v >> 24) & 255)


def _rid_to_tid_table(index: FmIndex) raises -> List[Int]:
    var out = List[Int]()
    var seen = Dict[String, Int]()
    var n_sq = 0
    var ci = 0
    while ci < index.contig_count():
        var canon = _canonical_contig(index.contig_name(ci))
        if canon not in seen:
            seen[canon] = n_sq
            n_sq += 1
        out.append(seen[canon])
        ci += 1
    return out^


def _put_i32(p: UnsafePointer[UInt8, MutAnyOrigin], off: Int, v: Int):
    var u = v
    if u < 0:
        u = u + 4294967296
    p[off] = UInt8(u & 255)
    p[off + 1] = UInt8((u >> 8) & 255)
    p[off + 2] = UInt8((u >> 16) & 255)
    p[off + 3] = UInt8((u >> 24) & 255)


def _put_u32(p: UnsafePointer[UInt8, MutAnyOrigin], off: Int, v: UInt32):
    p[off] = UInt8(Int(v) & 255)
    p[off + 1] = UInt8(Int(v >> 8) & 255)
    p[off + 2] = UInt8(Int(v >> 16) & 255)
    p[off + 3] = UInt8(Int(v >> 24) & 255)


def _nt16(b: UInt8) -> UInt8:
    if b == 65 or b == 97:
        return 1
    if b == 67 or b == 99:
        return 2
    if b == 71 or b == 103:
        return 4
    if b == 84 or b == 116:
        return 8
    return 15


def _nt16_comp(b: UInt8) -> UInt8:
    if b == 1:
        return 8
    if b == 8:
        return 1
    if b == 2:
        return 4
    if b == 4:
        return 2
    return 15


def _reg2bin(beg: Int, end: Int) -> Int:
    var e = end
    if e <= beg:
        e = beg + 1
    e -= 1
    if (beg >> 14) == (e >> 14):
        return 4681 + (beg >> 14)
    if (beg >> 17) == (e >> 17):
        return 585 + (beg >> 17)
    if (beg >> 20) == (e >> 20):
        return 73 + (beg >> 20)
    if (beg >> 23) == (e >> 23):
        return 9 + (beg >> 23)
    if (beg >> 26) == (e >> 26):
        return 1 + (beg >> 26)
    return 0


def _pack_bam_aln(
    p: UnsafePointer[UInt8, MutAnyOrigin],
    off0: Int,
    cap: Int,
    name_addr: Int,
    name_len: Int,
    flag: Int,
    tid: Int,
    pos0: Int,
    mapq: Int,
    sl: Int,
    sr: Int,
    seq_addr: Int,
    qlen: Int,
    qual_addr: Int,
    qual_len: Int,
    ntid: Int,
    npos: Int,
    tlen: Int,
    nm: Int,
    rg_addr: Int,
    rg_n: Int,
) raises -> Int:
    var rev = (flag & 16) != 0
    var slv = sl
    var srv = sr
    if slv < 0:
        slv = 0
    if srv < 0:
        srv = 0
    if slv + srv >= qlen:
        slv = 0
        srv = 0
    var mid = qlen - slv - srv
    if rev:
        var tmp_s = slv
        slv = srv
        srv = tmp_s
    var n_cigar = 0
    if tid >= 0 and (flag & 4) == 0:
        if slv > 0:
            n_cigar += 1
        if mid > 0:
            n_cigar += 1
        if srv > 0:
            n_cigar += 1
    var l_qname = name_len + 1
    var seq_packed_n = (qlen + 1) // 2
    var tags_n = 3 + rg_n + 1 + 4
    var block = 32 + l_qname + 4 * n_cigar + seq_packed_n + qlen + tags_n
    var need = off0 + 4 + block
    if need > cap:
        raise Error("BAM batch buffer overflow")
    var bin_v = 4680
    var tid_w = -1
    var pos_w = -1
    if tid >= 0 and (flag & 4) == 0:
        tid_w = tid
        pos_w = pos0
        var end = pos0 + mid
        if end <= pos0:
            end = pos0 + 1
        bin_v = _reg2bin(pos0, end)
    var ntid_w = ntid
    var npos_w = npos
    if ntid_w < 0:
        ntid_w = -1
        npos_w = -1
    var mq = mapq
    if mq < 0:
        mq = 0
    if mq > 255:
        mq = 255
    _put_i32(p, off0, block)
    _put_i32(p, off0 + 4, tid_w)
    _put_i32(p, off0 + 8, pos_w)
    _put_u32(
        p,
        off0 + 12,
        UInt32((bin_v << 16) | (mq << 8) | l_qname),
    )
    _put_u32(p, off0 + 16, UInt32((flag << 16) | n_cigar))
    _put_i32(p, off0 + 20, qlen)
    _put_i32(p, off0 + 24, ntid_w)
    _put_i32(p, off0 + 28, npos_w)
    _put_i32(p, off0 + 32, tlen)
    var name = _u8_at(name_addr)
    var seq = _u8_at(seq_addr)
    var qual = _u8_at(qual_addr)
    var rg = _u8_at(rg_addr)
    var off = off0 + 36
    var ni = 0
    while ni < name_len:
        p[off + ni] = name[ni]
        ni += 1
    p[off + ni] = 0
    off += l_qname
    if n_cigar > 0:
        if slv > 0:
            _put_u32(p, off, UInt32((slv << 4) | 4))
            off += 4
        if mid > 0:
            _put_u32(p, off, UInt32((mid << 4) | 0))
            off += 4
        if srv > 0:
            _put_u32(p, off, UInt32((srv << 4) | 4))
            off += 4
    var si = 0
    while si + 1 < qlen:
        var ia = si
        var ib = si + 1
        if rev:
            ia = qlen - 1 - si
            ib = qlen - 2 - si
        var a = _nt16(seq[ia])
        var b = _nt16(seq[ib])
        if rev:
            a = _nt16_comp(a)
            b = _nt16_comp(b)
        p[off] = (a << 4) | b
        off += 1
        si += 2
    if si < qlen:
        var ic = si
        if rev:
            ic = 0
        var c = _nt16(seq[ic])
        if rev:
            c = _nt16_comp(c)
        p[off] = c << 4
        off += 1
    var qi2 = 0
    if qual_len == qlen and qual_addr != 0:
        while qi2 < qlen:
            var srcq = qi2
            if rev:
                srcq = qlen - 1 - qi2
            var qb2 = qual[srcq]
            if qb2 >= 33:
                p[off + qi2] = qb2 - 33
            else:
                p[off + qi2] = 255
            qi2 += 1
    else:
        while qi2 < qlen:
            p[off + qi2] = 255
            qi2 += 1
    off += qlen
    p[off] = 82
    p[off + 1] = 71
    p[off + 2] = 90
    off += 3
    var ri = 0
    while ri < rg_n:
        p[off + ri] = rg[ri]
        ri += 1
    p[off + rg_n] = 0
    off += rg_n + 1
    p[off] = 78
    p[off + 1] = 77
    p[off + 2] = 67
    var nmv = nm
    if nmv < 0:
        nmv = 0
    if nmv > 255:
        nmv = 255
    p[off + 3] = UInt8(nmv)
    off += 4
    return off



def _emit_bam_from_batch(
    mut arena: FastqArena,
    paired: Bool,
    n1: Int,
    rid_addr: Int,
    pos_addr: Int,
    mapq_addr: Int,
    flag_addr: Int,
    sl_addr: Int,
    sr_addr: Int,
    nm_addr: Int,
    n1_name_a: Int,
    n1_name_n: Int,
    n1_orig_a: Int,
    n1_orig_n: Int,
    n1_qual_a: Int,
    n1_qual_n: Int,
    n2_name_a: Int,
    n2_name_n: Int,
    n2_orig_a: Int,
    n2_orig_n: Int,
    n2_qual_a: Int,
    n2_qual_n: Int,
    bam_p: UnsafePointer[UInt8, MutAnyOrigin],
    bam_cap: Int,
    rid_to_tid: List[Int],
    rg_p: UnsafePointer[UInt8, ...],
    rg_n: Int,
    coord_addr: Int,
    dhi_addr: Int,
    dlo_addr: Int,
    rec_addr: Int,
    len_addr: Int,
    score_addr: Int,
    pair_addr: Int,
    mut n_reads: Int,
    mut n_mapped: Int,
    n_batches: Int,
    meta_cap: Int,
) raises -> Int:
    fq_export_ptrs(
        arena,
        paired,
        n1_name_a,
        n1_name_n,
        n1_orig_a,
        n1_orig_n,
        n1_qual_a,
        n1_qual_n,
        n2_name_a,
        n2_name_n,
        n2_orig_a,
        n2_orig_n,
        n2_qual_a,
        n2_qual_n,
    )
    var host_rid = _i32_at(rid_addr)
    var host_pos = _u32_at(pos_addr)
    var host_mapq = _u32_at(mapq_addr)
    var host_flag = _i32_at(flag_addr)
    var host_sl = _u32_at(sl_addr)
    var host_sr = _u32_at(sr_addr)
    var host_nm = _u32_at(nm_addr)
    var name1_a = _u64_at(n1_name_a)
    var name1_n = _u32_at(n1_name_n)
    var orig1_a = _u64_at(n1_orig_a)
    var orig1_n = _u32_at(n1_orig_n)
    var qual1_a = _u64_at(n1_qual_a)
    var qual1_n = _u32_at(n1_qual_n)
    var name2_a = _u64_at(n2_name_a)
    var name2_n = _u32_at(n2_name_n)
    var orig2_a = _u64_at(n2_orig_a)
    var orig2_n = _u32_at(n2_orig_n)
    var qual2_a = _u64_at(n2_qual_a)
    var qual2_n = _u32_at(n2_qual_n)
    var h_coord = _u64_at(coord_addr)
    var h_dhi = _u64_at(dhi_addr)
    var h_dlo = _u64_at(dlo_addr)
    var h_rec = _u64_at(rec_addr)
    var h_len = _u32_at(len_addr)
    var h_score = _u32_at(score_addr)
    var h_pair = _u32_at(pair_addr)
    var rg_addr = Int(rg_p)
    var off = 0
    var pj = 0
    while pj < n1:
        var rid1 = Int(host_rid[pj])
        var pos1 = Int(host_pos[pj])
        var mq1 = Int(host_mapq[pj])
        var fl1 = Int(host_flag[pj]) | 1 | 64
        var sl1 = Int(host_sl[pj])
        var sr1 = Int(host_sr[pj])
        var nm1 = Int(host_nm[pj])
        var tid1 = -1
        if rid1 >= 0 and rid1 < len(rid_to_tid):
            tid1 = rid_to_tid[rid1]
        else:
            fl1 = fl1 | 4
        var qlen1 = Int(orig1_n[pj])
        var q1n = Int(qual1_n[pj])
        var n1len = Int(name1_n[pj])
        if not paired:
            if n_reads >= meta_cap:
                raise Error(
                    "FM sort cap exceeded; set METHYLGRAPHER_FM_SORT_CAP"
                )
            var rec_start = off
            off = _pack_bam_aln(
                bam_p,
                off,
                bam_cap,
                Int(name1_a[pj]),
                n1len,
                fl1,
                tid1,
                pos1,
                mq1,
                sl1,
                sr1,
                Int(orig1_a[pj]),
                qlen1,
                Int(qual1_a[pj]),
                q1n,
                -1,
                -1,
                0,
                nm1,
                rg_addr,
                rg_n,
            )
            h_coord[n_reads] = _coord_key(tid1, pos1, n_reads)
            if tid1 >= 0:
                h_dhi[n_reads] = _pack_end(
                    tid1,
                    _unclipped5(fl1, pos1, sl1, sr1, qlen1),
                    (fl1 >> 4) & 1,
                )
                h_dlo[n_reads] = 0
            else:
                h_dhi[n_reads] = ~UInt64(0)
                h_dlo[n_reads] = UInt64(n_reads)
            h_rec[n_reads] = (UInt64(n_batches) << 32) | UInt64(rec_start)
            h_len[n_reads] = UInt32(off - rec_start)
            var nm_c = nm1
            if nm_c < 0:
                nm_c = 0
            if nm_c > 255:
                nm_c = 255
            h_score[n_reads] = UInt32((mq1 << 8) | (255 - nm_c))
            h_pair[n_reads] = UInt32(n_reads)
            if tid1 >= 0:
                n_mapped += 1
            n_reads += 1
        else:
            var r2i = n1 + pj
            var rid2 = Int(host_rid[r2i])
            var pos2 = Int(host_pos[r2i])
            var mq2 = Int(host_mapq[r2i])
            var fl2 = Int(host_flag[r2i]) | 1 | 128
            var sl2 = Int(host_sl[r2i])
            var sr2 = Int(host_sr[r2i])
            var nm2 = Int(host_nm[r2i])
            var tid2 = -1
            if rid2 >= 0 and rid2 < len(rid_to_tid):
                tid2 = rid_to_tid[rid2]
            else:
                fl2 = fl2 | 4
            if tid1 < 0:
                fl2 = fl2 | 8
            if tid2 < 0:
                fl1 = fl1 | 8
            var ntid1 = -1
            var npos1 = -1
            var ntid2 = -1
            var npos2 = -1
            var tlen1 = 0
            var tlen2 = 0
            var qlen2 = Int(orig2_n[pj])
            if tid1 >= 0 and tid2 >= 0 and tid1 == tid2:
                fl1 = fl1 | 2
                fl2 = fl2 | 2
                ntid1 = tid1
                ntid2 = tid2
                npos1 = pos2
                npos2 = pos1
                var tl = pos2 - pos1
                if tl < 0:
                    tl = -tl
                tl = tl + qlen2
                if pos1 <= pos2:
                    tlen1 = tl
                    tlen2 = -tl
                else:
                    tlen1 = -tl
                    tlen2 = tl
            else:
                if tid1 >= 0:
                    ntid2 = tid1
                    npos2 = pos1
                if tid2 >= 0:
                    ntid1 = tid2
                    npos1 = pos2
            var q2n = Int(qual2_n[pj])
            var n2len = Int(name2_n[pj])
            if n_reads + 1 >= meta_cap:
                raise Error(
                    "FM sort cap exceeded; set METHYLGRAPHER_FM_SORT_CAP"
                )
            var rec_start1 = off
            off = _pack_bam_aln(
                bam_p,
                off,
                bam_cap,
                Int(name1_a[pj]),
                n1len,
                fl1,
                tid1,
                pos1,
                mq1,
                sl1,
                sr1,
                Int(orig1_a[pj]),
                qlen1,
                Int(qual1_a[pj]),
                q1n,
                ntid1,
                npos1,
                tlen1,
                nm1,
                rg_addr,
                rg_n,
            )
            var rec_len1 = off - rec_start1
            var rec_start2 = off
            off = _pack_bam_aln(
                bam_p,
                off,
                bam_cap,
                Int(name2_a[pj]),
                n2len,
                fl2,
                tid2,
                pos2,
                mq2,
                sl2,
                sr2,
                Int(orig2_a[pj]),
                qlen2,
                Int(qual2_a[pj]),
                q2n,
                ntid2,
                npos2,
                tlen2,
                nm2,
                rg_addr,
                rg_n,
            )
            var rec_len2 = off - rec_start2
            var i1 = n_reads
            var i2 = n_reads + 1
            var pid = UInt32(n_reads // 2)
            var nm_sum = nm1 + nm2
            if nm_sum < 0:
                nm_sum = 0
            if nm_sum > 255:
                nm_sum = 255
            var pscore = UInt32(((mq1 + mq2) << 8) | (255 - nm_sum))
            var dhi = ~UInt64(0)
            var dlo = UInt64(i1)
            if tid1 >= 0 or tid2 >= 0:
                var e1 = _pack_end(
                    tid1,
                    _unclipped5(fl1, pos1, sl1, sr1, qlen1),
                    (fl1 >> 4) & 1,
                )
                var e2 = _pack_end(
                    tid2,
                    _unclipped5(fl2, pos2, sl2, sr2, qlen2),
                    (fl2 >> 4) & 1,
                )
                if tid1 < 0:
                    dhi = e2
                    dlo = 0
                elif tid2 < 0:
                    dhi = e1
                    dlo = 0
                elif e1 <= e2:
                    dhi = e1
                    dlo = e2
                else:
                    dhi = e2
                    dlo = e1
            h_coord[i1] = _coord_key(tid1, pos1, i1)
            h_coord[i2] = _coord_key(tid2, pos2, i2)
            h_dhi[i1] = dhi
            h_dhi[i2] = dhi
            h_dlo[i1] = dlo
            h_dlo[i2] = dlo
            h_rec[i1] = (UInt64(n_batches) << 32) | UInt64(rec_start1)
            h_rec[i2] = (UInt64(n_batches) << 32) | UInt64(rec_start2)
            h_len[i1] = UInt32(rec_len1)
            h_len[i2] = UInt32(rec_len2)
            h_score[i1] = pscore
            h_score[i2] = pscore
            h_pair[i1] = pid
            h_pair[i2] = pid
            if tid1 >= 0:
                n_mapped += 1
            if tid2 >= 0:
                n_mapped += 1
            n_reads += 2
        pj += 1
    return off


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
        from std.gpu.host import (
            DeviceBuffer,
            DeviceContext,
            DeviceEvent,
            DeviceFunction,
            DeviceStream,
            HostBuffer,
        )
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
                                            elif (
                                                adj == 0
                                                and is_rev == 0
                                                and (nm > budget or alen < 24)
                                            ):
                                                # 1bp deletion on a coarse g grid.
                                                if bstart + qlen + 1 <= coff + clen:
                                                    var g = 0
                                                    while g <= qlen:
                                                        var nm_d = 1
                                                        var t = 0
                                                        while (
                                                            t < g and nm_d <= budget
                                                        ):
                                                            var rcd = Int(
                                                                (
                                                                    pac[
                                                                        (bstart + t)
                                                                        >> 2
                                                                    ]
                                                                    >> UInt8(
                                                                        (
                                                                            (
                                                                                ~(
                                                                                    bstart
                                                                                    + t
                                                                                )
                                                                            )
                                                                            & 3
                                                                        )
                                                                        << 1
                                                                    )
                                                                )
                                                                & 3
                                                            )
                                                            var qcd = Int(
                                                                codes[q_base + t]
                                                            )
                                                            if (
                                                                rcd > 3
                                                                or qcd > 3
                                                                or rcd != qcd
                                                            ):
                                                                nm_d += 1
                                                            t += 1
                                                        while (
                                                            t < qlen
                                                            and nm_d <= budget
                                                        ):
                                                            var rcd2 = Int(
                                                                (
                                                                    pac[
                                                                        (
                                                                            bstart
                                                                            + t
                                                                            + 1
                                                                        )
                                                                        >> 2
                                                                    ]
                                                                    >> UInt8(
                                                                        (
                                                                            (
                                                                                ~(
                                                                                    bstart
                                                                                    + t
                                                                                    + 1
                                                                                )
                                                                            )
                                                                            & 3
                                                                        )
                                                                        << 1
                                                                    )
                                                                )
                                                                & 3
                                                            )
                                                            var qcd2 = Int(
                                                                codes[q_base + t]
                                                            )
                                                            if (
                                                                rcd2 > 3
                                                                or qcd2 > 3
                                                                or rcd2 != qcd2
                                                            ):
                                                                nm_d += 1
                                                            t += 1
                                                        if (
                                                            nm_d <= budget
                                                            and nm_d < best_nm
                                                        ):
                                                            best_nm = nm_d
                                                            best_rid = crid
                                                            best_pos = local
                                                            best_sl = 0
                                                            best_sr = 0
                                                            best_mq = 20
                                                            best_flag = 0
                                                        g += 4
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

        def fm_mate_rescue_kernel(
            pac: UnsafePointer[UInt8, MutAnyOrigin],
            contig_off: UnsafePointer[UInt64, MutAnyOrigin],
            contig_len: UnsafePointer[UInt32, MutAnyOrigin],
            n_contigs: Int,
            l_pac: UInt64,
            codes: UnsafePointer[UInt8, MutAnyOrigin],
            read_lens: UnsafePointer[UInt32, MutAnyOrigin],
            max_len: Int,
            n_pairs: Int,
            window: Int,
            max_diff: Int,
            max_soft: Int,
            out_rid: UnsafePointer[Int32, MutAnyOrigin],
            out_pos: UnsafePointer[UInt32, MutAnyOrigin],
            out_mapq: UnsafePointer[UInt32, MutAnyOrigin],
            out_flag: UnsafePointer[Int32, MutAnyOrigin],
            out_sl: UnsafePointer[UInt32, MutAnyOrigin],
            out_sr: UnsafePointer[UInt32, MutAnyOrigin],
            out_nm: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            # If exactly one mate mapped, gapless-scan ±window on that contig
            # (prefer opposite strand). 1bp deletion only at the best locus.
            var pi = Int(block_idx.x * block_dim.x + thread_idx.x)
            if pi >= n_pairs:
                return
            var i1 = pi
            var i2 = pi + n_pairs
            var m1 = Int(out_rid[i1]) >= 0
            var m2 = Int(out_rid[i2]) >= 0
            if m1 == m2:
                return
            var src = i1
            var dst = i2
            if m2:
                src = i2
                dst = i1
            var crid = Int(out_rid[src])
            if crid < 0 or crid >= n_contigs:
                return
            var qlen = Int(read_lens[dst])
            if qlen < 24:
                return
            var coff = Int(contig_off[crid])
            var clen = Int(contig_len[crid])
            var mate_pos = Int(out_pos[src])
            var win = window
            if win < 32:
                win = 32
            if win > 1024:
                win = 1024
            var lo = mate_pos - win
            if lo < 0:
                lo = 0
            var hi = mate_pos + win
            if hi > clen - 24:
                hi = clen - 24
            if hi <= lo:
                return
            var budget = max_diff
            if budget < 2:
                budget = 2
            if budget > qlen // 2:
                budget = qlen // 2
            var q_base = dst * max_len
            var best_nm = 9999
            var best_pos = 0
            var best_flag = 0
            var scan_nm = 9999
            var scan_local = lo
            var scan_rev = 0
            var mate_rev = (Int(out_flag[src]) & 16) != 0
            var pass_s = 0
            while pass_s < 2:
                var is_rev = 1
                if pass_s == 0:
                    if mate_rev:
                        is_rev = 0
                    else:
                        is_rev = 1
                else:
                    if mate_rev:
                        is_rev = 1
                    else:
                        is_rev = 0
                var local = lo
                while local <= hi:
                    var bstart = coff + local
                    if bstart < 0 or UInt64(bstart + qlen) > l_pac:
                        local += 1
                        continue
                    var nm = 0
                    var j = 0
                    while j < qlen and nm <= budget + 4:
                        var ppos = bstart + j
                        var rb = Int(
                            (pac[ppos >> 2] >> UInt8(((~ppos) & 3) << 1)) & 3
                        )
                        var qc = Int(codes[q_base + j])
                        if is_rev != 0:
                            var qcr = Int(codes[q_base + (qlen - 1 - j)])
                            if qcr <= 3:
                                qc = 3 - qcr
                            else:
                                qc = 4
                        if rb > 3 or qc > 3 or rb != qc:
                            nm += 1
                        j += 1
                    if nm < scan_nm:
                        scan_nm = nm
                        scan_local = local
                        scan_rev = is_rev
                    if nm <= budget and nm < best_nm:
                        best_nm = nm
                        best_pos = local
                        best_flag = is_rev * 16
                    local += 1
                pass_s += 1
            if best_nm > budget and scan_nm < 9999:
                var bstart = coff + scan_local
                if bstart >= 0 and UInt64(bstart + qlen + 1) <= l_pac:
                    var g = 0
                    while g <= qlen:
                        var nm_d = 1
                        var t = 0
                        while t < g and nm_d <= budget:
                            var p0 = bstart + t
                            var rb0 = Int(
                                (pac[p0 >> 2] >> UInt8(((~p0) & 3) << 1)) & 3
                            )
                            var qc0 = Int(codes[q_base + t])
                            if scan_rev != 0:
                                var q0r = Int(codes[q_base + (qlen - 1 - t)])
                                if q0r <= 3:
                                    qc0 = 3 - q0r
                                else:
                                    qc0 = 4
                            if rb0 > 3 or qc0 > 3 or rb0 != qc0:
                                nm_d += 1
                            t += 1
                        while t < qlen and nm_d <= budget:
                            var p1 = bstart + t + 1
                            var rb1 = Int(
                                (pac[p1 >> 2] >> UInt8(((~p1) & 3) << 1)) & 3
                            )
                            var qc1 = Int(codes[q_base + t])
                            if scan_rev != 0:
                                var q1r = Int(codes[q_base + (qlen - 1 - t)])
                                if q1r <= 3:
                                    qc1 = 3 - q1r
                                else:
                                    qc1 = 4
                            if rb1 > 3 or qc1 > 3 or rb1 != qc1:
                                nm_d += 1
                            t += 1
                        if nm_d <= budget and nm_d < best_nm:
                            best_nm = nm_d
                            best_pos = scan_local
                            best_flag = scan_rev * 16
                        g += 4
            if best_nm <= budget:
                var mq: UInt32 = 20
                if best_nm == 0:
                    mq = 60
                elif best_nm <= 2:
                    mq = 40
                out_rid[dst] = Int32(crid)
                out_pos[dst] = UInt32(best_pos)
                out_mapq[dst] = mq
                out_flag[dst] = Int32(best_flag)
                out_sl[dst] = 0
                out_sr[dst] = 0
                out_nm[dst] = UInt32(best_nm)

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
        var rescue_win = _env_int("METHYLGRAPHER_FM_RESCUE_WIN", 512)
        var paired = fq2.byte_length() > 0
        var emit_bam = _path_is_bam(out_sam)
        var emit_kind = String("sam")
        if emit_bam:
            emit_kind = String("bam")
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
            " rescue_win=",
            rescue_win,
            " paired=",
            paired,
            " emit=",
            emit_kind,
        )

        var fh = Python.none()
        var bam_w = Python.none()
        var bam_arena = Python.none()
        var bam_runs = Python.none()
        var rid_to_tid = List[Int]()
        var bam_cap = 1
        var bam_blob = ctx.enqueue_create_host_buffer[DType.uint8](1)
        var rg_s = _env_str("METHYLGRAPHER_RG_ID", "mojo1")
        var sort_tile = 0
        var meta_cap = 1
        var use_runs = False
        if emit_bam:
            sort_tile = _env_int("METHYLGRAPHER_FM_SORT_TILE", 0)
            meta_cap = _env_int("METHYLGRAPHER_FM_SORT_CAP", 67108864)
            if sort_tile > 0:
                use_runs = True
                meta_cap = sort_tile
                var min_tile = batch_size
                if paired:
                    min_tile = batch_size * 2
                if meta_cap < min_tile:
                    raise Error(
                        "METHYLGRAPHER_FM_SORT_TILE must be >= records per FASTQ batch"
                    )
        var h_coord = ctx.enqueue_create_host_buffer[DType.uint64](meta_cap)
        var h_dhi = ctx.enqueue_create_host_buffer[DType.uint64](meta_cap)
        var h_dlo = ctx.enqueue_create_host_buffer[DType.uint64](meta_cap)
        var h_rec = ctx.enqueue_create_host_buffer[DType.uint64](meta_cap)
        var h_len = ctx.enqueue_create_host_buffer[DType.uint32](meta_cap)
        var h_score = ctx.enqueue_create_host_buffer[DType.uint32](meta_cap)
        var h_pair = ctx.enqueue_create_host_buffer[DType.uint32](meta_cap)
        var h_perm = ctx.enqueue_create_host_buffer[DType.uint32](meta_cap)
        var h_dup = ctx.enqueue_create_host_buffer[DType.uint32](meta_cap)
        var dupperm_n = 1
        if use_runs:
            dupperm_n = meta_cap
        var h_dupperm = ctx.enqueue_create_host_buffer[DType.uint32](dupperm_n)
        var md_raw = _env_str("METHYLGRAPHER_LINEAR_MARKDUP", "1").lower()
        var do_markdup = True
        if md_raw == "0" or md_raw == "false" or md_raw == "no" or md_raw == "off":
            do_markdup = False
        if emit_bam:
            bam_w = _open_bam_writer(index, out_sam)
            bam_arena = _open_bam_arena()
            if use_runs:
                bam_runs = _open_bam_run_store()
            rid_to_tid = _rid_to_tid_table(index)
            bam_cap = batch_size * 2 * 512
            if bam_cap < 1 << 20:
                bam_cap = 1 << 20
            bam_blob = ctx.enqueue_create_host_buffer[DType.uint8](bam_cap)
            print(
                "MojoLinear GPU-fm native BAM coordinate-sort cap=",
                meta_cap,
                " sort_tile=",
                sort_tile,
                " runs=",
                use_runs,
                " markdup=",
                do_markdup,
            )
        else:
            fh = open_text_write(out_sam)
            _write_sam_header_fm(fh, index)
        var fq_reader = Python.none()
        var fq_stream = FastqPairStream()
        var arena_cur = FastqArena()
        var arena_ready = FastqArena()
        var arena_spare = FastqArena()
        if emit_bam:
            fq_stream = fq_open(fq1, fq2, bs_r1, bs_r2)
            fq_reserve(arena_cur, batch_size, paired)
            fq_reserve(arena_ready, batch_size, paired)
            fq_reserve(arena_spare, batch_size, paired)
            print(
                "MojoLinear GPU-fm FASTQ=mojo-bulk overlap=pack||fq gpu=2-deep"
            )
        else:
            fq_reader = _open_fq_reader(fq1, fq2, bs_r1, bs_r2)

        var cap_n = batch_size * 2
        if cap_n < 2:
            cap_n = 2
        var max_len_cap = 1024
        var bases_cap = cap_n * max_len_cap
        var host_bases = ctx.enqueue_create_host_buffer[DType.uint8](bases_cap)
        var host_lens = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var dev_bases = ctx.enqueue_create_buffer[DType.uint8](bases_cap)
        var dev_codes = ctx.enqueue_create_buffer[DType.uint8](bases_cap)
        var dev_lens = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var dev_rid = ctx.enqueue_create_buffer[DType.int32](cap_n)
        var dev_pos = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var dev_mapq = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var dev_flag = ctx.enqueue_create_buffer[DType.int32](cap_n)
        var dev_sl = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var dev_sr = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var dev_nm = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var host_rid = ctx.enqueue_create_host_buffer[DType.int32](cap_n)
        var host_pos = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var host_mapq = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var host_flag = ctx.enqueue_create_host_buffer[DType.int32](cap_n)
        var host_sl = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var host_sr = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var host_nm = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var host_bases1 = ctx.enqueue_create_host_buffer[DType.uint8](bases_cap)
        var host_lens1 = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var dev_bases1 = ctx.enqueue_create_buffer[DType.uint8](bases_cap)
        var dev_codes1 = ctx.enqueue_create_buffer[DType.uint8](bases_cap)
        var dev_lens1 = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var dev_rid1 = ctx.enqueue_create_buffer[DType.int32](cap_n)
        var dev_pos1 = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var dev_mapq1 = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var dev_flag1 = ctx.enqueue_create_buffer[DType.int32](cap_n)
        var dev_sl1 = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var dev_sr1 = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var dev_nm1 = ctx.enqueue_create_buffer[DType.uint32](cap_n)
        var host_rid1 = ctx.enqueue_create_host_buffer[DType.int32](cap_n)
        var host_pos1 = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var host_mapq1 = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var host_flag1 = ctx.enqueue_create_host_buffer[DType.int32](cap_n)
        var host_sl1 = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var host_sr1 = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var host_nm1 = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var ev0 = ctx.create_event()
        var ev1 = ctx.create_event()
        var gpu_st = ctx.stream()
        var emit_rid = ctx.enqueue_create_host_buffer[DType.int32](cap_n)
        var emit_pos = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var emit_mapq = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var emit_flag = ctx.enqueue_create_host_buffer[DType.int32](cap_n)
        var emit_sl = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var emit_sr = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var emit_nm = ctx.enqueue_create_host_buffer[DType.uint32](cap_n)
        var ptr_cap = batch_size
        if ptr_cap < 1:
            ptr_cap = 1
        var h_n1_name_a = ctx.enqueue_create_host_buffer[DType.uint64](ptr_cap)
        var h_n1_name_n = ctx.enqueue_create_host_buffer[DType.uint32](ptr_cap)
        var h_n1_orig_a = ctx.enqueue_create_host_buffer[DType.uint64](ptr_cap)
        var h_n1_orig_n = ctx.enqueue_create_host_buffer[DType.uint32](ptr_cap)
        var h_n1_qual_a = ctx.enqueue_create_host_buffer[DType.uint64](ptr_cap)
        var h_n1_qual_n = ctx.enqueue_create_host_buffer[DType.uint32](ptr_cap)
        var h_n2_name_a = ctx.enqueue_create_host_buffer[DType.uint64](ptr_cap)
        var h_n2_name_n = ctx.enqueue_create_host_buffer[DType.uint32](ptr_cap)
        var h_n2_orig_a = ctx.enqueue_create_host_buffer[DType.uint64](ptr_cap)
        var h_n2_orig_n = ctx.enqueue_create_host_buffer[DType.uint32](ptr_cap)
        var h_n2_qual_a = ctx.enqueue_create_host_buffer[DType.uint64](ptr_cap)
        var h_n2_qual_n = ctx.enqueue_create_host_buffer[DType.uint32](ptr_cap)
        var h_seq_a = ctx.enqueue_create_host_buffer[DType.uint64](cap_n)
        var rg_bytes = rg_s.as_bytes()
        var rg_p = rg_bytes.unsafe_ptr()
        var rg_n = rg_s.byte_length()
        var paired_i = 0
        if paired:
            paired_i = 1

        var n_mapped = 0
        var n_reads = 0
        var n_total = 0
        var n_batches = 0
        var tile_chunks = 0
        var t_map0 = time_mod.perf_counter()
        var t_gpu = time_mod.perf_counter() * 0
        var t_emit = time_mod.perf_counter() * 0
        var t_fastq = time_mod.perf_counter() * 0
        comptime BLOCK = 256
        var k_pack = ctx.compile_function[pack_bases_kernel]()
        var k_seed = ctx.compile_function[fm_seed_extend_kernel]()
        var k_rescue = ctx.compile_function[fm_mate_rescue_kernel]()

        def _fm_fire_batch(
            ctx: DeviceContext,
            st: DeviceStream,
            k_pack: DeviceFunction,
            k_seed: DeviceFunction,
            k_rescue: DeviceFunction,
            mut host_bases: HostBuffer[DType.uint8],
            mut host_lens: HostBuffer[DType.uint32],
            mut host_rid: HostBuffer[DType.int32],
            mut host_pos: HostBuffer[DType.uint32],
            mut host_mapq: HostBuffer[DType.uint32],
            mut host_flag: HostBuffer[DType.int32],
            mut host_sl: HostBuffer[DType.uint32],
            mut host_sr: HostBuffer[DType.uint32],
            mut host_nm: HostBuffer[DType.uint32],
            mut dev_bases: DeviceBuffer[DType.uint8],
            mut dev_codes: DeviceBuffer[DType.uint8],
            mut dev_lens: DeviceBuffer[DType.uint32],
            mut dev_rid: DeviceBuffer[DType.int32],
            mut dev_pos: DeviceBuffer[DType.uint32],
            mut dev_mapq: DeviceBuffer[DType.uint32],
            mut dev_flag: DeviceBuffer[DType.int32],
            mut dev_sl: DeviceBuffer[DType.uint32],
            mut dev_sr: DeviceBuffer[DType.uint32],
            mut dev_nm: DeviceBuffer[DType.uint32],
            mut ev: DeviceEvent,
            n_bases: Int,
            n_seq: Int,
            max_len: Int,
            mut index_bwt: DeviceBuffer[DType.uint32],
            mut index_sa: DeviceBuffer[DType.uint64],
            mut index_pac: DeviceBuffer[DType.uint8],
            mut index_coff: DeviceBuffer[DType.uint64],
            mut index_clen: DeviceBuffer[DType.uint32],
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
            seed_len: Int,
            seed_stride: Int,
            max_occ: Int,
            max_diff: Int,
            max_soft: Int,
            max_adj: Int,
            rescue_win: Int,
            paired: Bool,
        ) raises:
            ctx.enqueue_copy(src_buf=host_bases, dst_buf=dev_bases)
            ctx.enqueue_copy(src_buf=host_lens, dst_buf=dev_lens)
            var grid_b = (n_bases + 256 - 1) // 256
            var grid_r = (n_seq + 256 - 1) // 256
            st.enqueue_function(
                k_pack,
                dev_bases.unsafe_ptr(),
                dev_codes.unsafe_ptr(),
                n_bases,
                grid_dim=grid_b,
                block_dim=256,
            )
            st.enqueue_function(
                k_seed,
                index_bwt.unsafe_ptr(),
                index_sa.unsafe_ptr(),
                index_pac.unsafe_ptr(),
                index_coff.unsafe_ptr(),
                index_clen.unsafe_ptr(),
                n_contigs,
                primary,
                seq_len,
                l_pac,
                l2_0,
                l2_1,
                l2_2,
                l2_3,
                l2_4,
                sa_intv,
                n_sa,
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
                block_dim=256,
            )
            if paired and rescue_win > 0:
                var n_pairs = n_seq // 2
                var grid_p = (n_pairs + 256 - 1) // 256
                st.enqueue_function(
                    k_rescue,
                    index_pac.unsafe_ptr(),
                    index_coff.unsafe_ptr(),
                    index_clen.unsafe_ptr(),
                    n_contigs,
                    l_pac,
                    dev_codes.unsafe_ptr(),
                    dev_lens.unsafe_ptr(),
                    max_len,
                    n_pairs,
                    rescue_win,
                    max_diff,
                    max_soft,
                    dev_rid.unsafe_ptr(),
                    dev_pos.unsafe_ptr(),
                    dev_mapq.unsafe_ptr(),
                    dev_flag.unsafe_ptr(),
                    dev_sl.unsafe_ptr(),
                    dev_sr.unsafe_ptr(),
                    dev_nm.unsafe_ptr(),
                    grid_dim=grid_p,
                    block_dim=256,
                )
            ctx.enqueue_copy(src_buf=dev_rid, dst_buf=host_rid)
            ctx.enqueue_copy(src_buf=dev_pos, dst_buf=host_pos)
            ctx.enqueue_copy(src_buf=dev_mapq, dst_buf=host_mapq)
            ctx.enqueue_copy(src_buf=dev_flag, dst_buf=host_flag)
            ctx.enqueue_copy(src_buf=dev_sl, dst_buf=host_sl)
            ctx.enqueue_copy(src_buf=dev_sr, dst_buf=host_sr)
            ctx.enqueue_copy(src_buf=dev_nm, dst_buf=host_nm)
            st.record_event(ev)


        var t_f0 = time_mod.perf_counter()
        var py_cur = Python.none()
        var n1_cur = 0
        if emit_bam:
            fq_read_batch(fq_stream, arena_cur, batch_size)
            n1_cur = arena_cur.n1
        else:
            py_cur = fq_reader.read_batch(batch_size)
            n1_cur = Int(py=py_cur.n1)
        t_fastq = t_fastq + (time_mod.perf_counter() - t_f0)
        var py_ready = Python.none()
        var n1_ready = 0
        var have_ready = False
        var gpu_live = False
        var gpu_g = 0
        var n_seq_g = 0


        while n1_cur > 0:
            var t_g0 = time_mod.perf_counter()
            var max_len = 0
            if emit_bam:
                max_len = arena_cur.max_len
            else:
                max_len = Int(py=py_cur.max_len)
            if max_len < seed_len:
                max_len = seed_len
            if max_len > max_len_cap:
                raise Error("read longer than FM base cap (1024)")
            if n1_cur > ptr_cap:
                raise Error("FASTQ batch larger than pointer tables")
            var n_seq = n1_cur
            if paired:
                n_seq = n1_cur * 2
            if n_seq > cap_n:
                raise Error("FASTQ batch larger than GPU buffers")
            var n_bases = n_seq * max_len

            var next_live = False
            var n_seq_next = 0
            var gpu_g2 = gpu_g
            var n1_next = 0
            var py_next = Python.none()
            var t_emit_ov = time_mod.perf_counter() * 0
            var t_overlap = time_mod.perf_counter() * 0
            var do_fire = False
            var fire_g = 0
            var fire_n_seq = n_seq
            var fire_max_len = max_len
            var fire_n_bases = n_bases

            if emit_bam:
                if not gpu_live:
                    if gpu_g == 0:
                        fq_pack_bases(
                            arena_cur,
                            paired,
                            Int(host_bases.unsafe_ptr()),
                            max_len,
                            Int(host_lens.unsafe_ptr()),
                        )
                    else:
                        fq_pack_bases(
                            arena_cur,
                            paired,
                            Int(host_bases1.unsafe_ptr()),
                            max_len,
                            Int(host_lens1.unsafe_ptr()),
                        )
                    do_fire = True
                    fire_g = gpu_g
                    n_seq_g = n_seq
                    gpu_live = True
            else:
                py_cur.zero_align(Int(host_bases.unsafe_ptr()), n_bases)
                py_cur.export_seq_ptrs(
                    Int(h_seq_a.unsafe_ptr()),
                    Int(host_lens.unsafe_ptr()),
                    paired_i,
                )
                var bases_addr = Int(host_bases.unsafe_ptr())
                var seq_ap = _u64_at(Int(h_seq_a.unsafe_ptr()))
                var si = 0
                while si < n_seq:
                    var ln = Int(host_lens[si])
                    if ln > 0:
                        memcpy(
                            dest=_u8_at(bases_addr + si * max_len),
                            src=_u8_at(Int(seq_ap[si])),
                            count=ln,
                        )
                    si += 1
                do_fire = True
                fire_g = 0

            if do_fire:
                if fire_g == 0:
                    _fm_fire_batch(
                        ctx,
                        gpu_st,
                        k_pack,
                        k_seed,
                        k_rescue,
                        host_bases,
                        host_lens,
                        host_rid,
                        host_pos,
                        host_mapq,
                        host_flag,
                        host_sl,
                        host_sr,
                        host_nm,
                        dev_bases,
                        dev_codes,
                        dev_lens,
                        dev_rid,
                        dev_pos,
                        dev_mapq,
                        dev_flag,
                        dev_sl,
                        dev_sr,
                        dev_nm,
                        ev0,
                        fire_n_bases,
                        fire_n_seq,
                        fire_max_len,
                        dev_bwt,
                        dev_sa,
                        dev_pac,
                        dev_coff,
                        dev_clen,
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
                        seed_len,
                        seed_stride,
                        max_occ,
                        max_diff,
                        max_soft,
                        max_adj,
                        rescue_win,
                        paired,
                    )
                else:
                    _fm_fire_batch(
                        ctx,
                        gpu_st,
                        k_pack,
                        k_seed,
                        k_rescue,
                        host_bases1,
                        host_lens1,
                        host_rid1,
                        host_pos1,
                        host_mapq1,
                        host_flag1,
                        host_sl1,
                        host_sr1,
                        host_nm1,
                        dev_bases1,
                        dev_codes1,
                        dev_lens1,
                        dev_rid1,
                        dev_pos1,
                        dev_mapq1,
                        dev_flag1,
                        dev_sl1,
                        dev_sr1,
                        dev_nm1,
                        ev1,
                        fire_n_bases,
                        fire_n_seq,
                        fire_max_len,
                        dev_bwt,
                        dev_sa,
                        dev_pac,
                        dev_coff,
                        dev_clen,
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
                        seed_len,
                        seed_stride,
                        max_occ,
                        max_diff,
                        max_soft,
                        max_adj,
                        rescue_win,
                        paired,
                    )
            do_fire = False
            var t_ov0 = time_mod.perf_counter()
            var fut = Python.none()
            if not emit_bam:
                fut = fq_reader.read_batch_async(batch_size)
            t_emit_ov = time_mod.perf_counter() * 0
            if emit_bam and have_ready:
                _maybe_flush_tile(
                    use_runs,
                    paired,
                    n1_ready,
                    n_reads,
                    n_total,
                    tile_chunks,
                    meta_cap,
                    api,
                    do_markdup,
                    bam_arena,
                    bam_runs,
                    Int(h_coord.unsafe_ptr()),
                    Int(h_dhi.unsafe_ptr()),
                    Int(h_dlo.unsafe_ptr()),
                    Int(h_score.unsafe_ptr()),
                    Int(h_pair.unsafe_ptr()),
                    Int(h_dup.unsafe_ptr()),
                    Int(h_perm.unsafe_ptr()),
                    Int(h_dupperm.unsafe_ptr()),
                    Int(h_rec.unsafe_ptr()),
                    Int(h_len.unsafe_ptr()),
                    Int(bam_blob.unsafe_ptr()),
                    bam_cap,
                )
                var ov_packed = List[Int](length=1, fill=0)
                var ov_fail = List[Int](length=2, fill=0)
                var ov_t = List[Float64](length=2, fill=Float64(0))

                @parameter
                def ov_work(i: Int):
                    var tw = _tick()
                    try:
                        if i == 0:
                            ov_packed[0] = _emit_bam_from_batch(
                                arena_ready,
                                paired,
                                n1_ready,
                                Int(emit_rid.unsafe_ptr()),
                                Int(emit_pos.unsafe_ptr()),
                                Int(emit_mapq.unsafe_ptr()),
                                Int(emit_flag.unsafe_ptr()),
                                Int(emit_sl.unsafe_ptr()),
                                Int(emit_sr.unsafe_ptr()),
                                Int(emit_nm.unsafe_ptr()),
                                Int(h_n1_name_a.unsafe_ptr()),
                                Int(h_n1_name_n.unsafe_ptr()),
                                Int(h_n1_orig_a.unsafe_ptr()),
                                Int(h_n1_orig_n.unsafe_ptr()),
                                Int(h_n1_qual_a.unsafe_ptr()),
                                Int(h_n1_qual_n.unsafe_ptr()),
                                Int(h_n2_name_a.unsafe_ptr()),
                                Int(h_n2_name_n.unsafe_ptr()),
                                Int(h_n2_orig_a.unsafe_ptr()),
                                Int(h_n2_orig_n.unsafe_ptr()),
                                Int(h_n2_qual_a.unsafe_ptr()),
                                Int(h_n2_qual_n.unsafe_ptr()),
                                bam_blob.unsafe_ptr(),
                                bam_cap,
                                rid_to_tid,
                                rg_p,
                                rg_n,
                                Int(h_coord.unsafe_ptr()),
                                Int(h_dhi.unsafe_ptr()),
                                Int(h_dlo.unsafe_ptr()),
                                Int(h_rec.unsafe_ptr()),
                                Int(h_len.unsafe_ptr()),
                                Int(h_score.unsafe_ptr()),
                                Int(h_pair.unsafe_ptr()),
                                n_reads,
                                n_mapped,
                                tile_chunks,
                                meta_cap,
                            )
                        else:
                            fq_read_batch(fq_stream, arena_spare, batch_size)
                    except e:
                        ov_fail[i] = 1
                        print("MojoLinear GPU-fm overlap worker", i, e)
                    ov_t[i] = _tick() - tw

                parallelize[ov_work](2, 2)
                if ov_fail[0] != 0 or ov_fail[1] != 0:
                    raise Error("FM pack||FASTQ overlap worker failed")
                _ = bam_arena.append_raw(
                    Int(bam_blob.unsafe_ptr()), ov_packed[0]
                )
                tile_chunks += 1
                t_emit_ov = ov_t[0]
                t_emit = t_emit + t_emit_ov
                t_fastq = t_fastq + ov_t[1]
                n1_next = arena_spare.n1
                n_batches += 1
                if n_batches == 1 or n_batches % 5 == 0:
                    print(
                        "MojoLinear GPU-fm progress batches=",
                        n_batches,
                        " gpu_reads≈",
                        n_total + n_reads,
                    )
            elif emit_bam:
                var t_f1 = time_mod.perf_counter()
                fq_read_batch(fq_stream, arena_spare, batch_size)
                n1_next = arena_spare.n1
                t_fastq = t_fastq + (time_mod.perf_counter() - t_f1)
            else:
                if have_ready:
                    var t_e0 = time_mod.perf_counter()
                    var batch1 = List[FastqRec]()
                    var batch2 = List[FastqRec]()
                    _ = _py_batch_to_recs(py_ready, paired, batch1, batch2)
                    var pj = 0
                    while pj < n1_ready:
                        var h1 = _hit_from_fm(
                            index,
                            batch1[pj].name,
                            batch1[pj].seq,
                            batch1[pj].original_seq,
                            Int(emit_rid[pj]),
                            Int(emit_pos[pj]),
                            Int(emit_mapq[pj]),
                            Int(emit_flag[pj]),
                            Int(emit_sl[pj]),
                            Int(emit_sr[pj]),
                        )
                        h1.qual = batch1[pj].qual
                        if not paired:
                            if h1.contig != "*":
                                n_mapped += 1
                            fh.write(_sam_with_rg_fm(h1) + "\n")
                            n_reads += 1
                        else:
                            var r2i = n1_ready + pj
                            var h2 = _hit_from_fm(
                                index,
                                batch2[pj].name,
                                batch2[pj].seq,
                                batch2[pj].original_seq,
                                Int(emit_rid[r2i]),
                                Int(emit_pos[r2i]),
                                Int(emit_mapq[r2i]),
                                Int(emit_flag[r2i]),
                                Int(emit_sl[r2i]),
                                Int(emit_sr[r2i]),
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
                    t_emit_ov = time_mod.perf_counter() - t_e0
                    t_emit = t_emit + t_emit_ov
                    n_batches += 1
                    if n_batches == 1 or n_batches % 5 == 0:
                        print(
                            "MojoLinear GPU-fm progress batches=",
                            n_batches,
                            " gpu_reads≈",
                            n_reads,
                        )
                py_next = fut.result()
                n1_next = Int(py=py_next.n1)
            t_overlap = time_mod.perf_counter() - t_ov0
            if not emit_bam:
                var t_fq_part = t_overlap - t_emit_ov
                if t_fq_part < 0:
                    t_fq_part = 0
                t_fastq = t_fastq + t_fq_part
            if emit_bam and n1_next > 0:
                gpu_g2 = 1 - gpu_g
                var max_len_n = arena_spare.max_len
                if max_len_n < seed_len:
                    max_len_n = seed_len
                if max_len_n > max_len_cap:
                    raise Error("read longer than FM base cap (1024)")
                if n1_next > ptr_cap:
                    raise Error("FASTQ batch larger than pointer tables")
                var n_seq_n = n1_next
                if paired:
                    n_seq_n = n1_next * 2
                if n_seq_n > cap_n:
                    raise Error("FASTQ batch larger than GPU buffers")
                var n_bases_n = n_seq_n * max_len_n
                if gpu_g2 == 0:
                    fq_pack_bases(
                        arena_spare,
                        paired,
                        Int(host_bases.unsafe_ptr()),
                        max_len_n,
                        Int(host_lens.unsafe_ptr()),
                    )
                else:
                    fq_pack_bases(
                        arena_spare,
                        paired,
                        Int(host_bases1.unsafe_ptr()),
                        max_len_n,
                        Int(host_lens1.unsafe_ptr()),
                    )
                do_fire = True
                fire_g = gpu_g2
                fire_n_seq = n_seq_n
                fire_max_len = max_len_n
                fire_n_bases = n_bases_n
                n_seq_next = n_seq_n
                next_live = True

            if do_fire:
                if fire_g == 0:
                    _fm_fire_batch(
                        ctx,
                        gpu_st,
                        k_pack,
                        k_seed,
                        k_rescue,
                        host_bases,
                        host_lens,
                        host_rid,
                        host_pos,
                        host_mapq,
                        host_flag,
                        host_sl,
                        host_sr,
                        host_nm,
                        dev_bases,
                        dev_codes,
                        dev_lens,
                        dev_rid,
                        dev_pos,
                        dev_mapq,
                        dev_flag,
                        dev_sl,
                        dev_sr,
                        dev_nm,
                        ev0,
                        fire_n_bases,
                        fire_n_seq,
                        fire_max_len,
                        dev_bwt,
                        dev_sa,
                        dev_pac,
                        dev_coff,
                        dev_clen,
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
                        seed_len,
                        seed_stride,
                        max_occ,
                        max_diff,
                        max_soft,
                        max_adj,
                        rescue_win,
                        paired,
                    )
                else:
                    _fm_fire_batch(
                        ctx,
                        gpu_st,
                        k_pack,
                        k_seed,
                        k_rescue,
                        host_bases1,
                        host_lens1,
                        host_rid1,
                        host_pos1,
                        host_mapq1,
                        host_flag1,
                        host_sl1,
                        host_sr1,
                        host_nm1,
                        dev_bases1,
                        dev_codes1,
                        dev_lens1,
                        dev_rid1,
                        dev_pos1,
                        dev_mapq1,
                        dev_flag1,
                        dev_sl1,
                        dev_sr1,
                        dev_nm1,
                        ev1,
                        fire_n_bases,
                        fire_n_seq,
                        fire_max_len,
                        dev_bwt,
                        dev_sa,
                        dev_pac,
                        dev_coff,
                        dev_clen,
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
                        seed_len,
                        seed_stride,
                        max_occ,
                        max_diff,
                        max_soft,
                        max_adj,
                        rescue_win,
                        paired,
                    )

            var t_w0 = time_mod.perf_counter()
            if emit_bam:
                if gpu_g == 0:
                    ev0.synchronize()
                    _copy_bytes(
                        Int(emit_rid.unsafe_ptr()), Int(host_rid.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_pos.unsafe_ptr()), Int(host_pos.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_mapq.unsafe_ptr()), Int(host_mapq.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_flag.unsafe_ptr()), Int(host_flag.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_sl.unsafe_ptr()), Int(host_sl.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_sr.unsafe_ptr()), Int(host_sr.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_nm.unsafe_ptr()), Int(host_nm.unsafe_ptr()), n_seq_g * 4
                    )
                else:
                    ev1.synchronize()
                    _copy_bytes(
                        Int(emit_rid.unsafe_ptr()), Int(host_rid1.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_pos.unsafe_ptr()), Int(host_pos1.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_mapq.unsafe_ptr()), Int(host_mapq1.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_flag.unsafe_ptr()), Int(host_flag1.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_sl.unsafe_ptr()), Int(host_sl1.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_sr.unsafe_ptr()), Int(host_sr1.unsafe_ptr()), n_seq_g * 4
                    )
                    _copy_bytes(
                        Int(emit_nm.unsafe_ptr()), Int(host_nm1.unsafe_ptr()), n_seq_g * 4
                    )
                t_gpu = t_gpu + (time_mod.perf_counter() - t_w0)
                gpu_live = next_live
                if next_live:
                    gpu_g = gpu_g2
                    n_seq_g = n_seq_next
            else:
                ctx.synchronize()
                t_gpu = t_gpu + (time_mod.perf_counter() - t_g0) - t_overlap
                _copy_bytes(
                    Int(emit_rid.unsafe_ptr()), Int(host_rid.unsafe_ptr()), n_seq * 4
                )
                _copy_bytes(
                    Int(emit_pos.unsafe_ptr()), Int(host_pos.unsafe_ptr()), n_seq * 4
                )
                _copy_bytes(
                    Int(emit_mapq.unsafe_ptr()), Int(host_mapq.unsafe_ptr()), n_seq * 4
                )
                _copy_bytes(
                    Int(emit_flag.unsafe_ptr()), Int(host_flag.unsafe_ptr()), n_seq * 4
                )
                _copy_bytes(
                    Int(emit_sl.unsafe_ptr()), Int(host_sl.unsafe_ptr()), n_seq * 4
                )
                _copy_bytes(
                    Int(emit_sr.unsafe_ptr()), Int(host_sr.unsafe_ptr()), n_seq * 4
                )
                _copy_bytes(
                    Int(emit_nm.unsafe_ptr()), Int(host_nm.unsafe_ptr()), n_seq * 4
                )
            if emit_bam:
                var tmp = arena_ready^
                arena_ready = arena_cur^
                arena_cur = arena_spare^
                arena_spare = tmp^
            else:
                py_ready = py_cur
                py_cur = py_next
            n1_ready = n1_cur
            have_ready = True
            n1_cur = n1_next

        if have_ready:
            var t_e1 = time_mod.perf_counter()
            if emit_bam:
                _maybe_flush_tile(
                    use_runs,
                    paired,
                    n1_ready,
                    n_reads,
                    n_total,
                    tile_chunks,
                    meta_cap,
                    api,
                    do_markdup,
                    bam_arena,
                    bam_runs,
                    Int(h_coord.unsafe_ptr()),
                    Int(h_dhi.unsafe_ptr()),
                    Int(h_dlo.unsafe_ptr()),
                    Int(h_score.unsafe_ptr()),
                    Int(h_pair.unsafe_ptr()),
                    Int(h_dup.unsafe_ptr()),
                    Int(h_perm.unsafe_ptr()),
                    Int(h_dupperm.unsafe_ptr()),
                    Int(h_rec.unsafe_ptr()),
                    Int(h_len.unsafe_ptr()),
                    Int(bam_blob.unsafe_ptr()),
                    bam_cap,
                )
                var packed_last = _emit_bam_from_batch(
                    arena_ready,
                    paired,
                    n1_ready,
                    Int(emit_rid.unsafe_ptr()),
                    Int(emit_pos.unsafe_ptr()),
                    Int(emit_mapq.unsafe_ptr()),
                    Int(emit_flag.unsafe_ptr()),
                    Int(emit_sl.unsafe_ptr()),
                    Int(emit_sr.unsafe_ptr()),
                    Int(emit_nm.unsafe_ptr()),
                    Int(h_n1_name_a.unsafe_ptr()),
                    Int(h_n1_name_n.unsafe_ptr()),
                    Int(h_n1_orig_a.unsafe_ptr()),
                    Int(h_n1_orig_n.unsafe_ptr()),
                    Int(h_n1_qual_a.unsafe_ptr()),
                    Int(h_n1_qual_n.unsafe_ptr()),
                    Int(h_n2_name_a.unsafe_ptr()),
                    Int(h_n2_name_n.unsafe_ptr()),
                    Int(h_n2_orig_a.unsafe_ptr()),
                    Int(h_n2_orig_n.unsafe_ptr()),
                    Int(h_n2_qual_a.unsafe_ptr()),
                    Int(h_n2_qual_n.unsafe_ptr()),
                    bam_blob.unsafe_ptr(),
                    bam_cap,
                    rid_to_tid,
                    rg_p,
                    rg_n,
                    Int(h_coord.unsafe_ptr()),
                    Int(h_dhi.unsafe_ptr()),
                    Int(h_dlo.unsafe_ptr()),
                    Int(h_rec.unsafe_ptr()),
                    Int(h_len.unsafe_ptr()),
                    Int(h_score.unsafe_ptr()),
                    Int(h_pair.unsafe_ptr()),
                    n_reads,
                    n_mapped,
                    tile_chunks,
                    meta_cap,
                )
                _ = bam_arena.append_raw(
                    Int(bam_blob.unsafe_ptr()), packed_last
                )
                tile_chunks += 1
            else:
                var batch1 = List[FastqRec]()
                var batch2 = List[FastqRec]()
                _ = _py_batch_to_recs(py_ready, paired, batch1, batch2)
                var pj = 0
                while pj < n1_ready:
                    var h1 = _hit_from_fm(
                        index,
                        batch1[pj].name,
                        batch1[pj].seq,
                        batch1[pj].original_seq,
                        Int(emit_rid[pj]),
                        Int(emit_pos[pj]),
                        Int(emit_mapq[pj]),
                        Int(emit_flag[pj]),
                        Int(emit_sl[pj]),
                        Int(emit_sr[pj]),
                    )
                    h1.qual = batch1[pj].qual
                    if not paired:
                        if h1.contig != "*":
                            n_mapped += 1
                        fh.write(_sam_with_rg_fm(h1) + "\n")
                        n_reads += 1
                    else:
                        var r2i = n1_ready + pj
                        var h2 = _hit_from_fm(
                            index,
                            batch2[pj].name,
                            batch2[pj].seq,
                            batch2[pj].original_seq,
                            Int(emit_rid[r2i]),
                            Int(emit_pos[r2i]),
                            Int(emit_mapq[r2i]),
                            Int(emit_flag[r2i]),
                            Int(emit_sl[r2i]),
                            Int(emit_sr[r2i]),
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
            t_emit = t_emit + (time_mod.perf_counter() - t_e1)
            n_batches += 1
            if n_batches == 1 or n_batches % 5 == 0:
                print(
                    "MojoLinear GPU-fm progress batches=",
                    n_batches,
                    " gpu_reads≈",
                    n_total + n_reads,
                )

        if emit_bam:
            fq_close(fq_stream)
        else:
            fq_reader.close()
        var t_sort = time_mod.perf_counter() * 0
        if emit_bam:
            var t_s0 = time_mod.perf_counter()
            var t_g0s = time_mod.perf_counter()
            var n_dups = 0
            if use_runs:
                if n_reads > 0:
                    _flush_sorted_bam_run(
                        api,
                        n_reads,
                        do_markdup,
                        bam_arena,
                        bam_runs,
                        Int(h_coord.unsafe_ptr()),
                        Int(h_dhi.unsafe_ptr()),
                        Int(h_dlo.unsafe_ptr()),
                        Int(h_score.unsafe_ptr()),
                        Int(h_pair.unsafe_ptr()),
                        Int(h_dup.unsafe_ptr()),
                        Int(h_perm.unsafe_ptr()),
                        Int(h_dupperm.unsafe_ptr()),
                        Int(h_rec.unsafe_ptr()),
                        Int(h_len.unsafe_ptr()),
                        Int(bam_blob.unsafe_ptr()),
                        bam_cap,
                    )
                    n_total += n_reads
                    n_reads = 0
                t_sort = time_mod.perf_counter() - t_s0
                t_g0s = time_mod.perf_counter()
                n_dups = Int(py=bam_runs.merge_into(bam_w, do_markdup))
                bam_w.close()
            else:
                if n_reads > 0:
                    gpu_sort_markdup(
                        api,
                        n_reads,
                        Int(h_coord.unsafe_ptr()),
                        Int(h_dhi.unsafe_ptr()),
                        Int(h_dlo.unsafe_ptr()),
                        Int(h_score.unsafe_ptr()),
                        Int(h_pair.unsafe_ptr()),
                        Int(h_dup.unsafe_ptr()),
                        Int(h_perm.unsafe_ptr()),
                        do_markdup,
                        0,
                    )
                t_sort = time_mod.perf_counter() - t_s0
                t_g0s = time_mod.perf_counter()
                if n_reads > 0:
                    _gather_perm_to_writer(
                        bam_w,
                        n_reads,
                        bam_arena,
                        Int(h_perm.unsafe_ptr()),
                        Int(h_rec.unsafe_ptr()),
                        Int(h_len.unsafe_ptr()),
                        Int(h_dup.unsafe_ptr()),
                        Int(bam_blob.unsafe_ptr()),
                        bam_cap,
                        True,
                    )
                n_total = n_reads
                bam_w.close()
            t_emit = t_emit + (time_mod.perf_counter() - t_g0s)
            var n_run_out = 0
            if use_runs:
                n_run_out = Int(py=bam_runs.n_runs())
            print(
                "MojoLinear GPU-fm sort_markdup_s=",
                t_sort,
                " gather_bgzf_s=",
                time_mod.perf_counter() - t_g0s,
                " arena_bytes=",
                Int(py=bam_arena.nbytes()),
                " runs=",
                n_run_out,
                " dups_marked=",
                n_dups,
            )
        else:
            fh.close()
        var t_map1 = time_mod.perf_counter()
        print(
            "wrote records -> ",
            out_sam,
            " mapped_records=",
            n_mapped,
            " reads=",
            n_total,
            " backend=gpu-fm",
        )
        print(
            "MojoLinear GPU-fm map_wall_s=",
            t_map1 - t_map0,
            " gpu_kernel_s=",
            t_gpu,
            " emit_s=",
            t_emit,
            " sort_s=",
            t_sort,
            " fastq_s=",
            t_fastq,
            " reads=",
            n_reads,
        )
        return n_mapped
