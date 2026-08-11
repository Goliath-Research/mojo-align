# Full-GPU dense-v1 linear WGBS map (NVIDIA DeviceContext).
#
# Resident on device (~27 GiB science): kmers + offsets + postings + sequences
# + contig_offsets. Per batch: 2-bit encode → locate → vote → gapless on GPU.
# Host only streams FASTQ and formats SAM from compact hit records.
#
# Locate default = binary search on sorted keys. Optional interpolation search
# via METHYLGRAPHER_LINEAR_LOCATE_ALGO=interp (keys are numeric 2-bit codes).

from std.collections import Dict, List
from std.python import Python, PythonObject
from std.sys import has_accelerator

from gpu_device import select_device
from gpu_kernels import _device_api, kernel_target_label, probe_device_context
from linear_extend import hit_to_sam_line, pair_hits, LinearHit
from linear_index import LinearIndex
from utility import open_text_write, reverse_complement


struct FastqRec(Copyable, Movable):
    var name: String
    var seq: String  # converted (align)
    var original_seq: String  # pre-conversion (GATK/Picard SEQ)
    var qual: String

    def __init__(
        out self,
        name: String,
        seq: String,
        qual: String = "*",
        original_seq: String = "",
    ):
        self.name = name
        self.seq = seq
        self.qual = qual
        if original_seq.byte_length() == 0:
            self.original_seq = seq
        else:
            self.original_seq = original_seq


def _strip_nl(mut s: String):
    while s.byte_length() > 0:
        var last = String(s[byte = s.byte_length() - 1 : s.byte_length()])
        if last == "\n" or last == "\r":
            s = String(s[byte = 0 : s.byte_length() - 1])
        else:
            break


def _bs_convert(seq: String, mode: String) raises -> String:
    if mode.byte_length() == 0 or mode == "none":
        return seq
    # Python C translate — conversion only; locate/vote/gapless stay on GPU.
    var builtins = Python.import_module("builtins")
    var str_mod = builtins.str
    if mode == "C2T":
        var tr = str_mod.maketrans("Cc", "Tt")
        return String(str_mod.translate(seq, tr))
    if mode == "G2A":
        var tr2 = str_mod.maketrans("Gg", "Aa")
        return String(str_mod.translate(seq, tr2))
    return seq


def _open_fastq(path: String) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var gzip = Python.import_module("gzip")
    var low = path.lower()
    if low.endswith(".gz") or low.endswith(".gzip"):
        return gzip.open(path, "rt")
    return builtins.open(path, "r")


def _read_one(fh: PythonObject) raises -> FastqRec:
    var n = String(fh.readline())
    if n.byte_length() == 0:
        return FastqRec("", "", "")
    var s = String(fh.readline())
    _ = String(fh.readline())
    var q = String(fh.readline())
    _strip_nl(n)
    _strip_nl(s)
    _strip_nl(q)
    if n.startswith("@"):
        n = String(n[byte = 1 : n.byte_length()])
    var bare = n
    var parts = n.split(" ")
    if len(parts) > 0:
        bare = String(parts[0])
    var slash = bare.split("/")
    if len(slash) > 0:
        bare = String(slash[0])
    if q.byte_length() == 0:
        q = String("*")
    return FastqRec(bare, s, q)


def _env_int(name: String, default: Int) raises -> Int:
    var os_mod = Python.import_module("os")
    var raw = String(os_mod.environ.get(name, String(default)))
    var n = Int(raw)
    if n < 1:
        return default
    return n


def _env_str(name: String, default: String) raises -> String:
    var os_mod = Python.import_module("os")
    var raw = String(os_mod.environ.get(name, default))
    if raw.byte_length() == 0:
        return default
    return raw


def _canonical_contig(name: String) raises -> String:
    """Strip bwameth ``f``/``r`` prefix so SAM matches the original reference."""
    if name.byte_length() >= 2:
        var p = String(name[byte = 0 : 1])
        if p == "f" or p == "r":
            return String(name[byte = 1 : name.byte_length()])
    return name


def _rg_id() raises -> String:
    return _env_str("METHYLGRAPHER_RG_ID", "mojo1")


def _write_sam_header(fh: PythonObject, index: LinearIndex) raises:
    # Unsorted here; orchestrator samtools sort → coordinate (GATK requires it).
    fh.write("@HD\tVN:1.6\tSO:unsorted\n")
    var n = index.contig_count()
    var seen = Dict[String, Int]()
    var ci = 0
    while ci < n:
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
    var rg = _rg_id()
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
    fh.write("@PG\tID:MojoFq2bamMeth\tPN:MojoFq2bamMeth\tVN:0.1.0-mojo\n")
    try:
        fh.flush()
    except:
        pass


def _require_hbm(science_bytes: Int, device: String) raises:
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, "/home/ubuntu/mojo-align/gpu-common/python")
    var mem = Python.import_module("gpu_mem")
    _ = mem.require_index_capacity(science_bytes, device=device, overhead=1.15)


def _hit_from_gpu(
    index: LinearIndex,
    name: String,
    align_seq: String,
    original_seq: String,
    cid: Int,
    start0: Int,
    mapq: Int,
    flag: Int,
    soft_l: Int,
    soft_r: Int,
) raises -> LinearHit:
    """Build SAM hit; SEQ is pre-conversion (GATK/Picard), RC if flag 0x10."""
    var qlen = align_seq.byte_length()
    var emit = original_seq
    if emit.byte_length() == 0:
        emit = align_seq
    if cid < 0:
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
        _canonical_contig(index.contig_name(cid)),
        start0 + 1,
        mapq,
        cigar,
        emit,
        "*",
    )


def _sam_with_rg(h: LinearHit) raises -> String:
    return hit_to_sam_line(h) + "\tRG:Z:" + _rg_id()


def map_fastq_dense_gpu_locate(
    mut index: LinearIndex,
    fq1: String,
    out_sam: String,
    device: String,
    fq2: String = "",
    bs_r1: String = "",
    bs_r2: String = "",
) raises -> Int:
    """Full-GPU dense map: locate + vote + gapless on device."""
    if not index.dense:
        raise Error("map_fastq_dense_gpu_locate requires dense-v1 index")
    if (
        index.kmers_addr == 0
        or index.offsets_addr == 0
        or index.postings_addr == 0
        or index.seq_addr == 0
        or index.contig_off_addr == 0
    ):
        raise Error("map_fastq_dense_gpu_locate: null mmap addresses")

    var resolved = select_device(device)
    var backend = probe_device_context(resolved)
    var target = kernel_target_label(resolved)
    print(
        "MojoLinear GPU-full device=",
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
            "MojoLinear GPU-full needs DeviceContext cuda/hip, got " + backend
        )

    comptime if not has_accelerator():
        raise Error("map_fastq_dense_gpu_locate requires accelerator build")
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
                        "MojoLinear GPU-full upload ",
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

        def locate_interp_kernel(
            keys: UnsafePointer[UInt64, MutAnyOrigin],
            kmers: UnsafePointer[UInt64, MutAnyOrigin],
            offsets: UnsafePointer[UInt64, MutAnyOrigin],
            out_start: UnsafePointer[UInt64, MutAnyOrigin],
            out_end: UnsafePointer[UInt64, MutAnyOrigin],
            n_slots: Int,
            n_table: Int,
            max_occ: Int,
        ):
            """Interpolation search on sorted numeric 2-bit keys."""
            var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
            if idx >= n_slots:
                return
            out_start[idx] = 0
            out_end[idx] = 0
            var key = keys[idx]
            if key == UInt64(0xFFFFFFFFFFFFFFFF) or n_table <= 0:
                return
            var lo = 0
            var hi = n_table - 1
            var steps = 0
            while lo <= hi and key >= kmers[lo] and key <= kmers[hi] and steps < 64:
                steps += 1
                if kmers[hi] == kmers[lo]:
                    if kmers[lo] == key:
                        hi = lo
                        break
                    return
                var span = kmers[hi] - kmers[lo]
                var mid = lo + Int(((key - kmers[lo]) * UInt64(hi - lo)) // span)
                if mid < lo:
                    mid = lo
                if mid > hi:
                    mid = hi
                var mk = kmers[mid]
                if mk == key:
                    lo = mid
                    hi = mid
                    break
                if mk < key:
                    lo = mid + 1
                else:
                    if mid == 0:
                        return
                    hi = mid - 1
            if lo > hi or lo >= n_table or kmers[lo] != key:
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
            """Reverse-complement 2-bit codes per read (strand 16 path)."""
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

        def vote_extend_kernel(
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
            strand_flag: Int32,
            out_cid: UnsafePointer[Int32, MutAnyOrigin],
            out_pos: UnsafePointer[UInt32, MutAnyOrigin],
            out_mapq: UnsafePointer[UInt32, MutAnyOrigin],
            out_flag: UnsafePointer[Int32, MutAnyOrigin],
            out_sl: UnsafePointer[UInt32, MutAnyOrigin],
            out_sr: UnsafePointer[UInt32, MutAnyOrigin],
            out_nm: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            """Vote top loci; accept with mismatches + end soft-clips (BWA-ish)."""
            var rid = Int(block_idx.x * block_dim.x + thread_idx.x)
            if rid >= n_reads:
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

            comptime TOP = 128
            comptime KEEP = 8
            var cand_key = InlineArray[UInt64, TOP](fill=UInt64(0xFFFFFFFFFFFFFFFF))
            var cand_n = InlineArray[Int32, TOP](fill=Int32(0))
            var n_cand = 0

            var slot = 0
            while slot < slots_per_read:
                var sidx = rid * slots_per_read + slot
                var a = Int(occ_start[sidx])
                var b = Int(occ_end[sidx])
                var q_off = Int(q_offs[sidx])
                if b > a and a >= 0 and b <= n_postings:
                    var pi = a
                    while pi < b:
                        var cid = Int(postings[pi * 2])
                        var pos = Int(postings[pi * 2 + 1])
                        if cid >= 0 and cid < n_contigs and pos >= q_off:
                            var start0 = pos - q_off
                            var vk = (UInt64(cid) << 32) | UInt64(start0)
                            var found = -1
                            var ci = 0
                            while ci < n_cand:
                                if cand_key[ci] == vk:
                                    found = ci
                                    break
                                ci += 1
                            if found >= 0:
                                cand_n[found] = cand_n[found] + 1
                            elif n_cand < TOP:
                                cand_key[n_cand] = vk
                                cand_n[n_cand] = 1
                                n_cand += 1
                        pi += 1
                slot += 1

            if n_cand == 0:
                return

            # Select up to KEEP highest-vote candidates.
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
                        var shift = n_pick
                        if shift > KEEP - 1:
                            shift = KEEP - 1
                        while shift > p:
                            pick_i[shift] = pick_i[shift - 1]
                            pick_n[shift] = pick_n[shift - 1]
                            shift -= 1
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

            var q_base = rid * max_len
            var budget = max_diff
            if budget < 2:
                budget = 2
            if budget > qlen // 2:
                budget = qlen // 2
            var clip_cap = max_soft
            if clip_cap > qlen // 4:
                clip_cap = qlen // 4

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
                    if bstart + qlen <= clen:
                        var ref_base = off0 + bstart
                        # Mismatch mask along the read.
                        var nm_all = 0
                        var j = 0
                        while j < qlen:
                            var rb = sequences[ref_base + j]
                            var rc: UInt8 = 255
                            if rb == 65 or rb == 97:
                                rc = 0
                            elif rb == 67 or rb == 99:
                                rc = 1
                            elif rb == 71 or rb == 103:
                                rc = 2
                            elif rb == 84 or rb == 116:
                                rc = 3
                            var qc = codes[q_base + j]
                            if rc > 3 or qc > 3 or rc != qc:
                                nm_all += 1
                            j += 1

                        var sl = 0
                        var sr = 0
                        var nm = nm_all
                        if nm_all > budget:
                            # Trim mismatch-heavy ends (soft-clip), re-score middle.
                            while sl < clip_cap:
                                var rb0 = sequences[ref_base + sl]
                                var rc0: UInt8 = 255
                                if rb0 == 65 or rb0 == 97:
                                    rc0 = 0
                                elif rb0 == 67 or rb0 == 99:
                                    rc0 = 1
                                elif rb0 == 71 or rb0 == 103:
                                    rc0 = 2
                                elif rb0 == 84 or rb0 == 116:
                                    rc0 = 3
                                var qc0 = codes[q_base + sl]
                                if rc0 <= 3 and qc0 <= 3 and rc0 == qc0:
                                    break
                                sl += 1
                            while sr < clip_cap:
                                var jr = qlen - 1 - sr
                                if jr <= sl:
                                    break
                                var rb1 = sequences[ref_base + jr]
                                var rc1: UInt8 = 255
                                if rb1 == 65 or rb1 == 97:
                                    rc1 = 0
                                elif rb1 == 67 or rb1 == 99:
                                    rc1 = 1
                                elif rb1 == 71 or rb1 == 103:
                                    rc1 = 2
                                elif rb1 == 84 or rb1 == 116:
                                    rc1 = 3
                                var qc1 = codes[q_base + jr]
                                if rc1 <= 3 and qc1 <= 3 and rc1 == qc1:
                                    break
                                sr += 1
                            nm = 0
                            var jm = sl
                            while jm < qlen - sr:
                                var rbm = sequences[ref_base + jm]
                                var rcm: UInt8 = 255
                                if rbm == 65 or rbm == 97:
                                    rcm = 0
                                elif rbm == 67 or rbm == 99:
                                    rcm = 1
                                elif rbm == 71 or rbm == 103:
                                    rcm = 2
                                elif rbm == 84 or rbm == 116:
                                    rcm = 3
                                var qcm = codes[q_base + jm]
                                if rcm > 3 or qcm > 3 or rcm != qcm:
                                    nm += 1
                                jm += 1

                        var alen = qlen - sl - sr
                        if nm <= budget and alen >= 32:
                            var mq: UInt32 = 20
                            if votes >= 3:
                                mq = 40
                            if votes >= 5:
                                mq = 60
                            if nm == 0 and sl == 0 and sr == 0:
                                mq = mq
                            elif nm > 2:
                                if mq > 20:
                                    mq = 20
                            out_cid[rid] = Int32(bcid)
                            out_pos[rid] = UInt32(bstart + sl)
                            out_mapq[rid] = mq
                            out_flag[rid] = strand_flag
                            out_sl[rid] = UInt32(sl)
                            out_sr[rid] = UInt32(sr)
                            out_nm[rid] = UInt32(nm)
                            return
                pk += 1

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
        if (
            index.kmers_size < kmers_bytes
            or index.offsets_size < offsets_bytes
            or index.postings_size < postings_bytes
            or index.seq_size < seq_bytes
            or index.contig_off_size < coff_bytes
        ):
            raise Error("MojoLinear GPU-full: mmap smaller than expected tables")

        var science = (
            kmers_bytes + offsets_bytes + postings_bytes + seq_bytes + coff_bytes
        )
        _require_hbm(science, resolved)
        print(
            "MojoLinear GPU-full upload science_bytes=",
            science,
            " n_keys=",
            n_table,
            " n_postings=",
            n_post,
            " n_bases=",
            seq_bytes,
        )

        var dev_kmers = ctx.enqueue_create_buffer[DType.uint64](n_table)
        var dev_offsets = ctx.enqueue_create_buffer[DType.uint64](n_off)
        var dev_postings = ctx.enqueue_create_buffer[DType.uint32](n_post * 2)
        var dev_seq = ctx.enqueue_create_buffer[DType.uint8](seq_bytes)
        var dev_coff = ctx.enqueue_create_buffer[DType.uint64](n_contigs + 1)

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
            "sequences",
        )
        upload_mmap_to_device(
            ctx,
            dev_coff.unsafe_ptr().bitcast[UInt8](),
            index.contig_off_addr,
            coff_bytes,
            "contig_offsets",
        )
        print("MojoLinear GPU-full index resident ok")

        var locate_algo = _env_str("METHYLGRAPHER_LINEAR_LOCATE_ALGO", "bsearch").lower()
        var use_interp = locate_algo == "interp" or locate_algo == "interpolation"
        print(
            "MojoLinear GPU-full locate_algo=",
            locate_algo,
            " (bsearch|interp)",
        )

        var fh = open_text_write(out_sam)
        _write_sam_header(fh, index)
        var fh1 = _open_fastq(fq1)
        var paired = fq2.byte_length() > 0
        var fh2 = fh1
        if paired:
            fh2 = _open_fastq(fq2)

        var batch_size = _env_int("METHYLGRAPHER_LINEAR_READ_BATCH", 4096)
        var seed_stride = _env_int("METHYLGRAPHER_LINEAR_SEED_STRIDE", 5)
        var max_occ = index.max_occ()
        # ~4% mismatches + end soft-clip — closes most of the exact-only map gap.
        var max_diff = _env_int("METHYLGRAPHER_LINEAR_MAX_DIFF", 6)
        var max_soft = _env_int("METHYLGRAPHER_LINEAR_MAX_SOFT", 8)
        # Reverse orientation is required even on bwameth f*/r* packs (BWA
        # still sets 0x10 against the converted contig). Set LINEAR_RC=0 only
        # for experiments.
        var rc_raw = _env_str("METHYLGRAPHER_LINEAR_RC", "1").lower()
        var do_rc = not (rc_raw == "0" or rc_raw == "false" or rc_raw == "no")
        var n_mapped = 0
        var n_reads = 0
        var n_batches = 0
        comptime BLOCK = 256

        print(
            "MojoLinear GPU-full map start batch=",
            batch_size,
            " stride=",
            seed_stride,
            " max_occ=",
            max_occ,
            " max_diff=",
            max_diff,
            " max_soft=",
            max_soft,
            " rc=",
            do_rc,
            " paired=",
            paired,
        )

        while True:
            var batch1 = List[FastqRec]()
            var batch2 = List[FastqRec]()
            var i = 0
            while i < batch_size:
                var r1 = _read_one(fh1)
                if r1.name.byte_length() == 0:
                    break
                if bs_r1.byte_length() > 0:
                    r1.seq = _bs_convert(r1.seq, bs_r1)
                batch1.append(r1^)
                if paired:
                    var r2 = _read_one(fh2)
                    if r2.name.byte_length() == 0:
                        raise Error("paired FASTQ length mismatch (R2 ended early)")
                    if bs_r2.byte_length() > 0:
                        r2.seq = _bs_convert(r2.seq, bs_r2)
                    batch2.append(r2^)
                i += 1
            if len(batch1) == 0:
                break

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
            ctx.enqueue_function[pack_bases_kernel](
                dev_bases.unsafe_ptr(),
                dev_codes.unsafe_ptr(),
                n_bases,
                grid_dim=grid_b,
                block_dim=BLOCK,
            )
            var grid_s = (n_slots + BLOCK - 1) // BLOCK
            var grid_r = (n_seq + BLOCK - 1) // BLOCK

            # Forward strand
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
            if use_interp:
                ctx.enqueue_function[locate_interp_kernel](
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
            else:
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
            ctx.enqueue_function[vote_extend_kernel](
                dev_start.unsafe_ptr(),
                dev_end.unsafe_ptr(),
                dev_qoff.unsafe_ptr(),
                dev_postings.unsafe_ptr(),
                n_post,
                dev_seq.unsafe_ptr(),
                dev_coff.unsafe_ptr(),
                n_contigs,
                dev_codes.unsafe_ptr(),
                dev_lens.unsafe_ptr(),
                max_len,
                slots_per,
                n_seq,
                max_diff,
                max_soft,
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

            if do_rc:
                # Reverse strand (RC codes → locate → extend)
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
                if use_interp:
                    ctx.enqueue_function[locate_interp_kernel](
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
                else:
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
                ctx.enqueue_function[vote_extend_kernel](
                    dev_start.unsafe_ptr(),
                    dev_end.unsafe_ptr(),
                    dev_qoff.unsafe_ptr(),
                    dev_postings.unsafe_ptr(),
                    n_post,
                    dev_seq.unsafe_ptr(),
                    dev_coff.unsafe_ptr(),
                    n_contigs,
                    dev_rc.unsafe_ptr(),
                    dev_lens.unsafe_ptr(),
                    max_len,
                    slots_per,
                    n_seq,
                    max_diff,
                    max_soft,
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
            else:
                # Mark RC outputs unmapped so host merge keeps FW hits.
                var host_cid_r_init = ctx.enqueue_create_host_buffer[DType.int32](
                    n_seq
                )
                var zi = 0
                while zi < n_seq:
                    host_cid_r_init[zi] = Int32(-1)
                    zi += 1
                ctx.enqueue_copy(src_buf=host_cid_r_init, dst_buf=dev_cid_r)

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
            var j = 0
            while j < n1:
                # Prefer lower NM, then higher mapq; FW vs RC.
                var use_rc = False
                var f_cid = Int(host_cid_f[j])
                var r_cid = Int(host_cid_r[j])
                if f_cid < 0 and r_cid >= 0:
                    use_rc = True
                elif f_cid >= 0 and r_cid >= 0:
                    var fnm = Int(host_nm_f[j])
                    var rnm = Int(host_nm_r[j])
                    if rnm < fnm or (
                        rnm == fnm and Int(host_mapq_r[j]) > Int(host_mapq_f[j])
                    ):
                        use_rc = True
                var h1: LinearHit
                if use_rc:
                    h1 = _hit_from_gpu(
                        index,
                        batch1[j].name,
                        batch1[j].seq,
                        batch1[j].original_seq,
                        r_cid,
                        Int(host_pos_r[j]),
                        Int(host_mapq_r[j]),
                        Int(host_flag_r[j]),
                        Int(host_sl_r[j]),
                        Int(host_sr_r[j]),
                    )
                else:
                    h1 = _hit_from_gpu(
                        index,
                        batch1[j].name,
                        batch1[j].seq,
                        batch1[j].original_seq,
                        f_cid,
                        Int(host_pos_f[j]),
                        Int(host_mapq_f[j]),
                        Int(host_flag_f[j]),
                        Int(host_sl_f[j]),
                        Int(host_sr_f[j]),
                    )
                h1.qual = batch1[j].qual
                if not paired:
                    if h1.contig != "*":
                        n_mapped += 1
                    fh.write(_sam_with_rg(h1) + "\n")
                    n_reads += 1
                else:
                    var r2i = n1 + j
                    var use_rc2 = False
                    var f2 = Int(host_cid_f[r2i])
                    var r2 = Int(host_cid_r[r2i])
                    if f2 < 0 and r2 >= 0:
                        use_rc2 = True
                    elif f2 >= 0 and r2 >= 0:
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
                            batch2[j].name,
                            batch2[j].seq,
                            batch2[j].original_seq,
                            r2,
                            Int(host_pos_r[r2i]),
                            Int(host_mapq_r[r2i]),
                            Int(host_flag_r[r2i]),
                            Int(host_sl_r[r2i]),
                            Int(host_sr_r[r2i]),
                        )
                    else:
                        h2 = _hit_from_gpu(
                            index,
                            batch2[j].name,
                            batch2[j].seq,
                            batch2[j].original_seq,
                            f2,
                            Int(host_pos_f[r2i]),
                            Int(host_mapq_f[r2i]),
                            Int(host_flag_f[r2i]),
                            Int(host_sl_f[r2i]),
                            Int(host_sr_f[r2i]),
                        )
                    h2.qual = batch2[j].qual
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
                j += 1

            n_batches += 1
            if n_batches == 1 or n_batches % 5 == 0:
                print(
                    "MojoLinear GPU-full progress batches=",
                    n_batches,
                    " reads=",
                    n_reads,
                    " mapped=",
                    n_mapped,
                )
                try:
                    fh.flush()
                except:
                    pass

        fh1.close()
        if paired:
            fh2.close()
        fh.close()
        print(
            "wrote SAM -> ",
            out_sam,
            " mapped_records=",
            n_mapped,
            " reads=",
            n_reads,
            " backend=gpu-full",
        )
        return n_mapped
