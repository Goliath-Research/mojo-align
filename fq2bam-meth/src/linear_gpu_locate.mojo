# Full-GPU dense-v1 linear WGBS map (DeviceContext: NVIDIA cuda / AMD hip).
#
# Resident on device: kmers + offsets + postings + sequences + contig_offsets.
# Per batch: 2-bit encode → locate → vote → mismatch/softclip/indel extend.
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


def _read_pe_batch(
    fh1: PythonObject,
    fh2: PythonObject,
    paired: Bool,
    batch_size: Int,
    bs_r1: String,
    bs_r2: String,
    mut batch1: List[FastqRec],
    mut batch2: List[FastqRec],
) raises:
    """Read up to batch_size PE (or SE) records into batch1/batch2. Empty ⇒ EOF."""
    batch1 = List[FastqRec]()
    batch2 = List[FastqRec]()
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
    indel_op: Int = 0,
    indel_at: Int = 0,
    indel_len: Int = 0,
) raises -> LinearHit:
    """Build SAM hit; SEQ is pre-conversion (GATK/Picard), RC if flag 0x10.

    indel_op: 0=none, 1=insertion (I), 2=deletion (D). indel_at is query
    bases matched before the indel within the non-softclipped middle.
    """
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
    if indel_op == 1 and indel_len > 0 and indel_at >= 0 and indel_at <= m:
        # Insertion consumes query bases only.
        var left = indel_at
        var right = m - indel_at - indel_len
        if right < 0:
            return LinearHit(name, 4, "*", 0, 0, "*", emit, "*")
        if left > 0:
            cigar = cigar + String(left) + "M"
        cigar = cigar + String(indel_len) + "I"
        if right > 0:
            cigar = cigar + String(right) + "M"
    elif indel_op == 2 and indel_len > 0 and indel_at >= 0 and indel_at <= m:
        # Deletion consumes reference only; middle query length stays m.
        var left_d = indel_at
        var right_d = m - indel_at
        if left_d > 0:
            cigar = cigar + String(left_d) + "M"
        cigar = cigar + String(indel_len) + "D"
        if right_d > 0:
            cigar = cigar + String(right_d) + "M"
    else:
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
            max_indel: Int,
            vote_occ: Int,
            # 0 = fast (vote + low-occ rescue); 1 = high-occ exact for unmapped only.
            pass_mode: Int,
            # When n_active >= 0, threads map through rid_map (compacted unmapped).
            # When n_active < 0, thread i handles read i (full batch).
            n_active: Int,
            rid_map: UnsafePointer[Int32, MutAnyOrigin],
            strand_flag: Int32,
            out_cid: UnsafePointer[Int32, MutAnyOrigin],
            out_pos: UnsafePointer[UInt32, MutAnyOrigin],
            out_mapq: UnsafePointer[UInt32, MutAnyOrigin],
            out_flag: UnsafePointer[Int32, MutAnyOrigin],
            out_sl: UnsafePointer[UInt32, MutAnyOrigin],
            out_sr: UnsafePointer[UInt32, MutAnyOrigin],
            out_nm: UnsafePointer[UInt32, MutAnyOrigin],
            out_iop: UnsafePointer[UInt32, MutAnyOrigin],
            out_iat: UnsafePointer[UInt32, MutAnyOrigin],
            out_ilen: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            """Two-pass extend: fast vote/rescue, then compacted high-occ for unmapped."""
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
            # Pass 1 safety: leave already-mapped reads alone.
            if pass_mode == 1 and Int(out_cid[rid]) >= 0:
                return
            out_cid[rid] = Int32(-1)
            out_pos[rid] = 0
            out_mapq[rid] = 0
            out_flag[rid] = Int32(4)
            out_sl[rid] = 0
            out_sr[rid] = 0
            out_nm[rid] = 0
            out_iop[rid] = 0
            out_iat[rid] = 0
            out_ilen[rid] = 0
            var qlen = Int(read_lens[rid])
            if qlen <= 0:
                return

            comptime TOP = 256
            comptime KEEP = 16
            var cand_key = InlineArray[UInt64, TOP](fill=UInt64(0xFFFFFFFFFFFFFFFF))
            var cand_n = InlineArray[Int32, TOP](fill=Int32(0))
            var n_cand = 0

            # Vote using seeds sorted by increasing occupancy (rare first).
            # Dual C2T packs are highly repetitive — processing common seeds
            # first fills TOP with decoys and drops the true locus.
            comptime MAX_SLOTS = 128
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
                    # Insert by ascending occ.
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

            # Cap used for majority vote (locate may return higher-occ seeds
            # for rescue). Default vote_occ << locate max_occ.
            var vote_cap = vote_occ
            if vote_cap < 1:
                vote_cap = 256

            # Pass 0: rare-seed vote. Pass 1: skip (high-occ path only).
            if pass_mode == 0:
                var oi = 0
                while oi < n_ord:
                    var occ_v = Int(ord_occ[oi])
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
                        pi += 1
                    oi += 1

            # Select up to KEEP highest-vote candidates.
            var pick_i = InlineArray[Int32, KEEP](fill=Int32(-1))
            var pick_n = InlineArray[Int32, KEEP](fill=Int32(0))
            var n_pick = 0
            var ci2 = 0
            while pass_mode == 0 and ci2 < n_cand:
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
            var indel_cap = max_indel
            if indel_cap < 0:
                indel_cap = 0
            if indel_cap > 4:
                indel_cap = 4
            var shift_cap = indel_cap
            if shift_cap < 2:
                shift_cap = 2
            if shift_cap > 6:
                shift_cap = 6

            # Best accepted among candidates (lowest edit cost, then votes).
            var best_cost = 9999
            var best_votes: Int32 = 0
            var best_cid = -1
            var best_pos = 0
            var best_sl = 0
            var best_sr = 0
            var best_iop = 0
            var best_iat = 0
            var best_ilen = 0

            var pk = 0
            while pk < n_pick and n_cand > 0:
                var ix = Int(pick_i[pk])
                var bkey = cand_key[ix]
                var votes = pick_n[pk]
                var bcid = Int(bkey >> 32)
                var bstart0 = Int(bkey & UInt64(0xFFFFFFFF))
                if bcid >= 0 and bcid < n_contigs and bstart0 >= 0:
                    var off0 = Int(contig_off[bcid])
                    var off1 = Int(contig_off[bcid + 1])
                    var clen = off1 - off0

                    var adj = -shift_cap
                    while adj <= shift_cap:
                        var bstart = bstart0 + adj
                        if bstart < 0 or bstart + qlen > clen:
                            adj += 1
                            continue
                        var ref_base = off0 + bstart

                        # Gapless + end soft-clip.
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
                            var cost = nm
                            if (
                                cost < best_cost
                                or (cost == best_cost and votes > best_votes)
                            ):
                                best_cost = cost
                                best_votes = votes
                                best_cid = bcid
                                best_pos = bstart + sl
                                best_sl = sl
                                best_sr = sr
                                best_iop = 0
                                best_iat = 0
                                best_ilen = 0

                        # Single indel only when this start fails gapless/softclip.
                        # Restrict to adj==0 (voted locus) to keep kernel latency sane.
                        if (
                            indel_cap > 0
                            and adj == 0
                            and (nm > budget or alen < 32)
                        ):
                            var dlen = 1
                            while dlen <= indel_cap:
                                # Deletion of dlen ref bases after query offset g.
                                if bstart + qlen + dlen <= clen:
                                    var g = 0
                                    while g <= qlen:
                                        var nm_d = dlen
                                        var t = 0
                                        while t < g and nm_d <= budget:
                                            var rbd = sequences[ref_base + t]
                                            var rcd: UInt8 = 255
                                            if rbd == 65 or rbd == 97:
                                                rcd = 0
                                            elif rbd == 67 or rbd == 99:
                                                rcd = 1
                                            elif rbd == 71 or rbd == 103:
                                                rcd = 2
                                            elif rbd == 84 or rbd == 116:
                                                rcd = 3
                                            var qcd = codes[q_base + t]
                                            if rcd > 3 or qcd > 3 or rcd != qcd:
                                                nm_d += 1
                                            t += 1
                                        while t < qlen and nm_d <= budget:
                                            var rbd2 = sequences[
                                                ref_base + t + dlen
                                            ]
                                            var rcd2: UInt8 = 255
                                            if rbd2 == 65 or rbd2 == 97:
                                                rcd2 = 0
                                            elif rbd2 == 67 or rbd2 == 99:
                                                rcd2 = 1
                                            elif rbd2 == 71 or rbd2 == 103:
                                                rcd2 = 2
                                            elif rbd2 == 84 or rbd2 == 116:
                                                rcd2 = 3
                                            var qcd2 = codes[q_base + t]
                                            if (
                                                rcd2 > 3
                                                or qcd2 > 3
                                                or rcd2 != qcd2
                                            ):
                                                nm_d += 1
                                            t += 1
                                        if nm_d <= budget and (
                                            nm_d < best_cost
                                            or (
                                                nm_d == best_cost
                                                and votes > best_votes
                                            )
                                        ):
                                            best_cost = nm_d
                                            best_votes = votes
                                            best_cid = bcid
                                            best_pos = bstart
                                            best_sl = 0
                                            best_sr = 0
                                            best_iop = 2
                                            best_iat = g
                                            best_ilen = dlen
                                        g += 1

                                # Insertion of dlen query bases after offset g.
                                if qlen > dlen and bstart + (qlen - dlen) <= clen:
                                    var gi = 0
                                    while gi <= qlen - dlen:
                                        var nm_i = dlen
                                        var ti = 0
                                        while ti < gi and nm_i <= budget:
                                            var rbi = sequences[ref_base + ti]
                                            var rci: UInt8 = 255
                                            if rbi == 65 or rbi == 97:
                                                rci = 0
                                            elif rbi == 67 or rbi == 99:
                                                rci = 1
                                            elif rbi == 71 or rbi == 103:
                                                rci = 2
                                            elif rbi == 84 or rbi == 116:
                                                rci = 3
                                            var qci = codes[q_base + ti]
                                            if rci > 3 or qci > 3 or rci != qci:
                                                nm_i += 1
                                            ti += 1
                                        var tj = gi + dlen
                                        var rj = gi
                                        while tj < qlen and nm_i <= budget:
                                            var rbi2 = sequences[ref_base + rj]
                                            var rci2: UInt8 = 255
                                            if rbi2 == 65 or rbi2 == 97:
                                                rci2 = 0
                                            elif rbi2 == 67 or rbi2 == 99:
                                                rci2 = 1
                                            elif rbi2 == 71 or rbi2 == 103:
                                                rci2 = 2
                                            elif rbi2 == 84 or rbi2 == 116:
                                                rci2 = 3
                                            var qci2 = codes[q_base + tj]
                                            if (
                                                rci2 > 3
                                                or qci2 > 3
                                                or rci2 != qci2
                                            ):
                                                nm_i += 1
                                            tj += 1
                                            rj += 1
                                        if nm_i <= budget and (
                                            nm_i < best_cost
                                            or (
                                                nm_i == best_cost
                                                and votes > best_votes
                                            )
                                        ):
                                            best_cost = nm_i
                                            best_votes = votes
                                            best_cid = bcid
                                            best_pos = bstart
                                            best_sl = 0
                                            best_sr = 0
                                            best_iop = 1
                                            best_iat = gi
                                            best_ilen = dlen
                                        gi += 1
                                dlen += 1

                        # Exact hit — stop searching further adj / candidates.
                        if best_cost == 0:
                            break
                        adj += 1

                    if best_cost == 0:
                        break
                pk += 1

            # Seed-rescue from rarest seeds.
            # Pass 0: low-occ gapless/softclip/indel only.
            # Pass 1: high-occ exact tally for still-unmapped reads.
            if pass_mode == 1 or best_cid < 0 or best_cost > 0:
                comptime HX = 48
                var hx_key = InlineArray[UInt64, HX](fill=UInt64(0xFFFFFFFFFFFFFFFF))
                var hx_n = InlineArray[Int32, HX](fill=Int32(0))
                var hx_min_occ = InlineArray[Int32, HX](fill=Int32(0x7FFFFFFF))
                var n_hx = 0
                var ri = 0
                var n_high_tried = 0
                while ri < n_ord:
                    var occ_r = Int(ord_occ[ri])
                    if occ_r <= 0:
                        ri += 1
                        continue
                    var high_occ = occ_r > vote_cap
                    # Pass split: skip the expensive path that belongs to the other pass.
                    if high_occ and pass_mode == 0:
                        ri += 1
                        continue
                    if (not high_occ) and pass_mode == 1:
                        ri += 1
                        continue
                    if high_occ:
                        # Try all high-occ seeds (sorted rare→common). Early
                        # exit once a locus has a clear multi-seed plurality.
                        n_high_tried += 1
                        if n_high_tried > 64:
                            ri += 1
                            continue
                    var slot_r = Int(ord_slot[ri])
                    var sidx_r = rid * slots_per_read + slot_r
                    var a_r = Int(occ_start[sidx_r])
                    var b_r = Int(occ_end[sidx_r])
                    var q_off_r = Int(q_offs[sidx_r])

                    if high_occ:
                        var pi_h = a_r
                        while pi_h < b_r:
                            var cid_h = Int(postings[pi_h * 2])
                            var pos_h = Int(postings[pi_h * 2 + 1])
                            if (
                                cid_h >= 0
                                and cid_h < n_contigs
                                and pos_h >= q_off_r
                            ):
                                var bs_h = pos_h - q_off_r
                                var o0_h = Int(contig_off[cid_h])
                                var o1_h = Int(contig_off[cid_h + 1])
                                if bs_h >= 0 and bs_h + qlen <= (o1_h - o0_h):
                                    var ref_h = o0_h + bs_h
                                    var nm_h = 0
                                    var jh = 0
                                    while jh < qlen and nm_h == 0:
                                        var rbh = sequences[ref_h + jh]
                                        var rch: UInt8 = 255
                                        if rbh == 65 or rbh == 97:
                                            rch = 0
                                        elif rbh == 67 or rbh == 99:
                                            rch = 1
                                        elif rbh == 71 or rbh == 103:
                                            rch = 2
                                        elif rbh == 84 or rbh == 116:
                                            rch = 3
                                        var qch = codes[q_base + jh]
                                        if rch > 3 or qch > 3 or rch != qch:
                                            nm_h = 1
                                        jh += 1
                                    if nm_h == 0:
                                        var vk_h = (UInt64(cid_h) << 32) | UInt64(
                                            bs_h
                                        )
                                        var found_h = -1
                                        var ci_h = 0
                                        while ci_h < n_hx:
                                            if hx_key[ci_h] == vk_h:
                                                found_h = ci_h
                                                break
                                            ci_h += 1
                                        if found_h >= 0:
                                            hx_n[found_h] = hx_n[found_h] + 1
                                            if Int32(occ_r) < hx_min_occ[found_h]:
                                                hx_min_occ[found_h] = Int32(occ_r)
                                        elif n_hx < HX:
                                            hx_key[n_hx] = vk_h
                                            hx_n[n_hx] = 1
                                            hx_min_occ[n_hx] = Int32(occ_r)
                                            n_hx += 1
                            pi_h += 1
                        # Early-stop high-occ scan once plurality is clear.
                        if n_hx > 0:
                            var b1 = 0
                            var b2 = 0
                            var zi = 0
                            while zi < n_hx:
                                var zv = Int(hx_n[zi])
                                if zv > b1:
                                    b2 = b1
                                    b1 = zv
                                elif zv > b2:
                                    b2 = zv
                                zi += 1
                            if b1 >= 3 and b1 > b2:
                                # Skip remaining high-occ seeds.
                                while ri + 1 < n_ord and Int(ord_occ[ri + 1]) > vote_cap:
                                    ri += 1
                        ri += 1
                        continue

                    # Low-occ rescue: gapless → softclip → single-indel.
                    var pi_r = a_r
                    while pi_r < b_r:
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
                            var clen_r = off1_r - off0_r
                            if bstart_r >= 0 and bstart_r + qlen <= clen_r:
                                var ref_r = off0_r + bstart_r
                                var nm_r = 0
                                var jr = 0
                                while jr < qlen and nm_r <= budget:
                                    var rb = sequences[ref_r + jr]
                                    var rc: UInt8 = 255
                                    if rb == 65 or rb == 97:
                                        rc = 0
                                    elif rb == 67 or rb == 99:
                                        rc = 1
                                    elif rb == 71 or rb == 103:
                                        rc = 2
                                    elif rb == 84 or rb == 116:
                                        rc = 3
                                    var qc = codes[q_base + jr]
                                    if rc > 3 or qc > 3 or rc != qc:
                                        nm_r += 1
                                    jr += 1

                                var sl_r = 0
                                var sr_r = 0
                                var cost_r = nm_r
                                var iop_r = 0
                                var iat_r = 0
                                var ilen_r = 0
                                var pos_out = bstart_r

                                if nm_r > budget and clip_cap > 0:
                                    while sl_r < clip_cap:
                                        var rb0 = sequences[ref_r + sl_r]
                                        var rc0: UInt8 = 255
                                        if rb0 == 65 or rb0 == 97:
                                            rc0 = 0
                                        elif rb0 == 67 or rb0 == 99:
                                            rc0 = 1
                                        elif rb0 == 71 or rb0 == 103:
                                            rc0 = 2
                                        elif rb0 == 84 or rb0 == 116:
                                            rc0 = 3
                                        var qc0 = codes[q_base + sl_r]
                                        if rc0 <= 3 and qc0 <= 3 and rc0 == qc0:
                                            break
                                        sl_r += 1
                                    while sr_r < clip_cap:
                                        var jrr = qlen - 1 - sr_r
                                        if jrr <= sl_r:
                                            break
                                        var rb1 = sequences[ref_r + jrr]
                                        var rc1: UInt8 = 255
                                        if rb1 == 65 or rb1 == 97:
                                            rc1 = 0
                                        elif rb1 == 67 or rb1 == 99:
                                            rc1 = 1
                                        elif rb1 == 71 or rb1 == 103:
                                            rc1 = 2
                                        elif rb1 == 84 or rb1 == 116:
                                            rc1 = 3
                                        var qc1 = codes[q_base + jrr]
                                        if rc1 <= 3 and qc1 <= 3 and rc1 == qc1:
                                            break
                                        sr_r += 1
                                    cost_r = 0
                                    var jm = sl_r
                                    while jm < qlen - sr_r:
                                        var rbm = sequences[ref_r + jm]
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
                                            cost_r += 1
                                        jm += 1
                                    pos_out = bstart_r + sl_r
                                    if qlen - sl_r - sr_r < 32:
                                        cost_r = 9999

                                # Single indel at seed locus (size 1..indel_cap).
                                if indel_cap > 0 and cost_r > budget:
                                    var dlen = 1
                                    while dlen <= indel_cap:
                                        if bstart_r + qlen + dlen <= clen_r:
                                            var g = q_off_r
                                            if g > qlen:
                                                g = qlen
                                            # Prefer indel near the seed offset.
                                            var g0 = g - 8
                                            if g0 < 0:
                                                g0 = 0
                                            var g1 = g + 8
                                            if g1 > qlen:
                                                g1 = qlen
                                            var gg = g0
                                            while gg <= g1:
                                                var nm_d = dlen
                                                var t = 0
                                                while t < gg and nm_d <= budget:
                                                    var rbd = sequences[
                                                        ref_r + t
                                                    ]
                                                    var rcd: UInt8 = 255
                                                    if rbd == 65 or rbd == 97:
                                                        rcd = 0
                                                    elif rbd == 67 or rbd == 99:
                                                        rcd = 1
                                                    elif (
                                                        rbd == 71 or rbd == 103
                                                    ):
                                                        rcd = 2
                                                    elif (
                                                        rbd == 84 or rbd == 116
                                                    ):
                                                        rcd = 3
                                                    var qcd = codes[q_base + t]
                                                    if (
                                                        rcd > 3
                                                        or qcd > 3
                                                        or rcd != qcd
                                                    ):
                                                        nm_d += 1
                                                    t += 1
                                                while t < qlen and nm_d <= budget:
                                                    var rbd2 = sequences[
                                                        ref_r + t + dlen
                                                    ]
                                                    var rcd2: UInt8 = 255
                                                    if rbd2 == 65 or rbd2 == 97:
                                                        rcd2 = 0
                                                    elif (
                                                        rbd2 == 67 or rbd2 == 99
                                                    ):
                                                        rcd2 = 1
                                                    elif (
                                                        rbd2 == 71
                                                        or rbd2 == 103
                                                    ):
                                                        rcd2 = 2
                                                    elif (
                                                        rbd2 == 84
                                                        or rbd2 == 116
                                                    ):
                                                        rcd2 = 3
                                                    var qcd2 = codes[q_base + t]
                                                    if (
                                                        rcd2 > 3
                                                        or qcd2 > 3
                                                        or rcd2 != qcd2
                                                    ):
                                                        nm_d += 1
                                                    t += 1
                                                if nm_d < cost_r:
                                                    cost_r = nm_d
                                                    sl_r = 0
                                                    sr_r = 0
                                                    iop_r = 2
                                                    iat_r = gg
                                                    ilen_r = dlen
                                                    pos_out = bstart_r
                                                gg += 1
                                        if (
                                            qlen > dlen
                                            and bstart_r + (qlen - dlen)
                                            <= clen_r
                                        ):
                                            var g2 = q_off_r
                                            if g2 > qlen - dlen:
                                                g2 = qlen - dlen
                                            var g0i = g2 - 8
                                            if g0i < 0:
                                                g0i = 0
                                            var g1i = g2 + 8
                                            if g1i > qlen - dlen:
                                                g1i = qlen - dlen
                                            var gi = g0i
                                            while gi <= g1i:
                                                var nm_i = dlen
                                                var ti = 0
                                                while ti < gi and nm_i <= budget:
                                                    var rbi = sequences[
                                                        ref_r + ti
                                                    ]
                                                    var rci: UInt8 = 255
                                                    if rbi == 65 or rbi == 97:
                                                        rci = 0
                                                    elif rbi == 67 or rbi == 99:
                                                        rci = 1
                                                    elif (
                                                        rbi == 71 or rbi == 103
                                                    ):
                                                        rci = 2
                                                    elif (
                                                        rbi == 84 or rbi == 116
                                                    ):
                                                        rci = 3
                                                    var qci = codes[q_base + ti]
                                                    if (
                                                        rci > 3
                                                        or qci > 3
                                                        or rci != qci
                                                    ):
                                                        nm_i += 1
                                                    ti += 1
                                                var tj = gi + dlen
                                                var rj = gi
                                                while (
                                                    tj < qlen and nm_i <= budget
                                                ):
                                                    var rbi2 = sequences[
                                                        ref_r + rj
                                                    ]
                                                    var rci2: UInt8 = 255
                                                    if rbi2 == 65 or rbi2 == 97:
                                                        rci2 = 0
                                                    elif (
                                                        rbi2 == 67 or rbi2 == 99
                                                    ):
                                                        rci2 = 1
                                                    elif (
                                                        rbi2 == 71
                                                        or rbi2 == 103
                                                    ):
                                                        rci2 = 2
                                                    elif (
                                                        rbi2 == 84
                                                        or rbi2 == 116
                                                    ):
                                                        rci2 = 3
                                                    var qci2 = codes[
                                                        q_base + tj
                                                    ]
                                                    if (
                                                        rci2 > 3
                                                        or qci2 > 3
                                                        or rci2 != qci2
                                                    ):
                                                        nm_i += 1
                                                    tj += 1
                                                    rj += 1
                                                if nm_i < cost_r:
                                                    cost_r = nm_i
                                                    sl_r = 0
                                                    sr_r = 0
                                                    iop_r = 1
                                                    iat_r = gi
                                                    ilen_r = dlen
                                                    pos_out = bstart_r
                                                gi += 1
                                        dlen += 1

                                if cost_r <= budget and (
                                    cost_r < best_cost or best_cid < 0
                                ):
                                    best_cost = cost_r
                                    best_votes = Int32(4)
                                    if occ_r <= 64:
                                        best_votes = 5
                                    best_cid = cid_r
                                    best_pos = pos_out
                                    best_sl = sl_r
                                    best_sr = sr_r
                                    best_iop = iop_r
                                    best_iat = iat_r
                                    best_ilen = ilen_r
                                    if cost_r == 0 and iop_r == 0:
                                        pi_r = b_r
                                        ri = n_ord
                        pi_r += 1
                    if best_cost == 0 and best_iop == 0 and best_cid >= 0:
                        break
                    ri += 1

                # Resolve high-occ exact tally: need ≥2 seeds and a plurality.
                # Tie-break by rarer seed evidence (lower min occ) so multi-copy
                # exact repeats prefer the locus supported by rarer kmers.
                if n_hx > 0:
                    var best_hi = 0
                    var second_hi = 0
                    var best_hx_i = -1
                    var best_mocc = 0x7FFFFFFF
                    var hi = 0
                    while hi < n_hx:
                        var vn = Int(hx_n[hi])
                        var mo = Int(hx_min_occ[hi])
                        if vn > best_hi:
                            second_hi = best_hi
                            best_hi = vn
                            best_mocc = mo
                            best_hx_i = hi
                        elif vn == best_hi and mo < best_mocc:
                            best_mocc = mo
                            best_hx_i = hi
                        elif vn > second_hi:
                            second_hi = vn
                        hi += 1
                    # Accept if ≥2 seeds agree, or a single exact locus is
                    # unique in the tally (second_hi==0). Ties map with low MAPQ.
                    if (
                        best_hx_i >= 0
                        and (
                            best_hi >= 2
                            or (best_hi >= 1 and second_hi == 0)
                        )
                        and (
                            best_cid < 0
                            or best_cost > 0
                            or Int(best_votes) < best_hi
                        )
                    ):
                        var bk = hx_key[best_hx_i]
                        best_cost = 0
                        if best_hi > second_hi:
                            best_votes = Int32(best_hi)
                            if best_votes > 5:
                                best_votes = 5
                        else:
                            best_votes = 1
                        best_cid = Int(bk >> 32)
                        best_pos = Int(bk & UInt64(0xFFFFFFFF))
                        best_sl = 0
                        best_sr = 0
                        best_iop = 0
                        best_iat = 0
                        best_ilen = 0

            if best_cid >= 0:
                var mq: UInt32 = 20
                if best_votes >= 3:
                    mq = 40
                if best_votes >= 5:
                    mq = 60
                if best_cost > 2:
                    if mq > 20:
                        mq = 20
                if best_iop != 0 and mq > 20:
                    mq = 20
                out_cid[rid] = Int32(best_cid)
                out_pos[rid] = UInt32(best_pos)
                out_mapq[rid] = mq
                out_flag[rid] = strand_flag
                out_sl[rid] = UInt32(best_sl)
                out_sr[rid] = UInt32(best_sr)
                out_nm[rid] = UInt32(best_cost)
                out_iop[rid] = UInt32(best_iop)
                out_iat[rid] = UInt32(best_iat)
                out_ilen[rid] = UInt32(best_ilen)

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
        var t_upload1 = time_mod.perf_counter()
        print(
            "MojoLinear GPU-full upload_wall_s=",
            t_upload1 - t_upload0,
        )

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

        # Large batches amortize kernel launch + host sync on high-HBM GPUs
        # (GH200-class). Per-batch working set is tens–hundreds of MiB vs ~50 GiB index.
        var batch_size = _env_int("METHYLGRAPHER_LINEAR_READ_BATCH", 16384)
        var seed_stride = _env_int("METHYLGRAPHER_LINEAR_SEED_STRIDE", 5)
        var max_occ = index.max_occ()
        # Majority-vote only uses seeds at or below vote_occ; higher-occ seeds
        # from locate are exact-only rescue candidates.
        var vote_occ = _env_int("METHYLGRAPHER_LINEAR_VOTE_OCC", 256)
        # ~4% mismatches + end soft-clip — closes most of the exact-only map gap.
        var max_diff = _env_int("METHYLGRAPHER_LINEAR_MAX_DIFF", 6)
        var max_soft = _env_int("METHYLGRAPHER_LINEAR_MAX_SOFT", 8)
        # Single-indel gapped extend (1..N bp I/D) after gapless/softclip.
        var max_indel = _env_int("METHYLGRAPHER_LINEAR_MAX_INDEL", 4)
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
            " vote_occ=",
            vote_occ,
            " max_diff=",
            max_diff,
            " max_soft=",
            max_soft,
            " max_indel=",
            max_indel,
            " rc=",
            do_rc,
            " paired=",
            paired,
            " prefetch_overlap=1",
        )
        var t_map0 = time_mod.perf_counter()

        # Prime first batch on host; later batches are read+packed while the
        # previous batch's pass-0 GPU kernels run (science-identical).
        var batch1 = List[FastqRec]()
        var batch2 = List[FastqRec]()
        _read_pe_batch(
            fh1, fh2, paired, batch_size, bs_r1, bs_r2, batch1, batch2
        )
        # Dummy host buffers; replaced before first H2D when batch non-empty.
        var host_bases = ctx.enqueue_create_host_buffer[DType.uint8](1)
        var host_lens = ctx.enqueue_create_host_buffer[DType.uint32](1)
        var n_seq = 0
        var max_len = 0
        var slots_per = 1
        var n_slots = 0
        var n_bases = 0
        var need_pack = True

        while len(batch1) > 0:
            if need_pack:
                var seqs = List[String]()
                for r in batch1:
                    seqs.append(r.seq)
                if paired:
                    for r in batch2:
                        seqs.append(r.seq)
                n_seq = len(seqs)
                max_len = 0
                var ri0 = 0
                while ri0 < n_seq:
                    var L0 = seqs[ri0].byte_length()
                    if L0 > max_len:
                        max_len = L0
                    ri0 += 1
                if max_len < k_len:
                    max_len = k_len
                slots_per = 1
                if max_len >= k_len:
                    slots_per = ((max_len - k_len) // seed_stride) + 1
                n_slots = n_seq * slots_per
                n_bases = n_seq * max_len
                host_bases = ctx.enqueue_create_host_buffer[DType.uint8](n_bases)
                host_lens = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
                var si = 0
                while si < n_seq:
                    var s = seqs[si]
                    var L = s.byte_length()
                    host_lens[si] = UInt32(L)
                    var base = si * max_len
                    var p = 0
                    while p < max_len:
                        if p < L:
                            host_bases[base + p] = UInt8(
                                ord(s[byte = p : p + 1])
                            )
                        else:
                            host_bases[base + p] = 78
                        p += 1
                    si += 1

            if n_batches == 0:
                # bases+codes+rc + keys/qoff/start/end + ~20×i32 hit fields ×2 strands
                var workset = (
                    n_bases * 3
                    + n_slots * (8 + 4 + 8 + 8)
                    + n_seq * 4 * 20
                )
                print(
                    "MojoLinear GPU-full batch_workset_bytes≈",
                    workset,
                    " n_seq=",
                    n_seq,
                    " n_slots=",
                    n_slots,
                )

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
            var dev_iop_f = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_iat_f = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_ilen_f = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_cid_r = ctx.enqueue_create_buffer[DType.int32](n_seq)
            var dev_pos_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_mapq_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_flag_r = ctx.enqueue_create_buffer[DType.int32](n_seq)
            var dev_sl_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_sr_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_nm_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_iop_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_iat_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_ilen_r = ctx.enqueue_create_buffer[DType.uint32](n_seq)

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
            # Dummy rid_map for full-batch (n_active < 0) launches.
            var host_rid_dummy = ctx.enqueue_create_host_buffer[DType.int32](1)
            host_rid_dummy[0] = Int32(0)
            var dev_rid_dummy = ctx.enqueue_create_buffer[DType.int32](1)
            ctx.enqueue_copy(src_buf=host_rid_dummy, dst_buf=dev_rid_dummy)

            # Pass 0 FW: vote + low-occ rescue (fast path, full batch).
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
                max_indel,
                vote_occ,
                0,
                -1,
                dev_rid_dummy.unsafe_ptr(),
                Int32(0),
                dev_cid_f.unsafe_ptr(),
                dev_pos_f.unsafe_ptr(),
                dev_mapq_f.unsafe_ptr(),
                dev_flag_f.unsafe_ptr(),
                dev_sl_f.unsafe_ptr(),
                dev_sr_f.unsafe_ptr(),
                dev_nm_f.unsafe_ptr(),
                dev_iop_f.unsafe_ptr(),
                dev_iat_f.unsafe_ptr(),
                dev_ilen_f.unsafe_ptr(),
                grid_dim=grid_r,
                block_dim=BLOCK,
            )

            # Overlap: read+pack next batch on host while pass-0 FW runs.
            var next_b1 = List[FastqRec]()
            var next_b2 = List[FastqRec]()
            _read_pe_batch(
                fh1, fh2, paired, batch_size, bs_r1, bs_r2, next_b1, next_b2
            )
            var next_host_bases = ctx.enqueue_create_host_buffer[DType.uint8](1)
            var next_host_lens = ctx.enqueue_create_host_buffer[DType.uint32](1)
            var next_n_seq = 0
            var next_max_len = 0
            var next_slots_per = 1
            var next_n_slots = 0
            var next_n_bases = 0
            var next_ready = False
            if len(next_b1) > 0:
                var next_seqs = List[String]()
                for r in next_b1:
                    next_seqs.append(r.seq)
                if paired:
                    for r in next_b2:
                        next_seqs.append(r.seq)
                next_n_seq = len(next_seqs)
                next_max_len = 0
                var nri = 0
                while nri < next_n_seq:
                    var nL = next_seqs[nri].byte_length()
                    if nL > next_max_len:
                        next_max_len = nL
                    nri += 1
                if next_max_len < k_len:
                    next_max_len = k_len
                next_slots_per = 1
                if next_max_len >= k_len:
                    next_slots_per = (
                        (next_max_len - k_len) // seed_stride
                    ) + 1
                next_n_slots = next_n_seq * next_slots_per
                next_n_bases = next_n_seq * next_max_len
                next_host_bases = ctx.enqueue_create_host_buffer[DType.uint8](
                    next_n_bases
                )
                next_host_lens = ctx.enqueue_create_host_buffer[DType.uint32](
                    next_n_seq
                )
                var nsi = 0
                while nsi < next_n_seq:
                    var ns = next_seqs[nsi]
                    var nLen = ns.byte_length()
                    next_host_lens[nsi] = UInt32(nLen)
                    var nbase = nsi * next_max_len
                    var np = 0
                    while np < next_max_len:
                        if np < nLen:
                            next_host_bases[nbase + np] = UInt8(
                                ord(ns[byte = np : np + 1])
                            )
                        else:
                            next_host_bases[nbase + np] = 78
                        np += 1
                    nsi += 1
                next_ready = True

            # Compact still-unmapped → Pass 1 high-occ (avoids warp divergence).
            # Pull cid+nm together so RC can reuse this snapshot when pass-1 is a no-op
            # (saves a second host sync on most batches).
            var host_cid_fw = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var host_nm_fw = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            ctx.enqueue_copy(src_buf=dev_cid_f, dst_buf=host_cid_fw)
            ctx.enqueue_copy(src_buf=dev_nm_f, dst_buf=host_nm_fw)
            ctx.synchronize()
            var n_unmap_f = 0
            var ui = 0
            while ui < n_seq:
                if Int(host_cid_fw[ui]) < 0:
                    n_unmap_f += 1
                ui += 1
            var ran_pass1_f = False
            if n_unmap_f > 0:
                var host_rid_f = ctx.enqueue_create_host_buffer[DType.int32](
                    n_unmap_f
                )
                var w = 0
                ui = 0
                while ui < n_seq:
                    if Int(host_cid_fw[ui]) < 0:
                        host_rid_f[w] = Int32(ui)
                        w += 1
                    ui += 1
                var dev_rid_f = ctx.enqueue_create_buffer[DType.int32](n_unmap_f)
                ctx.enqueue_copy(src_buf=host_rid_f, dst_buf=dev_rid_f)
                var grid_u = (n_unmap_f + BLOCK - 1) // BLOCK
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
                    max_indel,
                    vote_occ,
                    1,
                    n_unmap_f,
                    dev_rid_f.unsafe_ptr(),
                    Int32(0),
                    dev_cid_f.unsafe_ptr(),
                    dev_pos_f.unsafe_ptr(),
                    dev_mapq_f.unsafe_ptr(),
                    dev_flag_f.unsafe_ptr(),
                    dev_sl_f.unsafe_ptr(),
                    dev_sr_f.unsafe_ptr(),
                    dev_nm_f.unsafe_ptr(),
                    dev_iop_f.unsafe_ptr(),
                    dev_iat_f.unsafe_ptr(),
                    dev_ilen_f.unsafe_ptr(),
                    grid_dim=grid_u,
                    block_dim=BLOCK,
                )
                ran_pass1_f = True

            if do_rc:
                # Refresh FW hits only if pass-1 may have filled some unmapped.
                if ran_pass1_f:
                    ctx.enqueue_copy(src_buf=dev_cid_f, dst_buf=host_cid_fw)
                    ctx.enqueue_copy(src_buf=dev_nm_f, dst_buf=host_nm_fw)
                    ctx.synchronize()
                var n_need_rc = 0
                ui = 0
                while ui < n_seq:
                    # Exact FW hit (NM=0) is definitive — RC cannot improve NM.
                    if Int(host_cid_fw[ui]) < 0 or Int(host_nm_fw[ui]) > 0:
                        n_need_rc += 1
                    ui += 1
                if n_batches == 0:
                    print(
                        "MojoLinear GPU-full need_rc=",
                        n_need_rc,
                        "/",
                        n_seq,
                        " (first batch)",
                    )
                # Mark all RC outputs unmapped; compacted kernels fill need_rc.
                var host_cid_r_init = ctx.enqueue_create_host_buffer[DType.int32](
                    n_seq
                )
                var zi0 = 0
                while zi0 < n_seq:
                    host_cid_r_init[zi0] = Int32(-1)
                    zi0 += 1
                ctx.enqueue_copy(src_buf=host_cid_r_init, dst_buf=dev_cid_r)

                if n_need_rc > 0:
                    var host_rid_rc = ctx.enqueue_create_host_buffer[DType.int32](
                        n_need_rc
                    )
                    var wrc = 0
                    ui = 0
                    while ui < n_seq:
                        if Int(host_cid_fw[ui]) < 0 or Int(host_nm_fw[ui]) > 0:
                            host_rid_rc[wrc] = Int32(ui)
                            wrc += 1
                        ui += 1
                    var dev_rid_rc = ctx.enqueue_create_buffer[DType.int32](
                        n_need_rc
                    )
                    ctx.enqueue_copy(src_buf=host_rid_rc, dst_buf=dev_rid_rc)
                    var grid_rc = (n_need_rc + BLOCK - 1) // BLOCK

                    # Reverse strand locate still full-batch (slot layout); extend
                    # runs only on need_rc via rid_map.
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
                    # Pass 0 RC on need_rc only.
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
                        max_indel,
                        vote_occ,
                        0,
                        n_need_rc,
                        dev_rid_rc.unsafe_ptr(),
                        Int32(16),
                        dev_cid_r.unsafe_ptr(),
                        dev_pos_r.unsafe_ptr(),
                        dev_mapq_r.unsafe_ptr(),
                        dev_flag_r.unsafe_ptr(),
                        dev_sl_r.unsafe_ptr(),
                        dev_sr_r.unsafe_ptr(),
                        dev_nm_r.unsafe_ptr(),
                        dev_iop_r.unsafe_ptr(),
                        dev_iat_r.unsafe_ptr(),
                        dev_ilen_r.unsafe_ptr(),
                        grid_dim=grid_rc,
                        block_dim=BLOCK,
                    )
                    # Pass 1 RC: still-unmapped among need_rc.
                    var host_cid_r_compact = ctx.enqueue_create_host_buffer[
                        DType.int32
                    ](n_seq)
                    ctx.enqueue_copy(
                        src_buf=dev_cid_r, dst_buf=host_cid_r_compact
                    )
                    ctx.synchronize()
                    var n_unmap_r = 0
                    var uir = 0
                    while uir < n_need_rc:
                        var rr = Int(host_rid_rc[uir])
                        if Int(host_cid_r_compact[rr]) < 0:
                            n_unmap_r += 1
                        uir += 1
                    if n_unmap_r > 0:
                        var host_rid_r = ctx.enqueue_create_host_buffer[
                            DType.int32
                        ](n_unmap_r)
                        var wr = 0
                        uir = 0
                        while uir < n_need_rc:
                            var rr2 = Int(host_rid_rc[uir])
                            if Int(host_cid_r_compact[rr2]) < 0:
                                host_rid_r[wr] = Int32(rr2)
                                wr += 1
                            uir += 1
                        var dev_rid_r = ctx.enqueue_create_buffer[DType.int32](
                            n_unmap_r
                        )
                        ctx.enqueue_copy(src_buf=host_rid_r, dst_buf=dev_rid_r)
                        var grid_ur = (n_unmap_r + BLOCK - 1) // BLOCK
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
                            max_indel,
                            vote_occ,
                            1,
                            n_unmap_r,
                            dev_rid_r.unsafe_ptr(),
                            Int32(16),
                            dev_cid_r.unsafe_ptr(),
                            dev_pos_r.unsafe_ptr(),
                            dev_mapq_r.unsafe_ptr(),
                            dev_flag_r.unsafe_ptr(),
                            dev_sl_r.unsafe_ptr(),
                            dev_sr_r.unsafe_ptr(),
                            dev_nm_r.unsafe_ptr(),
                            dev_iop_r.unsafe_ptr(),
                            dev_iat_r.unsafe_ptr(),
                            dev_ilen_r.unsafe_ptr(),
                            grid_dim=grid_ur,
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
            var host_iop_f = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_iat_f = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_ilen_f = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_cid_r = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var host_pos_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_mapq_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_flag_r = ctx.enqueue_create_host_buffer[DType.int32](n_seq)
            var host_sl_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_sr_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_nm_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_iop_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_iat_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            var host_ilen_r = ctx.enqueue_create_host_buffer[DType.uint32](n_seq)
            ctx.enqueue_copy(src_buf=dev_cid_f, dst_buf=host_cid_f)
            ctx.enqueue_copy(src_buf=dev_pos_f, dst_buf=host_pos_f)
            ctx.enqueue_copy(src_buf=dev_mapq_f, dst_buf=host_mapq_f)
            ctx.enqueue_copy(src_buf=dev_flag_f, dst_buf=host_flag_f)
            ctx.enqueue_copy(src_buf=dev_sl_f, dst_buf=host_sl_f)
            ctx.enqueue_copy(src_buf=dev_sr_f, dst_buf=host_sr_f)
            ctx.enqueue_copy(src_buf=dev_nm_f, dst_buf=host_nm_f)
            ctx.enqueue_copy(src_buf=dev_iop_f, dst_buf=host_iop_f)
            ctx.enqueue_copy(src_buf=dev_iat_f, dst_buf=host_iat_f)
            ctx.enqueue_copy(src_buf=dev_ilen_f, dst_buf=host_ilen_f)
            ctx.enqueue_copy(src_buf=dev_cid_r, dst_buf=host_cid_r)
            ctx.enqueue_copy(src_buf=dev_pos_r, dst_buf=host_pos_r)
            ctx.enqueue_copy(src_buf=dev_mapq_r, dst_buf=host_mapq_r)
            ctx.enqueue_copy(src_buf=dev_flag_r, dst_buf=host_flag_r)
            ctx.enqueue_copy(src_buf=dev_sl_r, dst_buf=host_sl_r)
            ctx.enqueue_copy(src_buf=dev_sr_r, dst_buf=host_sr_r)
            ctx.enqueue_copy(src_buf=dev_nm_r, dst_buf=host_nm_r)
            ctx.enqueue_copy(src_buf=dev_iop_r, dst_buf=host_iop_r)
            ctx.enqueue_copy(src_buf=dev_iat_r, dst_buf=host_iat_r)
            ctx.enqueue_copy(src_buf=dev_ilen_r, dst_buf=host_ilen_r)
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
                        Int(host_iop_r[j]),
                        Int(host_iat_r[j]),
                        Int(host_ilen_r[j]),
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
                        Int(host_iop_f[j]),
                        Int(host_iat_f[j]),
                        Int(host_ilen_f[j]),
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
                            Int(host_iop_r[r2i]),
                            Int(host_iat_r[r2i]),
                            Int(host_ilen_r[r2i]),
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
                            Int(host_iop_f[r2i]),
                            Int(host_iat_f[r2i]),
                            Int(host_ilen_f[r2i]),
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

            # Promote prefetched next batch (already packed during pass-0).
            if next_ready:
                batch1 = next_b1^
                batch2 = next_b2^
                host_bases = next_host_bases^
                host_lens = next_host_lens^
                n_seq = next_n_seq
                max_len = next_max_len
                slots_per = next_slots_per
                n_slots = next_n_slots
                n_bases = next_n_bases
                need_pack = False
            else:
                batch1 = List[FastqRec]()
                batch2 = List[FastqRec]()
                need_pack = True

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
            " backend=gpu-full",
        )
        print(
            "MojoLinear GPU-full map_wall_s=",
            t_map1 - t_map0,
            " reads=",
            n_reads,
        )
        return n_mapped
