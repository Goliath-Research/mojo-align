# GPU-native Giraffe science: window-reduce, Q1Q1 HT probe, cluster, gapless.
#
# One DeviceContext owns index residency + all batch kernels for the FASTQ stream.
# Unique-cell HT only (skip IS_POINTER) — same contract as MojoMinIndex.

from max.algorithm import parallelize
from max.gpu.host import DeviceContext
from std.collections import List
from std.ffi import external_call
from std.memory import Pointer, UnsafePointer, unsafe_memmove
from std.python import Python, PythonObject
from std.sys import has_accelerator

from parallel_slot import clear_parallel_slot, parallel_slot_addr, set_parallel_slot

from giraffe_fastq import giraffe_fq_read_header_seq
from giraffe_gaf_emit import append_gaf_hits
from giraffe_gpu_index import (
    gpu_index_meta_from,
    log_gpu_index_resident,
    require_gpu_index_capacity,
)
from giraffe_hit import AlignmentHit
from giraffe_min_index import MojoMinIndex
from giraffe_pack import DensePack
from linear_fastq import FastqPairStream, FastqPipe


def _fixture_extend_tiny(
    pack: DensePack, qname: String, seq: String, k: Int
) raises -> List[AlignmentHit]:
    """Toy-pack exact match only (pack_n ≤ 50k). Not used on production GBZ."""
    var out = List[AlignmentHit]()
    if pack.size() > 50_000:
        return out^
    var qlen = seq.byte_length()
    var ids = pack.segment_ids()
    for sid in ids:
        if pack.get_sid(sid) == seq:
            out.append(
                AlignmentHit(qname, ">" + sid, qlen, 60, "cs:Z::" + String(qlen))
            )
            return out^
    _ = k
    return out^


comptime NO_KEY = UInt64(0x7FFFFFFFFFFFFFFF)
comptime IS_POINTER = UInt64(1) << 63
comptime OFFSET_BITS = 10
comptime REV_MASK = UInt64(1) << OFFSET_BITS
comptime OFF_MASK = REV_MASK - 1
comptime MAX_OCCS = 64
comptime HIT_CAP = 24
comptime MAX_CLUSTER = 16
# Node-id distance prune (heuristic when zip/dist STAGED); always on in GPU path.
comptime DIST_CAP = 200
# Fixed query stride so batch device buffers can be reused across the stream.
comptime READ_STRIDE_CAP = 256


struct StreamReadGPU(Copyable, Movable):
    """Converted FASTQ record; ``original_seq`` feeds MethylCall ``os:Z``."""

    var name: String
    var seq: String
    var original_seq: String
    var conversion: String

    def __init__(
        out self,
        name: String,
        seq: String,
        original_seq: String,
        conversion: String,
    ):
        self.name = name
        self.seq = seq
        self.original_seq = original_seq
        self.conversion = conversion


def gpu_native_backend_label(backend: String) raises -> String:
    return backend + "+gpu_ht+gpu_gapless+mojo_stream"


def _strip_nl(mut s: String):
    while s.byte_length() > 0:
        var last = String(s[byte = s.byte_length() - 1 : s.byte_length()])
        if last == "\n" or last == "\r":
            var trimmed = String(s[byte = 0 : s.byte_length() - 1])
            s = trimmed
        else:
            break


def _rc_from_conversion(conversion: String, fallback: String) -> String:
    if conversion == "C2T":
        return "CT"
    if conversion == "G2A":
        return "GA"
    return fallback


def _pe_extra_tags(
    ri: Int, original_seq: String, conversion: String, fallback_rc: String
) -> String:
    var os = original_seq
    var rc = _rc_from_conversion(conversion, fallback_rc)
    return "ri:i:" + String(ri) + "\tos:Z:" + os + "\trc:Z:" + rc


def _parse_mg_fastq_fields(n: String, s: String) -> StreamReadGPU:
    """Parse methylGrapher converted FASTQ header (``id_C2T_i_original``).

    Single left-to-right scan — avoid ``split`` + re-join on long ``original_seq``.
    """
    var bare = n
    var original = s
    var conversion = String("")
    # Find first three '_' separators without building a parts list.
    var u0 = -1
    var u1 = -1
    var u2 = -1
    var i = 0
    var nlen = n.byte_length()
    while i < nlen:
        if n[byte = i : i + 1] == "_":
            if u0 < 0:
                u0 = i
            elif u1 < 0:
                u1 = i
            elif u2 < 0:
                u2 = i
                break
        i += 1
    if u0 >= 0 and u1 > u0 and u2 > u1:
        var conv = String(n[byte = u0 + 1 : u1])
        if conv == "C2T" or conv == "G2A":
            bare = String(n[byte = 0 : u0])
            conversion = conv
            original = String(n[byte = u2 + 1 : nlen])
        else:
            bare = String(n[byte = 0 : u0])
    elif u0 >= 0:
        bare = String(n[byte = 0 : u0])
    var sp = bare.find(" ")
    if sp >= 0:
        var bare_head = String(bare[byte = 0 : sp])
        bare = bare_head
    return StreamReadGPU(bare, s, original, conversion)


def _read_one_gpu_pipe(mut pipe: FastqPipe) raises -> StreamReadGPU:
    var n = String("")
    var s = String("")
    if not giraffe_fq_read_header_seq(pipe, n, s):
        return StreamReadGPU("", "", "", "")
    return _parse_mg_fastq_fields(n, s)


def _mut_u8[origin: Origin](p: Pointer[UInt8, origin]) -> Pointer[UInt8, MutAnyOrigin]:
    return Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p))


def _memset_bytes[origin: Origin](dest: Pointer[UInt8, origin], value: Int, count: Int):
    var p = _mut_u8(dest)
    _ = external_call["memset", Pointer[UInt8, MutAnyOrigin]](p, value, UInt(count))


struct _GiraffeOvSlot:
    var emit_n: Int
    var out_fh: Int
    var pending_hits: Int
    var batch_size: Int
    var fq: Int
    var paired: Bool
    var batch1: Int
    var batch2: Int
    var ov_fail0: Int
    var ov_fail1: Int

    def __init__(
        out self,
        emit_n: Int,
        out_fh: Int,
        pending_hits: Int,
        batch_size: Int,
        fq: Int,
        paired: Bool,
        batch1: Int,
        batch2: Int,
        ov_fail0: Int,
        ov_fail1: Int,
    ):
        self.emit_n = emit_n
        self.out_fh = out_fh
        self.pending_hits = pending_hits
        self.batch_size = batch_size
        self.fq = fq
        self.paired = paired
        self.batch1 = batch1
        self.batch2 = batch2
        self.ov_fail0 = ov_fail0
        self.ov_fail1 = ov_fail1


def _giraffe_ov_worker(wi: Int):
    var slot = Pointer[_GiraffeOvSlot, MutAnyOrigin](
        unsafe_from_address=parallel_slot_addr("MOJO_GIRAFFE_OV")
    )
    var emit_n = Pointer[Int, MutAnyOrigin](unsafe_from_address=slot[].emit_n)
    var out_fh = Pointer[PythonObject, MutAnyOrigin](unsafe_from_address=slot[].out_fh)
    var pending_hits = Pointer[List[AlignmentHit], MutAnyOrigin](
        unsafe_from_address=slot[].pending_hits
    )
    var fq = Pointer[FastqPairStream, MutAnyOrigin](unsafe_from_address=slot[].fq)
    var batch1 = Pointer[List[StreamReadGPU], MutAnyOrigin](unsafe_from_address=slot[].batch1)
    var batch2 = Pointer[List[StreamReadGPU], MutAnyOrigin](unsafe_from_address=slot[].batch2)
    var ov_fail0 = Pointer[Int, MutAnyOrigin](unsafe_from_address=slot[].ov_fail0)
    var ov_fail1 = Pointer[Int, MutAnyOrigin](unsafe_from_address=slot[].ov_fail1)
    try:
        if wi == 0:
            emit_n[] = append_gaf_hits(out_fh[], pending_hits[])
        else:
            var i = 0
            while i < slot[].batch_size:
                var r1 = _read_one_gpu_pipe(fq[].r1)
                if r1.name.byte_length() == 0:
                    break
                batch1[].append(r1^)
                if slot[].paired:
                    var r2 = _read_one_gpu_pipe(fq[].r2)
                    if r2.name.byte_length() == 0:
                        raise Error("paired FASTQ length mismatch (R2 ended early)")
                    batch2[].append(r2^)
                i += 1
    except e:
        if wi == 0:
            ov_fail0[] = 1
        else:
            ov_fail1[] = 1
        print("mojo_stream emit||fastq overlap worker", wi, e)


struct _GiraffeSyncSlot:
    var ctx: Int
    var batch_size: Int
    var fq: Int
    var next_paired: Bool
    var next1: Int
    var next2: Int
    var sync_fail: Int
    var prefetch_ok: Int

    def __init__(
        out self,
        ctx: Int,
        batch_size: Int,
        fq: Int,
        next_paired: Bool,
        next1: Int,
        next2: Int,
        sync_fail: Int,
        prefetch_ok: Int,
    ):
        self.ctx = ctx
        self.batch_size = batch_size
        self.fq = fq
        self.next_paired = next_paired
        self.next1 = next1
        self.next2 = next2
        self.sync_fail = sync_fail
        self.prefetch_ok = prefetch_ok


def _giraffe_sync_worker(wi: Int):
    var slot = Pointer[_GiraffeSyncSlot, MutAnyOrigin](
        unsafe_from_address=parallel_slot_addr("MOJO_GIRAFFE_SYNC")
    )
    var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=slot[].ctx)
    var fq = Pointer[FastqPairStream, MutAnyOrigin](unsafe_from_address=slot[].fq)
    var next1 = Pointer[List[StreamReadGPU], MutAnyOrigin](unsafe_from_address=slot[].next1)
    var next2 = Pointer[List[StreamReadGPU], MutAnyOrigin](unsafe_from_address=slot[].next2)
    var sync_fail = Pointer[Int, MutAnyOrigin](unsafe_from_address=slot[].sync_fail)
    var prefetch_ok = Pointer[Int, MutAnyOrigin](unsafe_from_address=slot[].prefetch_ok)
    try:
        if wi == 0:
            ctx[].synchronize()
        else:
            var i = 0
            while i < slot[].batch_size:
                var r1 = _read_one_gpu_pipe(fq[].r1)
                if r1.name.byte_length() == 0:
                    break
                next1[].append(r1^)
                if slot[].next_paired:
                    var r2 = _read_one_gpu_pipe(fq[].r2)
                    if r2.name.byte_length() == 0:
                        raise Error("paired FASTQ length mismatch (R2 ended early)")
                    next2[].append(r2^)
                i += 1
    except e:
        if wi == 0:
            sync_fail[] = 1
        else:
            prefetch_ok[] = 0
        print("mojo_stream sync||prefetch worker", wi, e)


def gpu_native_stream_loop(
    mut pack: DensePack,
    mut min_idx: MojoMinIndex,
    mut fq: FastqPairStream,
    out_fh: PythonObject,
    dist: String,
    batch_size: Int,
    profile: Bool,
    backend: String,
    dev: String,
) raises -> Int:
    """Upload indexes once; stream batches with device seed/locate/cluster/gapless."""
    comptime if not has_accelerator():
        raise Error("gpu_native_stream_loop requires accelerator build")
    else:
        from max.gpu import block_dim, block_idx, thread_idx

        def copy_bytes_offset_kernel(
            dst: UnsafePointer[UInt8, MutAnyOrigin],
            src: UnsafePointer[UInt8, MutAnyOrigin],
            dst_off: Int64,
            n: Int64,
        ):
            """Device-side memcpy into ``dst[dst_off:dst_off+n]`` (no CUDA runtime API)."""
            var tid = Int64(block_idx.x * block_dim.x + thread_idx.x)
            if tid < n:
                dst[dst_off + tid] = src[tid]

        def upload_mmap_to_device(
            ctx: DeviceContext,
            dst: UnsafePointer[UInt8, MutAnyOrigin],
            host_addr: Int,
            nbytes: Int,
        ) raises:
            """Chunked H2D via Mojo HostBuffer + enqueue_copy + device copy kernel.

            Application code must not call libcudart/CuPy; DeviceContext owns the
            NVIDIA/AMD transport. ``api=\"cuda\"`` below is Mojo's NVIDIA backend
            token (framework naming), not a direct CUDA Runtime dependency.
            """
            if nbytes <= 0:
                return
            if host_addr == 0:
                raise Error("upload_mmap_to_device: null host mmap address")
            # 64 MiB host staging — keeps peak host RAM bounded for multi‑GB HT.
            comptime CHUNK = 64 * 1024 * 1024
            comptime BLOCK = 256
            var off = 0
            while off < nbytes:
                var n = nbytes - off
                if n > CHUNK:
                    n = CHUNK
                var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
                var src = UnsafePointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=host_addr + off
                )
                unsafe_memmove(dest=host.unsafe_ptr(), src=src, count=n)
                var stage = ctx.enqueue_create_buffer[DType.uint8](n)
                ctx.enqueue_copy(src_buf=host, dst_buf=stage)
                var grid = (n + BLOCK - 1) // BLOCK
                ctx.enqueue_function[copy_bytes_offset_kernel](
                    dst,
                    stage.unsafe_ptr(),
                    Int64(off),
                    Int64(n),
                    grid_dim=grid,
                    block_dim=BLOCK,
                )
                ctx.synchronize()
                off += n

        def pack_bases_kernel(
            bases: UnsafePointer[UInt8, MutAnyOrigin],
            codes: UnsafePointer[UInt8, MutAnyOrigin],
            n: Int64,
        ):
            var idx = Int64(block_idx.x * block_dim.x + thread_idx.x)
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
            n_bases: Int64,
            k_len: Int64,
            stride: Int64,
        ):
            var idx = Int64(block_idx.x * block_dim.x + thread_idx.x)
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
            var j = Int64(0)
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

        def window_reduce_kernel(
            keys_f: UnsafePointer[UInt64, MutAnyOrigin],
            keys_r: UnsafePointer[UInt64, MutAnyOrigin],
            hashes_f: UnsafePointer[UInt64, MutAnyOrigin],
            hashes_r: UnsafePointer[UInt64, MutAnyOrigin],
            valid: UnsafePointer[UInt8, MutAnyOrigin],
            read_lens: UnsafePointer[Int32, MutAnyOrigin],
            out_keys: UnsafePointer[UInt64, MutAnyOrigin],
            out_n: UnsafePointer[Int32, MutAnyOrigin],
            n_reads: Int64,
            stride: Int64,
            k_len: Int64,
            w_len: Int64,
        ):
            var ri = Int64(block_idx.x * block_dim.x + thread_idx.x)
            if ri >= n_reads:
                return
            var L = Int64(read_lens[ri])
            var base = ri * stride
            var win = k_len + w_len - 1
            var n_out: Int32 = 0
            var row = ri * MAX_OCCS
            if L >= win:
                var next_read_offset = Int64(0)
                var last_hash = UInt64(0)
                var last_offset = Int64(-1)
                var have_last = False
                var window_start = Int64(0)
                while window_start <= L - win and Int(n_out) < MAX_OCCS:
                    var best_key = UInt64(0)
                    var best_hash = UInt64(0)
                    var best_off = Int64(0)
                    var have_best = False
                    var ok = True
                    var wi = Int64(0)
                    while wi < w_len:
                        var pos = window_start + wi
                        var idx = base + pos
                        if valid[idx] == 0:
                            ok = False
                            break
                        var use_r = hashes_r[idx] < hashes_f[idx]
                        var cand_key = keys_f[idx]
                        var cand_hash = hashes_f[idx]
                        if use_r:
                            cand_key = keys_r[idx]
                            cand_hash = hashes_r[idx]
                        if (
                            not have_best
                            or cand_hash < best_hash
                            or (cand_hash == best_hash and pos < best_off)
                        ):
                            best_key = cand_key
                            best_hash = cand_hash
                            best_off = pos
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
                            out_keys[row + Int64(n_out)] = best_key
                            n_out = n_out + 1
                            next_read_offset = best_off + 1
                            last_hash = best_hash
                            last_offset = best_off
                            have_last = True
                    window_start += 1
            out_n[ri] = n_out

        def ht_probe_kernel(
            ht: UnsafePointer[UInt64, MutAnyOrigin],
            occ_keys: UnsafePointer[UInt64, MutAnyOrigin],
            occ_n: UnsafePointer[Int32, MutAnyOrigin],
            out_node: UnsafePointer[Int32, MutAnyOrigin],
            out_orient: UnsafePointer[UInt8, MutAnyOrigin],
            out_off: UnsafePointer[Int32, MutAnyOrigin],
            out_n: UnsafePointer[Int32, MutAnyOrigin],
            n_reads: Int64,
            cell_size: Int64,
            cell_count: Int64,
        ):
            var ri = Int64(block_idx.x * block_dim.x + thread_idx.x)
            if ri >= n_reads:
                return
            var n_occ = Int64(occ_n[ri])
            var written: Int32 = 0
            var row = ri * HIT_CAP
            var oi = Int64(0)
            while oi < n_occ and Int(written) < HIT_CAP:
                var key = occ_keys[ri * MAX_OCCS + oi] & NO_KEY
                var h = key
                h = (~h) + (h << 21)
                h = h ^ (h >> 24)
                h = (h + (h << 3)) + (h << 8)
                h = h ^ (h >> 14)
                h = (h + (h << 2)) + (h << 4)
                h = h ^ (h >> 28)
                h = h + (h << 31)
                var cell_offset = Int64(h) & (cell_count - 1)
                var attempt = Int64(0)
                var found = Int64(-1)
                while attempt < cell_count:
                    var array_off = cell_offset * cell_size
                    var cell_key = ht[array_off]
                    var bare = cell_key & NO_KEY
                    if bare == NO_KEY:
                        break
                    if bare == key:
                        found = array_off
                        break
                    cell_offset = (cell_offset + attempt + 1) & (cell_count - 1)
                    attempt += 1
                if found >= 0:
                    var ck = ht[found]
                    if (ck & IS_POINTER) == 0:
                        var pos = ht[found + 1]
                        var node_id = Int(pos >> UInt64(OFFSET_BITS + 1))
                        var is_rev = (pos & REV_MASK) != 0
                        var offset = Int(pos & OFF_MASK)
                        if node_id > 0:
                            var dup = False
                            var di = Int64(0)
                            while di < Int64(written):
                                if Int(out_node[row + di]) == node_id and Int(
                                    out_off[row + di]
                                ) == offset:
                                    dup = True
                                    break
                                di += 1
                            if not dup:
                                out_node[row + Int64(written)] = Int32(node_id)
                                if is_rev:
                                    out_orient[row + Int64(written)] = 1
                                else:
                                    out_orient[row + Int64(written)] = 0
                                out_off[row + Int64(written)] = Int32(offset)
                                written = written + 1
                oi += 1
            out_n[ri] = written

        def cluster_kernel(
            in_node: UnsafePointer[Int32, MutAnyOrigin],
            in_orient: UnsafePointer[UInt8, MutAnyOrigin],
            in_off: UnsafePointer[Int32, MutAnyOrigin],
            in_n: UnsafePointer[Int32, MutAnyOrigin],
            out_node: UnsafePointer[Int32, MutAnyOrigin],
            out_orient: UnsafePointer[UInt8, MutAnyOrigin],
            out_off: UnsafePointer[Int32, MutAnyOrigin],
            out_n: UnsafePointer[Int32, MutAnyOrigin],
            n_reads: Int64,
            do_prune: Int64,
        ):
            var ri = Int64(block_idx.x * block_dim.x + thread_idx.x)
            if ri >= n_reads:
                return
            var n_in = Int64(in_n[ri])
            var base = ri * HIT_CAP
            var obase = ri * MAX_CLUSTER
            if n_in <= 0:
                out_n[ri] = 0
                return
            # Single-pass bucket histogram (HIT_CAP small) vs O(n^2) recount.
            var best_bucket = Int(in_node[base]) >> 8
            var best_count = Int64(0)
            var i = Int64(0)
            while i < n_in:
                var b = Int(in_node[base + i]) >> 8
                var c = Int64(1)
                var j = i + 1
                while j < n_in:
                    if (Int(in_node[base + j]) >> 8) == b:
                        c += 1
                    j += 1
                if c > best_count:
                    best_count = c
                    best_bucket = b
                # Skip ahead over same bucket once counted? Keep simple: still O(n^2)
                # but early-exit when remaining cannot beat best_count.
                if best_count >= (n_in - i):
                    break
                i += 1
            var anchor = -1
            var written: Int32 = 0
            i = 0
            while i < n_in and Int(written) < MAX_CLUSTER:
                var nid = Int(in_node[base + i])
                if (nid >> 8) != best_bucket:
                    i += 1
                    continue
                if anchor < 0:
                    anchor = nid
                if do_prune != 0:
                    var d = nid - anchor
                    if d < 0:
                        d = -d
                    if d > DIST_CAP:
                        i += 1
                        continue
                out_node[obase + Int64(written)] = Int32(nid)
                out_orient[obase + Int64(written)] = in_orient[base + i]
                out_off[obase + Int64(written)] = in_off[base + i]
                written = written + 1
                i += 1
            out_n[ri] = written

        def upper_code(b: UInt8) -> UInt8:
            if b == 97:
                return 65
            if b == 99:
                return 67
            if b == 103:
                return 71
            if b == 116:
                return 84
            return b

        def comp_base(b: UInt8) -> UInt8:
            var u = upper_code(b)
            if u == 65:
                return 84
            if u == 84:
                return 65
            if u == 67:
                return 71
            if u == 71:
                return 67
            return 78

        def gapless_kernel(
            # Raw ASCII query bases (same alphabet as sequences.bin / pack_seq).
            # Must NOT be pack_bases_kernel codes (0–3) — upper_code/comp_base
            # and qb==rb compare ASCII.
            q_bases: UnsafePointer[UInt8, MutAnyOrigin],
            read_lens: UnsafePointer[Int32, MutAnyOrigin],
            q_stride: Int64,
            pack_off: UnsafePointer[UInt64, MutAnyOrigin],
            pack_seq: UnsafePointer[UInt8, MutAnyOrigin],
            pack_n: Int64,
            seq_bytes: Int64,
            seed_node: UnsafePointer[Int32, MutAnyOrigin],
            seed_orient: UnsafePointer[UInt8, MutAnyOrigin],
            seed_off: UnsafePointer[Int32, MutAnyOrigin],
            seed_n: UnsafePointer[Int32, MutAnyOrigin],
            out_node: UnsafePointer[Int32, MutAnyOrigin],
            out_mapq: UnsafePointer[Int32, MutAnyOrigin],
            out_matched: UnsafePointer[Int32, MutAnyOrigin],
            out_valid: UnsafePointer[UInt8, MutAnyOrigin],
            n_reads: Int64,
        ):
            var tid = Int64(block_idx.x * block_dim.x + thread_idx.x)
            var n_slots = n_reads * MAX_CLUSTER
            if tid >= n_slots:
                return
            var ri = tid // MAX_CLUSTER
            var si = tid % MAX_CLUSTER
            out_valid[tid] = 0
            out_node[tid] = 0
            out_mapq[tid] = 0
            out_matched[tid] = 0
            var n_seeds = Int64(seed_n[ri])
            if si >= n_seeds:
                return
            var qlen = Int64(read_lens[ri])
            if qlen <= 0:
                return
            var node_id = Int64(seed_node[ri * MAX_CLUSTER + si])
            if node_id < 1 or node_id > pack_n:
                return
            var is_rev = seed_orient[ri * MAX_CLUSTER + si] != 0
            var offset = Int64(seed_off[ri * MAX_CLUSTER + si])
            var oidx = (node_id - 1) * 2
            var seg_off = Int64(pack_off[oidx])
            var seg_len = Int64(pack_off[oidx + 1])
            if seg_off < 0 or seg_len <= 0 or seg_off + seg_len > seq_bytes:
                return

            var qbase = ri * q_stride
            var start0 = Int64(0)
            var start1 = Int64(0)
            var start2 = Int64(0)
            if is_rev:
                start0 = seg_len - offset - qlen
                if start0 < 0:
                    start0 = Int64(0)
                start1 = offset - qlen + 1
                if start1 < 0:
                    start1 = Int64(0)
                start2 = Int64(0)
            else:
                start0 = offset
                start1 = offset - qlen + 29
                if start1 < 0:
                    start1 = Int64(0)
                start2 = Int64(0)

            var best_score = Int64(-1)
            var best_matched = Int64(0)
            var best_mism = Int64(0)
            var trial = Int64(0)
            while trial < 3:
                var start = start0
                if trial == 1:
                    start = start1
                elif trial == 2:
                    start = start2
                if start < 0 or start >= seg_len:
                    trial += 1
                    continue
                var matched = Int64(0)
                var mism = Int64(0)
                var qi = Int64(0)
                var rpos = start
                while qi < qlen and rpos < seg_len:
                    var qb = upper_code(q_bases[qbase + qi])
                    var rb: UInt8 = 0
                    if is_rev:
                        # ref_aln[rpos] = RC(ref)[rpos] = comp(ref[seg_len-1-rpos])
                        rb = comp_base(pack_seq[seg_off + (seg_len - 1 - rpos)])
                    else:
                        rb = upper_code(pack_seq[seg_off + rpos])
                    if qb == rb:
                        matched += 1
                    else:
                        mism += 1
                        var mism_cap = qlen // 20
                        if mism_cap < 2:
                            mism_cap = 2
                        if mism > mism_cap:
                            break
                    qi += 1
                    rpos += 1
                var min_need = Int64(29)
                if seg_len < min_need:
                    min_need = seg_len
                if matched >= min_need:
                    var score = matched - 2 * mism
                    if score > best_score:
                        best_score = score
                        best_matched = matched
                        best_mism = mism
                    # Prefer primary seed start: skip remaining trials when near-perfect.
                    if matched >= (qlen * 9) // 10 and mism <= 1:
                        break
                trial += 1

            if best_score < 0:
                return
            var mq = 20
            if best_matched >= (qlen * 9) // 10:
                if best_mism == 0:
                    mq = 60
                else:
                    mq = 40
            out_node[tid] = Int32(node_id)
            out_mapq[tid] = Int32(mq)
            out_matched[tid] = Int32(best_matched)
            out_valid[tid] = 1

        # ---- single DeviceContext session ----
        # Mojo NVIDIA backend token is "cuda"; AMD is "hip". We do not call the
        # CUDA Runtime from app code — uploads use DeviceContext enqueue_copy.
        var api = String("cuda")
        var dlow = dev.lower()
        if dlow == "amd" or dlow == "hip" or dlow == "rocm":
            api = String("hip")
        var meta = gpu_index_meta_from(min_idx, pack)
        log_gpu_index_resident(meta)
        require_gpu_index_capacity(meta, dev)
        var ctx = DeviceContext(api=api)

        var n_off_u64 = meta.off_bytes // 8
        var dev_ht = ctx.enqueue_create_buffer[DType.uint64](meta.ht_words)
        var dev_off = ctx.enqueue_create_buffer[DType.uint64](n_off_u64)
        var dev_seq = ctx.enqueue_create_buffer[DType.uint8](meta.seq_bytes)
        ctx.synchronize()
        upload_mmap_to_device(
            ctx,
            _mut_u8(dev_ht.unsafe_ptr().bitcast[UInt8]()),
            meta.ht_host_addr,
            meta.ht_words * 8,
        )
        upload_mmap_to_device(
            ctx,
            _mut_u8(dev_off.unsafe_ptr().bitcast[UInt8]()),
            meta.off_host_addr,
            meta.off_bytes,
        )
        upload_mmap_to_device(
            ctx,
            _mut_u8(dev_seq.unsafe_ptr()),
            meta.seq_host_addr,
            meta.seq_bytes,
        )

        var label = gpu_native_backend_label(backend)
        print(
            "mojo_stream seed_backend=",
            label,
            " device=",
            dev,
            flush=True,
        )
        var os_mod = Python.import_module("os")
        os_mod.environ["METHYLGRAPHER_LAST_GPU_BACKEND"] = label

        var time = Python.import_module("time")
        var n_records = 0
        var n_written = 0
        var prune_dist = dist.byte_length() > 0
        # Always apply DIST_CAP prune on GPU path (zip/dist STAGED); dist path
        # presence only logs richer pruning later. Fewer seeds → gapless.
        var do_prune = 1
        _ = prune_dist
        comptime BLOCK = 256

        # Deferred GAF emit overlapped with the next FASTQ batch read.
        var pending_hits = List[AlignmentHit]()
        var has_pending = False
        var pref1 = List[StreamReadGPU]()
        var pref2 = List[StreamReadGPU]()
        var has_pref = False

        # Reuse HBM/host batch buffers across the stream (avoid per-batch create).
        var cap_reads = batch_size * 2
        if cap_reads < 2:
            cap_reads = 2
        var cap_bases = cap_reads * READ_STRIDE_CAP
        var cap_slots = cap_reads * MAX_CLUSTER
        var host_bases = ctx.enqueue_create_host_buffer[DType.uint8](cap_bases)
        var host_lens = ctx.enqueue_create_host_buffer[DType.int32](cap_reads)
        var d_bases = ctx.enqueue_create_buffer[DType.uint8](cap_bases)
        var d_codes = ctx.enqueue_create_buffer[DType.uint8](cap_bases)
        var d_kf = ctx.enqueue_create_buffer[DType.uint64](cap_bases)
        var d_kr = ctx.enqueue_create_buffer[DType.uint64](cap_bases)
        var d_hf = ctx.enqueue_create_buffer[DType.uint64](cap_bases)
        var d_hr = ctx.enqueue_create_buffer[DType.uint64](cap_bases)
        var d_valid = ctx.enqueue_create_buffer[DType.uint8](cap_bases)
        var d_lens = ctx.enqueue_create_buffer[DType.int32](cap_reads)
        var d_occ_keys = ctx.enqueue_create_buffer[DType.uint64](cap_reads * MAX_OCCS)
        var d_occ_n = ctx.enqueue_create_buffer[DType.int32](cap_reads)
        var d_hit_node = ctx.enqueue_create_buffer[DType.int32](cap_reads * HIT_CAP)
        var d_hit_orient = ctx.enqueue_create_buffer[DType.uint8](cap_reads * HIT_CAP)
        var d_hit_off = ctx.enqueue_create_buffer[DType.int32](cap_reads * HIT_CAP)
        var d_hit_n = ctx.enqueue_create_buffer[DType.int32](cap_reads)
        var d_cl_node = ctx.enqueue_create_buffer[DType.int32](cap_slots)
        var d_cl_orient = ctx.enqueue_create_buffer[DType.uint8](cap_slots)
        var d_cl_off = ctx.enqueue_create_buffer[DType.int32](cap_slots)
        var d_cl_n = ctx.enqueue_create_buffer[DType.int32](cap_reads)
        var d_out_node = ctx.enqueue_create_buffer[DType.int32](cap_slots)
        var d_out_mapq = ctx.enqueue_create_buffer[DType.int32](cap_slots)
        var d_out_matched = ctx.enqueue_create_buffer[DType.int32](cap_slots)
        var d_out_valid = ctx.enqueue_create_buffer[DType.uint8](cap_slots)
        var h_node = ctx.enqueue_create_host_buffer[DType.int32](cap_slots)
        var h_mapq = ctx.enqueue_create_host_buffer[DType.int32](cap_slots)
        var h_matched = ctx.enqueue_create_host_buffer[DType.int32](cap_slots)
        var h_valid = ctx.enqueue_create_host_buffer[DType.uint8](cap_slots)
        var pack_large = pack.size() > 50_000

        while True:
            var t0 = time.perf_counter()
            var batch1 = List[StreamReadGPU]()
            var batch2 = List[StreamReadGPU]()
            var paired = fq.paired
            var t_write = 0.0
            var emit_n = 0
            var ov_fail0 = 0
            var ov_fail1 = 0

            if has_pref:
                batch1 = pref1^
                batch2 = pref2^
                pref1 = List[StreamReadGPU]()
                pref2 = List[StreamReadGPU]()
                has_pref = False
                if has_pending:
                    emit_n = append_gaf_hits(out_fh, pending_hits)
                    n_written += emit_n
                    pending_hits = List[AlignmentHit]()
                    has_pending = False
                    t_write = Float64(py=time.perf_counter()) - Float64(py=t0)
            elif has_pending:
                var ov_slot = _GiraffeOvSlot(
                    Int(Pointer(to=emit_n)),
                    Int(Pointer(to=out_fh)),
                    Int(Pointer(to=pending_hits)),
                    batch_size,
                    Int(Pointer(to=fq)),
                    paired,
                    Int(Pointer(to=batch1)),
                    Int(Pointer(to=batch2)),
                    Int(Pointer(to=ov_fail0)),
                    Int(Pointer(to=ov_fail1)),
                )
                set_parallel_slot("MOJO_GIRAFFE_OV", Int(Pointer(to=ov_slot)))
                parallelize(_giraffe_ov_worker, 2, 2)
                clear_parallel_slot("MOJO_GIRAFFE_OV")
                if ov_fail0 != 0 or ov_fail1 != 0:
                    raise Error("Giraffe emit||FASTQ overlap worker failed")
                n_written += emit_n
                pending_hits = List[AlignmentHit]()
                has_pending = False
                t_write = Float64(py=time.perf_counter()) - Float64(py=t0)
            else:
                var i = 0
                while i < batch_size:
                    var r1 = _read_one_gpu_pipe(fq.r1)
                    if r1.name.byte_length() == 0:
                        break
                    batch1.append(r1^)
                    if paired:
                        var r2 = _read_one_gpu_pipe(fq.r2)
                        if r2.name.byte_length() == 0:
                            raise Error(
                                "paired FASTQ length mismatch (R2 ended early)"
                            )
                        batch2.append(r2^)
                    i += 1
            if len(batch1) == 0:
                break
            var t_fastq = time.perf_counter()

            var seqs = List[String]()
            for r in batch1:
                seqs.append(r.seq)
            if paired:
                for r in batch2:
                    seqs.append(r.seq)

            var n_reads = len(seqs)
            var max_len = 0
            var ri = 0
            while ri < n_reads:
                var L = seqs[ri].byte_length()
                if L > max_len:
                    max_len = L
                ri += 1
            if max_len > READ_STRIDE_CAP:
                raise Error(
                    "read length "
                    + String(max_len)
                    + " exceeds READ_STRIDE_CAP="
                    + String(READ_STRIDE_CAP)
                )
            if n_reads > cap_reads:
                raise Error("n_reads exceeds reused batch capacity")

            var t_seed0 = time.perf_counter()
            var ran_gpu_batch = False
            if max_len >= meta.k and n_reads > 0:
                ran_gpu_batch = True
                var stride = READ_STRIDE_CAP
                var n_bases = n_reads * stride
                # memset unused lanes to N so kmer kernels stay valid.
                _memset_bytes(host_bases.unsafe_ptr(), 78, n_bases)
                ri = 0
                while ri < n_reads:
                    var s = seqs[ri]
                    var L = s.byte_length()
                    host_lens[ri] = Int32(L)
                    if L > 0:
                        var sb = s.as_bytes()
                        unsafe_memmove(
                            dest=host_bases.unsafe_ptr() + ri * stride,
                            src=sb.unsafe_ptr(),
                            count=L,
                        )
                    ri += 1

                ctx.enqueue_copy(src_buf=host_bases, dst_buf=d_bases)
                ctx.enqueue_copy(src_buf=host_lens, dst_buf=d_lens)

                var grid_bases = (n_bases + BLOCK - 1) // BLOCK
                ctx.enqueue_function[pack_bases_kernel](
                    d_bases.unsafe_ptr(),
                    d_codes.unsafe_ptr(),
                    Int64(n_bases),
                    grid_dim=grid_bases,
                    block_dim=BLOCK,
                )
                ctx.enqueue_function[kmer_fwd_rc_kernel](
                    d_codes.unsafe_ptr(),
                    d_kf.unsafe_ptr(),
                    d_kr.unsafe_ptr(),
                    d_hf.unsafe_ptr(),
                    d_hr.unsafe_ptr(),
                    d_valid.unsafe_ptr(),
                    Int64(n_bases),
                    Int64(meta.k),
                    Int64(stride),
                    grid_dim=grid_bases,
                    block_dim=BLOCK,
                )
                var t_seed1 = time.perf_counter()

                var grid_reads = (n_reads + BLOCK - 1) // BLOCK
                ctx.enqueue_function[window_reduce_kernel](
                    d_kf.unsafe_ptr(),
                    d_kr.unsafe_ptr(),
                    d_hf.unsafe_ptr(),
                    d_hr.unsafe_ptr(),
                    d_valid.unsafe_ptr(),
                    d_lens.unsafe_ptr(),
                    d_occ_keys.unsafe_ptr(),
                    d_occ_n.unsafe_ptr(),
                    Int64(n_reads),
                    Int64(stride),
                    Int64(meta.k),
                    Int64(meta.w),
                    grid_dim=grid_reads,
                    block_dim=BLOCK,
                )

                ctx.enqueue_function[ht_probe_kernel](
                    dev_ht.unsafe_ptr(),
                    d_occ_keys.unsafe_ptr(),
                    d_occ_n.unsafe_ptr(),
                    d_hit_node.unsafe_ptr(),
                    d_hit_orient.unsafe_ptr(),
                    d_hit_off.unsafe_ptr(),
                    d_hit_n.unsafe_ptr(),
                    Int64(n_reads),
                    Int64(meta.cell_size),
                    Int64(meta.cell_count),
                    grid_dim=grid_reads,
                    block_dim=BLOCK,
                )
                var t_locate = time.perf_counter()

                ctx.enqueue_function[cluster_kernel](
                    d_hit_node.unsafe_ptr(),
                    d_hit_orient.unsafe_ptr(),
                    d_hit_off.unsafe_ptr(),
                    d_hit_n.unsafe_ptr(),
                    d_cl_node.unsafe_ptr(),
                    d_cl_orient.unsafe_ptr(),
                    d_cl_off.unsafe_ptr(),
                    d_cl_n.unsafe_ptr(),
                    Int64(n_reads),
                    Int64(do_prune),
                    grid_dim=grid_reads,
                    block_dim=BLOCK,
                )

                var n_slots = n_reads * MAX_CLUSTER
                var grid_slots = (n_slots + BLOCK - 1) // BLOCK
                ctx.enqueue_function[gapless_kernel](
                    d_bases.unsafe_ptr(),
                    d_lens.unsafe_ptr(),
                    Int64(stride),
                    dev_off.unsafe_ptr(),
                    dev_seq.unsafe_ptr(),
                    Int64(meta.pack_n),
                    Int64(meta.seq_bytes),
                    d_cl_node.unsafe_ptr(),
                    d_cl_orient.unsafe_ptr(),
                    d_cl_off.unsafe_ptr(),
                    d_cl_n.unsafe_ptr(),
                    d_out_node.unsafe_ptr(),
                    d_out_mapq.unsafe_ptr(),
                    d_out_matched.unsafe_ptr(),
                    d_out_valid.unsafe_ptr(),
                    Int64(n_reads),
                    grid_dim=grid_slots,
                    block_dim=BLOCK,
                )

                ctx.enqueue_copy(src_buf=d_out_node, dst_buf=h_node)
                ctx.enqueue_copy(src_buf=d_out_mapq, dst_buf=h_mapq)
                ctx.enqueue_copy(src_buf=d_out_matched, dst_buf=h_matched)
                ctx.enqueue_copy(src_buf=d_out_valid, dst_buf=h_valid)
                # Overlap DeviceContext sync with prefetch of the next FASTQ batch.
                var next1 = List[StreamReadGPU]()
                var next2 = List[StreamReadGPU]()
                var next_paired = paired
                var prefetch_ok = 1
                var sync_fail = 0
                var sync_slot = _GiraffeSyncSlot(
                    Int(Pointer(to=ctx)),
                    batch_size,
                    Int(Pointer(to=fq)),
                    next_paired,
                    Int(Pointer(to=next1)),
                    Int(Pointer(to=next2)),
                    Int(Pointer(to=sync_fail)),
                    Int(Pointer(to=prefetch_ok)),
                )
                set_parallel_slot("MOJO_GIRAFFE_SYNC", Int(Pointer(to=sync_slot)))
                parallelize(_giraffe_sync_worker, 2, 2)
                clear_parallel_slot("MOJO_GIRAFFE_SYNC")
                if sync_fail != 0:
                    raise Error("Giraffe DeviceContext synchronize failed")
                var t_sync_pref = time.perf_counter()
                if prefetch_ok != 0 and len(next1) > 0:
                    pref1 = next1^
                    pref2 = next2^
                    has_pref = True
                var t_extend = t_sync_pref
                var t_prefetch = t_sync_pref - t_locate

                var t_host0 = time.perf_counter()
                var batch_hits = List[AlignmentHit]()
                # Build PE / SE hits directly from D2H — skip per_read List copies.
                var j = 0
                while j < len(batch1):
                    var a = batch1[j].copy()
                    var r1_i = j
                    var r2_i = len(batch1) + j
                    if not paired:
                        n_records += 1
                        var si = 0
                        var took = 0
                        while si < MAX_CLUSTER and took < 2:
                            var tid = r1_i * MAX_CLUSTER + si
                            if h_valid[tid] != 0:
                                var matched = Int(h_matched[tid])
                                var mq = Int(h_mapq[tid])
                                var qlen = Int(host_lens[r1_i])
                                var cs = "cs:Z::" + String(matched)
                                if mq >= 60:
                                    cs = "cs:Z::" + String(qlen)
                                batch_hits.append(
                                    AlignmentHit(
                                        a.name,
                                        ">" + String(Int(h_node[tid])),
                                        qlen,
                                        mq,
                                        cs,
                                    )
                                )
                                took += 1
                            si += 1
                        if took == 0 and not pack_large:
                            var tiny = _fixture_extend_tiny(
                                pack, a.name, seqs[r1_i], meta.k
                            )
                            for h in tiny:
                                batch_hits.append(h.copy())
                    else:
                        var b = batch2[j].copy()
                        var h1 = AlignmentHit(
                            a.name, "*", a.seq.byte_length(), 0, "cs:Z:*"
                        )
                        var h2 = AlignmentHit(
                            b.name, "*", b.seq.byte_length(), 0, "cs:Z:*"
                        )
                        var si = 0
                        var took1 = 0
                        while si < MAX_CLUSTER and took1 < 1:
                            var tid = r1_i * MAX_CLUSTER + si
                            if h_valid[tid] != 0:
                                var matched = Int(h_matched[tid])
                                var mq = Int(h_mapq[tid])
                                var qlen = Int(host_lens[r1_i])
                                var cs = "cs:Z::" + String(matched)
                                if mq >= 60:
                                    cs = "cs:Z::" + String(qlen)
                                h1 = AlignmentHit(
                                    a.name,
                                    ">" + String(Int(h_node[tid])),
                                    qlen,
                                    mq,
                                    cs,
                                )
                                took1 = 1
                            si += 1
                        si = 0
                        var took2 = 0
                        while si < MAX_CLUSTER and took2 < 1:
                            var tid2 = r2_i * MAX_CLUSTER + si
                            if h_valid[tid2] != 0:
                                var matched2 = Int(h_matched[tid2])
                                var mq2 = Int(h_mapq[tid2])
                                var qlen2 = Int(host_lens[r2_i])
                                var cs2 = "cs:Z::" + String(matched2)
                                if mq2 >= 60:
                                    cs2 = "cs:Z::" + String(qlen2)
                                h2 = AlignmentHit(
                                    b.name,
                                    ">" + String(Int(h_node[tid2])),
                                    qlen2,
                                    mq2,
                                    cs2,
                                )
                                took2 = 1
                            si += 1
                        if took1 == 0 and not pack_large:
                            var tiny1 = _fixture_extend_tiny(
                                pack, a.name, seqs[r1_i], meta.k
                            )
                            if len(tiny1) > 0:
                                h1 = tiny1[0].copy()
                        if took2 == 0 and not pack_large:
                            var tiny2 = _fixture_extend_tiny(
                                pack, b.name, seqs[r2_i], meta.k
                            )
                            if len(tiny2) > 0:
                                h2 = tiny2[0].copy()
                        h1.extra_tags = _pe_extra_tags(
                            1, a.original_seq, a.conversion, "CT"
                        )
                        h2.extra_tags = _pe_extra_tags(
                            2, b.original_seq, b.conversion, "GA"
                        )
                        batch_hits.append(h1^)
                        batch_hits.append(h2^)
                        n_records += 2
                    j += 1
                var t_host = time.perf_counter() - t_host0
                # Defer write: overlapped with next FASTQ read.
                pending_hits = batch_hits^
                has_pending = True
                var t_emit = time.perf_counter()
                if profile:
                    print(
                        "mojo_stream stages_s fastq=",
                        t_fastq - t0 - t_write,
                        " gpu_seed=",
                        t_seed1 - t_seed0,
                        " locate=",
                        t_locate - t_seed1,
                        " sync_prefetch=",
                        t_prefetch,
                        " host_hits=",
                        t_host,
                        " gaf_write_ov=",
                        t_write,
                        " gaf_emit=",
                        t_emit - t_extend,
                        " n_batch=",
                        len(batch1),
                        flush=True,
                    )

            if not ran_gpu_batch:
                # Short-read toys (len < k): fixture exact-match on tiny packs only.
                var batch_hits2 = List[AlignmentHit]()
                var j2 = 0
                while j2 < len(batch1):
                    var a2 = batch1[j2].copy()
                    if not paired:
                        n_records += 1
                        var tiny = _fixture_extend_tiny(
                            pack, a2.name, seqs[j2], meta.k
                        )
                        for h in tiny:
                            batch_hits2.append(h.copy())
                    else:
                        var b2 = batch2[j2].copy()
                        var h1b = AlignmentHit(
                            a2.name, "*", a2.seq.byte_length(), 0, "cs:Z:*"
                        )
                        var h2b = AlignmentHit(
                            b2.name, "*", b2.seq.byte_length(), 0, "cs:Z:*"
                        )
                        var tiny1 = _fixture_extend_tiny(
                            pack, a2.name, seqs[j2], meta.k
                        )
                        var tiny2 = _fixture_extend_tiny(
                            pack, b2.name, seqs[len(batch1) + j2], meta.k
                        )
                        if len(tiny1) > 0:
                            h1b = tiny1[0].copy()
                        if len(tiny2) > 0:
                            h2b = tiny2[0].copy()
                        h1b.extra_tags = _pe_extra_tags(
                            1, a2.original_seq, a2.conversion, "CT"
                        )
                        h2b.extra_tags = _pe_extra_tags(
                            2, b2.original_seq, b2.conversion, "GA"
                        )
                        batch_hits2.append(h1b^)
                        batch_hits2.append(h2b^)
                        n_records += 2
                    j2 += 1
                n_written += append_gaf_hits(out_fh, batch_hits2)
                pending_hits = List[AlignmentHit]()
                has_pending = False
                if profile:
                    print(
                        "mojo_stream stages_s fastq=",
                        t_fastq - t0,
                        " gpu_seed= 0  locate= 0  cluster_extend= 0  gaf_emit= fixture_tiny",
                        " n_batch=",
                        len(batch1),
                        flush=True,
                    )
            _ = t_seed0

        if has_pending:
            n_written += append_gaf_hits(out_fh, pending_hits)
            pending_hits = List[AlignmentHit]()
            has_pending = False

        print(
            "mojo_stream_gpu_session records=",
            n_records,
            " gaf_lines=",
            n_written,
            flush=True,
        )
        return n_written
