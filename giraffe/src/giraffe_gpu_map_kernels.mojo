# GPU-native Giraffe science: window-reduce, Q1Q1 HT probe, cluster, gapless.
#
# One DeviceContext owns index residency + all batch kernels for the FASTQ stream.
# Unique-cell HT only (skip IS_POINTER) — same contract as MojoMinIndex.

from std.collections import List
from std.python import Python, PythonObject
from std.sys import has_accelerator

from giraffe_fastq import giraffe_fq_read_header_seq
from giraffe_gaf_emit import append_gaf_hits, flush_emit
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
comptime DIST_CAP = 200


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
            s = String(s[byte = 0 : s.byte_length() - 1])
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
    var bare = n
    var original = s
    var conversion = String("")
    var parts = n.split("_")
    if len(parts) >= 4:
        var conv = String(parts[1])
        if conv == "C2T" or conv == "G2A":
            bare = String(parts[0])
            conversion = conv
            original = String(parts[3])
            var pi = 4
            while pi < len(parts):
                original = original + "_" + String(parts[pi])
                pi += 1
        else:
            bare = String(parts[0])
    elif len(parts) > 0:
        bare = String(parts[0])
    var sp = bare.split(" ")
    if len(sp) > 0:
        bare = String(sp[0])
    return StreamReadGPU(bare, s, original, conversion)


def _read_one_gpu_pipe(mut pipe: FastqPipe) raises -> StreamReadGPU:
    var n = String("")
    var s = String("")
    if not giraffe_fq_read_header_seq(pipe, n, s):
        return StreamReadGPU("", "", "", "")
    return _parse_mg_fastq_fields(n, s)


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
        from std.gpu import block_dim, block_idx, thread_idx
        from std.gpu.host import DeviceContext
        from std.memory import UnsafePointer, memcpy

        def copy_bytes_offset_kernel(
            dst: UnsafePointer[UInt8, MutAnyOrigin],
            src: UnsafePointer[UInt8, MutAnyOrigin],
            dst_off: Int,
            n: Int,
        ):
            """Device-side memcpy into ``dst[dst_off:dst_off+n]`` (no CUDA runtime API)."""
            var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
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
            n_reads: Int,
            stride: Int,
            k_len: Int,
            w_len: Int,
        ):
            var ri = Int(block_idx.x * block_dim.x + thread_idx.x)
            if ri >= n_reads:
                return
            var L = Int(read_lens[ri])
            var base = ri * stride
            var win = k_len + w_len - 1
            var n_out: Int32 = 0
            var row = ri * MAX_OCCS
            if L >= win:
                var next_read_offset = 0
                var last_hash = UInt64(0)
                var last_offset = -1
                var have_last = False
                var window_start = 0
                while window_start <= L - win and Int(n_out) < MAX_OCCS:
                    var best_key = UInt64(0)
                    var best_hash = UInt64(0)
                    var best_off = 0
                    var have_best = False
                    var ok = True
                    var wi = 0
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
                            out_keys[row + Int(n_out)] = best_key
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
            n_reads: Int,
            cell_size: Int,
            cell_count: Int,
        ):
            var ri = Int(block_idx.x * block_dim.x + thread_idx.x)
            if ri >= n_reads:
                return
            var n_occ = Int(occ_n[ri])
            var written: Int32 = 0
            var row = ri * HIT_CAP
            var oi = 0
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
                var cell_offset = Int(h) & (cell_count - 1)
                var attempt = 0
                var found = -1
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
                            var di = 0
                            while di < Int(written):
                                if Int(out_node[row + di]) == node_id and Int(
                                    out_off[row + di]
                                ) == offset:
                                    dup = True
                                    break
                                di += 1
                            if not dup:
                                out_node[row + Int(written)] = Int32(node_id)
                                if is_rev:
                                    out_orient[row + Int(written)] = 1
                                else:
                                    out_orient[row + Int(written)] = 0
                                out_off[row + Int(written)] = Int32(offset)
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
            n_reads: Int,
            do_prune: Int,
        ):
            var ri = Int(block_idx.x * block_dim.x + thread_idx.x)
            if ri >= n_reads:
                return
            var n_in = Int(in_n[ri])
            var base = ri * HIT_CAP
            var obase = ri * MAX_CLUSTER
            if n_in <= 0:
                out_n[ri] = 0
                return
            var best_bucket = Int(in_node[base]) >> 8
            var best_count = 0
            var i = 0
            while i < n_in:
                var b = Int(in_node[base + i]) >> 8
                var c = 0
                var j = 0
                while j < n_in:
                    if (Int(in_node[base + j]) >> 8) == b:
                        c += 1
                    j += 1
                if c > best_count:
                    best_count = c
                    best_bucket = b
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
                out_node[obase + Int(written)] = Int32(nid)
                out_orient[obase + Int(written)] = in_orient[base + i]
                out_off[obase + Int(written)] = in_off[base + i]
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
            q_stride: Int,
            pack_off: UnsafePointer[UInt64, MutAnyOrigin],
            pack_seq: UnsafePointer[UInt8, MutAnyOrigin],
            pack_n: Int,
            seq_bytes: Int,
            seed_node: UnsafePointer[Int32, MutAnyOrigin],
            seed_orient: UnsafePointer[UInt8, MutAnyOrigin],
            seed_off: UnsafePointer[Int32, MutAnyOrigin],
            seed_n: UnsafePointer[Int32, MutAnyOrigin],
            out_node: UnsafePointer[Int32, MutAnyOrigin],
            out_mapq: UnsafePointer[Int32, MutAnyOrigin],
            out_matched: UnsafePointer[Int32, MutAnyOrigin],
            out_valid: UnsafePointer[UInt8, MutAnyOrigin],
            n_reads: Int,
        ):
            var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
            var n_slots = n_reads * MAX_CLUSTER
            if tid >= n_slots:
                return
            var ri = tid // MAX_CLUSTER
            var si = tid % MAX_CLUSTER
            out_valid[tid] = 0
            out_node[tid] = 0
            out_mapq[tid] = 0
            out_matched[tid] = 0
            var n_seeds = Int(seed_n[ri])
            if si >= n_seeds:
                return
            var qlen = Int(read_lens[ri])
            if qlen <= 0:
                return
            var node_id = Int(seed_node[ri * MAX_CLUSTER + si])
            if node_id < 1 or node_id > pack_n:
                return
            var is_rev = seed_orient[ri * MAX_CLUSTER + si] != 0
            var offset = Int(seed_off[ri * MAX_CLUSTER + si])
            var oidx = (node_id - 1) * 2
            var seg_off = Int(pack_off[oidx])
            var seg_len = Int(pack_off[oidx + 1])
            if seg_off < 0 or seg_len <= 0 or seg_off + seg_len > seq_bytes:
                return

            var qbase = ri * q_stride
            var start0 = 0
            var start1 = 0
            var start2 = 0
            if is_rev:
                start0 = seg_len - offset - qlen
                if start0 < 0:
                    start0 = 0
                start1 = offset - qlen + 1
                if start1 < 0:
                    start1 = 0
                start2 = 0
            else:
                start0 = offset
                start1 = offset - qlen + 29
                if start1 < 0:
                    start1 = 0
                start2 = 0

            var best_score = -1
            var best_matched = 0
            var best_mism = 0
            var trial = 0
            while trial < 3:
                var start = start0
                if trial == 1:
                    start = start1
                elif trial == 2:
                    start = start2
                if start < 0 or start >= seg_len:
                    trial += 1
                    continue
                var matched = 0
                var mism = 0
                var qi = 0
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
                var min_need = 29
                if seg_len < min_need:
                    min_need = seg_len
                if matched >= min_need:
                    var score = matched - 2 * mism
                    if score > best_score:
                        best_score = score
                        best_matched = matched
                        best_mism = mism
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
            dev_ht.unsafe_ptr().bitcast[UInt8](),
            meta.ht_host_addr,
            meta.ht_words * 8,
        )
        upload_mmap_to_device(
            ctx,
            dev_off.unsafe_ptr().bitcast[UInt8](),
            meta.off_host_addr,
            meta.off_bytes,
        )
        upload_mmap_to_device(
            ctx,
            dev_seq.unsafe_ptr(),
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
        var do_prune = 0
        if prune_dist:
            do_prune = 1
        comptime BLOCK = 256

        while True:
            var t0 = time.perf_counter()
            var batch1 = List[StreamReadGPU]()
            var batch2 = List[StreamReadGPU]()
            var paired = fq.paired
            var i = 0
            while i < batch_size:
                var r1 = _read_one_gpu_pipe(fq.r1)
                if r1.name.byte_length() == 0:
                    break
                batch1.append(r1^)
                if paired:
                    var r2 = _read_one_gpu_pipe(fq.r2)
                    if r2.name.byte_length() == 0:
                        raise Error("paired FASTQ length mismatch (R2 ended early)")
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

            var t_seed0 = time.perf_counter()
            var per_read_hits = List[List[AlignmentHit]]()
            ri = 0
            while ri < n_reads:
                per_read_hits.append(List[AlignmentHit]())
                ri += 1

            var ran_gpu_batch = False
            if max_len >= meta.k and n_reads > 0:
                ran_gpu_batch = True
                var n_bases = n_reads * max_len
                var host_bases = ctx.enqueue_create_host_buffer[DType.uint8](n_bases)
                var host_lens = ctx.enqueue_create_host_buffer[DType.int32](n_reads)
                ri = 0
                while ri < n_reads:
                    var s = seqs[ri]
                    var L = s.byte_length()
                    host_lens[ri] = Int32(L)
                    var base = ri * max_len
                    var p = 0
                    while p < max_len:
                        if p < L:
                            host_bases[base + p] = UInt8(ord(s[byte = p : p + 1]))
                        else:
                            host_bases[base + p] = 78
                        p += 1
                    ri += 1

                var d_bases = ctx.enqueue_create_buffer[DType.uint8](n_bases)
                var d_codes = ctx.enqueue_create_buffer[DType.uint8](n_bases)
                var d_kf = ctx.enqueue_create_buffer[DType.uint64](n_bases)
                var d_kr = ctx.enqueue_create_buffer[DType.uint64](n_bases)
                var d_hf = ctx.enqueue_create_buffer[DType.uint64](n_bases)
                var d_hr = ctx.enqueue_create_buffer[DType.uint64](n_bases)
                var d_valid = ctx.enqueue_create_buffer[DType.uint8](n_bases)
                var d_lens = ctx.enqueue_create_buffer[DType.int32](n_reads)
                ctx.enqueue_copy(src_buf=host_bases, dst_buf=d_bases)
                ctx.enqueue_copy(src_buf=host_lens, dst_buf=d_lens)

                var grid_bases = (n_bases + BLOCK - 1) // BLOCK
                ctx.enqueue_function[pack_bases_kernel](
                    d_bases.unsafe_ptr(),
                    d_codes.unsafe_ptr(),
                    n_bases,
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
                    n_bases,
                    meta.k,
                    max_len,
                    grid_dim=grid_bases,
                    block_dim=BLOCK,
                )
                var t_seed1 = time.perf_counter()

                var d_occ_keys = ctx.enqueue_create_buffer[DType.uint64](
                    n_reads * MAX_OCCS
                )
                var d_occ_n = ctx.enqueue_create_buffer[DType.int32](n_reads)
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
                    n_reads,
                    max_len,
                    meta.k,
                    meta.w,
                    grid_dim=grid_reads,
                    block_dim=BLOCK,
                )

                var d_hit_node = ctx.enqueue_create_buffer[DType.int32](
                    n_reads * HIT_CAP
                )
                var d_hit_orient = ctx.enqueue_create_buffer[DType.uint8](
                    n_reads * HIT_CAP
                )
                var d_hit_off = ctx.enqueue_create_buffer[DType.int32](
                    n_reads * HIT_CAP
                )
                var d_hit_n = ctx.enqueue_create_buffer[DType.int32](n_reads)
                ctx.enqueue_function[ht_probe_kernel](
                    dev_ht.unsafe_ptr(),
                    d_occ_keys.unsafe_ptr(),
                    d_occ_n.unsafe_ptr(),
                    d_hit_node.unsafe_ptr(),
                    d_hit_orient.unsafe_ptr(),
                    d_hit_off.unsafe_ptr(),
                    d_hit_n.unsafe_ptr(),
                    n_reads,
                    meta.cell_size,
                    meta.cell_count,
                    grid_dim=grid_reads,
                    block_dim=BLOCK,
                )
                var t_locate = time.perf_counter()

                var d_cl_node = ctx.enqueue_create_buffer[DType.int32](
                    n_reads * MAX_CLUSTER
                )
                var d_cl_orient = ctx.enqueue_create_buffer[DType.uint8](
                    n_reads * MAX_CLUSTER
                )
                var d_cl_off = ctx.enqueue_create_buffer[DType.int32](
                    n_reads * MAX_CLUSTER
                )
                var d_cl_n = ctx.enqueue_create_buffer[DType.int32](n_reads)
                ctx.enqueue_function[cluster_kernel](
                    d_hit_node.unsafe_ptr(),
                    d_hit_orient.unsafe_ptr(),
                    d_hit_off.unsafe_ptr(),
                    d_hit_n.unsafe_ptr(),
                    d_cl_node.unsafe_ptr(),
                    d_cl_orient.unsafe_ptr(),
                    d_cl_off.unsafe_ptr(),
                    d_cl_n.unsafe_ptr(),
                    n_reads,
                    do_prune,
                    grid_dim=grid_reads,
                    block_dim=BLOCK,
                )

                var n_slots = n_reads * MAX_CLUSTER
                var d_out_node = ctx.enqueue_create_buffer[DType.int32](n_slots)
                var d_out_mapq = ctx.enqueue_create_buffer[DType.int32](n_slots)
                var d_out_matched = ctx.enqueue_create_buffer[DType.int32](n_slots)
                var d_out_valid = ctx.enqueue_create_buffer[DType.uint8](n_slots)
                var grid_slots = (n_slots + BLOCK - 1) // BLOCK
                ctx.enqueue_function[gapless_kernel](
                    d_bases.unsafe_ptr(),
                    d_lens.unsafe_ptr(),
                    max_len,
                    dev_off.unsafe_ptr(),
                    dev_seq.unsafe_ptr(),
                    meta.pack_n,
                    meta.seq_bytes,
                    d_cl_node.unsafe_ptr(),
                    d_cl_orient.unsafe_ptr(),
                    d_cl_off.unsafe_ptr(),
                    d_cl_n.unsafe_ptr(),
                    d_out_node.unsafe_ptr(),
                    d_out_mapq.unsafe_ptr(),
                    d_out_matched.unsafe_ptr(),
                    d_out_valid.unsafe_ptr(),
                    n_reads,
                    grid_dim=grid_slots,
                    block_dim=BLOCK,
                )

                var h_node = ctx.enqueue_create_host_buffer[DType.int32](n_slots)
                var h_mapq = ctx.enqueue_create_host_buffer[DType.int32](n_slots)
                var h_matched = ctx.enqueue_create_host_buffer[DType.int32](n_slots)
                var h_valid = ctx.enqueue_create_host_buffer[DType.uint8](n_slots)
                ctx.enqueue_copy(src_buf=d_out_node, dst_buf=h_node)
                ctx.enqueue_copy(src_buf=d_out_mapq, dst_buf=h_mapq)
                ctx.enqueue_copy(src_buf=d_out_matched, dst_buf=h_matched)
                ctx.enqueue_copy(src_buf=d_out_valid, dst_buf=h_valid)
                ctx.synchronize()
                var t_extend = time.perf_counter()

                ri = 0
                while ri < n_reads:
                    var qname = String("")
                    if ri < len(batch1):
                        qname = batch1[ri].name
                    else:
                        qname = batch2[ri - len(batch1)].name
                    var qlen = seqs[ri].byte_length()
                    var row = List[AlignmentHit]()
                    var si = 0
                    while si < MAX_CLUSTER and len(row) < 2:
                        var tid = ri * MAX_CLUSTER + si
                        if h_valid[tid] != 0:
                            var matched = Int(h_matched[tid])
                            var mq = Int(h_mapq[tid])
                            var cs = "cs:Z::" + String(matched)
                            if mq >= 60:
                                cs = "cs:Z::" + String(qlen)
                            row.append(
                                AlignmentHit(
                                    qname,
                                    ">" + String(Int(h_node[tid])),
                                    qlen,
                                    mq,
                                    cs,
                                )
                            )
                        si += 1
                    if len(row) == 0:
                        row = _fixture_extend_tiny(pack, qname, seqs[ri], meta.k)
                    per_read_hits[ri] = row^
                    ri += 1

                var batch_hits = List[AlignmentHit]()
                var j = 0
                while j < len(batch1):
                    var a = batch1[j].copy()
                    var h1s = per_read_hits[j].copy()
                    if not paired:
                        n_records += 1
                        for h in h1s:
                            batch_hits.append(h.copy())
                    else:
                        var b = batch2[j].copy()
                        var h2s = per_read_hits[len(batch1) + j].copy()
                        var h1 = AlignmentHit(
                            a.name, "*", a.seq.byte_length(), 0, "cs:Z:*"
                        )
                        var h2 = AlignmentHit(
                            b.name, "*", b.seq.byte_length(), 0, "cs:Z:*"
                        )
                        if len(h1s) > 0:
                            h1 = h1s[0].copy()
                        if len(h2s) > 0:
                            h2 = h2s[0].copy()
                        h1.extra_tags = _pe_extra_tags(
                            1, a.original_seq, a.conversion, "CT"
                        )
                        h2.extra_tags = _pe_extra_tags(
                            2, b.original_seq, b.conversion, "GA"
                        )
                        batch_hits.append(h1^)
                        if len(h1s) > 1:
                            batch_hits.append(h1s[1].copy())
                        batch_hits.append(h2^)
                        if len(h2s) > 1:
                            batch_hits.append(h2s[1].copy())
                        n_records += 2
                    j += 1

                n_written += append_gaf_hits(out_fh, batch_hits)
                flush_emit(out_fh)
                var t_emit = time.perf_counter()
                if profile:
                    print(
                        "mojo_stream stages_s fastq=",
                        t_fastq - t0,
                        " gpu_seed=",
                        t_seed1 - t_seed0,
                        " locate=",
                        t_locate - t_seed1,
                        " cluster_extend=",
                        t_extend - t_locate,
                        " gaf_emit=",
                        t_emit - t_extend,
                        " n_batch=",
                        len(batch1),
                        flush=True,
                    )

            if not ran_gpu_batch:
                # Short-read toys (len < k): fixture exact-match on tiny packs only.
                ri = 0
                while ri < n_reads:
                    var qname2 = String("")
                    if ri < len(batch1):
                        qname2 = batch1[ri].name
                    else:
                        qname2 = batch2[ri - len(batch1)].name
                    per_read_hits[ri] = _fixture_extend_tiny(
                        pack, qname2, seqs[ri], meta.k
                    )
                    ri += 1
                var batch_hits2 = List[AlignmentHit]()
                var j2 = 0
                while j2 < len(batch1):
                    var a2 = batch1[j2].copy()
                    var h1s2 = per_read_hits[j2].copy()
                    if not paired:
                        n_records += 1
                        for h in h1s2:
                            batch_hits2.append(h.copy())
                    else:
                        var b2 = batch2[j2].copy()
                        var h2s2 = per_read_hits[len(batch1) + j2].copy()
                        var h1b = AlignmentHit(
                            a2.name, "*", a2.seq.byte_length(), 0, "cs:Z:*"
                        )
                        var h2b = AlignmentHit(
                            b2.name, "*", b2.seq.byte_length(), 0, "cs:Z:*"
                        )
                        if len(h1s2) > 0:
                            h1b = h1s2[0].copy()
                        if len(h2s2) > 0:
                            h2b = h2s2[0].copy()
                        h1b.extra_tags = _pe_extra_tags(
                            1, a2.original_seq, a2.conversion, "CT"
                        )
                        h2b.extra_tags = _pe_extra_tags(
                            2, b2.original_seq, b2.conversion, "GA"
                        )
                        batch_hits2.append(h1b^)
                        if len(h1s2) > 1:
                            batch_hits2.append(h1s2[1].copy())
                        batch_hits2.append(h2b^)
                        if len(h2s2) > 1:
                            batch_hits2.append(h2s2[1].copy())
                        n_records += 2
                    j2 += 1
                n_written += append_gaf_hits(out_fh, batch_hits2)
                flush_emit(out_fh)
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

        print(
            "mojo_stream_gpu_session records=",
            n_records,
            " gaf_lines=",
            n_written,
            flush=True,
        )
        return n_written
