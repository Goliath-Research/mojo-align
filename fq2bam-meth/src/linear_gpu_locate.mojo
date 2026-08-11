# GPU locate for dense-v1 linear WGBS (NVIDIA DeviceContext).
#
# One session: upload kmers.bin + offsets.bin once (~224 MiB), then per batch
# 2-bit encode (strided) + binary-search locate on device. Host walks rare
# postings from mmap and gapless-verifies — no String k-mer decode on hot path.

from std.collections import List
from std.python import Python, PythonObject
from std.sys import has_accelerator

from gpu_device import select_device
from gpu_kernels import _device_api, kernel_target_label, probe_device_context
from linear_extend import hit_to_sam_line, pair_hits, LinearHit
from linear_index import LinearIndex, SeedVote
from utility import open_text_write


struct FastqRec(Copyable, Movable):
    var name: String
    var seq: String
    var qual: String

    def __init__(out self, name: String, seq: String, qual: String = "*"):
        self.name = name
        self.seq = seq
        self.qual = qual


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
    var out = String("")
    var i = 0
    var n = seq.byte_length()
    while i < n:
        var ch = String(seq[byte = i : i + 1])
        var u = ch.upper()
        if mode == "C2T" and u == "C":
            out += "T"
        elif mode == "G2A" and u == "G":
            out += "A"
        else:
            out += ch
        i += 1
    return out^


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


def _write_sam_header(fh: PythonObject, index: LinearIndex) raises:
    fh.write("@HD\tVN:1.6\tSO:unsorted\n")
    var n = index.contig_count()
    var ci = 0
    while ci < n:
        fh.write(
            "@SQ\tSN:"
            + index.contig_name(ci)
            + "\tLN:"
            + String(index.contig_length(ci))
            + "\n"
        )
        ci += 1
    fh.write("@PG\tID:MojoFq2bamMeth\tPN:MojoFq2bamMeth\tVN:0.1.0-mojo\n")
    try:
        fh.flush()
    except:
        pass


def _hit_from_vote(
    index: LinearIndex, name: String, seq: String, vote: SeedVote
) raises -> LinearHit:
    var qlen = seq.byte_length()
    if vote.cid < 0 or vote.votes <= 0:
        return LinearHit(name, 4, "*", 0, 0, "*", seq, "*")
    var window = index.contig_window_id(vote.cid, vote.start, qlen)
    if window.byte_length() == qlen and window == seq:
        var mq = 20
        if vote.votes >= 3:
            mq = 40
        if vote.votes >= 5:
            mq = 60
        return LinearHit(
            name,
            0,
            index.contig_name(vote.cid),
            vote.start + 1,
            mq,
            String(qlen) + "M",
            seq,
            "*",
        )
    return LinearHit(name, 4, "*", 0, 0, "*", seq, "*")


def map_fastq_dense_gpu_locate(
    mut index: LinearIndex,
    fq1: String,
    out_sam: String,
    device: String,
    fq2: String = "",
    bs_r1: String = "",
    bs_r2: String = "",
) raises -> Int:
    """Stream PE/SE FASTQ with one DeviceContext locate session (dense-v1)."""
    if not index.dense:
        raise Error("map_fastq_dense_gpu_locate requires dense-v1 index")
    if index.kmers_addr == 0 or index.offsets_addr == 0:
        raise Error("map_fastq_dense_gpu_locate: null mmap addresses")

    var resolved = select_device(device)
    var backend = probe_device_context(resolved)
    var target = kernel_target_label(resolved)
    print(
        "MojoLinear GPU-locate device=",
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
            "MojoLinear GPU-locate needs DeviceContext cuda/hip, got " + backend
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
        ) raises:
            if nbytes <= 0:
                return
            if host_addr == 0:
                raise Error("upload_mmap_to_device: null host mmap address")
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

        def encode_strided_kernel(
            codes: UnsafePointer[UInt8, MutAnyOrigin],
            keys: UnsafePointer[UInt64, MutAnyOrigin],
            q_offs: UnsafePointer[UInt32, MutAnyOrigin],
            read_ids: UnsafePointer[UInt32, MutAnyOrigin],
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
            read_ids[idx] = UInt32(rid)
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

        var api = _device_api(resolved)
        var ctx = DeviceContext(api=api)
        var k_len = index.k
        var n_table = index.n_keys
        var n_off = n_table + 1
        var kmers_bytes = n_table * 8
        var offsets_bytes = n_off * 8
        if index.kmers_size < kmers_bytes or index.offsets_size < offsets_bytes:
            raise Error("MojoLinear GPU-locate: mmap smaller than n_keys tables")
        print(
            "MojoLinear GPU-locate upload kmers_bytes=",
            kmers_bytes,
            " offsets_bytes=",
            offsets_bytes,
            " n_keys=",
            n_table,
        )
        # Typed u64 device tables (avoid bitcast lifetime hazards).
        var dev_kmers = ctx.enqueue_create_buffer[DType.uint64](n_table)
        var dev_offsets = ctx.enqueue_create_buffer[DType.uint64](n_off)
        upload_mmap_to_device(
            ctx,
            dev_kmers.unsafe_ptr().bitcast[UInt8](),
            index.kmers_addr,
            kmers_bytes,
        )
        upload_mmap_to_device(
            ctx,
            dev_offsets.unsafe_ptr().bitcast[UInt8](),
            index.offsets_addr,
            offsets_bytes,
        )
        print("MojoLinear GPU-locate index resident ok")

        var fh = open_text_write(out_sam)
        _write_sam_header(fh, index)
        var fh1 = _open_fastq(fq1)
        var paired = fq2.byte_length() > 0
        var fh2 = fh1
        if paired:
            fh2 = _open_fastq(fq2)

        var batch_size = _env_int("METHYLGRAPHER_LINEAR_READ_BATCH", 2048)
        var seed_stride = _env_int("METHYLGRAPHER_LINEAR_SEED_STRIDE", 5)
        var max_occ = index.max_occ()
        var n_mapped = 0
        var n_reads = 0
        var n_batches = 0
        comptime BLOCK = 256

        print(
            "MojoLinear GPU-locate map start batch=",
            batch_size,
            " stride=",
            seed_stride,
            " max_occ=",
            max_occ,
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
            var dev_lens = ctx.enqueue_create_buffer[DType.uint32](n_seq)
            var dev_keys = ctx.enqueue_create_buffer[DType.uint64](n_slots)
            var dev_qoff = ctx.enqueue_create_buffer[DType.uint32](n_slots)
            var dev_rid = ctx.enqueue_create_buffer[DType.uint32](n_slots)
            var dev_start = ctx.enqueue_create_buffer[DType.uint64](n_slots)
            var dev_end = ctx.enqueue_create_buffer[DType.uint64](n_slots)

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
            ctx.enqueue_function[encode_strided_kernel](
                dev_codes.unsafe_ptr(),
                dev_keys.unsafe_ptr(),
                dev_qoff.unsafe_ptr(),
                dev_rid.unsafe_ptr(),
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

            var host_qoff = ctx.enqueue_create_host_buffer[DType.uint32](n_slots)
            var host_rid = ctx.enqueue_create_host_buffer[DType.uint32](n_slots)
            var host_start = ctx.enqueue_create_host_buffer[DType.uint64](n_slots)
            var host_end = ctx.enqueue_create_host_buffer[DType.uint64](n_slots)
            ctx.enqueue_copy(src_buf=dev_qoff, dst_buf=host_qoff)
            ctx.enqueue_copy(src_buf=dev_rid, dst_buf=host_rid)
            ctx.enqueue_copy(src_buf=dev_start, dst_buf=host_start)
            ctx.enqueue_copy(src_buf=dev_end, dst_buf=host_end)
            ctx.synchronize()

            # Per-read vote from rare posting ranges (host mmap postings).
            var n1 = len(batch1)
            var j = 0
            while j < n1:
                var starts1 = List[Int]()
                var ends1 = List[Int]()
                var qoffs1 = List[Int]()
                var slot = 0
                while slot < slots_per:
                    var idx1 = j * slots_per + slot
                    var a = Int(host_start[idx1])
                    var b = Int(host_end[idx1])
                    if b > a:
                        starts1.append(a)
                        ends1.append(b)
                        qoffs1.append(Int(host_qoff[idx1]))
                    slot += 1
                var vote1 = index.vote_dense_from_ranges(starts1, ends1, qoffs1)
                var h1 = _hit_from_vote(index, batch1[j].name, batch1[j].seq, vote1)
                h1.qual = batch1[j].qual

                if not paired:
                    if h1.contig != "*":
                        n_mapped += 1
                    fh.write(hit_to_sam_line(h1) + "\n")
                    n_reads += 1
                else:
                    var r2i = n1 + j
                    var starts2 = List[Int]()
                    var ends2 = List[Int]()
                    var qoffs2 = List[Int]()
                    var slot2 = 0
                    while slot2 < slots_per:
                        var idx2 = r2i * slots_per + slot2
                        var a2 = Int(host_start[idx2])
                        var b2 = Int(host_end[idx2])
                        if b2 > a2:
                            starts2.append(a2)
                            ends2.append(b2)
                            qoffs2.append(Int(host_qoff[idx2]))
                        slot2 += 1
                    var vote2 = index.vote_dense_from_ranges(
                        starts2, ends2, qoffs2
                    )
                    var h2 = _hit_from_vote(
                        index, batch2[j].name, batch2[j].seq, vote2
                    )
                    h2.qual = batch2[j].qual
                    var paired_hits = pair_hits(h1, h2)
                    var p1 = paired_hits.r1.copy()
                    var p2 = paired_hits.r2.copy()
                    if p1.contig != "*":
                        n_mapped += 1
                    if p2.contig != "*":
                        n_mapped += 1
                    fh.write(hit_to_sam_line(p1) + "\n")
                    fh.write(hit_to_sam_line(p2) + "\n")
                    n_reads += 2
                j += 1

            n_batches += 1
            if n_batches == 1 or n_batches % 5 == 0:
                print(
                    "MojoLinear GPU-locate progress batches=",
                    n_batches,
                    " reads=",
                    n_reads,
                    " mapped=",
                    n_mapped,
                    " slots=",
                    n_slots,
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
            " backend=gpu-locate",
        )
        return n_mapped
