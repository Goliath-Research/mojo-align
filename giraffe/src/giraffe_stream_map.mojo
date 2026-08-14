# Production GBZ stream map — Mojo hot path (no Python quartet_map / CuPy loop).
#
# NVIDIA/AMD GPU session (DeviceContext):
#   Upload .min HT + dense pack once -> per batch device window-reduce,
#   Q1Q1 HT probe, cluster, gapless -> D2H hits -> host GAF emit.
#   Banner: seed_backend=devicecontext-…+gpu_ht+gpu_gapless+mojo_stream
#
# CPU / GPU_REQUIRE=0 toys: host Mojo locate/cluster/gapless over mmap.

from std.collections import List
from std.python import Python, PythonObject
from std.sys import has_accelerator

from giraffe_device import require_device_or_raise
from giraffe_dist import cluster_seed_hits
from giraffe_fastq import giraffe_fq_close, giraffe_fq_open, giraffe_fq_read_header_seq
from giraffe_gaf_emit import append_gaf_hits, close_emit, emit_footer_log, open_gaf_write
from giraffe_sam_emit import close_qc_offsets
from giraffe_gapless import gapless_extend_with_pack
from giraffe_gpu_kernels import (
    kernel_target_label,
    probe_device_context,
)
from giraffe_gpu_map_kernels import gpu_native_stream_loop
from giraffe_hit import AlignmentHit
from giraffe_min_index import MojoMinIndex
from giraffe_minzip import locate_batch_hits_with_index
from giraffe_pack import DensePack
from linear_fastq import FastqPairStream, FastqPipe


struct StreamRead(Copyable, Movable):
    """One FASTQ record.

    ``seq`` is the converted body used for mapping. ``original_seq`` /
    ``conversion`` come from methylGrapher headers
    ``{qname}_{C2T|G2A}_{shard}_{original}`` and feed MethylCall ``os:Z`` /
    ``rc:Z`` (must not be the converted body).
    """

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
    if os.byte_length() == 0:
        os = String("")
    var rc = _rc_from_conversion(conversion, fallback_rc)
    return "ri:i:" + String(ri) + "\tos:Z:" + os + "\trc:Z:" + rc


def _parse_mg_fastq_fields(n: String, s: String) -> StreamRead:
    """Split methylGrapher converted-FASTQ header; fall back to body as os."""
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
    return StreamRead(bare, s, original, conversion)


def _read_one_pipe(mut pipe: FastqPipe) raises -> StreamRead:
    """One record via linear_fastq FastqPipe (pigz fd + libc read)."""
    var n = String("")
    var s = String("")
    if not giraffe_fq_read_header_seq(pipe, n, s):
        return StreamRead("", "", "", "")
    return _parse_mg_fastq_fields(n, s)


def _batch_size() raises -> Int:
    var os_mod = Python.import_module("os")
    var raw = String(os_mod.environ.get("METHYLGRAPHER_MOJO_READ_BATCH", "8192"))
    try:
        var n = Int(raw)
        if n < 1:
            return 8192
        return n
    except e:
        return 8192


def _stage_enabled() raises -> Bool:
    var os_mod = Python.import_module("os")
    var raw = String(os_mod.environ.get("METHYLGRAPHER_PROFILE_STAGES", "")).lower()
    return raw == "1" or raw == "true" or raw == "yes"


def _science_locate_batch(
    mut idx: MojoMinIndex, seqs: List[String], device: String
) raises -> List[List[String]]:
    """Mojo-native minimizer locate against an already-open mmap index."""
    return locate_batch_hits_with_index(idx, device, seqs, 24)


def _fixture_extend_small(
    pack: DensePack, qname: String, seq: String, k: Int
) raises -> List[AlignmentHit]:
    """Exact/k-mer fallback for toy packs only (mirrors quartet_map._fixture_extend)."""
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
    # Short k-mer majority over tiny pack.
    var best_sid = String("")
    var best_n = 0
    for sid in ids:
        var s = pack.get_sid(sid)
        if s.byte_length() < k:
            continue
        var count = 0
        var i = 0
        while i + k <= qlen:
            var mer = String(seq[byte = i : i + k])
            # substring check
            var j = 0
            var found = False
            while j + k <= s.byte_length():
                if String(s[byte = j : j + k]) == mer:
                    found = True
                    break
                j += 1
            if found:
                count += 1
            i += 1
        if count > best_n:
            best_n = count
            best_sid = sid.copy()
    if best_n > 0 and best_sid.byte_length() > 0:
        var mq = 20
        if best_n >= 2:
            mq = 40
        out.append(
            AlignmentHit(qname, ">" + best_sid, qlen, mq, "cs:Z::" + String(qlen))
        )
    return out^


def _map_one(
    pack: DensePack,
    qname: String,
    seq: String,
    seeds: List[String],
    dist: String,
    zipcodes: String,
    k: Int,
) raises -> List[AlignmentHit]:
    var clustered = cluster_seed_hits(seeds, dist, zipcodes)
    if len(clustered) > 16:
        var capped = List[String]()
        var i = 0
        while i < 16:
            capped.append(clustered[i].copy())
            i += 1
        clustered = capped^
    var hits = gapless_extend_with_pack(pack, qname, seq, clustered)
    if len(hits) > 0:
        return hits^
    return _fixture_extend_small(pack, qname, seq, k)

def _is_gpu_device(device: String) -> Bool:
    var d = device.lower()
    return (
        d == "nvidia"
        or d == "cuda"
        or d == "amd"
        or d == "hip"
        or d == "rocm"
    )



def _map_stream_gpu_session(
    mut pack: DensePack,
    mut min_idx: MojoMinIndex,
    mut fq: FastqPairStream,
    out_fh: PythonObject,
    dist: String,
    zipcodes: String,
    k: Int,
    batch_size: Int,
    profile: Bool,
    backend: String,
    dev: String,
) raises -> Int:
    """One DeviceContext: resident HT+pack + GPU seed/locate/cluster/gapless."""
    _ = zipcodes
    _ = k
    if not pack.contiguous or pack._use_py:
        raise Error(
            "GPU-native stream map requires contiguous dense-v1 mojo_segments pack"
        )
    comptime if has_accelerator():
        print(
            "mojo_min DeviceContext session device=",
            dev,
            " target=",
            kernel_target_label(dev),
            " backend=",
            backend,
            " k=",
            min_idx.k,
            " w=",
            min_idx.w,
            flush=True,
        )
        return gpu_native_stream_loop(
            pack,
            min_idx,
            fq,
            out_fh,
            dist,
            batch_size,
            profile,
            backend,
            dev,
        )
    raise Error(
        "mojo_stream_map GPU session requested but Mojo build has no accelerator"
    )


def map_fastq_stream_to_gaf(
    pack_dir: String,
    fq1: String,
    out_gaf: String,
    fq2: String,
    dist: String,
    min_path: String,
    zipcodes: String,
    k: Int,
    device: String,
) raises -> Int:
    """Stream FASTQ -> Mojo seed/cluster/extend -> GAF. Returns record count."""
    var dev = require_device_or_raise(device)
    var backend = probe_device_context(dev)
    var os_mod = Python.import_module("os")
    os_mod.environ["METHYLGRAPHER_GIRAFFE_DEVICE"] = dev
    os_mod.environ["METHYLGRAPHER_ALIGN_DEVICE"] = dev
    os_mod.environ["METHYLGRAPHER_LAST_GPU_BACKEND"] = backend

    # Fail closed: never claim GPU Align while falling back to CuPy/host-nvidia.
    if (
        backend.find("host-fallback") >= 0
        or backend.find("host-nvidia") >= 0
        or backend.find("cupy") >= 0
    ):
        var require = String(os_mod.environ.get("METHYLGRAPHER_GPU_REQUIRE", "")).lower()
        if (
            (require == "" or require == "1" or require == "true" or require == "yes")
            and (dev == "nvidia" or dev == "amd")
        ):
            raise Error(
                "mojo_stream_map refusing non-DeviceContext backend="
                + backend
                + " for device="
                + dev
                + " (CuPy/host-nvidia-fallback is not the production path)"
            )

    var batch_size = _batch_size()
    var profile = _stage_enabled()
    print(
        "mojo_stream_map device=",
        dev,
        " target=",
        kernel_target_label(dev),
        " backend=",
        backend,
        " batch=",
        batch_size,
        " pack=",
        pack_dir,
        flush=True,
    )

    var pack = DensePack(pack_dir)
    var min_idx = MojoMinIndex(min_path)
    print(
        "mojo_stream_map min_mmap k=",
        min_idx.k,
        " w=",
        min_idx.w,
        " cells=",
        min_idx.cell_count,
        flush=True,
    )
    var out_fh = open_gaf_write(out_gaf)
    var fq = giraffe_fq_open(fq1, fq2)
    var paired = fq.paired
    print("mojo_stream_map fastq=linear_fastq_pipe", flush=True)

    var gpu_session = (
        _is_gpu_device(dev)
        and (
            backend.startswith("devicecontext-cuda")
            or backend.startswith("devicecontext-hip")
        )
    )
    var require = String(os_mod.environ.get("METHYLGRAPHER_GPU_REQUIRE", "")).lower()
    var gpu_required = (
        require == ""
        or require == "1"
        or require == "true"
        or require == "yes"
    )
    if _is_gpu_device(dev) and gpu_required and not gpu_session:
        raise Error(
            "mojo_stream_map GPU_REQUIRE: need DeviceContext CUDA/HIP backend, got "
            + backend
            + " (host mmap locate/extend is not the production GPU path)"
        )
    if gpu_session:
        var n_wrt = _map_stream_gpu_session(
            pack,
            min_idx,
            fq,
            out_fh,
            dist,
            zipcodes,
            k,
            batch_size,
            profile,
            backend,
            dev,
        )
        giraffe_fq_close(fq)
        close_emit(out_fh)
        min_idx.close()
        emit_footer_log()
        close_qc_offsets()
        print(
            "mojo_stream_map done gaf_lines=",
            n_wrt,
            " -> ",
            out_gaf,
            flush=True,
        )
        return n_wrt

    var n_records = 0
    var n_written = 0
    var time = Python.import_module("time")

    while True:
        var t0 = time.perf_counter()
        var batch1 = List[StreamRead]()
        var batch2 = List[StreamRead]()
        var i = 0
        while i < batch_size:
            var r1 = _read_one_pipe(fq.r1)
            if r1.name.byte_length() == 0:
                break
            batch1.append(r1^)
            if paired:
                var r2 = _read_one_pipe(fq.r2)
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

        # Science locate: DeviceContext minimizer hashing + Mojo mmap HT probe.
        var t_gpu = time.perf_counter()
        var seed_hits = _science_locate_batch(min_idx, seqs, dev)
        var t_locate = time.perf_counter()
        _ = k

        var batch_hits = List[AlignmentHit]()
        var j = 0
        while j < len(batch1):
            var a = batch1[j].copy()
            var seeds_a = seed_hits[j].copy()
            var h1s = _map_one(pack, a.name, a.seq, seeds_a, dist, zipcodes, k)
            if not paired:
                n_records += 1
                for h in h1s:
                    batch_hits.append(h.copy())
            else:
                var b = batch2[j].copy()
                var seeds_b = seed_hits[len(batch1) + j].copy()
                var h2s = _map_one(pack, b.name, b.seq, seeds_b, dist, zipcodes, k)
                var h1 = AlignmentHit(a.name, "*", a.seq.byte_length(), 0, "cs:Z:*")
                var h2 = AlignmentHit(b.name, "*", b.seq.byte_length(), 0, "cs:Z:*")
                if len(h1s) > 0:
                    h1 = h1s[0].copy()
                if len(h2s) > 0:
                    h2 = h2s[0].copy()
                # PE tags for MethylCall: os:Z must be original (not converted body).
                h1.extra_tags = _pe_extra_tags(1, a.original_seq, a.conversion, "CT")
                h2.extra_tags = _pe_extra_tags(2, b.original_seq, b.conversion, "GA")
                # Primary before secondary (vg giraffe -M 2 / MethylCall convention).
                batch_hits.append(h1^)
                if len(h1s) > 1:
                    batch_hits.append(h1s[1].copy())
                batch_hits.append(h2^)
                if len(h2s) > 1:
                    batch_hits.append(h2s[1].copy())
                n_records += 2
            j += 1
        var t_extend = time.perf_counter()

        n_written += append_gaf_hits(out_fh, batch_hits)
        var t_emit = time.perf_counter()

        if profile:
            print(
                "mojo_stream stages_s fastq=",
                t_fastq - t0,
                " gpu_seed=",
                t_gpu - t_fastq,
                " locate=",
                t_locate - t_gpu,
                " cluster_extend=",
                t_extend - t_locate,
                " gaf_emit=",
                t_emit - t_extend,
                " n_batch=",
                len(batch1),
            )

    giraffe_fq_close(fq)
    close_emit(out_fh)
    min_idx.close()
    emit_footer_log()
    close_qc_offsets()
    print(
        "mojo_stream_map done records=",
        n_records,
        " gaf_lines=",
        n_written,
        " -> ",
        out_gaf,
        flush=True,
    )
    return n_records
