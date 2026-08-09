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
from giraffe_gaf_emit import append_gaf_hits, open_gaf_write
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


struct StreamRead(Copyable, Movable):
    var name: String
    var seq: String

    def __init__(out self, name: String, seq: String):
        self.name = name
        self.seq = seq


def _strip_nl(mut s: String):
    while s.byte_length() > 0:
        var last = String(s[byte = s.byte_length() - 1 : s.byte_length()])
        if last == "\n" or last == "\r":
            s = String(s[byte = 0 : s.byte_length() - 1])
        else:
            break


def _open_fastq(path: String) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var gzip = Python.import_module("gzip")
    var low = path.lower()
    if low.endswith(".gz") or low.endswith(".gzip"):
        return gzip.open(path, "rt")
    return builtins.open(path, "r")


def _read_one(fh: PythonObject) raises -> StreamRead:
    var n = String(fh.readline())
    if n.byte_length() == 0:
        return StreamRead("", "")
    var s = String(fh.readline())
    _ = String(fh.readline())
    _ = String(fh.readline())
    _strip_nl(n)
    _strip_nl(s)
    if n.startswith("@"):
        n = String(n[byte = 1 : n.byte_length()])
    var bare = n
    var parts = n.split("_")
    if len(parts) > 0:
        bare = String(parts[0])
    var sp = bare.split(" ")
    if len(sp) > 0:
        bare = String(sp[0])
    return StreamRead(bare, s)


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
    fh1: PythonObject,
    fh2: PythonObject,
    paired: Bool,
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
            fh1,
            fh2,
            paired,
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
    var fh1 = _open_fastq(fq1)
    var paired = fq2.byte_length() > 0
    var fh2 = fh1
    if paired:
        fh2 = _open_fastq(fq2)

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
            fh1,
            fh2,
            paired,
            out_fh,
            dist,
            zipcodes,
            k,
            batch_size,
            profile,
            backend,
            dev,
        )
        fh1.close()
        if paired:
            fh2.close()
        out_fh.close()
        min_idx.close()
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
            var r1 = _read_one(fh1)
            if r1.name.byte_length() == 0:
                break
            batch1.append(r1^)
            if paired:
                var r2 = _read_one(fh2)
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
                # PE tags for MethylCall (primary pair).
                h1.extra_tags = "ri:i:1\tos:Z:" + a.seq + "\trc:Z:CT"
                h2.extra_tags = "ri:i:2\tos:Z:" + b.seq + "\trc:Z:GA"
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
        out_fh.flush()
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

    fh1.close()
    if paired:
        fh2.close()
    out_fh.close()
    min_idx.close()
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
