# Device-resident Giraffe index metadata helpers.
#
# DeviceBuffer allocation + H2D live in the stream-map GPU session
# (``giraffe_gpu_map_kernels.upload_mmap_to_device`` via Mojo DeviceContext).
# This module only builds upload metadata from host mmaps and runs HBM preflight.

from std.python import Python

from giraffe_min_index import MojoMinIndex
from giraffe_pack import DensePack


struct GpuIndexMeta(Copyable, Movable):
    """Host-side metadata for a device-resident GBZ index pair."""

    var k: Int
    var w: Int
    var cell_size: Int
    var cell_count: Int
    var payload_size: Int
    var ht_words: Int
    var ht_host_addr: Int
    var pack_n: Int
    var off_host_addr: Int
    var off_bytes: Int
    var seq_host_addr: Int
    var seq_bytes: Int
    var contiguous: Bool

    def __init__(
        out self,
        k: Int,
        w: Int,
        cell_size: Int,
        cell_count: Int,
        payload_size: Int,
        ht_words: Int,
        ht_host_addr: Int,
        pack_n: Int,
        off_host_addr: Int,
        off_bytes: Int,
        seq_host_addr: Int,
        seq_bytes: Int,
        contiguous: Bool,
    ):
        self.k = k
        self.w = w
        self.cell_size = cell_size
        self.cell_count = cell_count
        self.payload_size = payload_size
        self.ht_words = ht_words
        self.ht_host_addr = ht_host_addr
        self.pack_n = pack_n
        self.off_host_addr = off_host_addr
        self.off_bytes = off_bytes
        self.seq_host_addr = seq_host_addr
        self.seq_bytes = seq_bytes
        self.contiguous = contiguous


def gpu_index_meta_from(
    min_idx: MojoMinIndex, pack: DensePack
) raises -> GpuIndexMeta:
    """Build upload metadata from open host mmap indexes."""
    if not pack.contiguous or pack._use_py:
        raise Error(
            "GpuGiraffeIndex requires contiguous dense-v1 pack (mojo_mmap)"
        )
    if min_idx.addr == 0 or pack.off_addr == 0 or pack.seq_addr == 0:
        raise Error("GpuGiraffeIndex: null mmap address")
    var ht_words = min_idx.cell_count * min_idx.cell_size
    var ht_host = min_idx.addr + min_idx.ht_data_offset
    return GpuIndexMeta(
        min_idx.k,
        min_idx.w,
        min_idx.cell_size,
        min_idx.cell_count,
        min_idx.payload_size,
        ht_words,
        ht_host,
        pack.n,
        pack.off_addr,
        pack.off_size,
        pack.seq_addr,
        pack.seq_size,
        True,
    )


def gpu_index_science_bytes(meta: GpuIndexMeta) raises -> Int:
    """HT + pack offsets + sequence bytes (device slabs, no Mojo overhead)."""
    return meta.ht_words * 8 + meta.off_bytes + meta.seq_bytes


def gpu_index_resident_gib(meta: GpuIndexMeta) raises -> Float64:
    return Float64(gpu_index_science_bytes(meta)) / Float64(1024 * 1024 * 1024)


def log_gpu_index_resident(meta: GpuIndexMeta) raises:
    print(
        "gpu_index_resident_gib=",
        gpu_index_resident_gib(meta),
        " ht_words=",
        meta.ht_words,
        " pack_n=",
        meta.pack_n,
        " seq_bytes=",
        meta.seq_bytes,
        flush=True,
    )


def require_gpu_index_capacity(meta: GpuIndexMeta, device: String) raises:
    """Query free HBM (nvidia-smi / rocm-smi) before DeviceContext index upload.

    Raises a clear capacity Error instead of a late driver OOM during
    ``enqueue_create_buffer`` for the multi‑GiB HT.
    """
    var mem = Python.import_module("engine.gpu_mem")
    try:
        _ = mem.require_index_capacity(
            gpu_index_science_bytes(meta), device=device
        )
    except e:
        raise Error(String(e))
