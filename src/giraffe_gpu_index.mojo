# Device-resident Giraffe index metadata + H2D bootstrap helpers.
#
# DeviceBuffer allocation stays in the stream-map GPU session (lifetime).
# This module builds upload metadata from host mmaps and fills device pointers
# via chunked ``cudaMemcpy`` (``engine.gpu_h2d``).

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


def gpu_index_resident_gib(meta: GpuIndexMeta) raises -> Float64:
    var bytes_total = meta.ht_words * 8 + meta.off_bytes + meta.seq_bytes
    return Float64(bytes_total) / Float64(1024 * 1024 * 1024)


def h2d_fill(dev_ptr: Int, host_ptr: Int, nbytes: Int) raises:
    """Chunked host→device memcpy into an already-allocated device pointer."""
    if nbytes <= 0:
        return
    if dev_ptr == 0 or host_ptr == 0:
        raise Error("h2d_fill: null pointer")
    var h2d = Python.import_module("engine.gpu_h2d")
    h2d.memcpy_htod_chunked(dev_ptr, host_ptr, nbytes)


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
