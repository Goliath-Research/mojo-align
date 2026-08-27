# Centroid stream kernels (scatter-add + bin histogram) for MethylPipeline.
# Device select comes from gpu-common. Host loops match numeric/python/centroid_kernels.py.

from std.python import Python, PythonObject
from std.sys import has_accelerator

from gpu_device import require_device_or_raise, select_device
from gpu_kernels import _device_api, probe_device_context


def probe_numeric_device(device: String) raises -> String:
    var resolved = select_device(device)
    return probe_device_context(resolved)


def require_numeric_device(device: String) raises -> String:
    return require_device_or_raise(device)


def scatter_add_u32_host(
    acc: PythonObject, idx: PythonObject, values: PythonObject
) raises:
    """Host scatter-add (uint32) via NumPy add.at — parity with Python API."""
    var np = Python.import_module("numpy")
    np.add.at(acc, np.asarray(idx, dtype=np.intp), np.asarray(values, dtype=np.uint32))


def scatter_add_f32_host(
    acc: PythonObject, idx: PythonObject, values: PythonObject
) raises:
    var np = Python.import_module("numpy")
    np.add.at(acc, np.asarray(idx, dtype=np.intp), np.asarray(values, dtype=np.float32))


def bin_histogram_add_host(
    bin_counts: PythonObject, pos_idx: PythonObject, bin_idx: PythonObject
) raises:
    var np = Python.import_module("numpy")
    # Python.tuple(a, b) is Mojo's multi-arg constructor. builtins.tuple(a, b)
    # would call Python's tuple() which accepts only one iterable.
    np.add.at(
        bin_counts,
        Python.tuple(
            np.asarray(pos_idx, dtype=np.intp), np.asarray(bin_idx, dtype=np.intp)
        ),
        1,
    )


def scatter_add_u32_on_device(
    device: String, acc: PythonObject, idx: PythonObject, values: PythonObject
) raises:
    """DeviceContext add when an accelerator is present; else host add.at."""
    var resolved = select_device(device)
    var backend = probe_device_context(resolved)
    comptime if has_accelerator():
        from std.atomic import Atomic
        from std.gpu import block_dim, block_idx, thread_idx
        from std.gpu.host import DeviceContext
        from std.memory import UnsafePointer

        def scatter_u32_kernel(
            dst: UnsafePointer[UInt32, MutAnyOrigin],
            index: UnsafePointer[Int32, MutAnyOrigin],
            src: UnsafePointer[UInt32, MutAnyOrigin],
            n: Int,
        ):
            var i = Int(block_idx.x * block_dim.x + thread_idx.x)
            if i >= n:
                return
            var j = Int(index[i])
            # Duplicate indices must accumulate (np.add.at). A plain
            # load-add-store drops concurrent writes to the same j.
            _ = Atomic[DType.uint32].fetch_add(dst + j, src[i])

        if backend.startswith("devicecontext-cuda") or backend.startswith(
            "devicecontext-hip"
        ):
            var np = Python.import_module("numpy")
            var acc_h = np.ascontiguousarray(acc, dtype=np.uint32)
            var idx_h = np.ascontiguousarray(idx, dtype=np.int32)
            var val_h = np.ascontiguousarray(values, dtype=np.uint32)
            var n_acc = Int(py=acc_h.shape[0])
            var n = Int(py=idx_h.shape[0])
            if n <= 0:
                return
            var api = _device_api(resolved)
            var ctx = DeviceContext(api=api)
            var host_acc = ctx.enqueue_create_host_buffer[DType.uint32](n_acc)
            var host_idx = ctx.enqueue_create_host_buffer[DType.int32](n)
            var host_val = ctx.enqueue_create_host_buffer[DType.uint32](n)
            var i = 0
            while i < n_acc:
                host_acc[i] = UInt32(Int(py=acc_h[i]))
                i += 1
            i = 0
            while i < n:
                host_idx[i] = Int32(Int(py=idx_h[i]))
                host_val[i] = UInt32(Int(py=val_h[i]))
                i += 1
            var dev_acc = ctx.enqueue_create_buffer[DType.uint32](n_acc)
            var dev_idx = ctx.enqueue_create_buffer[DType.int32](n)
            var dev_val = ctx.enqueue_create_buffer[DType.uint32](n)
            ctx.enqueue_copy(src_buf=host_acc, dst_buf=dev_acc)
            ctx.enqueue_copy(src_buf=host_idx, dst_buf=dev_idx)
            ctx.enqueue_copy(src_buf=host_val, dst_buf=dev_val)
            comptime BLOCK = 256
            var grid = (n + BLOCK - 1) // BLOCK
            ctx.enqueue_function[scatter_u32_kernel](
                dev_acc.unsafe_ptr(),
                dev_idx.unsafe_ptr(),
                dev_val.unsafe_ptr(),
                n,
                grid_dim=grid,
                block_dim=BLOCK,
            )
            ctx.enqueue_copy(src_buf=dev_acc, dst_buf=host_acc)
            ctx.synchronize()
            i = 0
            while i < n_acc:
                acc_h[i] = host_acc[i]
                i += 1
            return

    scatter_add_u32_host(acc, idx, values)


def _smoke_host_kernels() raises:
    """Exercise host add.at paths, including 2-D histogram index tuples."""
    var np = Python.import_module("numpy")
    var acc = np.zeros(4, dtype=np.uint32)
    scatter_add_u32_host(
        acc,
        np.array([1, 1, 2], dtype=np.intp),
        np.array([3, 4, 5], dtype=np.uint32),
    )
    if Int(py=acc[1]) != 7 or Int(py=acc[2]) != 5:
        raise Error("scatter_add_u32_host duplicate-index smoke failed")
    var counts = np.zeros(Python.tuple(2, 4), dtype=np.uint32)
    bin_histogram_add_host(
        counts,
        np.array([0, 0, 1], dtype=np.intp),
        np.array([1, 1, 3], dtype=np.intp),
    )
    if Int(py=counts[0][1]) != 2 or Int(py=counts[1][3]) != 1:
        raise Error("bin_histogram_add_host index-tuple smoke failed")


def main() raises:
    var os_mod = Python.import_module("os")
    var device = String(os_mod.environ.get("METHYLGRAPHER_ALIGN_DEVICE", "auto"))
    var backend = probe_numeric_device(device)
    print("numeric centroid probe backend=", backend)
    _smoke_host_kernels()
    print("numeric centroid host smoke ok")
