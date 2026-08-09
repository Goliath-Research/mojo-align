"""Removed: CUDA Runtime H2D bootstrap.

Index upload is Mojo-native (``giraffe_gpu_map_kernels.upload_mmap_to_device``:
HostBuffer + ``DeviceContext.enqueue_copy`` + device copy kernel). Do not import
``libcudart`` / CuPy on the production Giraffe path.
"""

from __future__ import annotations


def memcpy_htod(*_a, **_k):  # noqa: ANN001
    raise RuntimeError(
        "engine.gpu_h2d CUDA Runtime path removed; use Mojo DeviceContext "
        "upload_mmap_to_device in giraffe_gpu_map_kernels.mojo"
    )


def memcpy_htod_chunked(*_a, **_k):  # noqa: ANN001
    raise RuntimeError(
        "engine.gpu_h2d CUDA Runtime path removed; use Mojo DeviceContext "
        "upload_mmap_to_device in giraffe_gpu_map_kernels.mojo"
    )
