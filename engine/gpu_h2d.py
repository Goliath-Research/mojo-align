"""Chunked CUDA H2D bootstrap for Mojo DeviceContext index residency.

Used only to upload mmap'd ``.min`` HT / dense-pack slabs into a device pointer
obtained from Mojo ``DeviceBuffer.unsafe_ptr()``. Science kernels stay in Mojo.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

_cudart: Optional[ctypes.CDLL] = None

cudaMemcpyHostToDevice = 1

_CANDIDATES = (
    "libcudart.so",
    "libcudart.so.12",
    "libcudart.so.11",
    "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
    "/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
    "/usr/local/cuda/lib64/libcudart.so",
    "/usr/lib/x86_64-linux-gnu/libcudart.so",
    "/usr/lib/aarch64-linux-gnu/libcudart.so",
)


def _lib() -> ctypes.CDLL:
    global _cudart
    if _cudart is not None:
        return _cudart
    env = os.environ.get("METHYLGRAPHER_CUDART_PATH", "").strip()
    paths = ([env] if env else []) + list(_CANDIDATES)
    last_err: Exception | None = None
    for p in paths:
        if not p:
            continue
        try:
            if p.startswith("/") and not Path(p).is_file():
                continue
            lib = ctypes.CDLL(p)
            lib.cudaMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.cudaMemcpy.restype = ctypes.c_int
            _cudart = lib
            return lib
        except OSError as e:
            last_err = e
            continue
    raise OSError(
        f"libcudart.so not found (set METHYLGRAPHER_CUDART_PATH); last={last_err}"
    )


def memcpy_htod(dst_device: int, src_host: int, nbytes: int) -> None:
    """``cudaMemcpy`` host→device. Raises ``OSError`` on CUDA error."""
    if nbytes <= 0:
        return
    if dst_device == 0 or src_host == 0:
        raise OSError("memcpy_htod: null pointer")
    rc = _lib().cudaMemcpy(
        ctypes.c_void_p(int(dst_device)),
        ctypes.c_void_p(int(src_host)),
        ctypes.c_size_t(int(nbytes)),
        cudaMemcpyHostToDevice,
    )
    if rc != 0:
        raise OSError(rc, f"cudaMemcpy H2D failed rc={rc} nbytes={nbytes}")


def memcpy_htod_chunked(
    dst_device: int, src_host: int, nbytes: int, chunk: int = 64 << 20
) -> None:
    """Upload large slabs in chunks (default 64 MiB)."""
    off = 0
    n = int(nbytes)
    step = max(int(chunk), 1 << 20)
    while off < n:
        m = min(step, n - off)
        memcpy_htod(int(dst_device) + off, int(src_host) + off, m)
        off += m
