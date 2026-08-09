"""libc mmap open for Mojo-native MinimizerIndex HT probe.

Returns a raw address Mojo can wrap with ``UnsafePointer``; keeps the fd
alive until ``close_min_mmap``.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, Tuple

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

PROT_READ = 1
MAP_PRIVATE = 2
MAP_FAILED = ctypes.c_void_p(-1).value

# fd -> (addr, size) for munmap on close
_OPEN: dict[int, tuple[int, int]] = {}


def open_min_mmap(path: str | Path) -> Tuple[int, int, Any, int]:
    """Return ``(addr, size, None, fd)``. Keep ``fd`` until close."""
    p = Path(path)
    size = p.stat().st_size
    fd = os.open(p, os.O_RDONLY)
    addr = _libc.mmap(None, size, PROT_READ, MAP_PRIVATE, fd, 0)
    if addr is None or addr == MAP_FAILED or addr == 0:
        err = ctypes.get_errno()
        os.close(fd)
        raise OSError(err, f"mmap failed for {p}")
    a = int(addr)
    _OPEN[int(fd)] = (a, int(size))
    return a, int(size), None, int(fd)


def close_min_mmap(_mm: Any, fd: int) -> None:
    """Unmap + close fd registered by ``open_min_mmap``."""
    ent = _OPEN.pop(int(fd), None)
    if ent is not None:
        a, n = ent
        _libc.munmap(ctypes.c_void_p(a), n)
    try:
        os.close(int(fd))
    except OSError:
        pass
