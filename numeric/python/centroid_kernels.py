"""Centroid stream kernels: host reference + optional DeviceContext probe.

Methylutils imports this module when ``gpu_backend=mojo``. The host loops match
the Mojo DeviceContext kernels in ``numeric/src/centroid_kernels.mojo``.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def _device_requested() -> str:
    raw = os.environ.get("METHYLGRAPHER_GIRAFFE_DEVICE") or os.environ.get(
        "METHYLGRAPHER_ALIGN_DEVICE", "auto"
    )
    return str(raw).strip().lower() or "auto"


def probe_device() -> str:
    """Best-effort DeviceContext label (``host-python`` when Mojo is not invoked)."""
    try:
        from gpu_mem import hbm_info  # type: ignore

        info = hbm_info("auto")
        source = getattr(info, "source", "") or ""
        if source in {"nvidia-smi", "nvml"}:
            return "devicecontext-cuda"
        if source == "rocm-smi":
            return "devicecontext-hip"
    except Exception:
        pass
    return "host-python"


def require_device() -> str:
    """Fail closed when METHYLGRAPHER_GPU_REQUIRE is on and the device probe misses."""
    backend = probe_device()
    require = os.environ.get("METHYLGRAPHER_GPU_REQUIRE", "").strip().lower()
    if require not in {"1", "true", "yes", "on"}:
        return backend
    device = _device_requested()
    if device in {"nvidia", "cuda"} and not backend.startswith("devicecontext-cuda"):
        raise RuntimeError(
            "gpu_backend=mojo requires DeviceContext CUDA "
            f"(device={device} backend={backend})"
        )
    if device in {"amd", "hip", "rocm"} and not backend.startswith("devicecontext-hip"):
        raise RuntimeError(
            "gpu_backend=mojo requires DeviceContext HIP "
            f"(device={device} backend={backend})"
        )
    return backend


def merge_centroid_positions(
    existing_pos: np.ndarray,
    existing_size: int,
    sample_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Match ``MethylCentroidBuilder.add_sample`` new-position detection.

    Returns ``(sample_pos as uint32, is_new bool mask, n_new)``.
    """
    pos = np.asarray(sample_pos, dtype=np.uint32)
    size = int(existing_size)
    if size <= 0:
        existing = np.asarray(existing_pos, dtype=np.uint32)
        max_pos = existing[-1] if existing.size else np.uint32(0)
        pos_capped = np.minimum(pos, max_pos)
        idx = np.searchsorted(existing[:0], pos_capped)
        first = existing[0] if existing.size else np.uint32(0)
        is_new = (pos > max_pos) | (first != pos)
        return pos, is_new, int(is_new.sum())
    existing = np.asarray(existing_pos[:size], dtype=np.uint32)
    max_pos = existing[-1]
    pos_capped = np.minimum(pos, max_pos)
    idx = np.searchsorted(existing, pos_capped)
    idx = np.clip(idx, 0, size - 1)
    is_new = (pos > max_pos) | (existing[idx] != pos)
    return pos, is_new, int(is_new.sum())


def scatter_add_u32(acc: np.ndarray, idx: np.ndarray, values: np.ndarray) -> None:
    np.add.at(acc, np.asarray(idx, dtype=np.intp), np.asarray(values, dtype=np.uint32))


def scatter_add_f32(acc: np.ndarray, idx: np.ndarray, values: np.ndarray) -> None:
    np.add.at(acc, np.asarray(idx, dtype=np.intp), np.asarray(values, dtype=np.float32))


def digitize_bins(mean: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(bin_edges, dtype=np.float64)
    n_bins = int(edges.size) - 1
    if n_bins <= 1:
        return np.zeros(len(mean), dtype=np.intp)
    bin_idx = np.digitize(np.asarray(mean, dtype=np.float64), edges[1:-1])
    return np.clip(bin_idx, 0, n_bins - 1).astype(np.intp)


def bin_histogram_add(
    bin_counts: np.ndarray, pos_idx: np.ndarray, bin_idx: np.ndarray
) -> None:
    np.add.at(
        bin_counts,
        (np.asarray(pos_idx, dtype=np.intp), np.asarray(bin_idx, dtype=np.intp)),
        1,
    )
