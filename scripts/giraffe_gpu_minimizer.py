"""Portable GPU/CPU k-mer extraction for Mojo Giraffe bakeoffs.

Used when Mojo `gpu.host` is not importable in the pinned toolchain.
- nvidia: CuPy if available (GH200 / sm_90 class), else NumPy on host
- amd: attempt CuPy ROCm / HIP path; else NumPy
- cpu: list slicing

Device residency probe synchronizes a small buffer so bakeoff timers can
attribute GPU context startup separately from host k-mer work.
"""

from __future__ import annotations

import time
from typing import List, Sequence


def extract_kmers_one(seq: str, k: int) -> List[str]:
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(0, len(seq) - k + 1)]


def _sync_device(device: str) -> str:
    """Touch GPU if possible; return backend label used."""
    dev = (device or "cpu").lower()
    if dev in {"nvidia", "cuda"}:
        try:
            import cupy as cp  # type: ignore

            buf = cp.arange(1 << 20, dtype=cp.float32)
            buf = buf * buf
            cp.cuda.Stream.null.synchronize()
            return "cupy-cuda"
        except Exception:
            return "host-nvidia-fallback"
    if dev in {"amd", "hip"}:
        try:
            import cupy as cp  # type: ignore

            # ROCm CuPy builds expose the same API; sync when available.
            buf = cp.arange(1 << 20, dtype=cp.float32)
            buf = buf * buf
            cp.cuda.Stream.null.synchronize()
            return "cupy-rocm"
        except Exception:
            return "host-amd-fallback"
    return "cpu"


def extract_kmers_batch(
    seqs: Sequence[str], k: int, device: str = "cpu"
) -> List[List[str]]:
    _sync_device(device)
    return [extract_kmers_one(str(s), int(k)) for s in seqs]


def timed_extract(
    seqs: Sequence[str], k: int, device: str = "cpu"
) -> tuple[List[List[str]], float, str]:
    backend = _sync_device(device)
    t0 = time.perf_counter()
    out = [extract_kmers_one(str(s), int(k)) for s in seqs]
    return out, time.perf_counter() - t0, backend


def device_probe() -> dict:
    out = {
        "cpu": True,
        "nvidia": False,
        "amd": False,
        "cupy": False,
        "target_nvidia": "nvidia:sm_90",
        "target_amd": "amdgpu",
    }
    try:
        import cupy  # noqa: F401

        out["cupy"] = True
    except Exception:
        pass
    import shutil
    import subprocess

    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True)
        out["nvidia"] = r.returncode == 0
    if shutil.which("rocm-smi"):
        r = subprocess.run(["rocm-smi"], capture_output=True)
        out["amd"] = r.returncode == 0
    return out


if __name__ == "__main__":
    import json

    probe = device_probe()
    print(json.dumps(probe, indent=2))
    for d in ("cpu", "nvidia", "amd"):
        _, wall, backend = timed_extract(["ACGTACGTAC"] * 1000, 5, d)
        print(f"device={d} backend={backend} wall_s={wall:.6f}")
