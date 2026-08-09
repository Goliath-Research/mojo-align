"""GPU HBM capacity preflight (NVIDIA via nvidia-smi; AMD via rocm-smi).

Queries free HBM through driver management tools *before* Mojo DeviceContext
allocates, so capacity failures are explicit. Does not call the CUDA Runtime.
(Mojo's NVIDIA backend token is still named ``cuda`` inside DeviceContext —
that is framework naming, not an app-level cudart dependency.)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import NamedTuple, Optional


class GpuHbmInfo(NamedTuple):
    free_bytes: int
    total_bytes: int
    source: str  # nvidia-smi | nvml | rocm-smi


def _gib(n: int) -> float:
    return float(n) / float(1024**3)


def hbm_info_nvidia() -> Optional[GpuHbmInfo]:
    """Free/total HBM for GPU 0 via nvidia-smi (no CUDA context required)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    line = r.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        free_mib = int(float(parts[0]))
        total_mib = int(float(parts[1]))
    except ValueError:
        return None
    return GpuHbmInfo(free_mib << 20, total_mib << 20, "nvidia-smi")


def hbm_info_rocm() -> Optional[GpuHbmInfo]:
    """Best-effort free/total via rocm-smi (AMD)."""
    if shutil.which("rocm-smi") is None:
        return None
    try:
        r = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    text = r.stdout + "\n" + r.stderr
    # Typical: "GPU[0] ... Total Memory (B): 17163091968" / "Total Used Memory (B): …"
    totals = [int(x) for x in re.findall(r"Total Memory \(B\):\s*(\d+)", text)]
    useds = [int(x) for x in re.findall(r"Total Used Memory \(B\):\s*(\d+)", text)]
    if not totals:
        return None
    total = totals[0]
    used = useds[0] if useds else 0
    free = max(total - used, 0)
    return GpuHbmInfo(free, total, "rocm-smi")


def hbm_info(device: str = "nvidia") -> GpuHbmInfo:
    """Return free/total HBM or raise with a clear probe error."""
    d = (device or "nvidia").strip().lower()
    if d in {"nvidia", "cuda", "auto", ""}:
        info = hbm_info_nvidia()
        if info is not None:
            return info
        raise RuntimeError(
            "GPU HBM preflight: nvidia-smi did not report memory.free/total "
            "(is the NVIDIA driver up?)"
        )
    if d in {"amd", "hip", "rocm"}:
        info = hbm_info_rocm()
        if info is not None:
            return info
        raise RuntimeError(
            "GPU HBM preflight: rocm-smi did not report VRAM "
            "(is ROCm installed on PATH?)"
        )
    raise RuntimeError(f"GPU HBM preflight: unsupported device={device!r}")


def require_index_capacity(
    science_bytes: int,
    *,
    device: str = "nvidia",
    overhead: Optional[float] = None,
) -> dict:
    """Fail closed if free HBM cannot hold the resident index (+ overhead).

    Parameters
    ----------
    science_bytes:
        HT + pack offsets + sequence bytes (device slabs).
    overhead:
        Multiplier for Mojo DeviceContext working set beyond science slabs.
        Observed ~85 GiB HBM for ~37 GiB science on GH200 (~2.3×).
        Env ``METHYLGRAPHER_GPU_RESIDENT_OVERHEAD`` overrides (default 2.4).
        Set ``METHYLGRAPHER_GPU_MEM_PREFLIGHT=0`` to skip.
    """
    if os.environ.get("METHYLGRAPHER_GPU_MEM_PREFLIGHT", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return {"skipped": True}

    if science_bytes <= 0:
        raise RuntimeError("GPU HBM preflight: science_bytes must be > 0")

    if overhead is None:
        raw = os.environ.get("METHYLGRAPHER_GPU_RESIDENT_OVERHEAD", "").strip()
        overhead = float(raw) if raw else 2.4
    if overhead < 1.0:
        raise RuntimeError(
            f"GPU HBM preflight: overhead must be >= 1.0 (got {overhead})"
        )

    info = hbm_info(device)
    need_science = int(science_bytes)
    need_total = int(float(science_bytes) * float(overhead))
    report = {
        "skipped": False,
        "source": info.source,
        "free_bytes": info.free_bytes,
        "total_bytes": info.total_bytes,
        "science_bytes": need_science,
        "need_bytes": need_total,
        "overhead": float(overhead),
        "free_gib": _gib(info.free_bytes),
        "total_gib": _gib(info.total_bytes),
        "science_gib": _gib(need_science),
        "need_gib": _gib(need_total),
    }
    print(
        "gpu_hbm_preflight source=",
        info.source,
        " free_gib=",
        round(report["free_gib"], 3),
        " total_gib=",
        round(report["total_gib"], 3),
        " science_gib=",
        round(report["science_gib"], 3),
        " need_gib=",
        round(report["need_gib"], 3),
        " overhead=",
        overhead,
        flush=True,
    )

    if info.free_bytes < need_science:
        raise RuntimeError(
            "GPU HBM insufficient for Giraffe index slabs: need "
            f"{report['science_gib']:.2f} GiB free for HT+pack+seq, have "
            f"{report['free_gib']:.2f} GiB free / {report['total_gib']:.2f} GiB total "
            f"(via {info.source}). Free the GPU (one Align per GH200; no leftover "
            "Mojo/DeviceContext) before retrying."
        )
    if info.free_bytes < need_total:
        raise RuntimeError(
            "GPU HBM insufficient for Mojo DeviceContext working set: science "
            f"slabs are {report['science_gib']:.2f} GiB but observed residency is "
            f"~{overhead:.1f}× that (~{report['need_gib']:.2f} GiB need); have "
            f"{report['free_gib']:.2f} GiB free / {report['total_gib']:.2f} GiB total "
            f"(via {info.source}). Set METHYLGRAPHER_GPU_RESIDENT_OVERHEAD to tune; "
            "ensure no other process holds HBM."
        )
    return report


def wait_for_hbm_free(
    min_free_gib: float,
    *,
    device: str = "nvidia",
    timeout_s: float = 180.0,
    poll_s: float = 2.0,
) -> dict:
    """Block until free HBM ≥ ``min_free_gib`` or raise after ``timeout_s``.

    Used between dual-graph C2T/G2A Mojo processes and by the host worker before
    starting Align docker — CUDA can reclaim HBM a few seconds after process exit.
    """
    import time

    if min_free_gib <= 0:
        raise RuntimeError("wait_for_hbm_free: min_free_gib must be > 0")
    need = int(float(min_free_gib) * float(1024**3))
    deadline = time.monotonic() + float(timeout_s)
    last: Optional[GpuHbmInfo] = None
    while True:
        last = hbm_info(device)
        if last.free_bytes >= need:
            report = {
                "free_gib": _gib(last.free_bytes),
                "total_gib": _gib(last.total_bytes),
                "min_free_gib": float(min_free_gib),
                "source": last.source,
            }
            print(
                "gpu_hbm_wait ok free_gib=",
                round(report["free_gib"], 3),
                " min_free_gib=",
                min_free_gib,
                " source=",
                last.source,
                flush=True,
            )
            return report
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "GPU HBM did not reclaim in time: need "
                f"{min_free_gib:.1f} GiB free, have {_gib(last.free_bytes):.2f} GiB free / "
                f"{_gib(last.total_bytes):.2f} GiB total after {timeout_s:.0f}s "
                f"(via {last.source}). Kill leftover Mojo/Align containers on this GPU."
            )
        time.sleep(max(0.2, float(poll_s)))
