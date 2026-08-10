"""GPU HBM preflight helpers (no device required for pure logic tests)."""

from __future__ import annotations

import pytest

from engine import gpu_mem


def test_require_index_capacity_skips_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METHYLGRAPHER_GPU_MEM_PREFLIGHT", "0")
    assert gpu_mem.require_index_capacity(37 << 30) == {"skipped": True}


def test_require_index_capacity_fails_closed_on_low_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METHYLGRAPHER_GPU_MEM_PREFLIGHT", "1")
    monkeypatch.delenv("METHYLGRAPHER_GPU_HBM_FRACTION", raising=False)
    monkeypatch.setenv("METHYLGRAPHER_GPU_RESIDENT_OVERHEAD", "2.4")

    def fake_info(device: str = "nvidia") -> gpu_mem.GpuHbmInfo:
        # 10 GiB free / 96 GiB total — cannot hold 37 GiB science slabs
        return gpu_mem.GpuHbmInfo(10 << 30, 96 << 30, "nvidia-smi")

    monkeypatch.setattr(gpu_mem, "hbm_info", fake_info)
    with pytest.raises(RuntimeError, match="insufficient for Giraffe index slabs"):
        gpu_mem.require_index_capacity(37 << 30, device="nvidia")


def test_require_index_capacity_overhead_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METHYLGRAPHER_GPU_MEM_PREFLIGHT", "1")
    monkeypatch.delenv("METHYLGRAPHER_GPU_HBM_FRACTION", raising=False)
    monkeypatch.setenv("METHYLGRAPHER_GPU_RESIDENT_OVERHEAD", "2.4")

    def fake_info(device: str = "nvidia") -> gpu_mem.GpuHbmInfo:
        # 50 GiB free: enough for 37 GiB science, not for 2.4× working set
        return gpu_mem.GpuHbmInfo(50 << 30, 96 << 30, "nvidia-smi")

    monkeypatch.setattr(gpu_mem, "hbm_info", fake_info)
    with pytest.raises(RuntimeError, match="DeviceContext working set"):
        gpu_mem.require_index_capacity(37 << 30, device="nvidia")


def test_require_index_capacity_fraction_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METHYLGRAPHER_GPU_MEM_PREFLIGHT", "1")
    monkeypatch.delenv("METHYLGRAPHER_GPU_RESIDENT_OVERHEAD", raising=False)
    monkeypatch.setenv("METHYLGRAPHER_GPU_HBM_FRACTION", "0.90")

    def fake_info(device: str = "nvidia") -> gpu_mem.GpuHbmInfo:
        # 65 GiB free / 96 GiB total: below 90% budget (~86.4 GiB)
        return gpu_mem.GpuHbmInfo(65 << 30, 96 << 30, "nvidia-smi")

    monkeypatch.setattr(gpu_mem, "hbm_info", fake_info)
    with pytest.raises(RuntimeError, match="free-fraction budget"):
        gpu_mem.require_index_capacity(37 << 30, device="nvidia")


def test_require_index_capacity_fraction_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METHYLGRAPHER_GPU_MEM_PREFLIGHT", "1")
    monkeypatch.setenv("METHYLGRAPHER_GPU_HBM_FRACTION", "0.90")

    def fake_info(device: str = "nvidia") -> gpu_mem.GpuHbmInfo:
        return gpu_mem.GpuHbmInfo(95 << 30, 96 << 30, "nvidia-smi")

    monkeypatch.setattr(gpu_mem, "hbm_info", fake_info)
    out = gpu_mem.require_index_capacity(37 << 30, device="nvidia")
    assert out["budget_mode"] == "fraction"
    assert out["hbm_fraction"] == pytest.approx(0.90)
    assert out["need_bytes"] == int((96 << 30) * 0.90)


def test_require_index_capacity_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METHYLGRAPHER_GPU_MEM_PREFLIGHT", "1")
    monkeypatch.delenv("METHYLGRAPHER_GPU_HBM_FRACTION", raising=False)
    monkeypatch.setenv("METHYLGRAPHER_GPU_RESIDENT_OVERHEAD", "2.4")

    def fake_info(device: str = "nvidia") -> gpu_mem.GpuHbmInfo:
        return gpu_mem.GpuHbmInfo(95 << 30, 96 << 30, "nvidia-smi")

    monkeypatch.setattr(gpu_mem, "hbm_info", fake_info)
    out = gpu_mem.require_index_capacity(37 << 30, device="nvidia")
    assert out["skipped"] is False
    assert out["source"] == "nvidia-smi"
    assert out["free_bytes"] == 95 << 30
    assert out["budget_mode"] == "overhead"


def test_require_index_capacity_requires_budget_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METHYLGRAPHER_GPU_MEM_PREFLIGHT", "1")
    monkeypatch.delenv("METHYLGRAPHER_GPU_HBM_FRACTION", raising=False)
    monkeypatch.delenv("METHYLGRAPHER_GPU_RESIDENT_OVERHEAD", raising=False)

    def fake_info(device: str = "nvidia") -> gpu_mem.GpuHbmInfo:
        return gpu_mem.GpuHbmInfo(95 << 30, 96 << 30, "nvidia-smi")

    monkeypatch.setattr(gpu_mem, "hbm_info", fake_info)
    with pytest.raises(RuntimeError, match="HBM_FRACTION"):
        gpu_mem.require_index_capacity(37 << 30, device="nvidia")


def test_wait_for_hbm_free_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_info(device: str = "nvidia") -> gpu_mem.GpuHbmInfo:
        return gpu_mem.GpuHbmInfo(95 << 30, 96 << 30, "nvidia-smi")

    monkeypatch.setattr(gpu_mem, "hbm_info", fake_info)
    out = gpu_mem.wait_for_hbm_free(90.0, device="nvidia", timeout_s=1.0, poll_s=0.1)
    assert out["free_gib"] > 90.0


def test_wait_for_hbm_free_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_info(device: str = "nvidia") -> gpu_mem.GpuHbmInfo:
        return gpu_mem.GpuHbmInfo(8 << 30, 96 << 30, "nvidia-smi")

    monkeypatch.setattr(gpu_mem, "hbm_info", fake_info)
    with pytest.raises(RuntimeError, match="did not reclaim"):
        gpu_mem.wait_for_hbm_free(90.0, device="nvidia", timeout_s=0.3, poll_s=0.1)
