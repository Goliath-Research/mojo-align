"""GPU/CPU minimizer seed helpers for Mojo Giraffe.

Used by ``engine.quartet_map`` when ``device=nvidia|amd`` so the production
GBZ stream path actually runs on the accelerator (CuPy), not host Python loops.

Also provides a tiny DeviceContext residency probe bridge for bakeoffs when
Mojo kernels are unavailable inside the Python process.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from engine.minimizer_index import (
    MinimizerOcc,
    PACK,
    encode_key64,
    wang_hash_64,
)

_PACK_ARR = np.full(256, 255, dtype=np.uint8)
for _b, _c in PACK.items():
    _PACK_ARR[int(_b)] = int(_c)


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
    if dev in {"amd", "hip", "rocm"}:
        try:
            import cupy as cp  # type: ignore

            buf = cp.arange(1 << 20, dtype=cp.float32)
            buf = buf * buf
            cp.cuda.Stream.null.synchronize()
            return "cupy-rocm"
        except Exception:
            pass
        try:
            import ctypes

            ctypes.CDLL("libamdhip64.so")
            return "hip-host-kernels"
        except Exception:
            return "host-amd-fallback"
    return "cpu"


def extract_kmers_batch(
    seqs: Sequence[str], k: int, device: str = "cpu"
) -> List[List[str]]:
    backend = _sync_device(device)
    os.environ["METHYLGRAPHER_LAST_GPU_BACKEND"] = backend
    return [extract_kmers_one(str(s), int(k)) for s in seqs]


def timed_extract(
    seqs: Sequence[str], k: int, device: str = "cpu"
) -> tuple[List[List[str]], float, str]:
    backend = _sync_device(device)
    t0 = time.perf_counter()
    out = [extract_kmers_one(str(s), int(k)) for s in seqs]
    return out, time.perf_counter() - t0, backend


def _minimizers_cpu(seq: str, k: int, w: int) -> List[MinimizerOcc]:
    """Parity path matching MinimizerIndex.minimizers (forward windows)."""
    s = seq.encode("ascii", errors="ignore")
    total = len(s)
    win = k + w - 1
    if total < win:
        return []
    results: List[MinimizerOcc] = []
    next_read_offset = 0
    last_hash: Optional[int] = None
    last_offset = -1
    for window_start in range(0, total - win + 1):
        best: Optional[MinimizerOcc] = None
        for i in range(w):
            pos = window_start + i
            chunk = s[pos : pos + k]
            fk = encode_key64(chunk, k)
            if fk is None:
                best = None
                break
            rk = 0
            ok = True
            for b in reversed(chunk):
                p = PACK.get(b)
                if p is None:
                    ok = False
                    break
                rk = (rk << 2) | (p ^ 3)
            if not ok:
                best = None
                break
            fh = wang_hash_64(fk)
            rh = wang_hash_64(rk)
            if rh < fh:
                cand = MinimizerOcc(rk, rh, pos, True)
            else:
                cand = MinimizerOcc(fk, fh, pos, False)
            if best is None or cand.hash < best.hash or (
                cand.hash == best.hash and cand.offset < best.offset
            ):
                best = cand
        if best is None:
            continue
        if not results or last_hash == best.hash or last_offset < best.offset:
            if best.offset >= next_read_offset:
                occ = best
                if occ.is_reverse:
                    occ = MinimizerOcc(occ.key, occ.hash, occ.offset + k - 1, True)
                results.append(occ)
                next_read_offset = best.offset + 1
                last_hash = best.hash
                last_offset = best.offset
    results.sort(key=lambda m: m.offset)
    return results


def _wang_hash_vec(keys: "object") -> "object":
    import cupy as cp  # type: ignore

    key = keys.astype(cp.uint64)
    mask = cp.uint64((1 << 64) - 1)
    key = key & mask
    key = ((~key) + (key << cp.uint64(21))) & mask
    key ^= key >> cp.uint64(24)
    key = (key + (key << cp.uint64(3)) + (key << cp.uint64(8))) & mask
    key ^= key >> cp.uint64(14)
    key = (key + (key << cp.uint64(2)) + (key << cp.uint64(4))) & mask
    key ^= key >> cp.uint64(28)
    key = (key + (key << cp.uint64(31))) & mask
    return key


def minimizers_batch_gpu(
    seqs: Sequence[str],
    *,
    k: int,
    w: int,
    device: str = "nvidia",
) -> Tuple[List[List[MinimizerOcc]], str]:
    """Compute Giraffe-style minimizers for a batch; prefer CuPy on NVIDIA/AMD."""
    backend = _sync_device(device)
    os.environ["METHYLGRAPHER_LAST_GPU_BACKEND"] = backend
    if backend.startswith("cupy-"):
        try:
            return _minimizers_batch_cupy(seqs, k=k, w=w), backend
        except Exception as exc:  # pragma: no cover - device/runtime variance
            print(f"CuPy minimizer batch failed ({exc}); CPU fallback", flush=True)
            backend = "host-minimizer-fallback"
            os.environ["METHYLGRAPHER_LAST_GPU_BACKEND"] = backend
    out = [_minimizers_cpu(s, k, w) for s in seqs]
    return out, backend


def _minimizers_batch_cupy(
    seqs: Sequence[str], *, k: int, w: int
) -> List[List[MinimizerOcc]]:
    """Vectorized per-read minimizer extraction on GPU (encode + wang hash)."""
    import cupy as cp  # type: ignore

    win = k + w - 1
    results: List[List[MinimizerOcc]] = []
    for seq in seqs:
        s = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
        total = int(s.shape[0])
        if total < win:
            results.append([])
            continue
        codes_h = _PACK_ARR[s]
        codes = cp.asarray(codes_h)
        n_pos = total - k + 1
        # Build forward keys via rolling pack on GPU
        keys_f = cp.zeros(n_pos, dtype=cp.uint64)
        valid = cp.ones(n_pos, dtype=cp.bool_)
        for j in range(k):
            c = codes[j : j + n_pos]
            bad = c > 3
            valid &= ~bad
            keys_f = (keys_f << cp.uint64(2)) | c.astype(cp.uint64)
        # RC key consumes bases in reverse order within each k-mer
        keys_r = cp.zeros(n_pos, dtype=cp.uint64)
        for j in range(k):
            c = codes[(k - 1 - j) : (k - 1 - j) + n_pos]
            keys_r = (keys_r << cp.uint64(2)) | (c.astype(cp.uint64) ^ cp.uint64(3))

        hf = _wang_hash_vec(keys_f)
        hr = _wang_hash_vec(keys_r)
        use_r = hr < hf
        best_key = cp.where(use_r, keys_r, keys_f)
        best_hash = cp.where(use_r, hr, hf)
        best_rev = use_r
        best_off = cp.arange(n_pos, dtype=cp.int32)

        # Per window: argmin hash across w k-mers
        n_win = total - win + 1
        # Gather window candidates — loop windows on host over GPU arrays for
        # correct Giraffe dedup semantics (parity with CPU path).
        hf_h = cp.asnumpy(best_hash)
        key_h = cp.asnumpy(best_key)
        rev_h = cp.asnumpy(best_rev)
        valid_h = cp.asnumpy(valid)
        cp.cuda.Stream.null.synchronize()

        occs: List[MinimizerOcc] = []
        next_read_offset = 0
        last_hash: Optional[int] = None
        last_offset = -1
        for window_start in range(n_win):
            best: Optional[MinimizerOcc] = None
            ok = True
            for i in range(w):
                pos = window_start + i
                if not valid_h[pos]:
                    ok = False
                    break
                cand = MinimizerOcc(
                    int(key_h[pos]),
                    int(hf_h[pos]),
                    int(pos),
                    bool(rev_h[pos]),
                )
                if best is None or cand.hash < best.hash or (
                    cand.hash == best.hash and cand.offset < best.offset
                ):
                    best = cand
            if not ok or best is None:
                continue
            if not occs or last_hash == best.hash or last_offset < best.offset:
                if best.offset >= next_read_offset:
                    occ = best
                    if occ.is_reverse:
                        occ = MinimizerOcc(
                            occ.key, occ.hash, occ.offset + k - 1, True
                        )
                    occs.append(occ)
                    next_read_offset = best.offset + 1
                    last_hash = best.hash
                    last_offset = best.offset
        occs.sort(key=lambda m: m.offset)
        results.append(occs)
    return results


def device_probe() -> dict:
    out = {
        "cpu": True,
        "nvidia": False,
        "amd": False,
        "cupy": False,
        "backend": "cpu",
        "target_nvidia": "nvidia:sm_90",
        "target_amd": "amdgpu:gfx942",
    }
    try:
        import cupy as cp  # type: ignore

        out["cupy"] = True
        out["nvidia"] = int(cp.cuda.runtime.getDeviceCount()) > 0
        if out["nvidia"]:
            out["backend"] = _sync_device("nvidia")
    except Exception:
        pass
    if not out["nvidia"]:
        try:
            import subprocess

            r = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
            )
            out["nvidia"] = r.returncode == 0 and bool(r.stdout.strip())
        except Exception:
            pass
    try:
        import subprocess

        r = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            check=False,
        )
        out["amd"] = r.returncode == 0
    except Exception:
        pass
    return out
