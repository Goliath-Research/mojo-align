#!/usr/bin/env python3
"""System-Python GPU minimizer worker (CuPy).

Invoked by ``engine.quartet_map`` via subprocess so Align can use the image's
system ``cupy-cuda12x`` install even though Mojo ``PYTHONHOME`` points at the
trimmed pixi env (no pip / no CuPy).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from giraffe_gpu_minimizer import minimizers_batch_gpu  # noqa: E402


def main() -> int:
    payload = json.load(sys.stdin)
    seqs = [str(s) for s in payload.get("seqs", [])]
    k = int(payload["k"])
    w = int(payload["w"])
    device = str(payload.get("device", "nvidia"))
    occs_batch, backend = minimizers_batch_gpu(seqs, k=k, w=w, device=device)
    out = {
        "backend": backend,
        "occs": [
            [
                {
                    "key": int(o.key),
                    "hash": int(o.hash),
                    "offset": int(o.offset),
                    "is_reverse": bool(o.is_reverse),
                }
                for o in occs
            ]
            for occs in occs_batch
        ],
    }
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
