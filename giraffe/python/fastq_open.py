"""Production FASTQ open for Mojo Giraffe stream map.

pigz/gzip decompress via pipe + large text buffer — avoids Python ``gzip.open``
readline overhead on Buffy-scale inputs. Mojo calls ``open_fastq_text``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, TextIO


class _PigzTextHandle:
    """Text stream that keeps the decompress process alive until close()."""

    __slots__ = ("_proc", "_fh")

    def __init__(self, proc: subprocess.Popen[str]):
        assert proc.stdout is not None
        self._proc = proc
        self._fh = proc.stdout

    def readline(self, *args: Any, **kwargs: Any) -> str:
        return self._fh.readline(*args, **kwargs)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass

    def __enter__(self) -> "_PigzTextHandle":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def open_fastq_text(path: str) -> TextIO[str] | _PigzTextHandle:
    """Open FASTQ for Mojo ``readline`` loop (plain or .gz)."""
    low = path.lower()
    if not (low.endswith(".gz") or low.endswith(".gzip")):
        return open(path, "r", buffering=8 * 1024 * 1024, encoding="utf-8", errors="replace")
    decomp = shutil.which("pigz") or shutil.which("gzip")
    if decomp is None:
        raise RuntimeError(f"pigz or gzip required for gzip FASTQ: {path}")
    cmd = [decomp, "-dc", path]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=8 * 1024 * 1024,
    )
    return _PigzTextHandle(proc)
