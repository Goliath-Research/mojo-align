"""Process-local state for MojoGiraffe QC SAM emit (offsets handle + counters).

Kept in Python so Mojo can mutate via interop; the hot path still formats SAM
in Mojo and looks up segment offsets via mmap (no FASTQ dict).
"""

from __future__ import annotations

from typing import Any, Optional

_offsets: Any = None
_mapped: int = 0
_skip_no_seq: int = 0
_skip_no_anchor: int = 0


def reset() -> None:
    global _mapped, _skip_no_seq, _skip_no_anchor
    _mapped = 0
    _skip_no_seq = 0
    _skip_no_anchor = 0


def set_offsets(obj: Any) -> None:
    global _offsets
    _offsets = obj
    reset()


def get_offsets() -> Any:
    return _offsets


def clear_offsets() -> None:
    global _offsets
    _offsets = None


def bump_mapped() -> int:
    global _mapped
    _mapped += 1
    return _mapped


def bump_skip_no_seq() -> None:
    global _skip_no_seq
    _skip_no_seq += 1


def bump_skip_no_anchor() -> None:
    global _skip_no_anchor
    _skip_no_anchor += 1


def summary() -> str:
    return (
        f"mapped={_mapped} skip_no_seq={_skip_no_seq} "
        f"skip_no_anchor={_skip_no_anchor}"
    )


def mapped_count() -> int:
    return _mapped
