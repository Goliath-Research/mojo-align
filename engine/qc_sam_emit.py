"""Fast batch SAM formatting for MojoGiraffe QC (called from Mojo via Python).

Avoids per-base Mojo string concat for QUAL and batches NFS writes.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

from engine import grch38_offsets as off

_BUF: List[str] = []
_BUF_CHARS = 0
_BUF_FLUSH = 4 * 1024 * 1024  # 4 MiB


def _project_path(path: str) -> Optional[Tuple[str, int]]:
    """First path segment present in the GRCh38 offset table → (chrom, pos1)."""
    if not path or path == "*":
        return None
    n = len(path)
    i = 0
    while i < n:
        ch = path[i]
        if ch not in "><":
            i += 1
            continue
        j = i + 1
        while j < n and path[j] not in "><":
            j += 1
        try:
            sid = int(path[i + 1 : j])
        except ValueError:
            i = j
            continue
        info = off.lookup(sid)
        if info is not None:
            chrom, start0, _length = info
            return chrom, int(start0) + 1
        i = j
    return None


def _tag_value(extras: str, prefix: str) -> str:
    if not extras:
        return ""
    if extras.startswith(prefix):
        idx = 0
    else:
        needle = "\t" + prefix
        at = extras.find(needle)
        if at < 0:
            return ""
        idx = at + 1
    start = idx + len(prefix)
    tab = extras.find("\t", start)
    return extras[start:] if tab < 0 else extras[start:tab]


def _flush(fh: Any) -> None:
    global _BUF, _BUF_CHARS
    if not _BUF:
        return
    fh.write("".join(_BUF))
    _BUF = []
    _BUF_CHARS = 0


def open_sam(path: str, offsets_root: str) -> Any:
    global _BUF, _BUF_CHARS
    _BUF = []
    _BUF_CHARS = 0
    off.open_table(offsets_root)
    print(f"grch38_offsets open root={offsets_root}", flush=True)
    fh = open(path, "w", buffering=8 * 1024 * 1024)
    fh.write("@HD\tVN:1.6\tSO:unsorted\n")
    for sn, ln in zip(off.chroms(), off.chrom_lens()):
        fh.write(f"@SQ\tSN:{sn}\tLN:{max(int(ln), 1)}\n")
    return fh


def close_sam(fh: Any) -> None:
    try:
        _flush(fh)
        fh.flush()
        fh.close()
    finally:
        off.close_table()


def append_hits(
    fh: Any, rows: Sequence[Tuple[Any, Any, Any, Any]]
) -> int:
    """rows: (query_name, path, mapq, extra_tags). Returns mapped count."""
    global _BUF, _BUF_CHARS
    n = 0
    for qname, path, mapq, extras in rows:
        qname_s = str(qname)
        path_s = str(path)
        if not path_s or path_s == "*":
            continue
        seq = _tag_value(str(extras), "os:Z:")
        if not seq:
            off.bump_skip_no_seq()
            continue
        proj = _project_path(path_s)
        if proj is None:
            off.bump_skip_no_anchor()
            continue
        chrom, pos1 = proj
        # QUAL '*' — restore_original_sequences uses FASTQ, not BAM qualities.
        line = (
            f"{qname_s}\t0\t{chrom}\t{pos1}\t{int(mapq)}\t{len(seq)}M"
            f"\t*\t0\t0\t{seq}\t*\n"
        )
        _BUF.append(line)
        _BUF_CHARS += len(line)
        n += 1
        mapped = off.bump_mapped()
        if mapped % 1_000_000 == 0:
            print(f"mojo_qc_sam progress {off.summary()}", flush=True)
        if _BUF_CHARS >= _BUF_FLUSH:
            _flush(fh)
    return n


def summary() -> str:
    return off.summary()
