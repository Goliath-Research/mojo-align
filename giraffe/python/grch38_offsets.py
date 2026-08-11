"""Dense grch38-dense-v1 segment offset table (mmap) + buffered QC SAM emit.

SAM formatting lives here so MojoGiraffe QC can use the already-mounted
``engine/grch38_offsets.py`` overlay without requiring a new runner mount.
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

RECORD = struct.Struct("<IIQ")
UNSET = 0xFFFFFFFF

# SAM flags used for QC PE emit (ri:i from Mojo map kernels).
_FLAG_PAIRED = 0x1
_FLAG_PROPER_PAIR = 0x2
_FLAG_READ1 = 0x40
_FLAG_READ2 = 0x80


class Grch38OffsetsTable:
    __slots__ = ("root", "max_id", "chroms", "chrom_lens", "_mm", "_fh")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != "grch38-dense-v1":
            raise RuntimeError(f"unsupported offsets format: {meta.get('format')}")
        self.max_id = int(meta["max_id"])
        self.chroms: List[str] = []
        self.chrom_lens: List[int] = []
        chroms_path = self.root / "chroms.tsv"
        if chroms_path.is_file():
            for line in chroms_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split("\t")
                self.chroms.append(parts[0])
                self.chrom_lens.append(int(parts[1]) if len(parts) > 1 and parts[1] else 0)
        rec = self.root / "records.bin"
        self._fh = open(rec, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        expect = (self.max_id + 1) * 16
        if len(self._mm) < expect:
            raise RuntimeError(f"records.bin too small: {len(self._mm)} < {expect}")

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass

    def lookup(self, seg_id: int) -> Optional[Tuple[str, int, int]]:
        """Return (chrom, start0, length) or None."""
        if seg_id < 0 or seg_id > self.max_id:
            return None
        cidx, length, start = RECORD.unpack_from(self._mm, seg_id * 16)
        if cidx == UNSET or cidx >= len(self.chroms):
            return None
        return self.chroms[cidx], int(start), int(length)


_TABLE: Optional[Grch38OffsetsTable] = None
_mapped = 0
_skip_no_seq = 0
_skip_no_anchor = 0
_BUF: List[str] = []
_BUF_CHARS = 0
_BUF_FLUSH = 4 * 1024 * 1024


def open_table(root: str) -> None:
    global _TABLE, _mapped, _skip_no_seq, _skip_no_anchor
    close_table()
    _TABLE = Grch38OffsetsTable(root)
    _mapped = _skip_no_seq = _skip_no_anchor = 0


def close_table() -> None:
    global _TABLE
    if _TABLE is not None:
        _TABLE.close()
        _TABLE = None


def chroms() -> List[str]:
    return list(_TABLE.chroms) if _TABLE else []


def chrom_lens() -> List[int]:
    return list(_TABLE.chrom_lens) if _TABLE else []


def lookup(seg_id: int) -> Optional[Tuple[str, int, int]]:
    if _TABLE is None:
        return None
    return _TABLE.lookup(int(seg_id))


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


def _project_path(path: str) -> Optional[Tuple[str, int]]:
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
        info = lookup(sid)
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


def _read_index(extras: str) -> Optional[int]:
    """Parse Mojo ``ri:i:1|2`` tag (which FASTQ mate produced the hit)."""
    raw = _tag_value(extras, "ri:i:")
    if raw == "1":
        return 1
    if raw == "2":
        return 2
    return None


def _flush(fh: Any) -> None:
    global _BUF, _BUF_CHARS
    if not _BUF:
        return
    fh.write("".join(_BUF))
    _BUF = []
    _BUF_CHARS = 0


def open_sam(path: str, offsets_root: str) -> Any:
    """Open buffered QC SAM + offset table (Mojo QC path)."""
    global _BUF, _BUF_CHARS
    _BUF = []
    _BUF_CHARS = 0
    open_table(offsets_root)
    print(f"grch38_offsets open root={offsets_root}", flush=True)
    fh = open(path, "w", buffering=8 * 1024 * 1024)
    fh.write("@HD\tVN:1.6\tSO:unsorted\n")
    for sn, ln in zip(chroms(), chrom_lens()):
        fh.write(f"@SQ\tSN:{sn}\tLN:{max(int(ln), 1)}\n")
    return fh


def close_sam(fh: Any) -> None:
    try:
        _flush(fh)
        fh.flush()
        fh.close()
    finally:
        close_table()


def append_hits(fh: Any, rows: Sequence[Sequence[Any]]) -> int:
    """rows: (query_name, path, mapq, extra_tags). Returns mapped count.

    Uses ``ri:i`` from Mojo map kernels for READ1/READ2 + PAIRED bits. When both
    mates of a qname project in the same batch, also set PROPER_PAIR and mate
    RNEXT/PNEXT/TLEN so samtools flagstat PE rates are meaningful.
    """
    global _BUF, _BUF_CHARS

    prepared: List[Tuple[str, str, int, int, str, Optional[int]]] = []
    for row in rows:
        qname_s = str(row[0])
        path_s = str(row[1])
        mapq = int(row[2])
        extras = str(row[3]) if len(row) > 3 else ""
        if not path_s or path_s == "*":
            continue
        seq = _tag_value(extras, "os:Z:")
        if not seq:
            bump_skip_no_seq()
            continue
        proj = _project_path(path_s)
        if proj is None:
            bump_skip_no_anchor()
            continue
        chrom, pos1 = proj
        prepared.append((qname_s, chrom, pos1, mapq, seq, _read_index(extras)))

    mates_by_q: Dict[str, Dict[int, int]] = {}
    for idx, (qname_s, _chrom, _pos1, _mapq, _seq, ri) in enumerate(prepared):
        if ri is None:
            continue
        mates_by_q.setdefault(qname_s, {})[ri] = idx

    n = 0
    for idx, (qname_s, chrom, pos1, mapq, seq, ri) in enumerate(prepared):
        flag = 0
        rnext = "*"
        pnext = 0
        tlen = 0
        if ri == 1:
            flag |= _FLAG_PAIRED | _FLAG_READ1
        elif ri == 2:
            flag |= _FLAG_PAIRED | _FLAG_READ2
        mate_idx = None
        if ri is not None:
            mate_ri = 2 if ri == 1 else 1
            mate_idx = mates_by_q.get(qname_s, {}).get(mate_ri)
        if mate_idx is not None:
            _mq, mate_chrom, mate_pos, _mmapq, mate_seq, _mri = prepared[mate_idx]
            flag |= _FLAG_PROPER_PAIR
            rnext = "=" if mate_chrom == chrom else mate_chrom
            pnext = int(mate_pos)
            # Template length: 5' of this read to 3' of mate on same contig.
            if mate_chrom == chrom:
                this_end = pos1 + len(seq)
                mate_end = mate_pos + len(mate_seq)
                if pos1 <= mate_pos:
                    tlen = mate_end - pos1
                else:
                    tlen = -(this_end - mate_pos)

        # QUAL '*' — restore_original_sequences uses FASTQ, not BAM qualities.
        line = (
            f"{qname_s}\t{flag}\t{chrom}\t{pos1}\t{mapq}\t{len(seq)}M"
            f"\t{rnext}\t{pnext}\t{tlen}\t{seq}\t*\n"
        )
        _BUF.append(line)
        _BUF_CHARS += len(line)
        n += 1
        mapped = bump_mapped()
        if mapped % 1_000_000 == 0:
            print(f"mojo_qc_sam progress {summary()}", flush=True)
        if _BUF_CHARS >= _BUF_FLUSH:
            _flush(fh)
    return n
