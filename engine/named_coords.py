"""GBZ chopped-node → GFA named-coordinates translation for Mojo Giraffe GAF.

Align packs are built with ``vg convert --no-translation`` so node ids match
``.min`` Positions (≈145M chopped nodes). MethylCall + ``cpg.tsv`` use the
companion ``*.wl.gfa`` segment id space (≈60M). Classic ``vg giraffe
--named-coordinates`` performs this mapping at emit time; Mojo must too.

Binary index layout (``*.named_coords/``)::

- ``meta.json`` — counts + provenance
- ``nodes.bin`` — for node_id in 1..N: little-endian ``uint32 seg_id``,
  ``uint32 offset_in_segment`` (entry 0 unused / zero)
"""

from __future__ import annotations

import json
import mmap
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

_U32U32 = struct.Struct("<II")
_U64 = struct.Struct("<Q")
_PATH_NODE_RE = re.compile(r"([><])([^><\s]+)")

FORMAT_V1 = "gbz-to-gfa-v1"


def _node_length_from_pack(offsets_mm: memoryview, node_id: int) -> int:
    """Dense pack offsets.bin: per node (offset, length) uint64 pairs."""
    if node_id < 1:
        raise ValueError(f"node_id must be >= 1, got {node_id}")
    base = (node_id - 1) * 16
    if base + 16 > len(offsets_mm):
        raise ValueError(f"node_id {node_id} past pack end")
    return int(_U64.unpack_from(offsets_mm, base + 8)[0])


def build_named_coords_index(
    translation_tsv: str | Path,
    pack_dir: str | Path,
    out_dir: str | Path,
) -> Path:
    """Invert ``vg gbwt --translation`` T-lines into a dense node→(seg,off) index."""
    translation_tsv = Path(translation_tsv)
    pack_dir = Path(pack_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    off_path = pack_dir / "offsets.bin"
    meta_pack = json.loads((pack_dir / "meta.json").read_text(encoding="utf-8"))
    n_pack = int(meta_pack.get("n_segments") or (off_path.stat().st_size // 16))

    # First pass: count coverage / max node
    max_node = 0
    n_t = 0
    with translation_tsv.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("T\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[1]:
                continue
            n_t += 1
            for tok in parts[2].split(","):
                tok = tok.strip()
                if not tok:
                    continue
                nid = int(tok)
                if nid > max_node:
                    max_node = nid
    if max_node <= 0:
        raise RuntimeError(f"no T-nodes in {translation_tsv}")

    n_nodes = max(max_node, n_pack)
    nodes_path = out / "nodes.bin"
    filled = 0
    max_gfa_seg = 0
    size = (n_nodes + 1) * _U32U32.size
    # Pre-allocate zeros, then mmap read/write (portable vs ACCESS_WRITE quirks).
    with nodes_path.open("wb") as out_fh:
        out_fh.truncate(size)
        out_fh.flush()
    with off_path.open("rb") as off_fh, nodes_path.open("r+b") as out_fh:
        off_mm = mmap.mmap(off_fh.fileno(), 0, access=mmap.ACCESS_READ)
        out_mm = mmap.mmap(out_fh.fileno(), size, access=mmap.ACCESS_WRITE)
        try:
            with translation_tsv.open(
                "r", encoding="utf-8", errors="replace"
            ) as fh:
                for line in fh:
                    if not line.startswith("T\t"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3 or not parts[1]:
                        continue
                    try:
                        seg_id = int(parts[1])
                    except ValueError:
                        # Non-numeric GFA names unsupported in dense u32 index.
                        continue
                    if seg_id > max_gfa_seg:
                        max_gfa_seg = seg_id
                    offset = 0
                    for tok in parts[2].split(","):
                        tok = tok.strip()
                        if not tok:
                            continue
                        nid = int(tok)
                        if nid < 1 or nid > n_nodes:
                            raise RuntimeError(
                                f"node {nid} out of range 1..{n_nodes}"
                            )
                        _U32U32.pack_into(
                            out_mm, nid * _U32U32.size, seg_id, offset
                        )
                        offset += _node_length_from_pack(
                            memoryview(off_mm), nid
                        )
                        filled += 1
                        if filled % 5_000_000 == 0:
                            print(
                                f"named_coords: filled {filled} nodes…",
                                flush=True,
                            )
        finally:
            out_mm.flush()
            out_mm.close()
            off_mm.close()

    meta = {
        "format": FORMAT_V1,
        "n_nodes": n_nodes,
        "n_translation_lines": n_t,
        "n_filled": filled,
        "max_gfa_segment": max_gfa_seg,
        "pack": str(pack_dir.resolve()),
        "translation": str(translation_tsv.resolve()),
        "pack_n_segments": n_pack,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"named_coords: wrote {out} n_nodes={n_nodes} filled={filled} T_lines={n_t}",
        flush=True,
    )
    return out


class NamedCoordsIndex:
    """mmap lookup: GBZ node id → (gfa_segment_id, offset_in_segment)."""

    def __init__(self, index_dir: str | Path):
        self.root = Path(index_dir)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        if self.meta.get("format") != FORMAT_V1:
            raise ValueError(f"unsupported named_coords format in {self.root}")
        self.n_nodes = int(self.meta["n_nodes"])
        self._fh = (self.root / "nodes.bin").open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        expected = (self.n_nodes + 1) * _U32U32.size
        if len(self._mm) < expected:
            raise ValueError(
                f"nodes.bin too small: {len(self._mm)} < {expected} in {self.root}"
            )

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fh.close()

    def __enter__(self) -> "NamedCoordsIndex":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def lookup(self, node_id: int) -> Tuple[int, int]:
        if node_id < 1 or node_id > self.n_nodes:
            raise KeyError(node_id)
        seg, off = _U32U32.unpack_from(self._mm, node_id * _U32U32.size)
        if seg == 0:
            raise KeyError(node_id)
        return int(seg), int(off)


def parse_gaf_path(path: str) -> List[Tuple[str, str]]:
    """Return list of (orient, node_id_str) for a GAF path column."""
    return [(m.group(1), m.group(2)) for m in _PATH_NODE_RE.finditer(path)]


def translate_path(
    path: str, index: NamedCoordsIndex
) -> Tuple[str, int]:
    """Map a GBZ-node path to a GFA-segment path + path_start.

    Collapses consecutive chopped nodes that tile the same GFA segment.
    ``path_start`` is the offset of the first node within the first GFA
    segment (MethylCall indexes into the concatenation of full segment
    sequences on the path).
    """
    nodes = parse_gaf_path(path)
    if not nodes:
        return path, 0

    out_parts: List[str] = []
    path_start = 0
    prev_seg: Optional[int] = None
    first = True
    for orient, nid_s in nodes:
        try:
            nid = int(nid_s)
        except ValueError as exc:
            raise ValueError(f"non-numeric path node {nid_s!r} in {path!r}") from exc
        seg, off = index.lookup(nid)
        if first:
            path_start = off
            first = False
        if prev_seg is None or seg != prev_seg:
            out_parts.append(f"{orient}{seg}")
            prev_seg = seg
        # Same segment continuation: keep single path element; pstart already set.
    return "".join(out_parts), path_start


def translate_gaf_line(line: str, index: NamedCoordsIndex) -> str:
    """Rewrite one GAF row to named-coordinates (path + pstart/pend/plen)."""
    raw = line.rstrip("\n")
    if not raw or raw.startswith("#"):
        return line if line.endswith("\n") else line + "\n"
    parts = raw.split("\t")
    if len(parts) < 12:
        return line if line.endswith("\n") else line + "\n"
    path = parts[5]
    if path in {"*", ""}:
        return line if line.endswith("\n") else line + "\n"
    try:
        qlen = int(parts[1])
    except ValueError:
        qlen = 0
    try:
        old_pstart = int(parts[7])
    except ValueError:
        old_pstart = 0

    new_path, node_off = translate_path(path, index)
    # Preserve any intra-node pstart Mojo may have set (usually 0).
    pstart = node_off + old_pstart
    try:
        old_pend = int(parts[8])
        span = max(0, old_pend - old_pstart)
    except ValueError:
        span = qlen
    if span <= 0:
        span = qlen
    pend = pstart + span

    # plen: keep span-compatible length; MethylCall recomputes pend from cs/rl.
    parts[5] = new_path
    parts[6] = str(max(pend, int(parts[6]) if parts[6].isdigit() else pend))
    parts[7] = str(pstart)
    parts[8] = str(pend)
    return "\t".join(parts) + "\n"


def translate_gaf_file(
    in_gaf: str | Path,
    out_gaf: str | Path,
    index: NamedCoordsIndex,
) -> int:
    """Stream-translate a GAF file. Returns line count."""
    in_gaf = Path(in_gaf)
    out_gaf = Path(out_gaf)
    out_gaf.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with in_gaf.open("r", encoding="utf-8", errors="replace") as fin, out_gaf.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            fout.write(translate_gaf_line(line, index))
            n += 1
            if n % 2_000_000 == 0:
                print(f"named_coords: translated {n} GAF lines…", flush=True)
    return n


def default_index_dir(segments_cache: str | Path | None = None) -> Path:
    root = Path(
        segments_cache
        or os.environ.get("METHYLGRAPHER_MOJO_SEGMENTS_CACHE", "").strip()
        or "/work/cache/mojo_segments"
    )
    return root / "hprc-d9-bs.wl.gbz_to_gfa.named_coords"


def index_ready(index_dir: str | Path) -> bool:
    root = Path(index_dir)
    return (root / "meta.json").is_file() and (root / "nodes.bin").is_file()
