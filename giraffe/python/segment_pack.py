"""Dense mmap-friendly segment pack for Mojo Giraffe extend.

Layout under ``{name}.mojo_segments/``:

- ``meta.json`` — format, counts, source paths
- ``ids.txt`` — one segment id per line (order = index)
- ``offsets.bin`` — uint64 little-endian pairs ``(offset, length)`` per segment
- ``sequences.bin`` — concatenated raw ASCII sequences

Also accepts legacy ``segments.jsonl`` (jsonl-v1) for readiness checks.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

FORMAT_DENSE_V1 = "dense-v1"
FORMAT_JSONL_V1 = "jsonl-v1"
_U64 = struct.Struct("<Q")


def pack_dir_candidates(gbz_path: str) -> List[Path]:
    candidates = [Path(gbz_path + ".mojo_segments")]
    name = Path(gbz_path).name + ".mojo_segments"
    env = os.environ.get("METHYLGRAPHER_MOJO_SEGMENTS_CACHE", "").strip()
    roots: List[Path] = []
    if env:
        roots.append(Path(env))
    for cand in (
        Path("/work/cache/mojo_segments"),
        Path("/lambda/nfs/Work/cache/mojo_segments"),
    ):
        if cand not in roots:
            roots.append(cand)
    for root in roots:
        candidates.append(root / name)
    return candidates


def pack_ready(cache: Path) -> bool:
    meta = cache / "meta.json"
    if not meta.is_file():
        return False
    if (cache / "sequences.bin").is_file() and (cache / "offsets.bin").is_file():
        return True
    if (cache / "segments.jsonl").is_file():
        return True
    return False


def resolve_pack(gbz_path: str) -> Optional[Path]:
    for cache in pack_dir_candidates(gbz_path):
        if pack_ready(cache):
            return cache
    return None


def segment_cache_ready(gbz_path: str) -> bool:
    return resolve_pack(gbz_path) is not None


def build_dense_pack_from_gbz(
    gbz_path: str,
    out_dir: str | Path,
    *,
    vg_path: Optional[str] = None,
) -> Path:
    """Stream ``vg convert -f`` S-lines into a dense pack (GBZ node-id aligned)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vg = vg_path or os.environ.get("VG_PATH", "").strip() or "vg"
    # --no-translation: emit GBZ handlegraph node ids (must match .min Position ids).
    cmd = [
        vg,
        "convert",
        "-f",
        "--gbwtgraph-algorithm",
        "--no-translation",
        gbz_path,
    ]
    print(f"segment_pack: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert proc.stdout is not None
    n = 0
    total_bp = 0
    with (
        (out / "ids.txt").open("w", encoding="utf-8") as ids_fh,
        (out / "offsets.bin").open("wb") as off_fh,
        (out / "sequences.bin").open("wb") as seq_fh,
    ):
        for line in proc.stdout:
            if not line.startswith("S\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            seg_id, seq = parts[1], parts[2]
            raw = seq.encode("ascii", errors="ignore")
            off_fh.write(_U64.pack(total_bp))
            off_fh.write(_U64.pack(len(raw)))
            seq_fh.write(raw)
            ids_fh.write(seg_id + "\n")
            total_bp += len(raw)
            n += 1
            if n % 2_000_000 == 0:
                print(f"segment_pack: {n} nodes, {total_bp} bp", flush=True)
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"vg convert failed ({rc}): {stderr[:800]}")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "format": FORMAT_DENSE_V1,
                "n_segments": n,
                "n_bp": total_bp,
                "gbz": str(Path(gbz_path).resolve()),
                "source": "vg-convert",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"segment_pack: wrote {out} n_segments={n} n_bp={total_bp}", flush=True)
    return out


def build_dense_pack_from_gfa(
    gfa_path: str,
    out_dir: str | Path,
    *,
    source_gbz: str = "",
) -> Path:
    """Stream GFA S-lines into a dense pack (ids/offsets/sequences/meta)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "ids.txt"
    off_path = out / "offsets.bin"
    seq_path = out / "sequences.bin"
    meta_path = out / "meta.json"

    n = 0
    total_bp = 0
    with (
        open(gfa_path, "r", encoding="utf-8", errors="replace") as gfa,
        ids_path.open("w", encoding="utf-8") as ids_fh,
        off_path.open("wb") as off_fh,
        seq_path.open("wb") as seq_fh,
    ):
        for line in gfa:
            if not line.startswith("S\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            seg_id, seq = parts[1], parts[2]
            if not seg_id:
                continue
            raw = seq.encode("ascii", errors="ignore")
            off_fh.write(_U64.pack(total_bp))
            off_fh.write(_U64.pack(len(raw)))
            seq_fh.write(raw)
            ids_fh.write(seg_id + "\n")
            total_bp += len(raw)
            n += 1
            if n % 5_000_000 == 0:
                print(f"segment_pack: {n} segments, {total_bp} bp", flush=True)

    meta = {
        "format": FORMAT_DENSE_V1,
        "n_segments": n,
        "n_bp": total_bp,
        "gfa": str(Path(gfa_path).resolve()),
        "gbz": source_gbz,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"segment_pack: wrote {out} n_segments={n} n_bp={total_bp}", flush=True)
    return out


class SegmentPack:
    """mmap-backed id → sequence lookup."""

    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)
        meta_path = self.root / "meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        self.format = str(self.meta.get("format", ""))
        self._id_to_idx: Dict[str, int] = {}
        self._ids: List[str] = []
        self._offsets: Optional[memoryview] = None
        self._seqs: Optional[memoryview] = None
        self._jsonl: Optional[Dict[str, str]] = None
        self._load()

    def _load(self) -> None:
        dense = (self.root / "sequences.bin").is_file() and (self.root / "offsets.bin").is_file()
        if dense:
            # Prefer mmap for production packs (multi-GB).
            # If ids are contiguous 1..N, skip the giant dict (145M nodes).
            self._off_fh = (self.root / "offsets.bin").open("rb")
            self._seq_fh = (self.root / "sequences.bin").open("rb")
            self._off_mm = mmap.mmap(self._off_fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._seq_mm = mmap.mmap(self._seq_fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._offsets = memoryview(self._off_mm)
            self._seqs = memoryview(self._seq_mm)
            n = len(self._off_mm) // 16
            self._n = n
            self._contiguous = False
            self._ids = []
            self._id_to_idx = {}
            # Contiguous 1..N: meta + first/last id probes (constant time).
            meta_n = int(self.meta.get("n_segments") or 0)
            first = ""
            last = ""
            with (self.root / "ids.txt").open("rb") as fh:
                first = fh.readline().decode("utf-8", errors="replace").strip()
                if n > 1:
                    fh.seek(0, os.SEEK_END)
                    size = fh.tell()
                    # Read a small tail window to get the last line.
                    win = min(256, size)
                    fh.seek(-win, os.SEEK_END)
                    tail = fh.read(win).decode("utf-8", errors="replace")
                    last = tail.strip().splitlines()[-1].strip() if tail.strip() else ""
            if first == "1" and last == str(n) and (meta_n == 0 or meta_n == n):
                self._contiguous = True
            if not self._contiguous:
                ids = (self.root / "ids.txt").read_text(encoding="utf-8").splitlines()
                self._ids = ids
                self._id_to_idx = {sid: i for i, sid in enumerate(ids)}
            self.format = FORMAT_DENSE_V1
            return
        jsonl = self.root / "segments.jsonl"
        if jsonl.is_file():
            out: Dict[str, str] = {}
            with jsonl.open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    out[str(row["id"])] = str(row["seq"])
            self._jsonl = out
            self._ids = list(out.keys())
            self._id_to_idx = {sid: i for i, sid in enumerate(self._ids)}
            self.format = FORMAT_JSONL_V1
            return
        raise FileNotFoundError(f"no dense or jsonl pack in {self.root}")

    def __len__(self) -> int:
        if getattr(self, "_contiguous", False):
            return int(getattr(self, "_n", 0))
        return len(self._ids)

    def ids(self) -> List[str]:
        if getattr(self, "_contiguous", False):
            return [str(i) for i in range(1, self._n + 1)]
        return list(self._ids)

    def get(self, seg_id: str) -> Optional[str]:
        if getattr(self, "_contiguous", False):
            try:
                nid = int(seg_id)
            except ValueError:
                return None
            if nid < 1 or nid > self._n:
                return None
            return self.get_by_index(nid - 1)
        idx = self._id_to_idx.get(str(seg_id))
        if idx is None:
            return None
        return self.get_by_index(idx)

    def get_by_index(self, idx: int) -> str:
        if self._jsonl is not None:
            return self._jsonl[self._ids[idx]]
        assert self._offsets is not None and self._seqs is not None
        base = idx * 16
        offset = _U64.unpack_from(self._offsets, base)[0]
        length = _U64.unpack_from(self._offsets, base + 8)[0]
        return bytes(self._seqs[offset : offset + length]).decode("ascii")

    def items(self) -> Iterator[Tuple[str, str]]:
        n = len(self)
        for i in range(n):
            sid = str(i + 1) if getattr(self, "_contiguous", False) else self._ids[i]
            yield sid, self.get_by_index(i)

    def as_dict(self) -> Dict[str, str]:
        return {sid: seq for sid, seq in self.items()}
