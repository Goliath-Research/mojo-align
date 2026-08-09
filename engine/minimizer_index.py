"""mmap reader for gbwtgraph MinimizerIndex v11 (``Q1Q1`` / ``.shortread.withzip.min``).

Pure-Python locate path for Mojo Giraffe seed stage — no libvg FFI.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
TAG_Q1Q1 = 0x31513151
NO_KEY = 0x7FFFFFFFFFFFFFFF
IS_POINTER = 1 << 63
KEY_MASK = NO_KEY
OFFSET_BITS = 10
REV_MASK = 1 << OFFSET_BITS
OFF_MASK = REV_MASK - 1
PACK = {ord("A"): 0, ord("C"): 1, ord("G"): 2, ord("T"): 3,
        ord("a"): 0, ord("c"): 1, ord("g"): 2, ord("t"): 3}


def wang_hash_64(key: int) -> int:
    """gbwt::wang_hash_64 (Thomas Wang) — used by gbwtgraph Key64::hash()."""
    key &= (1 << 64) - 1
    key = ((~key) + (key << 21)) & ((1 << 64) - 1)  # (key << 21) - key - 1
    key ^= key >> 24
    key = (key + (key << 3) + (key << 8)) & ((1 << 64) - 1)  # key * 265
    key ^= key >> 14
    key = (key + (key << 2) + (key << 4)) & ((1 << 64) - 1)  # key * 21
    key ^= key >> 28
    key = (key + (key << 31)) & ((1 << 64) - 1)
    return key


@dataclass(frozen=True)
class MinHit:
    node_id: int
    is_rev: bool
    offset: int
    payload0: int = 0
    payload1: int = 0


@dataclass(frozen=True)
class MinimizerOcc:
    key: int
    hash: int
    offset: int
    is_reverse: bool


def decode_position(code: int) -> Tuple[int, bool, int]:
    node_id = code >> (OFFSET_BITS + 1)
    is_rev = bool(code & REV_MASK)
    offset = code & OFF_MASK
    return node_id, is_rev, offset


def encode_key64(seq: bytes, k: int) -> Optional[int]:
    if len(seq) != k:
        return None
    key = 0
    for b in seq:
        p = PACK.get(b if isinstance(b, int) else ord(b))
        if p is None:
            return None
        key = ((key << 2) | p) & ((1 << (2 * k)) - 1)
    return key


class MinimizerIndex:
    """Memory-map a ``.withzip.min`` file and locate seeds."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = self.path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.tag = _U32.unpack_from(self._mm, 0)[0]
        self.version = _U32.unpack_from(self._mm, 4)[0]
        if self.tag != TAG_Q1Q1:
            raise ValueError(f"not a MinimizerIndex (tag={self.tag:#x}) at {path}")
        (
            self.k,
            self.w,
            self.n_keys,
            self.unused,
            self.capacity,
            self.n_values,
            self.n_unique,
            self.flags,
        ) = struct.unpack_from("<8Q", self._mm, 8)
        self.key_bits = self.flags & 0xFF
        self.payload_size = (self.flags >> 12) & 0xF
        self.uses_syncmers = bool(self.flags & 0x0100)
        if self.key_bits != 64:
            raise ValueError(f"only 64-bit keys supported (got {self.key_bits})")
        self.cell_size = 1 + 1 + self.payload_size  # key + pos + payload words
        # Tags (gbwt::Tags / StringArray SDSL) currently fixed 40 bytes for
        # payload=zipcodes indexes produced by vg 1.70 — HT size marker at 112.
        self.ht_size_offset = 112
        self.ht_words = _U64.unpack_from(self._mm, self.ht_size_offset)[0]
        if self.ht_words % self.cell_size != 0:
            raise ValueError(
                f"hash table words {self.ht_words} not divisible by cell_size {self.cell_size}"
            )
        self.cell_count = self.ht_words // self.cell_size
        if self.cell_count == 0 or (self.cell_count & (self.cell_count - 1)) != 0:
            raise ValueError(f"cell_count must be power-of-two, got {self.cell_count}")
        self.ht_data_offset = self.ht_size_offset + 8
        self.ht_bytes = self.ht_words * 8
        self._lists_offset = self.ht_data_offset + self.ht_bytes
        # pointer-cell array_offset (in words) → file offset of occurrence vector
        self._ptr_file_off: dict[int, int] = {}
        self._ptr_indexed = False
        import os

        # Unique keys dominate; defer multi-hit list index until a pointer hit.
        if os.environ.get("METHYLGRAPHER_MIN_INDEX_POINTERS", "lazy").strip().lower() in {
            "1",
            "true",
            "yes",
            "eager",
        }:
            self._index_pointer_lists()

    def close(self) -> None:
        self._mm.close()
        self._fh.close()

    def __enter__(self) -> "MinimizerIndex":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _index_pointer_lists(self) -> None:
        """Walk HT once; record file offsets of multi-hit vectors (cell order)."""
        if self._ptr_indexed:
            return
        off = self._lists_offset
        mm = self._mm
        cs = self.cell_size
        base = self.ht_data_offset
        for array_off in range(0, self.ht_words, cs):
            key = _U64.unpack_from(mm, base + array_off * 8)[0]
            if not (key & IS_POINTER):
                continue
            self._ptr_file_off[array_off] = off
            n = _U64.unpack_from(mm, off)[0]
            off += 8 + n * 8
        self._frequent_offset = off
        self._ptr_indexed = True

    def _cell_key(self, array_off: int) -> int:
        return _U64.unpack_from(self._mm, self.ht_data_offset + array_off * 8)[0]

    def _cell_words(self, array_off: int) -> Tuple[int, ...]:
        base = self.ht_data_offset + array_off * 8
        return tuple(
            _U64.unpack_from(self._mm, base + i * 8)[0] for i in range(self.cell_size)
        )

    def find_offset(self, key: int) -> Optional[int]:
        key &= KEY_MASK
        h = wang_hash_64(key)
        cell_count = self.cell_count
        cell_offset = h & (cell_count - 1)
        for attempt in range(cell_count):
            array_off = cell_offset * self.cell_size
            cell_key = self._cell_key(array_off)
            bare = cell_key & KEY_MASK
            if bare == NO_KEY or bare == key:
                if bare == NO_KEY:
                    return None
                return array_off
            cell_offset = (cell_offset + attempt + 1) & (cell_count - 1)
        return None

    def _hits_at(self, array_off: int) -> List[MinHit]:
        words = self._cell_words(array_off)
        key = words[0]
        out: List[MinHit] = []
        value_size = 1 + self.payload_size
        if key & IS_POINTER:
            # Full pointer-list index is O(cell_count) over a 32 GB table — too
            # expensive to build lazily mid-map. Skip multi-hit keys unless the
            # operator eagerly indexed (METHYLGRAPHER_MIN_INDEX_POINTERS=eager).
            if not self._ptr_indexed:
                return out
            file_off = self._ptr_file_off.get(array_off)
            if file_off is None:
                return out
            n = _U64.unpack_from(self._mm, file_off)[0]
            data = file_off + 8
            for i in range(0, n, value_size):
                pos = _U64.unpack_from(self._mm, data + i * 8)[0]
                p0 = (
                    _U64.unpack_from(self._mm, data + (i + 1) * 8)[0]
                    if self.payload_size > 0
                    else 0
                )
                p1 = (
                    _U64.unpack_from(self._mm, data + (i + 2) * 8)[0]
                    if self.payload_size > 1
                    else 0
                )
                nid, rev, off = decode_position(pos)
                if nid:
                    out.append(MinHit(nid, rev, off, p0, p1))
            return out
        pos = words[1]
        p0 = words[2] if self.payload_size > 0 else 0
        p1 = words[3] if self.payload_size > 1 else 0
        nid, rev, off = decode_position(pos)
        if nid:
            out.append(MinHit(nid, rev, off, p0, p1))
        return out

    def find(self, key: int) -> List[MinHit]:
        array_off = self.find_offset(key)
        if array_off is None:
            return []
        return self._hits_at(array_off)

    def window_bp(self) -> int:
        return self.k + self.w - 1

    def minimizers(self, seq: str) -> List[MinimizerOcc]:
        """Giraffe-style (k,w) minimizers (forward windows; reverse-complement aware)."""
        if self.uses_syncmers:
            raise NotImplementedError("syncmer indexes not supported yet")
        s = seq.encode("ascii", errors="ignore")
        total = len(s)
        win = self.window_bp()
        if total < win:
            return []
        k = self.k
        # Collect per-window minimum hash among k-mers / revcomps.
        # Simplified but hash-compatible with Key64 wang hash on packed keys.
        results: List[MinimizerOcc] = []
        next_read_offset = 0
        last_hash: Optional[int] = None
        last_offset = -1
        for window_start in range(0, total - win + 1):
            best: Optional[MinimizerOcc] = None
            for i in range(self.w):
                pos = window_start + i
                chunk = s[pos : pos + k]
                fk = encode_key64(chunk, k)
                if fk is None:
                    best = None
                    break
                # reverse complement key
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
            if (
                not results
                or last_hash == best.hash
                or last_offset < best.offset
            ):
                if best.offset >= next_read_offset:
                    occ = best
                    if occ.is_reverse:
                        occ = MinimizerOcc(
                            occ.key, occ.hash, occ.offset + k - 1, True
                        )
                    results.append(occ)
                    next_read_offset = best.offset + 1
                    last_hash = best.hash
                    last_offset = best.offset
        results.sort(key=lambda m: m.offset)
        return results

    def locate_from_minimizers(
        self, occs: Sequence[MinimizerOcc], *, hit_cap: int = 32
    ) -> List[MinHit]:
        """Graph hits from precomputed minimizers (CPU or GPU seed stage)."""
        hits: List[MinHit] = []
        seen: set[Tuple[int, int, int]] = set()
        for occ in occs:
            found = self.find(occ.key)
            if len(found) > hit_cap:
                found = found[:hit_cap]
            for h in found:
                key = (h.node_id, int(h.is_rev), h.offset)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(h)
        return hits

    def locate_read(self, seq: str, *, hit_cap: int = 32) -> List[MinHit]:
        """Minimizers → graph hits (capped per minimizer)."""
        return self.locate_from_minimizers(self.minimizers(seq), hit_cap=hit_cap)

    def locate_key_batches(
        self, keys_batch: Sequence[Sequence[int]], *, hit_cap: int = 24
    ) -> List[List[str]]:
        """Batch HT lookup for Mojo-native minimizer keys → ``node:orient:offset``.

        One Python call per FASTQ batch (avoids per-key Mojo↔Python round-trips).
        """
        out: List[List[str]] = []
        for keys in keys_batch:
            hits: List[str] = []
            seen: set[Tuple[int, int, int]] = set()
            for key in keys:
                found = self.find(int(key))
                if len(found) > hit_cap:
                    found = found[:hit_cap]
                for h in found:
                    sk = (h.node_id, int(h.is_rev), h.offset)
                    if sk in seen:
                        continue
                    seen.add(sk)
                    hits.append(f"{h.node_id}:{int(h.is_rev)}:{h.offset}")
            out.append(hits)
        return out


def probe_minimizer(path: str | Path) -> dict:
    with MinimizerIndex(path) as idx:
        return {
            "path": str(path),
            "version": idx.version,
            "k": idx.k,
            "w": idx.w,
            "n_keys": idx.n_keys,
            "capacity": idx.capacity,
            "cell_count": idx.cell_count,
            "payload_size": idx.payload_size,
            "ht_data_offset": idx.ht_data_offset,
            "n_pointer_lists": len(idx._ptr_file_off),
        }
