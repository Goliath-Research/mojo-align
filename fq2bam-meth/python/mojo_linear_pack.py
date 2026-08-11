"""Dense Mojo linear k-mer pack (dense-v1) — fleet sibling of bwameth.

Layout under ``${REF}.mojo_linear_k${k}/``:

  meta.json      format=dense-v1, k, n_keys, n_postings, contigs
  kmers.bin      uint64 LE, sorted unique 2-bit-encoded k-mers
  offsets.bin    uint64 LE, length n_keys+1 (CSR starts into postings)
  postings.bin   uint32 LE pairs (contig_id, pos) — 8 bytes/posting

k ≤ 31 (fits in uint64 with 2 bits/base). Ambiguous bases skip the window.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

_BASE_CODE = np.full(256, 255, dtype=np.uint8)
_BASE_CODE[ord("A")] = 0
_BASE_CODE[ord("C")] = 1
_BASE_CODE[ord("G")] = 2
_BASE_CODE[ord("T")] = 3
_BASE_CODE[ord("a")] = 0
_BASE_CODE[ord("c")] = 1
_BASE_CODE[ord("g")] = 2
_BASE_CODE[ord("t")] = 3


def pack_is_complete(cache_dir: Path) -> bool:
    return (
        (cache_dir / "meta.json").is_file()
        and (cache_dir / "kmers.bin").is_file()
        and (cache_dir / "offsets.bin").is_file()
        and (cache_dir / "postings.bin").is_file()
    )


def _ensure_ref_link(cache_dir: Path, c2t_fasta: Path) -> None:
    """Point ``cache_dir/ref.fa`` at the C2T FASTA (symlink preferred)."""
    ref_link = cache_dir / "ref.fa"
    c2t_fasta = Path(c2t_fasta)
    if ref_link.exists() or ref_link.is_symlink():
        return
    try:
        ref_link.symlink_to(c2t_fasta.resolve())
    except OSError:
        import shutil

        shutil.copy2(c2t_fasta, ref_link)


def ensure_dense_pack(
    *,
    c2t_fasta: Path,
    cache_dir: Path,
    k: int = 15,
    log: Optional[Callable[[str], None]] = None,
) -> Path:
    """Return a complete dense-v1 pack, building under an exclusive flock if missing.

    Concurrent workers serialize on ``${cache_dir}.lock``. After the lock is
    acquired the pack is re-checked so only the first builder runs.
    Set ``METHYLGRAPHER_LINEAR_CACHE_BUILD=0`` to fail instead of building.
    """
    c2t_fasta = Path(c2t_fasta)
    cache_dir = Path(cache_dir)

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg, flush=True)

    if pack_is_complete(cache_dir):
        _ensure_ref_link(cache_dir, c2t_fasta)
        return cache_dir

    auto = os.environ.get("METHYLGRAPHER_LINEAR_CACHE_BUILD", "1").strip().lower()
    if auto in ("0", "false", "no", "off"):
        raise RuntimeError(
            f"Mojo dense-v1 pack missing at {cache_dir} "
            f"(METHYLGRAPHER_LINEAR_CACHE_BUILD={auto})"
        )

    if not c2t_fasta.is_file():
        raise FileNotFoundError(f"C2T FASTA required to build pack: {c2t_fasta}")

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(cache_dir) + ".lock")
    _log(f"mojo_linear_pack: waiting for lock {lock_path}")
    t0 = time.time()
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        _log(f"mojo_linear_pack: acquired lock after {time.time() - t0:.1f}s")
        if pack_is_complete(cache_dir):
            _log(f"mojo_linear_pack: pack already complete (other worker) → {cache_dir}")
            _ensure_ref_link(cache_dir, c2t_fasta)
            return cache_dir
        _log(f"mojo_linear_pack: building dense-v1 k={k} → {cache_dir}")
        build_dense_pack(c2t_fasta, cache_dir, k=k)
        _ensure_ref_link(cache_dir, c2t_fasta)
        lock_fh.write(f"built k={k} t={time.time():.0f}\n")
        lock_fh.flush()
    return cache_dir


def read_fasta_contigs(path: Path) -> List[Tuple[str, bytes]]:
    contigs: List[Tuple[str, bytes]] = []
    name = ""
    chunks: List[bytes] = []
    with path.open("rb") as fh:
        for raw in fh:
            if raw.startswith(b">"):
                if name:
                    contigs.append((name, b"".join(chunks).upper()))
                header = raw[1:].decode("ascii", errors="replace").strip()
                name = header.split()[0] if header else "unnamed"
                chunks = []
            else:
                chunks.append(raw.strip())
    if name:
        contigs.append((name, b"".join(chunks).upper()))
    return contigs


def encode_kmers_contig(seq: bytes, k: int, contig_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (kmers u64, locs u64 packed as contig_id<<32|pos) for one contig."""
    n = len(seq)
    if n < k:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    codes = _BASE_CODE[np.frombuffer(seq, dtype=np.uint8)].astype(np.uint64)
    n_win = n - k + 1
    # Window invalid if any base > 3 (N/other → 255 in table).
    bad = codes > 3
    # rolling OR of bad flags across window via cumsum trick
    bad_u32 = bad.astype(np.uint32)
    cumbad = np.zeros(n + 1, dtype=np.uint32)
    cumbad[1:] = np.cumsum(bad_u32)
    window_bad = (cumbad[k:] - cumbad[:-k]) > 0

    # 2-bit pack: sum codes[i+j] << 2*(k-1-j)
    kmers = np.zeros(n_win, dtype=np.uint64)
    for j in range(k):
        shift = np.uint64(2 * (k - 1 - j))
        kmers |= codes[j : j + n_win] << shift

    pos = np.arange(n_win, dtype=np.uint64)
    locs = (np.uint64(contig_id) << np.uint64(32)) | pos
    keep = ~window_bad
    # Also drop any window that still has high bits from 255 codes
    keep &= kmers < (np.uint64(1) << np.uint64(2 * k))
    return kmers[keep], locs[keep]


def build_dense_pack(c2t_fasta: Path, out_dir: Path, k: int = 15) -> dict:
    if k < 1 or k > 31:
        raise ValueError(f"k must be in 1..31, got {k}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading FASTA {c2t_fasta}", flush=True)
    contigs = read_fasta_contigs(c2t_fasta)
    if not contigs:
        raise RuntimeError(f"no contigs in {c2t_fasta}")
    names = [n for n, _ in contigs]
    print(f"contigs={len(contigs)} bases={sum(len(s) for _, s in contigs)}", flush=True)

    kmer_parts: List[np.ndarray] = []
    loc_parts: List[np.ndarray] = []
    for cid, (name, seq) in enumerate(contigs):
        km, loc = encode_kmers_contig(seq, k, cid)
        if km.size:
            kmer_parts.append(km)
            loc_parts.append(loc)
        if (cid + 1) % 5 == 0 or cid + 1 == len(contigs):
            print(f"  encoded {cid + 1}/{len(contigs)} ({name}, {len(seq)} bp)", flush=True)

    if not kmer_parts:
        raise RuntimeError("no valid k-mers (check alphabet / k)")

    kmers = np.concatenate(kmer_parts)
    locs = np.concatenate(loc_parts)
    del kmer_parts, loc_parts
    n_postings = int(kmers.size)
    print(f"sorting {n_postings} postings…", flush=True)
    order = np.argsort(kmers, kind="mergesort")
    kmers = kmers[order]
    locs = locs[order]
    del order

    print("CSR unique keys…", flush=True)
    # Boundaries where kmer changes
    change = np.ones(n_postings, dtype=bool)
    change[1:] = kmers[1:] != kmers[:-1]
    starts = np.nonzero(change)[0].astype(np.uint64)
    uniq = kmers[starts]
    n_keys = int(uniq.size)
    offsets = np.empty(n_keys + 1, dtype=np.uint64)
    offsets[:-1] = starts
    offsets[-1] = np.uint64(n_postings)

    # Expand locs → postings u32 pairs
    print(f"writing bins n_keys={n_keys} n_postings={n_postings}", flush=True)
    contig_ids = (locs >> np.uint64(32)).astype(np.uint32)
    positions = (locs & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    del locs
    postings = np.empty(n_postings * 2, dtype=np.uint32)
    postings[0::2] = contig_ids
    postings[1::2] = positions
    del contig_ids, positions

    (out_dir / "kmers.bin").write_bytes(uniq.astype("<u8", copy=False).tobytes())
    (out_dir / "offsets.bin").write_bytes(offsets.astype("<u8", copy=False).tobytes())
    (out_dir / "postings.bin").write_bytes(postings.astype("<u4", copy=False).tobytes())
    (out_dir / "contigs.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    meta = {
        "format": "dense-v1",
        "k": k,
        "n_keys": n_keys,
        "n_postings": n_postings,
        "contigs": names,
        "c2t_fasta": str(c2t_fasta),
        "endian": "little",
        "kmer_encoding": "2bit-ACGT",
        "posting": "u32_contig_id,u32_pos0",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    # Remove obsolete text cache if present
    legacy = out_dir / "hits.tsv"
    if legacy.is_file():
        legacy.unlink()
    print(f"OK dense-v1 → {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build dense-v1 Mojo linear k-mer pack")
    p.add_argument("--ref", required=True, help="C2T FASTA (or raw FASTA to index as-is)")
    p.add_argument("--out", required=True, help="Output pack directory")
    p.add_argument("-k", type=int, default=15)
    args = p.parse_args(argv)
    build_dense_pack(Path(args.ref), Path(args.out), k=args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
