"""GBZ-native index access for Mojo Giraffe.

Production map path: dense segment pack + MinimizerIndex (``.min``) + zip/dist
via ``engine.quartet_map``. Fixture-scale ``extend_exact`` is quarantined for
tiny packs / empty minimizer indexes only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from engine.quartet_map import map_fastq_to_gaf as quartet_map_fastq_to_gaf
from engine.quartet_map import mojo_giraffe_ready
from engine.segment_pack import (
    SegmentPack,
    pack_ready,
    resolve_pack,
    segment_cache_ready,
)


def resolve_gbz_quartet(index_prefix: str) -> Optional[Dict[str, str]]:
    """Return paths for gbz/dist/min/zipcodes when all four exist."""
    p = Path(index_prefix)
    _ = p
    gbz = Path(f"{index_prefix}.giraffe.gbz")
    dist = Path(f"{index_prefix}.dist")
    min1 = Path(f"{index_prefix}.min")
    min2 = Path(f"{index_prefix}.shortread.withzip.min")
    zipc = Path(f"{index_prefix}.shortread.zipcodes")
    if not gbz.is_file():
        return None
    if not dist.is_file():
        return None
    min_path = min2 if min2.is_file() else (min1 if min1.is_file() else None)
    if min_path is None:
        return None
    if min2.is_file() and not zipc.is_file():
        return None
    return {
        "gbz": str(gbz.resolve()),
        "dist": str(dist.resolve()),
        "min": str(min_path.resolve()),
        "zipcodes": str(zipc.resolve()) if zipc.is_file() else "",
        "index_prefix": index_prefix,
    }


def _vg_bin() -> str:
    return os.environ.get("VG_PATH", "").strip() or shutil.which("vg") or "vg"


def stream_gbz_to_segments(gbz_path: str, vg_path: Optional[str] = None) -> Dict[str, str]:
    """Stream ``vg convert -f gfa`` and collect S-line segments only."""
    vg = vg_path or _vg_bin()
    cmd = [vg, "convert", "-f", gbz_path]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert proc.stdout is not None
    segments: Dict[str, str] = {}
    for line in proc.stdout:
        if not line.startswith("S\t"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        segments[parts[1]] = parts[2]
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"vg convert failed ({rc}): {stderr[:500]}")
    if not segments:
        raise RuntimeError(f"no segments decoded from GBZ {gbz_path}")
    return segments


def _default_cache_roots() -> List[Path]:
    roots: List[Path] = []
    env = os.environ.get("METHYLGRAPHER_MOJO_SEGMENTS_CACHE", "").strip()
    if env:
        roots.append(Path(env))
    for cand in (
        Path("/work/cache/mojo_segments"),
        Path("/lambda/nfs/Work/cache/mojo_segments"),
        Path(tempfile.gettempdir()) / "mojo_segments",
    ):
        if cand not in roots:
            roots.append(cand)
    return roots


def segment_cache_dir(gbz_path: str) -> Path:
    sibling = Path(gbz_path + ".mojo_segments")
    if sibling.is_dir() and pack_ready(sibling):
        return sibling
    try:
        parent = Path(gbz_path).resolve().parent
        if os.access(parent, os.W_OK):
            return sibling
    except OSError:
        pass
    name = Path(gbz_path).name + ".mojo_segments"
    for root in _default_cache_roots():
        return root / name
    return sibling


# Re-export readiness helpers used by align_backends
resolve_segment_cache = resolve_pack


def ensure_segment_cache(
    gbz_path: str,
    *,
    force: bool = False,
    vg_path: Optional[str] = None,
) -> Path:
    """Build or reuse on-disk segment pack (legacy jsonl via vg convert)."""
    existing = None if force else resolve_pack(gbz_path)
    if existing is not None:
        return existing
    cache = segment_cache_dir(gbz_path)
    meta = cache / "meta.json"
    pack = cache / "segments.jsonl"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        name = Path(gbz_path).name + ".mojo_segments"
        last_err: Optional[BaseException] = None
        cache = None  # type: ignore[assignment]
        for root in _default_cache_roots():
            cand = root / name
            try:
                cand.mkdir(parents=True, exist_ok=True)
                cache = cand
                break
            except OSError as exc:
                last_err = exc
        if cache is None:
            raise RuntimeError(
                f"cannot create mojo segment cache for {gbz_path}: {last_err}"
            ) from last_err
        meta = cache / "meta.json"
        pack = cache / "segments.jsonl"
    segments = stream_gbz_to_segments(gbz_path, vg_path=vg_path)
    with pack.open("w", encoding="utf-8") as fh:
        for seg_id, seq in segments.items():
            fh.write(json.dumps({"id": seg_id, "seq": seq}, separators=(",", ":")) + "\n")
    meta.write_text(
        json.dumps(
            {
                "gbz": str(Path(gbz_path).resolve()),
                "n_segments": len(segments),
                "format": "jsonl-v1",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return cache


def load_segments(gbz_path: str, *, vg_path: Optional[str] = None) -> Dict[str, str]:
    """Load segments from dense/jsonl pack or direct convert for tiny GBZ."""
    gbz = Path(gbz_path)
    max_direct = int(
        os.environ.get("METHYLGRAPHER_MOJO_GBZ_DIRECT_MAX_BYTES", str(64 * 1024 * 1024))
    )
    if gbz.is_file() and gbz.stat().st_size <= max_direct:
        try:
            return stream_gbz_to_segments(str(gbz), vg_path=vg_path)
        except Exception:
            pass
    ready = resolve_pack(str(gbz))
    if ready is None:
        cache = ensure_segment_cache(str(gbz), vg_path=vg_path)
    else:
        cache = ready
    return SegmentPack(cache).as_dict()


def extract_kmers(seq: str, k: int) -> List[str]:
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(0, len(seq) - k + 1)]


def build_kmer_index(segments: Dict[str, str], k: int = 5) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for seg_id, seq in segments.items():
        for i, mer in enumerate(extract_kmers(seq, k)):
            idx.setdefault(mer, []).append(f"{seg_id}:{i}")
    return idx


def seed_hits(kmer_index: Dict[str, List[str]], seq: str, k: int) -> List[str]:
    hits: List[str] = []
    for mer in extract_kmers(seq, k):
        hits.extend(kmer_index.get(mer, []))
    return hits


def extend_exact(segments: Dict[str, str], qname: str, seq: str, k: int = 5) -> List[dict]:
    """QUARANTINED fixture-scale exact / majority-seed extend — not production."""
    out: List[dict] = []
    qlen = len(seq)
    for seg_id, s in segments.items():
        if s == seq:
            out.append(
                {
                    "query_name": qname,
                    "path": f">{seg_id}",
                    "qlen": qlen,
                    "mapq": 60,
                    "cs_tag": f"cs:Z::{qlen}",
                    "extra_tags": "",
                }
            )
    if out:
        return out
    if len(segments) > 50_000:
        return out
    counts: Dict[str, int] = {}
    kidx = build_kmer_index(segments, k)
    for h in seed_hits(kidx, seq, k):
        seg = h.split(":", 1)[0]
        counts[seg] = counts.get(seg, 0) + 1
    if not counts:
        return out
    best_seg, best_n = max(counts.items(), key=lambda kv: kv[1])
    mq = 40 if best_n >= 2 else 20
    out.append(
        {
            "query_name": qname,
            "path": f">{best_seg}",
            "qlen": qlen,
            "mapq": mq,
            "cs_tag": f"cs:Z::{qlen}",
            "extra_tags": "",
        }
    )
    return out


def format_gaf_line(hit: dict) -> str:
    qlen = str(hit["qlen"])
    line = (
        f"{hit['query_name']}\t{qlen}\t0\t{qlen}\t+\t{hit['path']}\t"
        f"{qlen}\t0\t{qlen}\t{qlen}\t{qlen}\t{hit['mapq']}\t{hit['cs_tag']}"
    )
    extra = hit.get("extra_tags") or ""
    if extra:
        line += "\t" + extra
    return line


def map_gbz_fastq_to_gaf(
    *,
    gbz: str,
    fq1: str,
    out_gaf: str,
    fq2: str = "",
    dist: str = "",
    min_path: str = "",
    zipcodes: str = "",
    k: int = 5,
    device: str = "cpu",
) -> int:
    """Deprecated name — delegates to quartet mapper (min/zip/dist + dense pack)."""
    return quartet_map_fastq_to_gaf(
        gbz=gbz,
        fq1=fq1,
        out_gaf=out_gaf,
        fq2=fq2,
        dist=dist,
        min_path=min_path,
        zipcodes=zipcodes,
        k=k,
        device=device,
    )


def probe_gbz(gbz_path: str) -> dict:
    p = Path(gbz_path)
    cache = segment_cache_dir(gbz_path)
    return {
        "exists": p.is_file(),
        "bytes": p.stat().st_size if p.is_file() else 0,
        "cache": str(cache),
        "cache_ready": pack_ready(cache) if cache.exists() else False,
        "vg": _vg_bin(),
        "mojo_ready_env": mojo_giraffe_ready(),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="GBZ helper for Mojo Giraffe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_cache = sub.add_parser("cache", help="Build segment cache from GBZ")
    p_cache.add_argument("--gbz", required=True)
    p_cache.add_argument("--force", action="store_true")
    p_map = sub.add_parser("map", help="Map FASTQ → GAF via quartet")
    p_map.add_argument("--gbz", required=True)
    p_map.add_argument("--dist", default="")
    p_map.add_argument("--min", dest="min_path", default="")
    p_map.add_argument("--zipcodes", default="")
    p_map.add_argument("--fq1", required=True)
    p_map.add_argument("--fq2", default="")
    p_map.add_argument("--out_gaf", required=True)
    p_map.add_argument("-k", type=int, default=5)
    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--gbz", required=True)
    args = ap.parse_args()
    if args.cmd == "cache":
        d = ensure_segment_cache(args.gbz, force=args.force)
        print(d)
    elif args.cmd == "probe":
        print(json.dumps(probe_gbz(args.gbz), indent=2))
    else:
        n = map_gbz_fastq_to_gaf(
            gbz=args.gbz,
            dist=args.dist,
            min_path=args.min_path,
            zipcodes=args.zipcodes,
            fq1=args.fq1,
            fq2=args.fq2,
            out_gaf=args.out_gaf,
            k=args.k,
        )
        print(f"wrote {n} hits → {args.out_gaf}")
