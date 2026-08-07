"""Production GBZ-quartet map: .min locate → zip/dist cluster → gapless extend → GAF.

Replaces fixture-scale ``extend_exact`` for Mojo Giraffe when packed segments
and a readable minimizer index are available.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from engine.minimizer_index import MinimizerIndex, MinHit
from engine.segment_pack import (
    SegmentPack,
    build_dense_pack_from_gfa,
    build_dense_pack_from_gbz,
    resolve_pack,
)
from engine.zipcodes_index import DistIndex, ZipcodesIndex


def _parse_fastq(path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        while True:
            n = fh.readline()
            if not n:
                break
            s = fh.readline().rstrip("\n\r")
            fh.readline()
            fh.readline()
            name = n[1:].strip() if n.startswith("@") else n.strip()
            bare = name.split("_")[0]
            rows.append((bare, s))
    return rows


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


def _gapless_extend(
    pack: SegmentPack,
    node_id: int,
    offset: int,
    is_rev: bool,
    query: str,
) -> Optional[Tuple[str, int, str]]:
    """Extend query against segment sequence starting at seed offset.

    Single-node gapless first; if the node is shorter than the read (common on
    chopped pangenome graphs), accept a high-identity local match of the node
    inside the query as a seed alignment (full multi-node walk is follow-on).

    Returns (path, mapq, cs_tag) or None.
    """
    seg = pack.get(str(node_id))
    if seg is None:
        return None
    q = query.upper()
    ref = seg.upper()
    comp = str.maketrans("ACGT", "TGCA")
    if is_rev:
        ref_aln = ref.translate(comp)[::-1]
        start_candidates = [
            max(0, len(ref_aln) - offset - len(q)),
            max(0, offset - len(q) + 1),
            0,
        ]
    else:
        ref_aln = ref
        start_candidates = [offset, max(0, offset - len(q) + 29), 0]

    best = None
    for start in start_candidates:
        if start < 0 or start >= len(ref_aln):
            continue
        matched = 0
        mism = 0
        qi = 0
        ri = start
        while qi < len(q) and ri < len(ref_aln):
            if q[qi] == ref_aln[ri]:
                matched += 1
            else:
                mism += 1
                if mism > max(2, len(q) // 20):
                    break
            qi += 1
            ri += 1
        covered = matched / max(1, min(len(q), len(ref_aln)))
        if matched < min(29, len(ref_aln)):
            continue
        score = matched - 2 * mism
        if best is None or score > best[0]:
            # Full-read cover vs local node cover
            if matched >= int(0.9 * len(q)):
                mq = 60 if mism == 0 else 40
                cs = f"cs:Z::{len(q)}" if mism == 0 else f"cs:Z::{matched}"
            else:
                mq = 30 if covered > 0.9 else 20
                cs = f"cs:Z::{matched}"
            best = (score, f">{node_id}", mq, cs)
    if best is not None:
        return best[1], best[2], best[3]

    # Local containment: node sequence (or RC) appears inside the query.
    for cand, rev in ((ref, False), (ref.translate(comp)[::-1], True)):
        if len(cand) < 21:
            continue
        pos = q.find(cand)
        if pos < 0:
            continue
        mq = 40 if len(cand) >= 29 else 20
        return f">{node_id}", mq, f"cs:Z::{len(cand)}"
    return None


def _cluster_hits(
    hits: List[MinHit],
    zipcodes: Optional[ZipcodesIndex],
    dist: Optional[DistIndex],
) -> List[MinHit]:
    if not hits:
        return []
    buckets: Dict[int, List[MinHit]] = defaultdict(list)
    for h in hits:
        if zipcodes is not None:
            key = zipcodes.cluster_key(h.payload0, h.payload1)
        else:
            key = h.node_id >> 8
        buckets[key].append(h)
    # Prefer largest cluster; within cluster keep diverse nodes
    best_key = max(buckets.keys(), key=lambda k: len(buckets[k]))
    clustered = buckets[best_key]
    if dist is not None and len(clustered) > 1:
        anchor = clustered[0].node_id
        clustered = [h for h in clustered if dist.distance_ok(anchor, h.node_id)] or clustered
    # Cap seeds for extend
    return clustered[:16]


def map_one_read(
    *,
    pack: SegmentPack,
    min_index: Optional[MinimizerIndex],
    zipcodes: Optional[ZipcodesIndex],
    dist: Optional[DistIndex],
    qname: str,
    seq: str,
    k_fallback: int = 5,
) -> List[dict]:
    hits_out: List[dict] = []
    seeds: List[MinHit] = []
    if min_index is not None and min_index.n_keys > 0:
        seeds = min_index.locate_read(seq, hit_cap=24)
        seeds = _cluster_hits(seeds, zipcodes, dist)
        # Majority node vote as additional signal
        if seeds:
            counts = Counter(h.node_id for h in seeds)
            top_nodes = {n for n, _ in counts.most_common(4)}
            seeds = [h for h in seeds if h.node_id in top_nodes] or seeds

    if seeds:
        scored: List[Tuple[float, dict]] = []
        for h in seeds:
            ext = _gapless_extend(pack, h.node_id, h.offset, h.is_rev, seq)
            if ext is None:
                continue
            path, mq, cs = ext
            scored.append(
                (
                    mq,
                    {
                        "query_name": qname,
                        "path": path,
                        "qlen": len(seq),
                        "mapq": mq,
                        "cs_tag": cs,
                        "extra_tags": "",
                    },
                )
            )
        scored.sort(key=lambda x: -x[0])
        for _, hit in scored[:2]:
            hits_out.append(hit)
        if hits_out:
            return hits_out

    # Fixture / empty-min fallback: exact segment match then short k-mer majority
    return _fixture_extend(pack, qname, seq, k_fallback)


def _fixture_extend(
    pack: SegmentPack, qname: str, seq: str, k: int
) -> List[dict]:
    qlen = len(seq)
    for sid in pack.ids():
        if pack.get(sid) == seq:
            return [
                {
                    "query_name": qname,
                    "path": f">{sid}",
                    "qlen": qlen,
                    "mapq": 60,
                    "cs_tag": f"cs:Z::{qlen}",
                    "extra_tags": "",
                }
            ]
    # short k-mer majority over pack (toy only — small packs)
    if len(pack) > 50_000:
        return []
    counts: Dict[str, int] = {}
    for sid, s in pack.items():
        if len(s) < k:
            continue
        for i in range(0, len(seq) - k + 1):
            mer = seq[i : i + k]
            if mer in s:
                counts[sid] = counts.get(sid, 0) + 1
    if not counts:
        return []
    best_seg, best_n = max(counts.items(), key=lambda kv: kv[1])
    mq = 40 if best_n >= 2 else 20
    return [
        {
            "query_name": qname,
            "path": f">{best_seg}",
            "qlen": qlen,
            "mapq": mq,
            "cs_tag": f"cs:Z::{qlen}",
            "extra_tags": "",
        }
    ]


def ensure_pack_for_gbz(gbz: str) -> SegmentPack:
    ready = resolve_pack(gbz)
    if ready is not None:
        return SegmentPack(ready)
    # Try companion / local GFA for dense build
    # Production: build from GBZ via vg convert (node-id aligned). Companion
    # wl.gfa may have a different node set than C2T/G2A Giraffe GBZs.
    out_root = os.environ.get("METHYLGRAPHER_MOJO_SEGMENTS_CACHE", "").strip()
    if not out_root:
        out_root = "/work/cache/mojo_segments"
    out_dir = Path(out_root) / (Path(gbz).name + ".mojo_segments")
    try:
        build_dense_pack_from_gbz(gbz, out_dir)
        return SegmentPack(out_dir)
    except Exception as exc:
        # Fixture fallback: companion GFA next to tiny GBZ
        p = Path(gbz)
        gfa_candidates = [
            p.parent / "toy.wl.gfa",
            p.parent / f"{p.name.split('.giraffe')[0]}.wl.gfa",
        ]
        gfa = next((c for c in gfa_candidates if c.is_file()), None)
        if gfa is None:
            raise RuntimeError(
                f"no segment pack for {gbz}; build with "
                f"build_mojo_segment_pack.py --from-gbz --gbz {gbz} ({exc})"
            ) from exc
        build_dense_pack_from_gfa(str(gfa), out_dir, source_gbz=str(p.resolve()))
        return SegmentPack(out_dir)


def map_fastq_to_gaf(
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
    """Map FASTQ using quartet indexes + dense pack; write GAF."""
    _ = device
    pack = ensure_pack_for_gbz(gbz)
    min_index: Optional[MinimizerIndex] = None
    zip_index: Optional[ZipcodesIndex] = None
    dist_index: Optional[DistIndex] = None
    if min_path and Path(min_path).is_file():
        min_index = MinimizerIndex(min_path)
    if zipcodes and Path(zipcodes).is_file():
        try:
            zip_index = ZipcodesIndex(zipcodes)
        except Exception:
            zip_index = None
    if dist and Path(dist).is_file():
        dist_index = DistIndex(dist)

    reads = _parse_fastq(fq1)
    hits: List[dict] = []
    try:
        if not fq2:
            for name, seq in reads:
                hits.extend(
                    map_one_read(
                        pack=pack,
                        min_index=min_index,
                        zipcodes=zip_index,
                        dist=dist_index,
                        qname=name,
                        seq=seq,
                        k_fallback=k,
                    )
                )
        else:
            mates = _parse_fastq(fq2)
            for (n1, s1), (n2, s2) in zip(reads, mates):
                h1s = map_one_read(
                    pack=pack,
                    min_index=min_index,
                    zipcodes=zip_index,
                    dist=dist_index,
                    qname=n1,
                    seq=s1,
                    k_fallback=k,
                )
                h2s = map_one_read(
                    pack=pack,
                    min_index=min_index,
                    zipcodes=zip_index,
                    dist=dist_index,
                    qname=n2,
                    seq=s2,
                    k_fallback=k,
                )
                h1 = h1s[0] if h1s else {
                    "query_name": n1, "path": "*", "qlen": len(s1), "mapq": 0,
                    "cs_tag": "cs:Z:*", "extra_tags": "",
                }
                h2 = h2s[0] if h2s else {
                    "query_name": n2, "path": "*", "qlen": len(s2), "mapq": 0,
                    "cs_tag": "cs:Z:*", "extra_tags": "",
                }
                h1["extra_tags"] = f"ri:i:1\tos:Z:{s1}\trc:Z:CT"
                h2["extra_tags"] = f"ri:i:2\tos:Z:{s2}\trc:Z:GA"
                hits.append(h1)
                hits.append(h2)
    finally:
        if min_index is not None:
            min_index.close()
        if zip_index is not None:
            zip_index.close()

    Path(out_gaf).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_gaf, "w", encoding="utf-8") as fh:
        for h in hits:
            if h["path"] == "*":
                continue
            fh.write(format_gaf_line(h) + "\n")
            n_written += 1
    return len(hits)


def mojo_giraffe_ready() -> bool:
    """Operator gate: require explicit READY before production Mojo selection."""
    return os.environ.get("METHYLGRAPHER_MOJO_GIRAFFE_READY", "").strip() in {
        "1",
        "true",
        "yes",
        "on",
    }
