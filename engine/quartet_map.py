"""Oracle / reference GBZ-quartet map (Python).

Production Align uses Mojo ``giraffe_stream_map`` (see ``src/giraffe_gbz.mojo``).
This module remains for:
  - ``ensure_pack_for_gbz`` (one-time dense pack resolve/build)
  - ``mojo_giraffe_ready`` selection gate
  - parity / bakeoff oracle via ``map_fastq_to_gaf``
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

from engine.minimizer_index import MinimizerIndex, MinHit
from engine.segment_pack import (
    SegmentPack,
    build_dense_pack_from_gfa,
    build_dense_pack_from_gbz,
    resolve_pack,
)
from engine.stage_timer import StageTimer
from engine.zipcodes_index import DistIndex, ZipcodesIndex

# Bound peak RAM on Buffy-scale FASTQs (hundreds of GB uncompressed).
# Operator override: METHYLGRAPHER_MOJO_READ_BATCH (pairs / SE reads per chunk).
_DEFAULT_READ_BATCH = 8192


class FastqRec(NamedTuple):
    """Converted FASTQ record; ``original_seq`` feeds MethylCall ``os:Z``."""

    name: str
    seq: str
    original_seq: str
    conversion: str  # C2T / G2A / ""


def _read_batch_size() -> int:
    raw = os.environ.get("METHYLGRAPHER_MOJO_READ_BATCH", "").strip()
    if not raw:
        return _DEFAULT_READ_BATCH
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_READ_BATCH
    return max(1, n)


def _parse_mg_fastq_header(name: str, body: str) -> FastqRec:
    """Parse ``{qname}_{C2T|G2A}_{shard}_{original}``; else os falls back to body."""
    parts = name.split("_")
    if len(parts) >= 4 and parts[1] in ("C2T", "G2A"):
        bare = parts[0].split(" ", 1)[0]
        original = "_".join(parts[3:])
        return FastqRec(bare, body, original, parts[1])
    bare = parts[0].split(" ", 1)[0] if parts else name
    return FastqRec(bare, body, body, "")


def _rc_from_conversion(conversion: str, fallback: str) -> str:
    if conversion == "C2T":
        return "CT"
    if conversion == "G2A":
        return "GA"
    return fallback


def _pe_extra_tags(ri: int, rec: FastqRec, fallback_rc: str) -> str:
    rc = _rc_from_conversion(rec.conversion, fallback_rc)
    return f"ri:i:{ri}\tos:Z:{rec.original_seq}\trc:Z:{rc}"


def _iter_fastq(path: str) -> Iterator[FastqRec]:
    """Stream FASTQ records without loading the file into memory."""
    with open(path, encoding="utf-8") as fh:
        while True:
            n = fh.readline()
            if not n:
                break
            s = fh.readline().rstrip("\n\r")
            fh.readline()
            fh.readline()
            name = n[1:].strip() if n.startswith("@") else n.strip()
            yield _parse_mg_fastq_header(name, s)


def _parse_fastq(path: str) -> List[FastqRec]:
    """Load entire FASTQ (tests / tiny fixtures only — do not use on Buffy)."""
    return list(_iter_fastq(path))


def _iter_fastq_batches(
    fq1: str, fq2: str = "", *, batch_size: Optional[int] = None
) -> Iterator[List[Tuple[FastqRec, Optional[FastqRec]]]]:
    """Yield batches of (read1, optional read2)."""
    bs = batch_size if batch_size is not None else _read_batch_size()
    it1 = _iter_fastq(fq1)
    it2 = _iter_fastq(fq2) if fq2 else None
    batch: List[Tuple[FastqRec, Optional[FastqRec]]] = []
    while True:
        try:
            r1 = next(it1)
        except StopIteration:
            break
        r2: Optional[FastqRec] = None
        if it2 is not None:
            try:
                r2 = next(it2)
            except StopIteration as exc:
                raise RuntimeError(
                    f"paired FASTQ length mismatch: {fq2} ended before {fq1}"
                ) from exc
        batch.append((r1, r2))
        if len(batch) >= bs:
            yield batch
            batch = []
    if batch:
        yield batch
    if it2 is not None:
        try:
            next(it2)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                f"paired FASTQ length mismatch: {fq1} ended before {fq2}"
            )


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

    # Multi-node heuristic for chopped graphs: try neighboring node ids when
    # the seed node is shorter than the read (common on HPRC d9-bs).
    if len(ref) < len(q) and len(pack) <= 50_000_000:
        path_parts = [f">{node_id}"]
        covered = best[0] if best else 0
        qi = covered if isinstance(covered, int) else 0
        # Prefer forward neighbor ids in the dense pack.
        for nxt in (node_id + 1, node_id - 1, node_id + 2):
            if nxt == node_id or nxt < 0:
                continue
            nseg = pack.get(str(nxt))
            if not nseg:
                continue
            nref = nseg.upper()
            if is_rev:
                nref = nref.translate(comp)[::-1]
            rem = q[qi:]
            if not rem:
                break
            matched = 0
            for a, b in zip(rem, nref):
                if a != b:
                    break
                matched += 1
            if matched < 15:
                continue
            path_parts.append(f">{nxt}")
            qi += matched
            if qi >= int(0.9 * len(q)):
                mq = 50 if qi >= len(q) else 35
                return "".join(path_parts), mq, f"cs:Z::{qi}"
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
    seeds: Optional[List[MinHit]] = None,
) -> List[dict]:
    hits_out: List[dict] = []
    if seeds is None:
        seeds = []
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

    # Fixture / empty-min fallback (toy packs only). Production packs are
    # 100M+ nodes — never scan them for exact/k-mer fallback (was ~80s/read).
    return _fixture_extend(pack, qname, seq, k_fallback)


def _seed_batch_hits(
    *,
    min_index: Optional[MinimizerIndex],
    seqs: List[str],
    device: str,
) -> Tuple[List[List[MinHit]], str]:
    """GPU (CuPy) or CPU minimizer seed → per-read MinHit lists.

    Prefer **in-process** ``giraffe_gpu_minimizer.minimizers_batch_gpu`` (no JSON
    IPC). Fall back to ``gpu_seed_worker.py`` when Mojo PYTHONHOME blocks CuPy,
    then to host ``locate_read``.
    """
    if min_index is None or min_index.n_keys == 0 or not seqs:
        return [[] for _ in seqs], "no-index"
    dev = (device or "cpu").strip().lower() or "cpu"
    backend = "cpu"
    if dev in {"nvidia", "cuda", "amd", "hip", "rocm"}:
        # 1) In-process CuPy / host minimizer (no subprocess JSON).
        try:
            import sys
            from engine.minimizer_index import MinimizerOcc

            scripts = Path(__file__).resolve().parents[1] / "scripts"
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            from giraffe_gpu_minimizer import minimizers_batch_gpu  # type: ignore

            occs_batch, backend = minimizers_batch_gpu(
                seqs, k=int(min_index.k), w=int(min_index.w), device=dev
            )
            hits = [
                min_index.locate_from_minimizers(list(occs), hit_cap=24)
                for occs in occs_batch
            ]
            if len(hits) != len(seqs):
                raise RuntimeError(
                    f"in-process GPU seed returned {len(hits)} rows for {len(seqs)} seqs"
                )
            print(f"quartet_map seed_backend={backend}+inprocess", flush=True)
            return hits, backend + "+inprocess"
        except Exception as exc_in:
            print(f"in-process GPU seed unavailable ({exc_in}); try worker", flush=True)
        # 2) System-python worker (CuPy outside Mojo PYTHONHOME).
        try:
            import json
            import subprocess
            from engine.minimizer_index import MinimizerOcc

            worker = Path(__file__).resolve().parents[1] / "scripts" / "gpu_seed_worker.py"
            if not worker.is_file():
                worker = Path("/opt/methylgrapher-mojo/scripts/gpu_seed_worker.py")
            payload = {
                "seqs": seqs,
                "k": int(min_index.k),
                "w": int(min_index.w),
                "device": dev,
            }
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in {"PYTHONHOME", "PYTHONPATH", "LD_PRELOAD"}
            }
            env["PYTHONPATH"] = "/opt/methylgrapher-mojo/scripts:/opt/methylgrapher-mojo"
            proc = subprocess.run(
                ["/usr/bin/python3", str(worker)],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"gpu_seed_worker exit {proc.returncode}: {proc.stderr[-500:]}"
                )
            data = json.loads(proc.stdout)
            backend = str(data.get("backend") or "cupy")
            hits: List[List[MinHit]] = []
            for occs_raw in data.get("occs", []):
                occs = [
                    MinimizerOcc(
                        int(o["key"]),
                        int(o["hash"]),
                        int(o["offset"]),
                        bool(o["is_reverse"]),
                    )
                    for o in occs_raw
                ]
                hits.append(min_index.locate_from_minimizers(occs, hit_cap=24))
            if len(hits) != len(seqs):
                raise RuntimeError(
                    f"gpu_seed_worker returned {len(hits)} rows for {len(seqs)} seqs"
                )
            print(f"quartet_map seed_backend={backend}+worker", flush=True)
            return hits, backend + "+worker"
        except Exception as exc:
            print(f"GPU minimizer seed unavailable ({exc}); CPU locate", flush=True)
            backend = "host-locate-fallback"
    hits = [min_index.locate_read(s, hit_cap=24) for s in seqs]
    return hits, backend


def _fixture_extend(
    pack: SegmentPack, qname: str, seq: str, k: int
) -> List[dict]:
    # Guard FIRST: production dense packs must not enter pack.ids()/items().
    if len(pack) > 50_000:
        return []
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
    """Map FASTQ using quartet indexes + dense pack; write GAF.

    Streams reads in batches (``METHYLGRAPHER_MOJO_READ_BATCH``, default 8192)
    and writes GAF incrementally so Buffy-scale FASTQs do not OOM.

    ``device`` selects the seed backend: ``nvidia``/``amd`` use CuPy GPU
    minimizer extraction (plus Mojo DeviceContext warmup from the caller);
    ``cpu`` keeps the host Python locate path.
    """
    dev = (device or os.environ.get("METHYLGRAPHER_GIRAFFE_DEVICE") or "cpu")
    dev = str(dev).strip().lower() or "cpu"
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

    Path(out_gaf).parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    n_written = 0
    batch_size = _read_batch_size()
    seed_backend = "unset"
    timer = StageTimer.from_env()
    workers_raw = os.environ.get("METHYLGRAPHER_MOJO_EXTEND_WORKERS", "4").strip()
    try:
        extend_workers = max(1, int(workers_raw))
    except ValueError:
        extend_workers = 4
    print(
        f"quartet_map device={dev} batch={batch_size} extend_workers={extend_workers} "
        f"mojo_backend={os.environ.get('METHYLGRAPHER_LAST_GPU_BACKEND', '')}",
        flush=True,
    )
    try:
        with open(out_gaf, "w", encoding="utf-8") as fh:
            for batch in _iter_fastq_batches(fq1, fq2, batch_size=batch_size):
                with timer.stage("fastq_batch"):
                    flat_seqs: List[str] = []
                    for r1, r2 in batch:
                        flat_seqs.append(r1.seq)
                        if r2 is not None:
                            flat_seqs.append(r2.seq)
                with timer.stage("seed_locate"):
                    seed_hits, seed_backend = _seed_batch_hits(
                        min_index=min_index, seqs=flat_seqs, device=dev
                    )
                os.environ["METHYLGRAPHER_LAST_SEED_BACKEND"] = seed_backend

                def _map_pair(item):
                    (r1, r2), si_local = item
                    if r2 is None:
                        hits = map_one_read(
                            pack=pack,
                            min_index=min_index,
                            zipcodes=zip_index,
                            dist=dist_index,
                            qname=r1.name,
                            seq=r1.seq,
                            k_fallback=k,
                            seeds=seed_hits[si_local],
                        )
                        return 1, hits
                    h1s = map_one_read(
                        pack=pack,
                        min_index=min_index,
                        zipcodes=zip_index,
                        dist=dist_index,
                        qname=r1.name,
                        seq=r1.seq,
                        k_fallback=k,
                        seeds=seed_hits[si_local],
                    )
                    h2s = map_one_read(
                        pack=pack,
                        min_index=min_index,
                        zipcodes=zip_index,
                        dist=dist_index,
                        qname=r2.name,
                        seq=r2.seq,
                        k_fallback=k,
                        seeds=seed_hits[si_local + 1],
                    )
                    h1 = h1s[0] if h1s else {
                        "query_name": r1.name, "path": "*", "qlen": len(r1.seq),
                        "mapq": 0, "cs_tag": "cs:Z:*", "extra_tags": "",
                    }
                    h2 = h2s[0] if h2s else {
                        "query_name": r2.name, "path": "*", "qlen": len(r2.seq),
                        "mapq": 0, "cs_tag": "cs:Z:*", "extra_tags": "",
                    }
                    h1["extra_tags"] = _pe_extra_tags(1, r1, "CT")
                    h2["extra_tags"] = _pe_extra_tags(2, r2, "GA")
                    return 2, [h1, h2]

                work = []
                si = 0
                for r1, r2 in batch:
                    work.append(((r1, r2), si))
                    si += 1 if r2 is None else 2

                with timer.stage("cluster_extend"):
                    if extend_workers <= 1 or len(work) < 8:
                        mapped = [_map_pair(w) for w in work]
                    else:
                        with ThreadPoolExecutor(max_workers=extend_workers) as pool:
                            mapped = list(pool.map(_map_pair, work))

                with timer.stage("gaf_emit"):
                    for n_rec, hits in mapped:
                        n_records += n_rec
                        for h in hits:
                            if h["path"] == "*":
                                continue
                            fh.write(format_gaf_line(h) + "\n")
                            n_written += 1
                fh.flush()
    finally:
        if min_index is not None:
            min_index.close()
        if zip_index is not None:
            zip_index.close()
        timer.write()

    # Return mapped+unmapped record count (PE = 2 per pair) for CLI parity.
    _ = n_written
    return n_records


def mojo_giraffe_ready() -> bool:
    """Whether Mojo GBZ may be selected for ``gpu_giraffe`` / ``mojo_giraffe``.

    Default **on** (production Mojo path). Opt out with
    ``METHYLGRAPHER_MOJO_GIRAFFE_READY=0`` / ``false`` / ``off`` to force
    ``vg_autoscale`` while keeping ``align_engine=gpu_giraffe``.
    """
    raw = os.environ.get("METHYLGRAPHER_MOJO_GIRAFFE_READY", "1").strip().lower()
    if raw in {"0", "false", "no", "off", "vg"}:
        return False
    return True
