#!/usr/bin/env python3
"""Compare align.linear.parabricks vs align.linear.mojo BAM concordance.

Expects the per-path sample layout from mojo-align:

  <sample-dir>/
    align.linear.parabricks/<sampleId>.bam
    align.linear.mojo/<sampleId>.bam

Primary gate: |Δ mapped_rate| ≤ --max-delta (default 0.02), matching
MethylPipeline mojo-fq2bam concordance gates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


ALIGN_PARABRICKS = "align.linear.parabricks"
ALIGN_MOJO = "align.linear.mojo"


def _run_flagstat(bam: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        ["samtools", "flagstat", str(bam)],
        capture_output=True,
        text=True,
        check=True,
    )
    total = mapped = primary_mapped = 0
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            n = int(parts[0])
        except ValueError:
            continue
        if "in total" in line:
            total = n
        elif "primary mapped" in line:
            primary_mapped = n
        elif "mapped (" in line and "primary" not in line:
            mapped = n
    if total <= 0:
        raise RuntimeError(f"no total reads in flagstat for {bam}")
    return {
        "total": total,
        "mapped": mapped,
        "primary_mapped": primary_mapped or mapped,
        "mapped_rate": mapped / total,
        "primary_mapped_rate": (primary_mapped or mapped) / total,
        "raw": proc.stdout,
    }


def _idxstats_counts(bam: Path) -> Dict[str, int]:
    proc = subprocess.run(
        ["samtools", "idxstats", str(bam)],
        capture_output=True,
        text=True,
        check=True,
    )
    out: Dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        chrom, _length, mapped = parts[0], parts[1], parts[2]
        if chrom == "*":
            continue
        out[chrom] = int(mapped)
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank Spearman without numpy (small chrom vectors)."""
    n = len(xs)
    if n < 2:
        return 1.0

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    den_x = sum((rx[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    den_y = sum((ry[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return 1.0 if xs == ys else 0.0
    return num / (den_x * den_y)


def resolve_bams(
    sample_dir: Path, sample_id: str
) -> Tuple[Path, Path]:
    clara = sample_dir / ALIGN_PARABRICKS / f"{sample_id}.bam"
    mojo = sample_dir / ALIGN_MOJO / f"{sample_id}.bam"
    return clara, mojo


def compare(
    *,
    sample_dir: Path,
    sample_id: str,
    max_delta: float,
    min_idxstats_spearman: float,
) -> Dict[str, Any]:
    clara_bam, mojo_bam = resolve_bams(sample_dir, sample_id)
    for bam in (clara_bam, mojo_bam):
        if not bam.is_file():
            raise FileNotFoundError(f"missing BAM: {bam}")
        bai = Path(str(bam) + ".bai")
        if not bai.is_file():
            # try .bam.bai already covered; also csi
            subprocess.run(
                ["samtools", "index", str(bam)], check=False, capture_output=True
            )

    f_clara = _run_flagstat(clara_bam)
    f_mojo = _run_flagstat(mojo_bam)
    delta = abs(f_clara["mapped_rate"] - f_mojo["mapped_rate"])

    idx_clara = _idxstats_counts(clara_bam)
    idx_mojo = _idxstats_counts(mojo_bam)
    # Gate on primary assembly contigs (1–22, X, Y, MT). Alt/decoy/random
    # contigs are sparse on 100k smokes and dominate all-contig Spearman noise
    # without reflecting WGBS science concordance.
    primary = {str(i) for i in range(1, 23)} | {"X", "Y", "MT", "chrM", "chrX", "chrY"}
    primary |= {f"chr{i}" for i in range(1, 23)}
    chroms_all = sorted(set(idx_clara) | set(idx_mojo))
    chroms = [c for c in chroms_all if c in primary]
    if len(chroms) < 2:
        chroms = chroms_all
    xs = [float(idx_clara.get(c, 0)) for c in chroms]
    ys = [float(idx_mojo.get(c, 0)) for c in chroms]
    spearman = _spearman(xs, ys) if chroms else 1.0
    xs_all = [float(idx_clara.get(c, 0)) for c in chroms_all]
    ys_all = [float(idx_mojo.get(c, 0)) for c in chroms_all]
    spearman_all = _spearman(xs_all, ys_all) if chroms_all else 1.0

    mapped_pass = delta <= max_delta
    idx_pass = spearman >= min_idxstats_spearman
    report: Dict[str, Any] = {
        "sampleId": sample_id,
        "sampleDir": str(sample_dir),
        "paths": {
            ALIGN_PARABRICKS: str(clara_bam),
            ALIGN_MOJO: str(mojo_bam),
        },
        "clara": {
            "mapped_rate": f_clara["mapped_rate"],
            "primary_mapped_rate": f_clara["primary_mapped_rate"],
            "total": f_clara["total"],
            "mapped": f_clara["mapped"],
        },
        "mojo": {
            "mapped_rate": f_mojo["mapped_rate"],
            "primary_mapped_rate": f_mojo["primary_mapped_rate"],
            "total": f_mojo["total"],
            "mapped": f_mojo["mapped"],
        },
        "abs_delta_mapped_rate": delta,
        "max_delta": max_delta,
        "idxstats_spearman": spearman,
        "idxstats_spearman_all_contigs": spearman_all,
        "idxstats_chroms": chroms,
        "min_idxstats_spearman": min_idxstats_spearman,
        "gates": {
            "mapped_rate": mapped_pass,
            "idxstats_spearman": idx_pass,
        },
        "pass": mapped_pass and idx_pass,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sample-dir",
        type=Path,
        required=True,
        help="Sample root containing align.linear.parabricks/ and align.linear.mojo/",
    )
    p.add_argument("--sample-id", required=True)
    p.add_argument(
        "--max-delta",
        type=float,
        default=0.02,
        help="Max |Δ| mapped_rate (default 0.02)",
    )
    p.add_argument(
        "--min-idxstats-spearman",
        type=float,
        default=0.95,
        help="Min Spearman on per-contig mapped counts (default 0.95)",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write report JSON (default: <sample-dir>/linear_parity_report.json)",
    )
    args = p.parse_args(argv)

    try:
        report = compare(
            sample_dir=args.sample_dir.resolve(),
            sample_id=args.sample_id,
            max_delta=args.max_delta,
            min_idxstats_spearman=args.min_idxstats_spearman,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out = args.out_json or (args.sample_dir / "linear_parity_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}", file=sys.stderr)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
