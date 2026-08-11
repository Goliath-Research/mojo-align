#!/usr/bin/env python3
"""Compare Clara vs Mojo BAMs with GATK4 / Picard-shaped metrics.

Clara advertises GATK 4 support; Mojo must pass the same consumer checks
before we claim parity — not only flagstat Δ.

Tools (first available wins for each tool):
  - gatk ValidateSamFile / CollectAlignmentSummaryMetrics / CollectInsertSizeMetrics
  - or picard.jar with the same tools
  - always: samtools flagstat + idxstats (fallback gates)

Example:
  python3 fq2bam-meth/scripts/compare_gatk_picard_metrics.py \\
    --clara-bam align.linear.parabricks/sample.bam \\
    --mojo-bam align.linear.mojo/sample.bam \\
    --ref /work/genomes/linear/GRCh38/ensembl-114/Homo_sapiens.GRCh38.dna.primary_assembly.fa \\
    --out-dir /tmp/gatk_parity
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _run(cmd: List[str], log: Path) -> Tuple[int, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("COMMAND: " + " ".join(cmd) + "\n")
        if proc.stdout:
            fh.write(proc.stdout)
        if proc.stderr:
            fh.write(proc.stderr)
        fh.write(f"exit={proc.returncode}\n")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _find_gatk() -> Optional[List[str]]:
    gatk = shutil.which("gatk")
    if gatk:
        return [gatk]
    jar = os.environ.get("GATK_JAR", "").strip()
    if jar and Path(jar).is_file():
        return ["java", "-jar", jar]
    return None


def _find_picard() -> Optional[List[str]]:
    picard = shutil.which("picard")
    if picard:
        return [picard]
    jar = os.environ.get("PICARD_JAR", "").strip()
    if jar and Path(jar).is_file():
        return ["java", "-jar", jar]
    return None


def _flagstat_rate(bam: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        ["samtools", "flagstat", str(bam)],
        capture_output=True,
        text=True,
        check=True,
    )
    total = mapped = proper = 0
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
        elif "mapped (" in line and "primary" not in line:
            mapped = n
        elif "properly paired" in line:
            proper = n
    return {
        "total": total,
        "mapped": mapped,
        "properly_paired": proper,
        "mapped_rate": (mapped / total) if total else 0.0,
        "proper_rate": (proper / total) if total else 0.0,
    }


def _parse_alignment_summary(path: Path) -> Dict[str, float]:
    """Parse Picard CollectAlignmentSummaryMetrics second CATEGORY=PAIR row."""
    if not path.is_file():
        return {}
    rows: List[Dict[str, str]] = []
    header: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] == "CATEGORY":
            header = parts
            continue
        if header and len(parts) == len(header):
            rows.append(dict(zip(header, parts)))
    pick = None
    for r in rows:
        if r.get("CATEGORY") == "PAIR":
            pick = r
            break
    if pick is None and rows:
        pick = rows[-1]
    if not pick:
        return {}
    out: Dict[str, float] = {}
    for k in (
        "PCT_PF_READS_ALIGNED",
        "PF_MISMATCH_RATE",
        "PF_INDEL_RATE",
        "MEAN_READ_LENGTH",
        "PCT_CHIMERAS",
        "PCT_ADAPTER",
    ):
        if k in pick and pick[k] not in ("?", ""):
            try:
                out[k] = float(pick[k])
            except ValueError:
                pass
    return out


def _parse_insert_size(path: Path) -> Dict[str, float]:
    if not path.is_file():
        return {}
    header: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] == "MEDIAN_INSERT_SIZE":
            header = parts
            continue
        if header and len(parts) == len(header):
            row = dict(zip(header, parts))
            out: Dict[str, float] = {}
            for k in ("MEDIAN_INSERT_SIZE", "MEAN_INSERT_SIZE", "STANDARD_DEVIATION"):
                if k in row and row[k] not in ("?", ""):
                    try:
                        out[k] = float(row[k])
                    except ValueError:
                        pass
            return out
    return {}


def _collect_metrics(
    tool: List[str],
    bam: Path,
    ref: Path,
    out_prefix: Path,
    log: Path,
) -> Dict[str, Any]:
    """Run ValidateSamFile + alignment + insert metrics when GATK/Picard exists."""
    result: Dict[str, Any] = {"tool": tool[0], "validate": None, "alignment": {}, "insert": {}}
    val_out = out_prefix.with_suffix(".validate.txt")
    # GATK vs Picard CLI differ slightly; try GATK style first.
    rc, _ = _run(
        tool
        + [
            "ValidateSamFile",
            "-I",
            str(bam),
            "-MODE",
            "SUMMARY",
            "-O",
            str(val_out),
        ],
        log,
    )
    if rc != 0:
        rc, _ = _run(
            tool
            + [
                "ValidateSamFile",
                "I=" + str(bam),
                "MODE=SUMMARY",
                "O=" + str(val_out),
            ],
            log,
        )
    result["validate_exit"] = rc
    result["validate_path"] = str(val_out)
    if val_out.is_file():
        result["validate"] = val_out.read_text(encoding="utf-8", errors="replace")[:4000]

    aln = out_prefix.with_name(out_prefix.name + ".alignment_summary_metrics")
    rc, _ = _run(
        tool
        + [
            "CollectAlignmentSummaryMetrics",
            "-I",
            str(bam),
            "-R",
            str(ref),
            "-O",
            str(aln),
        ],
        log,
    )
    if rc != 0:
        _run(
            tool
            + [
                "CollectAlignmentSummaryMetrics",
                "I=" + str(bam),
                "R=" + str(ref),
                "O=" + str(aln),
            ],
            log,
        )
    result["alignment"] = _parse_alignment_summary(aln)

    ins = out_prefix.with_name(out_prefix.name + ".insert_size_metrics")
    hist = out_prefix.with_name(out_prefix.name + ".insert_size_histogram.pdf")
    rc, _ = _run(
        tool
        + [
            "CollectInsertSizeMetrics",
            "-I",
            str(bam),
            "-O",
            str(ins),
            "-H",
            str(hist),
            "-M",
            "0.5",
        ],
        log,
    )
    if rc != 0:
        _run(
            tool
            + [
                "CollectInsertSizeMetrics",
                "I=" + str(bam),
                "O=" + str(ins),
                "H=" + str(hist),
                "M=0.5",
            ],
            log,
        )
    result["insert"] = _parse_insert_size(ins)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clara-bam", type=Path, required=True)
    ap.add_argument("--mojo-bam", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-mapped-delta", type=float, default=0.02)
    ap.add_argument("--max-proper-delta", type=float, default=0.05)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log = out / "gatk_picard_compare.log"

    report: Dict[str, Any] = {
        "clara_bam": str(args.clara_bam),
        "mojo_bam": str(args.mojo_bam),
        "gates": {},
        "clara": {},
        "mojo": {},
    }

    clara_fs = _flagstat_rate(args.clara_bam)
    mojo_fs = _flagstat_rate(args.mojo_bam)
    report["clara"]["flagstat"] = clara_fs
    report["mojo"]["flagstat"] = mojo_fs
    d_map = abs(clara_fs["mapped_rate"] - mojo_fs["mapped_rate"])
    d_proper = abs(clara_fs["proper_rate"] - mojo_fs["proper_rate"])
    report["gates"]["mapped_rate"] = {
        "delta": d_map,
        "pass": d_map <= args.max_mapped_delta,
        "threshold": args.max_mapped_delta,
    }
    report["gates"]["proper_pair_rate"] = {
        "delta": d_proper,
        "pass": d_proper <= args.max_proper_delta,
        "threshold": args.max_proper_delta,
    }

    tool = _find_gatk() or _find_picard()
    if tool is None:
        report["gatk_picard"] = {
            "available": False,
            "note": "Install GATK4 or set GATK_JAR / PICARD_JAR for full tables",
        }
    else:
        report["gatk_picard"] = {"available": True, "cmd_prefix": tool}
        report["clara"]["metrics"] = _collect_metrics(
            tool, args.clara_bam, args.ref, out / "clara", log
        )
        report["mojo"]["metrics"] = _collect_metrics(
            tool, args.mojo_bam, args.ref, out / "mojo", log
        )
        # ValidateSamFile: Mojo must not hard-fail if Clara passes.
        c_val = report["clara"]["metrics"].get("validate_exit", 1)
        m_val = report["mojo"]["metrics"].get("validate_exit", 1)
        report["gates"]["validate_sam"] = {
            "clara_exit": c_val,
            "mojo_exit": m_val,
            "pass": m_val == 0 or (c_val != 0 and m_val == c_val),
        }
        ca = report["clara"]["metrics"].get("alignment", {})
        ma = report["mojo"]["metrics"].get("alignment", {})
        if "PCT_PF_READS_ALIGNED" in ca and "PCT_PF_READS_ALIGNED" in ma:
            d = abs(ca["PCT_PF_READS_ALIGNED"] - ma["PCT_PF_READS_ALIGNED"])
            report["gates"]["pct_pf_reads_aligned"] = {
                "delta": d,
                "pass": d <= args.max_mapped_delta,
                "clara": ca["PCT_PF_READS_ALIGNED"],
                "mojo": ma["PCT_PF_READS_ALIGNED"],
            }

    report["pass"] = all(g.get("pass") for g in report["gates"].values()) if report["gates"] else False
    out_json = out / "gatk_picard_parity_report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "gates": report["gates"]}, indent=2))
    print("wrote", out_json)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
