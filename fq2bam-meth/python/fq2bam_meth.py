"""MojoFq2bamMeth — portable linear WGBS Align (Clara fq2bam_meth substitute).

Directional C↔T / G↔A conversion + **Mojo linear GPU mapper** (NVIDIA / AMD)
against a C2T-converted reference, then samtools sort + index. Emits a
Parabricks-shaped metrics JSON subset and qc-metrics directory consumed by
methyl_alignment_qc.

Mapper selection (``METHYLGRAPHER_LINEAR_MAPPER``):

- ``mojo`` (default when GPU/device path ready) — ``src/linear_mapper.mojo``
- ``bwa`` — CPU fallback via BWA-MEM

Device selection mirrors Giraffe: ``-device auto|cpu|nvidia|amd`` /
``METHYLGRAPHER_ALIGN_DEVICE``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _convert_base(base: str, mode: str) -> str:
    b = base.upper()
    if mode == "C2T":
        return "T" if b == "C" else base
    if mode == "G2A":
        return "A" if b == "G" else base
    return base


def convert_fastq(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(dst).endswith(".gz") else open
    with _open_text(src) as fin, opener(dst, "wt", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i % 4 == 1:
                fout.write("".join(_convert_base(ch, mode) for ch in line.rstrip("\n")) + "\n")
            else:
                fout.write(line if line.endswith("\n") else line + "\n")


def convert_fasta_c2t(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", errors="replace") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if line.startswith(">"):
                fout.write(line)
            else:
                fout.write("".join("T" if ch in "Cc" else ch for ch in line))


def _run(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.stdout:
            log.write(proc.stdout)
        if proc.stderr:
            log.write(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                proc.stderr.strip() or proc.stdout.strip() or f"command failed: {cmd[0]}"
            )


def _ensure_bwa_index(fasta: Path, log_path: Path) -> None:
    if (Path(str(fasta) + ".bwt")).is_file() or (Path(str(fasta) + ".bwt.2bit.64")).is_file():
        return
    bwa = shutil.which("bwa")
    if not bwa:
        raise RuntimeError("bwa not found on PATH (required for BWA fallback)")
    _run([bwa, "index", str(fasta)], log_path)


def _parse_flagstat(text: str) -> Dict[str, int]:
    out = {"total": 0, "mapped": 0, "paired": 0, "properly_paired": 0, "duplicates": 0}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            n = int(parts[0])
        except ValueError:
            continue
        if "in total" in line:
            out["total"] = n
        elif "mapped (" in line:
            out["mapped"] = n
        elif "paired in sequencing" in line:
            out["paired"] = n
        elif "properly paired" in line:
            out["properly_paired"] = n
        elif "duplicates" in line:
            out["duplicates"] = n
    return out


def _parse_samtools_stats(text: str) -> Dict[str, float]:
    """Extract useful SN fields from ``samtools stats``."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("SN"):
            continue
        # SN\tkey:\tvalue
        m = re.match(r"SN\t([^:]+):\t([0-9.]+)", line)
        if not m:
            continue
        key = m.group(1).strip().lower().replace(" ", "_")
        try:
            out[key] = float(m.group(2))
        except ValueError:
            continue
    return out


def write_parabricks_shaped_metrics(
    *,
    sample_id: str,
    out_json: Path,
    flagstat: Dict[str, int],
    stats: Optional[Dict[str, float]] = None,
    mean_insert: Optional[float] = None,
    deamination_qscore: int = 5,
    mapper: str = "mojo",
    device: str = "auto",
) -> None:
    """Parabricks-shaped JSON for methyl_alignment_qc.wgbs_parabricks_qc."""
    stats = stats or {}
    total = max(int(flagstat.get("total") or 0), 1)
    mapped = int(flagstat.get("mapped") or 0)
    pf = total
    bases_mapped = int(stats.get("bases_mapped_(cigar)", 0) or 0)
    if bases_mapped <= 0:
        bases_mapped = mapped * 100
    avg_qual = float(stats.get("average_quality", 36.0) or 36.0)
    insert = mean_insert
    if insert is None:
        insert = float(stats.get("insert_size_average", 200.0) or 200.0)
    # Cycle qualities: constant from samtools average (not Clara per-cycle).
    mean_quality = [float(avg_qual)] * 100
    q30_frac = min(max(avg_qual / 40.0, 0.0), 1.0)
    placeholders = ["pre_adapter_summaries.TOTAL_QSCORE", "gc_bias_summary"]
    payload = {
        "sample_id": sample_id,
        "engine": "mojo_fq2bam_meth",
        "mapper": mapper,
        "device": device,
        "metrics_source": "samtools+placeholders",
        "placeholder_fields": placeholders,
        "quality_yield": {
            "total_reads": total,
            "pf_reads": pf,
            "pf_bases": bases_mapped if bases_mapped > 0 else pf * 100,
            "pf_q30_bases": int((bases_mapped if bases_mapped > 0 else pf * 100) * q30_frac),
        },
        "mean_quality_by_cycle": {"mean_quality": mean_quality},
        "gc_bias_summary": {"at_dropout": 1.0, "gc_dropout": 1.0},
        "insert_size_metrics": {"median_insert_size": float(insert)},
        "pre_adapter_summaries": {
            "ARTIFACT_NAME": ["Deamination", "OxoG"],
            "TOTAL_QSCORE": [int(deamination_qscore), 40],
        },
        "alignment_summary": {
            "total_reads": total,
            "mapped_reads": mapped,
            "mapped_rate": mapped / total,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _repo_root() -> Path:
    # fq2bam-meth/python/fq2bam_meth.py → mojo-align root
    return Path(__file__).resolve().parents[2]


def _gpu_require() -> bool:
    raw = os.environ.get("METHYLGRAPHER_GPU_REQUIRE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_linear_mapper(device: str) -> str:
    """Return ``mojo`` or ``bwa`` based on env + device availability."""
    raw = os.environ.get("METHYLGRAPHER_LINEAR_MAPPER", "").strip().lower()
    if _gpu_require() and raw in {"bwa"}:
        raise RuntimeError(
            "METHYLGRAPHER_GPU_REQUIRE=1 forbids LINEAR_MAPPER=bwa "
            "(bakeoff fail-closed)"
        )
    if raw in {"mojo", "bwa"}:
        return raw
    if raw in {"auto", ""}:
        # Prefer Mojo linear kernels unless explicitly forced to BWA.
        # CPU device still uses Mojo CPU DeviceContext path (not BWA).
        return "mojo"
    raise RuntimeError(
        "METHYLGRAPHER_LINEAR_MAPPER must be 'mojo', 'bwa', or 'auto' "
        f"(got {raw!r})"
    )


def _mojo_bin() -> List[str]:
    """Return argv prefix to run Mojo under pixi when available."""
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "run", "mojo"]
    mojo = shutil.which("mojo")
    if mojo:
        return [mojo]
    raise RuntimeError("mojo / pixi not found (required for Mojo linear mapper)")


def run_mojo_linear_map(
    *,
    c2t_ref: Path,
    c2t_r1: Path,
    g2a_r2: Path,
    out_sam: Path,
    device: str,
    k: int,
    cache_dir: Path,
    log: Path,
) -> None:
    root = _repo_root()
    cmd = _mojo_bin() + [
        "-I",
        "gpu-common/src",
        "-I",
        "fq2bam-meth/src",
        "-I",
        "giraffe/src",
        "-I",
        "methylgrapher/src",
        "fq2bam-meth/src/linear_mapper.mojo",
        "-ref",
        str(c2t_ref),
        "-fq1",
        str(c2t_r1),
        "-fq2",
        str(g2a_r2),
        "-out_sam",
        str(out_sam),
        "-device",
        device,
        "-k",
        str(k),
        "-cache_dir",
        str(cache_dir),
    ]
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(cmd) + "\n")
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.stdout:
            handle.write(proc.stdout)
        if proc.stderr:
            handle.write(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                proc.stderr.strip()
                or proc.stdout.strip()
                or f"Mojo linear mapper failed (exit {proc.returncode})"
            )


def run_bwa_mem_stream(
    *,
    bwa: str,
    samtools: str,
    c2t_ref: Path,
    c2t_r1: Path,
    g2a_r2: Path,
    bam_unsorted: Path,
    threads: int,
    log: Path,
) -> None:
    """Stream BWA SAM → samtools view (never buffer full SAM in Python)."""
    bwa_cmd = [bwa, "mem", "-t", str(threads), str(c2t_ref), str(c2t_r1), str(g2a_r2)]
    view_cmd = [samtools, "view", "-bS", "-o", str(bam_unsorted), "-"]
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(bwa_cmd) + " | " + " ".join(view_cmd) + "\n")
        bwa_proc = subprocess.Popen(
            bwa_cmd,
            stdout=subprocess.PIPE,
            stderr=handle,
        )
        assert bwa_proc.stdout is not None
        view_proc = subprocess.run(
            view_cmd,
            stdin=bwa_proc.stdout,
            stderr=handle,
            check=False,
        )
        bwa_proc.stdout.close()
        bwa_rc = bwa_proc.wait()
    if bwa_rc != 0:
        raise RuntimeError(f"bwa mem failed (exit {bwa_rc}); see {log}")
    if view_proc.returncode != 0:
        raise RuntimeError(
            f"samtools view failed (exit {view_proc.returncode}); see {log}"
        )


def run_mojo_fq2bam_meth(
    *,
    fq1: Path,
    fq2: Path,
    reference_fasta: Path,
    out_bam: Path,
    out_qc_dir: Path,
    sample_id: str,
    threads: int = 16,
    device: str = "auto",
    work_dir: Optional[Path] = None,
    log_path: Optional[Path] = None,
    k: int = 15,
) -> Dict[str, str]:
    samtools = shutil.which("samtools")
    if not samtools:
        raise RuntimeError("MojoFq2bamMeth requires samtools on PATH")

    work = Path(work_dir or tempfile.mkdtemp(prefix="mojo_fq2bam_"))
    work.mkdir(parents=True, exist_ok=True)
    log = Path(log_path or (work / "mojo_fq2bam_meth.log"))
    mapper = resolve_linear_mapper(device)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"device={device} threads={threads} mapper={mapper}\n")

    try:
        scripts = Path(__file__).resolve().parents[2] / "giraffe" / "scripts"
        sys.path.insert(0, str(scripts))
        from giraffe_gpu_minimizer import device_probe, _sync_device  # type: ignore

        probe = device_probe()
        backend = _sync_device(device if device != "auto" else "cpu")
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"device_probe={json.dumps(probe)} sync_backend={backend}\n")
    except Exception as exc:  # pragma: no cover
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"device_probe_skipped={exc}\n")

    c2t_ref = work / (reference_fasta.name + ".C2T.fa")
    if not c2t_ref.is_file():
        convert_fasta_c2t(reference_fasta, c2t_ref)

    c2t_r1 = work / "C2T.R1.fastq.gz"
    g2a_r2 = work / "G2A.R2.fastq.gz"
    convert_fastq(fq1, c2t_r1, "C2T")
    convert_fastq(fq2, g2a_r2, "G2A")

    bam_unsorted = work / "aligned.bam"
    used_mapper = mapper
    if mapper == "mojo":
        try:
            out_sam = work / "aligned.sam"
            cache_dir = work / "mojo_linear_index"
            run_mojo_linear_map(
                c2t_ref=c2t_ref,
                c2t_r1=c2t_r1,
                g2a_r2=g2a_r2,
                out_sam=out_sam,
                device=device,
                k=k,
                cache_dir=cache_dir,
                log=log,
            )
            _run(
                [samtools, "view", "-bS", str(out_sam), "-o", str(bam_unsorted)],
                log,
            )
        except Exception as exc:
            if _gpu_require():
                raise RuntimeError(
                    f"Mojo linear mapper failed under METHYLGRAPHER_GPU_REQUIRE=1 "
                    f"(no BWA fallback): {exc}"
                ) from exc
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"mojo_linear_failed={exc}; falling back to bwa\n")
            bwa = shutil.which("bwa")
            if not bwa:
                raise RuntimeError(
                    f"Mojo linear mapper failed and bwa unavailable: {exc}"
                ) from exc
            _ensure_bwa_index(c2t_ref, log)
            run_bwa_mem_stream(
                bwa=bwa,
                samtools=samtools,
                c2t_ref=c2t_ref,
                c2t_r1=c2t_r1,
                g2a_r2=g2a_r2,
                bam_unsorted=bam_unsorted,
                threads=threads,
                log=log,
            )
            used_mapper = "bwa_fallback"
    else:
        bwa = shutil.which("bwa")
        if not bwa:
            raise RuntimeError("MojoFq2bamMeth BWA fallback requires bwa on PATH")
        _ensure_bwa_index(c2t_ref, log)
        run_bwa_mem_stream(
            bwa=bwa,
            samtools=samtools,
            c2t_ref=c2t_ref,
            c2t_r1=c2t_r1,
            g2a_r2=g2a_r2,
            bam_unsorted=bam_unsorted,
            threads=threads,
            log=log,
        )
        used_mapper = "bwa"

    out_bam.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [samtools, "sort", "-@", str(max(1, threads // 2)), "-o", str(out_bam), str(bam_unsorted)],
        log,
    )
    _run([samtools, "index", str(out_bam)], log)

    fs = subprocess.run(
        [samtools, "flagstat", str(out_bam)], capture_output=True, text=True, check=False
    )
    flagstat = _parse_flagstat(fs.stdout or "")
    st = subprocess.run(
        [samtools, "stats", str(out_bam)], capture_output=True, text=True, check=False
    )
    stats = _parse_samtools_stats(st.stdout or "")

    out_qc_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = out_bam.parent / f"{sample_id}.json"
    write_parabricks_shaped_metrics(
        sample_id=sample_id,
        out_json=metrics_json,
        flagstat=flagstat,
        stats=stats,
        mapper=used_mapper,
        device=device,
    )
    (out_qc_dir / "alignment_summary.json").write_text(
        json.dumps({"flagstat": flagstat, "stats": stats, "mapper": used_mapper}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(metrics_json, out_qc_dir / f"{sample_id}.json")

    return {
        "bamPath": str(out_bam),
        "metricsJson": str(metrics_json),
        "qcMetricsDir": str(out_qc_dir),
        "logPath": str(log),
        "engine": "mojo_fq2bam_meth",
        "mapper": used_mapper,
        "device": device,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MojoFq2bamMeth portable linear WGBS Align")
    p.add_argument("-fq1", required=True)
    p.add_argument("-fq2", required=True)
    p.add_argument("-ref", required=True)
    p.add_argument("-out_bam", required=True)
    p.add_argument("-out_qc_dir", required=True)
    p.add_argument("-sample_id", required=True)
    p.add_argument("-t", type=int, default=int(os.environ.get("METHYLGRAPHER_BWA_THREADS", "16")))
    p.add_argument("-device", default=os.environ.get("METHYLGRAPHER_ALIGN_DEVICE", "auto"))
    p.add_argument("-work_dir", default=None)
    p.add_argument("-k", type=int, default=int(os.environ.get("METHYLGRAPHER_LINEAR_K", "15")))
    args = p.parse_args(argv)
    run_mojo_fq2bam_meth(
        fq1=Path(args.fq1),
        fq2=Path(args.fq2),
        reference_fasta=Path(args.ref),
        out_bam=Path(args.out_bam),
        out_qc_dir=Path(args.out_qc_dir),
        sample_id=args.sample_id,
        threads=args.t,
        device=args.device,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        k=args.k,
    )
    print("MojoFq2bamMeth OK", args.out_bam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
