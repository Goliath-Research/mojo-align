"""Bismark-style XM/XG methylation tags for linear WGBS BAMs.

XM is computed from the original (pre-align-conversion) read sequence versus
the reference along the CIGAR. XG is CT for R1 / GA for R2 (bwa-meth convention).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

_CT_METH = {ord("C"), ord("c")}
_CT_UNMETH = {ord("T"), ord("t")}
_GA_METH = {ord("G"), ord("g")}
_GA_UNMETH = {ord("A"), ord("a")}


def write_meth_tags_enabled() -> bool:
    raw = os.environ.get("METHYLGRAPHER_WRITE_METH_TAGS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def xg_from_flag(flag: int) -> str:
    """Return XG string: CT for read1 / unpaired, GA for read2."""
    if flag & 0x80:  # BAM_FREAD2
        return "GA"
    return "CT"


def encode_bismark_xm(
    ref_bases: str,
    read_seq: str,
    cigar_ops: Sequence[Tuple[int, int]],
    xg: str,
) -> str:
    """Build an XM:Z string the same length as ``read_seq``.

    ``cigar_ops`` is a list of ``(op, length)`` where op is BAM CIGAR op
    (0=M, 1=I, 2=D, 3=N, 4=S, 5=H, 7==, 8=X).
    """
    xm = ["."] * len(read_seq)
    qpos = 0
    rpos = 0
    ct = str(xg).upper().startswith("C")
    for op, length in cigar_ops:
        if op in (0, 7, 8):  # M / = / X
            for _ in range(length):
                if qpos < len(read_seq) and rpos < len(ref_bases):
                    ref = ref_bases[rpos].upper()
                    base = read_seq[qpos]
                    if ct and ref == "C":
                        if ord(base) in _CT_METH:
                            xm[qpos] = "Z"
                        elif ord(base) in _CT_UNMETH:
                            xm[qpos] = "z"
                    elif (not ct) and ref == "G":
                        if ord(base) in _GA_METH:
                            xm[qpos] = "Z"
                        elif ord(base) in _GA_UNMETH:
                            xm[qpos] = "z"
                qpos += 1
                rpos += 1
        elif op == 1:  # I
            qpos += length
        elif op in (2, 3):  # D / N
            rpos += length
        elif op == 4:  # S
            qpos += length
        elif op == 5:  # H
            pass
        else:
            qpos += length
    return "".join(xm)


def meth_call_from_xm(xm_char: str) -> int:
    """Return 1 methylated, 0 unmethylated, -1 not a cytosine call."""
    if xm_char in "ZXHU":
        return 1
    if xm_char in "zxhu":
        return 0
    return -1


def meth_call_from_seq_xg(ref: str, base: str, xg: str) -> int:
    """Sequence+XG cytosine call (MethylExtractor fallback)."""
    ref_u = ref.upper()
    base_u = base.upper()
    ct = str(xg).upper().startswith("C")
    if ct and ref_u == "C":
        if base_u == "C":
            return 1
        if base_u == "T":
            return 0
    if (not ct) and ref_u == "G":
        if base_u == "G":
            return 1
        if base_u == "A":
            return 0
    return -1


def _parse_cigar(cigar: str) -> List[Tuple[int, int]]:
    ops = {"M": 0, "I": 1, "D": 2, "N": 3, "S": 4, "H": 5, "P": 6, "=": 7, "X": 8}
    out: List[Tuple[int, int]] = []
    n = 0
    for ch in cigar:
        if ch.isdigit():
            n = n * 10 + int(ch)
            continue
        out.append((ops.get(ch, 0), n or 0))
        n = 0
    return out


def _load_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name = ""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name:
                seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def annotate_bam_meth_tags(
    bam_path: Path,
    reference_fasta: Path,
    *,
    samtools: str | None = None,
    log: Path | None = None,
) -> Path:
    """Rewrite ``bam_path`` in place with XM:Z and XG:Z tags (samtools required)."""
    st = samtools or shutil.which("samtools")
    if not st:
        raise RuntimeError("annotate_bam_meth_tags requires samtools on PATH")
    bam = Path(bam_path)
    fasta = _load_fasta(Path(reference_fasta))
    view = subprocess.run(
        [st, "view", "-h", str(bam)],
        check=True,
        capture_output=True,
        text=True,
    )
    out_lines: List[str] = []
    for line in view.stdout.splitlines():
        if not line or line.startswith("@"):
            out_lines.append(line)
            continue
        parts = line.split("\t")
        if len(parts) < 11:
            out_lines.append(line)
            continue
        flag = int(parts[1])
        if flag & 4:
            out_lines.append(line)
            continue
        chrom = parts[2]
        pos0 = int(parts[3]) - 1
        cigar = parts[5]
        seq = parts[9]
        xg = xg_from_flag(flag)
        ref_seq = fasta.get(chrom, "")
        ops = _parse_cigar(cigar)
        ref_span = sum(ln for op, ln in ops if op in (0, 2, 3, 7, 8))
        ref_slice = ref_seq[pos0 : pos0 + ref_span] if ref_seq else ""
        xm = encode_bismark_xm(ref_slice, seq, ops, xg) if ref_slice else ""
        tags = [t for t in parts[11:] if not t.startswith(("XM:Z:", "XG:Z:"))]
        if xm:
            tags.append(f"XM:Z:{xm}")
        tags.append(f"XG:Z:{xg}")
        out_lines.append("\t".join(parts[:11] + tags))

    tmp = Path(tempfile.mkstemp(suffix=".bam", prefix="meth_tags_")[1])
    try:
        proc = subprocess.run(
            [st, "view", "-b", "-o", str(tmp), "-"],
            input="\n".join(out_lines) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                proc.stderr.strip() or "samtools view failed while writing methylation tags"
            )
        tmp.replace(bam)
        subprocess.run([st, "index", str(bam)], check=False)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    if log is not None:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"POST: wrote XM:Z/XG:Z methylation tags onto {bam}\n")
    return bam
