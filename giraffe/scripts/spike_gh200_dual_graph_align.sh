#!/usr/bin/env bash
# Phase 0 GH200 spike: Parabricks GAF contract check + optional tiny dual vg GAF.
# Usage:
#   scripts/spike_gh200_dual_graph_align.sh
#   SPIKE_RUN_VG=1 SPIKE_READ_PAIRS=2000 scripts/spike_gh200_dual_graph_align.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="${SPIKE_OUT:-$ROOT/docs/spike_gh200_out}"
mkdir -p "$REPORT_DIR"
PARABRICKS_IMAGE="${METHYL_PARABRICKS_IMAGE:-nvcr.io/nvidia/clara/clara-parabricks:4.7.0-1}"
MG_IMAGE="${METHYL_METHYLGRAPHER_IMAGE:-epimethyl/methylgrapher:1.70-mojo}"
INDEX_PREFIX="${SPIKE_INDEX_PREFIX:-/work/genomes/pangenome/GRCh38/d9-bs/1.70/hprc-d9-bs}"
FQ1="${SPIKE_FQ1:-/work/samples/DBCST-051425-111148-DS20M/DBCST-051425-111148-DS20M_1.fastq.gz}"
FQ2="${SPIKE_FQ2:-/work/samples/DBCST-051425-111148-DS20M/DBCST-051425-111148-DS20M_2.fastq.gz}"
NPAIRS="${SPIKE_READ_PAIRS:-2000}"

{
  echo "# GH200 Phase 0 spike log"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi: unavailable"
  echo
  echo "## Parabricks giraffe I/O options (GAF search)"
  docker run --rm "$PARABRICKS_IMAGE" pbrun giraffe --help 2>&1 \
    | grep -Eih 'out-bam|gaf|gam|named|output-format|--out' || true
  if docker run --rm "$PARABRICKS_IMAGE" pbrun giraffe --help 2>&1 | grep -qiE -- '--out-gaf|output-format.*gaf|\bgaf\b'; then
    echo "VERDICT: Parabricks appears to expose GAF — re-check manually (unexpected on 4.7.0-1)."
  else
    echo "VERDICT: NO-GO — Parabricks giraffe is BAM-only (no GAF / named-coordinates)."
  fi
} | tee "$REPORT_DIR/phase0_parabricks_probe.txt"

if [[ "${SPIKE_RUN_VG:-0}" != "1" ]]; then
  echo "Skipping vg tiny dual-map (set SPIKE_RUN_VG=1 to run)."
  exit 0
fi

if [[ ! -f "$FQ1" || ! -f "$FQ2" ]]; then
  echo "SPIKE_RUN_VG=1 but FASTQs missing: $FQ1 $FQ2" >&2
  exit 1
fi

WORK="$REPORT_DIR/vg_tiny_work"
rm -rf "$WORK"
mkdir -p "$WORK"
# Extract NPAIRS * 4 FASTQ lines (paired).
python3 - <<PY
import gzip
from pathlib import Path
n = int("$NPAIRS") * 4
for src, dst in [("$FQ1", "$WORK/r1.fastq"), ("$FQ2", "$WORK/r2.fastq")]:
    with gzip.open(src, "rt") as fin, open(dst, "w") as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)
print("wrote", n // 4, "pairs")
PY

# BS convert (directional): R1 C→T, R2 G→A — same as methylGrapher.
awk 'NR%4==2{gsub(/C/,"T")}1' "$WORK/r1.fastq" > "$WORK/C2T.R1.fastq"
awk 'NR%4==2{gsub(/G/,"A")}1' "$WORK/r2.fastq" > "$WORK/G2A.R2.fastq"

run_vg() {
  local ref="$1" out="$2"
  local prefix="${INDEX_PREFIX}.wl.${ref}"
  /usr/bin/time -f "elapsed_sec=%e max_rss_kb=%M" -o "$WORK/${ref}.time" \
    docker run --rm --user "$(id -u):$(id -g)" \
      -v "$(dirname "$INDEX_PREFIX"):$(dirname "$INDEX_PREFIX")" \
      -v "$WORK:$WORK" \
      "$MG_IMAGE" \
      vg giraffe -p -t "${SPIKE_THREADS:-32}" -o gaf -M 2 --named-coordinates \
      -Z "${prefix}.giraffe.gbz" \
      -d "${prefix}.dist" \
      -m "${prefix}.shortread.withzip.min" \
      -z "${prefix}.shortread.zipcodes" \
      -f "$WORK/C2T.R1.fastq" -f "$WORK/G2A.R2.fastq" \
      > "$out" 2>"$WORK/${ref}.giraffe.log" || true
}

echo "Running tiny dual vg giraffe GAF (pairs=$NPAIRS)..."
run_vg C2T "$WORK/c2t.gaf"
run_vg G2A "$WORK/g2a.gaf"
{
  echo "## Tiny dual vg GAF"
  echo "pairs=$NPAIRS"
  cat "$WORK/C2T.time" 2>/dev/null || echo "C2T: failed"
  cat "$WORK/G2A.time" 2>/dev/null || echo "G2A: failed"
  wc -l "$WORK/c2t.gaf" "$WORK/g2a.gaf" 2>/dev/null || true
} | tee "$REPORT_DIR/phase0_vg_tiny.txt"

echo "Phase 0 artifacts under $REPORT_DIR"
