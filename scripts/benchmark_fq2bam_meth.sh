#!/usr/bin/env bash
# Wall-clock harness for MojoFq2bamMeth (toy fixture by default).
#
# Usage:
#   scripts/benchmark_fq2bam_meth.sh [R1] [R2] [REF] [DEVICE]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.pixi/bin:$PATH"

FIX="$REPO_ROOT/tests/data/fq2bam_fixture"
R1="${1:-$FIX/R1.fastq}"
R2="${2:-$FIX/R2.fastq}"
REF="${3:-$FIX/ref.fa}"
DEVICE="${4:-cpu}"
K="${METHYLGRAPHER_LINEAR_K:-8}"

run_one() {
    local mapper="$1"
    local work
    work="$(mktemp -d)"
    export METHYLGRAPHER_LINEAR_MAPPER="$mapper"
    echo "=== mapper=$mapper device=$DEVICE ==="
    /usr/bin/time -v "$REPO_ROOT/bin/methylGrapher" MojoFq2bamMeth \
        -fq1 "$R1" -fq2 "$R2" -ref "$REF" \
        -out_bam "$work/out.bam" -out_qc_dir "$work/qc" -sample_id bench \
        -device "$DEVICE" -k "$K" -t "${THREADS:-8}" -work_dir "$work/w" \
        2>"$work/time.txt" || true
    echo "-- flagstat --"
    samtools flagstat "$work/out.bam" 2>/dev/null | head -6 || true
    echo "-- time (Elapsed / Maximum resident) --"
    rg -n "Elapsed \(wall clock\)|Maximum resident" "$work/time.txt" || cat "$work/time.txt" | tail -20
    rm -rf "$work"
}

run_one mojo
if command -v bwa >/dev/null 2>&1; then
    run_one bwa
else
    echo "bwa not on PATH; skip BWA baseline"
fi

echo
echo "Clara pbrun fq2bam_meth comparison is an operator step on NGC hosts;"
echo "see docs/BENCHMARK_FQ2BAM_METH.md."
