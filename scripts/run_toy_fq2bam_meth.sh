#!/usr/bin/env bash
# Smoke-test MojoFq2bamMeth against tests/data/fq2bam_fixture/.
#
# Usage:
#   scripts/run_toy_fq2bam_meth.sh [python|mojo] [cpu|nvidia|amd|auto]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$REPO_ROOT"

ENGINE="${1:-python}"
DEVICE="${2:-cpu}"
if [[ "$ENGINE" != "python" && "$ENGINE" != "mojo" ]]; then
    echo "usage: $0 [python|mojo] [cpu|nvidia|amd|auto]" >&2
    exit 1
fi

FIX="$REPO_ROOT/tests/data/fq2bam_fixture"
WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

if [[ "$ENGINE" == "mojo" ]]; then
    export METHYLGRAPHER_ENGINE=mojo
else
    unset METHYLGRAPHER_ENGINE 2>/dev/null || true
fi

# Prefer Mojo linear mapper (BWA only if mojo fails / LINEAR_MAPPER=bwa).
export METHYLGRAPHER_LINEAR_MAPPER="${METHYLGRAPHER_LINEAR_MAPPER:-mojo}"

OUT_BAM="$WORK/toy.bam"
OUT_QC="$WORK/qc"
echo "== MojoFq2bamMeth toy (engine=$ENGINE device=$DEVICE mapper=$METHYLGRAPHER_LINEAR_MAPPER) =="
"$REPO_ROOT/bin/methylGrapher" MojoFq2bamMeth \
    -fq1 "$FIX/R1.fastq" \
    -fq2 "$FIX/R2.fastq" \
    -ref "$FIX/ref.fa" \
    -out_bam "$OUT_BAM" \
    -out_qc_dir "$OUT_QC" \
    -sample_id toy \
    -device "$DEVICE" \
    -k 8 \
    -t 2 \
    -work_dir "$WORK/work"

echo
samtools flagstat "$OUT_BAM" | head -8
echo
echo "== metrics =="
python3 -c "import json; print(json.dumps(json.load(open('$OUT_QC/toy.json')), indent=2))"

if [[ "${KEEP_WORK_DIR:-0}" == "1" ]]; then
    trap - EXIT
    echo "KEEP_WORK_DIR=1: kept $WORK"
fi
