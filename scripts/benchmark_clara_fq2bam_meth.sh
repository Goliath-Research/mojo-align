#!/usr/bin/env bash
# Clara fq2bam_meth vs MojoFq2bamMeth wall-clock harness (GH200 / NGC).
#
# Records Clara baseline when ``pbrun`` is available; always runs Mojo with
# METHYLGRAPHER_GPU_REQUIRE=1 (fail-closed — no silent BWA).
#
# Usage:
#   scripts/benchmark_clara_fq2bam_meth.sh [R1] [R2] [REF] [DEVICE]
#
# Outputs JSON summary under OUT_DIR (default /tmp/clara_fq2bam_bench).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.pixi/bin:$PATH"

FIX="$REPO_ROOT/tests/data/fq2bam_fixture"
R1="${1:-$FIX/R1.fastq}"
R2="${2:-$FIX/R2.fastq}"
REF="${3:-$FIX/ref.fa}"
DEVICE="${4:-nvidia}"
OUT_DIR="${OUT_DIR:-/tmp/clara_fq2bam_bench}"
K="${METHYLGRAPHER_LINEAR_K:-8}"
THREADS="${THREADS:-8}"
mkdir -p "$OUT_DIR"

SUMMARY="$OUT_DIR/summary.json"
: >"$OUT_DIR/clara.time.txt"
: >"$OUT_DIR/mojo.time.txt"

echo "=== MojoFq2bamMeth (GPU_REQUIRE=1) device=$DEVICE ==="
export METHYLGRAPHER_GPU_REQUIRE=1
export METHYLGRAPHER_LINEAR_MAPPER=mojo
MOJO_WORK="$(mktemp -d -p "$OUT_DIR" mojo.XXXXXX)"
MOJO_T0=$(date +%s.%N)
set +e
/usr/bin/time -v "$REPO_ROOT/bin/methylGrapher" MojoFq2bamMeth \
  -fq1 "$R1" -fq2 "$R2" -ref "$REF" \
  -out_bam "$MOJO_WORK/out.bam" -out_qc_dir "$MOJO_WORK/qc" -sample_id mojo_bench \
  -device "$DEVICE" -k "$K" -t "$THREADS" -work_dir "$MOJO_WORK/w" \
  >"$OUT_DIR/mojo.log" 2>"$OUT_DIR/mojo.time.txt"
MOJO_RC=$?
set -e
MOJO_T1=$(date +%s.%N)
MOJO_WALL=$(python3 -c "print(f'{float('$MOJO_T1')-float('$MOJO_T0'):.4f}')")
echo "mojo wall_s=$MOJO_WALL rc=$MOJO_RC"

CLARA_WALL=""
CLARA_RC=127
if command -v pbrun >/dev/null 2>&1; then
  echo "=== Clara pbrun fq2bam_meth ==="
  CLARA_WORK="$(mktemp -d -p "$OUT_DIR" clara.XXXXXX)"
  CLARA_T0=$(date +%s.%N)
  set +e
  /usr/bin/time -v pbrun fq2bam_meth \
    --ref "$REF" \
    --in-fq "$R1" "$R2" \
    --out-bam "$CLARA_WORK/out.bam" \
    --tmp-dir "$CLARA_WORK/tmp" \
    >"$OUT_DIR/clara.log" 2>"$OUT_DIR/clara.time.txt"
  CLARA_RC=$?
  set -e
  CLARA_T1=$(date +%s.%N)
  CLARA_WALL=$(python3 -c "print(f'{float('$CLARA_T1')-float('$CLARA_T0'):.4f}')")
  echo "clara wall_s=$CLARA_WALL rc=$CLARA_RC"
else
  echo "pbrun not on PATH — Clara baseline PENDING (operator on NGC/GH200)"
  echo "PENDING_CLARA_BASELINE" >"$OUT_DIR/clara.PENDING"
fi

python3 - <<PY
import json
from pathlib import Path
out = Path("$SUMMARY")
payload = {
    "r1": "$R1",
    "r2": "$R2",
    "ref": "$REF",
    "device": "$DEVICE",
    "mojo_wall_s": float("$MOJO_WALL") if "$MOJO_WALL" else None,
    "mojo_rc": int("$MOJO_RC"),
    "clara_wall_s": float("$CLARA_WALL") if "$CLARA_WALL" else None,
    "clara_rc": int("$CLARA_RC"),
    "gate": "mojo_wall < clara_wall (strict) when both succeed",
    "gpu_require": True,
}
if payload["mojo_wall_s"] and payload["clara_wall_s"]:
    payload["pass_strict_lt"] = payload["mojo_wall_s"] < payload["clara_wall_s"]
    payload["ratio_mojo_over_clara"] = round(
        payload["mojo_wall_s"] / payload["clara_wall_s"], 4
    )
else:
    payload["pass_strict_lt"] = None
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "wrote $SUMMARY"
echo "See docs/BENCHMARK_FQ2BAM_METH.md to record operator numbers."
