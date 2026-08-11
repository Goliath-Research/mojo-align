#!/usr/bin/env bash
# Run align.linear.parabricks vs align.linear.mojo on the NVIDIA Parabricks
# tutorial sample (or a local FASTQ pair) and score concordance.
#
# Official Clara sample (fq2bam tutorial inputs — also valid for fq2bam_meth):
#   wget -O parabricks_sample.tar.gz \
#     "https://s3.amazonaws.com/parabricks.sample/parabricks_sample.tar.gz"
#   tar xvf parabricks_sample.tar.gz
#   export PARABRICKS_SAMPLE=$PWD/parabricks_sample
#
# Usage:
#   fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh [options]
#
# Options:
#   --sample-dir DIR     Default: /tmp/samples/parabricks_sample
#   --sample-id ID       Default: parabricks_sample
#   --parabricks-sample DIR   Or set PARABRICKS_SAMPLE
#   --max-pairs N        Subset first N PE pairs (0 = all; default 50000)
#   --device DEV         Mojo device (default nvidia)
#   --image IMAGE        Clara docker image (default nvcr.io/nvidia/clara/clara-parabricks:4.5.1-1)
#   --skip-clara         Only run Mojo (compare against existing Clara BAM)
#   --skip-mojo          Only run Clara
#   --compare-only       Skip both aligners; only score existing BAMs
#   --toy                Use fq2bam-meth toy fixture (no Clara sample / no pbrun)
#
# Layout written:
#   <sample-dir>/
#     *.fastq.gz (shared inputs, optional subset)
#     align.linear.parabricks/<sample-id>.bam
#     align.linear.mojo/<sample-id>.bam
#     linear_parity_report.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.pixi/bin:${PATH:-}"

SAMPLE_DIR="${SAMPLE_DIR:-/work/samples/parabricks_sample}"
SAMPLE_ID="${SAMPLE_ID:-parabricks_sample}"
PB_SAMPLE="${PARABRICKS_SAMPLE:-}"
MAX_PAIRS="${MAX_PAIRS:-50000}"
DEVICE="${DEVICE:-nvidia}"
PB_IMAGE="${METHYL_PARABRICKS_IMAGE:-nvcr.io/nvidia/clara/clara-parabricks:4.5.1-1}"
RUN_CLARA=1
RUN_MOJO=1
COMPARE=1
USE_TOY=0
THREADS="${THREADS:-16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample-dir) SAMPLE_DIR="$2"; shift 2 ;;
    --sample-id) SAMPLE_ID="$2"; shift 2 ;;
    --parabricks-sample) PB_SAMPLE="$2"; shift 2 ;;
    --max-pairs) MAX_PAIRS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --image) PB_IMAGE="$2"; shift 2 ;;
    --skip-clara) RUN_CLARA=0; shift ;;
    --skip-mojo) RUN_MOJO=0; shift ;;
    --compare-only) RUN_CLARA=0; RUN_MOJO=0; shift ;;
    --toy) USE_TOY=1; shift ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

ALIGN_PB="align.linear.parabricks"
ALIGN_MOJO="align.linear.mojo"
mkdir -p "$SAMPLE_DIR" "$SAMPLE_DIR/$ALIGN_PB" "$SAMPLE_DIR/$ALIGN_MOJO"

if [[ "$RUN_CLARA" -eq 0 && "$RUN_MOJO" -eq 0 ]]; then
  echo "=== Concordance only ($ALIGN_PB vs $ALIGN_MOJO) ==="
  python3 "$REPO_ROOT/fq2bam-meth/scripts/compare_linear_align_parity.py" \
    --sample-dir "$SAMPLE_DIR" \
    --sample-id "$SAMPLE_ID" \
    --out-json "$SAMPLE_DIR/linear_parity_report.json"
  exit $?
fi

if [[ "$USE_TOY" -eq 1 ]]; then
  R1_SRC="$REPO_ROOT/fq2bam-meth/tests/data/fq2bam_fixture/R1.fastq"
  R2_SRC="$REPO_ROOT/fq2bam-meth/tests/data/fq2bam_fixture/R2.fastq"
  REF="$REPO_ROOT/fq2bam-meth/tests/data/fq2bam_fixture/ref.fa"
  SAMPLE_ID="${SAMPLE_ID:-toy}"
  MAX_PAIRS=0
  echo "NOTE: --toy uses the tiny AT-rich fixture (smoke only; not Clara sample)"
else
  if [[ -z "$PB_SAMPLE" ]]; then
    for cand in \
      /work/samples/parabricks_sample \
      "$SAMPLE_DIR" \
      "$PWD/parabricks_sample" \
      "$HOME/parabricks_sample" \
      /workdir/parabricks_sample \
      /data/parabricks_sample; do
      if [[ -d "$cand/Data" && -d "$cand/Ref" ]]; then
        PB_SAMPLE="$cand"
        break
      fi
    done
  fi
  if [[ -z "$PB_SAMPLE" || ! -d "$PB_SAMPLE/Data" ]]; then
    cat >&2 <<EOF
ERROR: Parabricks sample not found under /work/samples/parabricks_sample.

Fetch once:

  fq2bam-meth/scripts/fetch_parabricks_sample.sh

Or pass --parabricks-sample DIR / --toy for the local smoke fixture.
EOF
    exit 2
  fi
  # Default sample-dir to the staged NVIDIA tree when still at the default path.
  if [[ "$SAMPLE_DIR" == "/work/samples/parabricks_sample" ]]; then
    SAMPLE_DIR="$PB_SAMPLE"
  fi
  R1_SRC="$PB_SAMPLE/Data/sample_1.fq.gz"
  R2_SRC="$PB_SAMPLE/Data/sample_2.fq.gz"
  REF="$PB_SAMPLE/Ref/Homo_sapiens_assembly38.fasta"
  for f in "$R1_SRC" "$R2_SRC" "$REF"; do
    [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
  done
fi

# Materialize inputs under work/ — never write into Data/ (immutable source).
mkdir -p "$SAMPLE_DIR/work"
R1_SRC="$(readlink -f "$R1_SRC")"
R2_SRC="$(readlink -f "$R2_SRC")"
R1="$SAMPLE_DIR/work/${SAMPLE_ID}_R1.fastq.gz"
R2="$SAMPLE_DIR/work/${SAMPLE_ID}_R2.fastq.gz"
if [[ "$MAX_PAIRS" -gt 0 ]]; then
  echo "Subsetting first $MAX_PAIRS PE pairs → $SAMPLE_DIR/work"
  # Refuse to clobber source FASTQs if paths ever collide.
  if [[ "$R1" -ef "$R1_SRC" || "$R2" -ef "$R2_SRC" ]]; then
    echo "ERROR: subset output path resolves to source FASTQ; aborting" >&2
    exit 2
  fi
  python3 - <<PY
from pathlib import Path
import gzip
import os

def open_text(p: Path):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else p.open("r")

n = int("$MAX_PAIRS")
r1_src, r2_src = Path("$R1_SRC"), Path("$R2_SRC")
r1_out, r2_out = Path("$R1"), Path("$R2")
for src, out in ((r1_src, r1_out), (r2_src, r2_out)):
    if out.exists() and src.resolve() == out.resolve():
        raise SystemExit("subset output would overwrite source FASTQ")
r1_out.parent.mkdir(parents=True, exist_ok=True)
with open_text(r1_src) as f1, open_text(r2_src) as f2, \
     gzip.open(r1_out, "wt") as o1, gzip.open(r2_out, "wt") as o2:
    for i in range(n):
        for _ in range(4):
            l1 = f1.readline()
            l2 = f2.readline()
            if not l1 or not l2:
                raise SystemExit(f"EOF before {n} pairs (got {i})")
            o1.write(l1)
            o2.write(l2)
print(f"wrote {n} pairs → {r1_out} / {r2_out}")
PY
else
  # Point at full source files without copying
  R1="$R1_SRC"
  R2="$R2_SRC"
fi

run_clara() {
  local out_bam="$SAMPLE_DIR/$ALIGN_PB/${SAMPLE_ID}.bam"
  local work="$SAMPLE_DIR/$ALIGN_PB/work"
  mkdir -p "$work"
  echo "=== $ALIGN_PB (Clara pbrun fq2bam_meth) ==="
  if command -v pbrun >/dev/null 2>&1; then
    pbrun fq2bam_meth \
      --ref "$REF" \
      --in-fq "$R1" "$R2" \
      --out-bam "$out_bam" \
      --tmp-dir "$work/tmp" \
      2>&1 | tee "$SAMPLE_DIR/$ALIGN_PB/${SAMPLE_ID}.fq2bam_meth.log"
  elif command -v docker >/dev/null 2>&1; then
    local host_sample host_ref
    host_sample="$(cd "$SAMPLE_DIR" && pwd)"
    host_ref="$(cd "$(dirname "$REF")" && pwd)"
    docker run --rm --gpus all \
      -v "$host_sample:/sample" \
      -v "$host_ref:/ref:ro" \
      "$PB_IMAGE" \
      pbrun fq2bam_meth \
        --ref "/ref/$(basename "$REF")" \
        --in-fq "/sample/$(basename "$R1")" "/sample/$(basename "$R2")" \
        --out-bam "/sample/$ALIGN_PB/${SAMPLE_ID}.bam" \
        --tmp-dir "/sample/$ALIGN_PB/work/tmp" \
      2>&1 | tee "$SAMPLE_DIR/$ALIGN_PB/${SAMPLE_ID}.fq2bam_meth.log"
  else
    echo "ERROR: neither pbrun nor docker available for Clara arm" >&2
    exit 3
  fi
  if [[ ! -f "${out_bam}.bai" && ! -f "${out_bam}.csi" ]]; then
    samtools index "$out_bam" || true
  fi
}

run_mojo() {
  local out_bam="$SAMPLE_DIR/$ALIGN_MOJO/${SAMPLE_ID}.bam"
  local work="$SAMPLE_DIR/$ALIGN_MOJO/work"
  local qc="$SAMPLE_DIR/$ALIGN_MOJO/qc"
  mkdir -p "$work" "$qc"
  echo "=== $ALIGN_MOJO (MojoFq2bamMeth) ==="
  export PYTHONPATH="${REPO_ROOT}/methylgrapher:${REPO_ROOT}/giraffe/scripts:${REPO_ROOT}/giraffe/python:${REPO_ROOT}/fq2bam-meth/python:${REPO_ROOT}/gpu-common/python${PYTHONPATH:+:$PYTHONPATH}"
  export METHYLGRAPHER_GPU_REQUIRE="${METHYLGRAPHER_GPU_REQUIRE:-1}"
  export METHYLGRAPHER_LINEAR_MAPPER=mojo
  # Toy fixture uses short k; Clara sample uses default 15.
  local k_args=()
  if [[ "$USE_TOY" -eq 1 ]]; then
    k_args=(-k 8)
    export METHYLGRAPHER_GPU_REQUIRE=0
  fi
  "$REPO_ROOT/bin/methylGrapher" MojoFq2bamMeth \
    -fq1 "$R1" -fq2 "$R2" -ref "$REF" \
    -out_bam "$out_bam" -out_qc_dir "$qc" -sample_id "$SAMPLE_ID" \
    -device "$DEVICE" -t "$THREADS" -work_dir "$work" \
    "${k_args[@]}" \
    2>&1 | tee "$SAMPLE_DIR/$ALIGN_MOJO/${SAMPLE_ID}.fq2bam_meth.log"
  if [[ ! -f "${out_bam}.bai" ]]; then
    samtools index "$out_bam" || true
  fi
}

if [[ "$RUN_CLARA" -eq 1 ]]; then
  run_clara
fi
if [[ "$RUN_MOJO" -eq 1 ]]; then
  run_mojo
fi

CLARA_BAM="$SAMPLE_DIR/$ALIGN_PB/${SAMPLE_ID}.bam"
MOJO_BAM="$SAMPLE_DIR/$ALIGN_MOJO/${SAMPLE_ID}.bam"
if [[ "$COMPARE" -eq 1 ]]; then
  if [[ -f "$CLARA_BAM" && -f "$MOJO_BAM" ]]; then
    echo "=== Concordance ($ALIGN_PB vs $ALIGN_MOJO) ==="
    python3 "$REPO_ROOT/fq2bam-meth/scripts/compare_linear_align_parity.py" \
      --sample-dir "$SAMPLE_DIR" \
      --sample-id "$SAMPLE_ID" \
      --out-json "$SAMPLE_DIR/linear_parity_report.json"
  else
    echo "Skip compare — need both BAMs:" >&2
    echo "  clara: $CLARA_BAM ($([[ -f $CLARA_BAM ]] && echo ok || echo missing))" >&2
    echo "  mojo:  $MOJO_BAM ($([[ -f $MOJO_BAM ]] && echo ok || echo missing))" >&2
    if [[ "$RUN_CLARA" -eq 0 || "$RUN_MOJO" -eq 0 ]]; then
      echo "Re-run without --skip-* or pass --compare-only once both arms exist." >&2
      exit 0
    fi
    exit 2
  fi
fi
