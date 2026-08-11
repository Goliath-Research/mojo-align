#!/usr/bin/env bash
# Benchmark Mojo Giraffe (cpu/nvidia/amd) vs vg on a fixture FASTQ×GFA.
# Writes timings to stdout; see docs/BENCHMARK_GIRAFFE.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GFA="${GFA:-tests/data/toy.wl.gfa}"
FQ1="${FQ1:-tests/data/giraffe_fixture/R1.fastq}"
FQ2="${FQ2:-tests/data/giraffe_fixture/R2.fastq}"
OUT_DIR="${OUT_DIR:-/tmp/giraffe_bench}"
mkdir -p "$OUT_DIR"

run_mojo() {
  local device="$1"
  local out="$OUT_DIR/mojo_${device}.gaf"
  local t0 t1 wall
  t0=$(date +%s.%N)
  pixi run mojo -I src src/main.mojo MojoGiraffe \
    -gfa "$GFA" -fq1 "$FQ1" -fq2 "$FQ2" -out_gaf "$out" -device "$device" -k 5 \
    >/dev/null
  t1=$(date +%s.%N)
  wall=$(python3 -c "print(f'{float('$t1')-float('$t0'):.4f}')")
  echo "mojo device=${device} wall_s=${wall} out=${out}"
}

echo "=== Mojo Giraffe fixture bench ==="
run_mojo cpu
run_mojo nvidia || echo "mojo device=nvidia skipped/failed"
run_mojo amd || echo "mojo device=amd skipped/failed (no ROCm node)"

echo "=== Device probe ==="
python3 scripts/giraffe_gpu_minimizer.py

echo "=== Golden GAF parity (cpu) ==="
python3 scripts/giraffe_gaf_parity.py \
  --mojo "$OUT_DIR/mojo_cpu.gaf" \
  --golden tests/data/giraffe_fixture/golden.gaf \
  --require-extra-tags
