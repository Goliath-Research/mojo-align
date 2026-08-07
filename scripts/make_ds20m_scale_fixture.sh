#!/usr/bin/env bash
# Build a larger PE fixture from toy segments for Mojo Giraffe scale smoke
# (not a real 20M-read cohort — see docs/BENCHMARK_GIRAFFE.md for DS20M recipe).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
N="${1:-2000}"
OUT="${2:-$ROOT/tests/data/giraffe_fixture/ds_scale}"
mkdir -p "$OUT"
python3 - <<PY
from pathlib import Path
n = int("$N")
out = Path("$OUT")
segs = ["ACGTACGTAC", "GTACGTACGT", "TTTTAAAACC"]
r1, r2 = [], []
for i in range(n):
    s1 = segs[i % 2]
    s2 = segs[(i + 1) % 3]
    r1 += [f"@read{i}_C2T_{i}_{s1}", s1, "+", "I" * len(s1)]
    r2 += [f"@read{i}_G2A_{i}_{s2}", s2, "+", "I" * len(s2)]
(out / "R1.fastq").write_text("\n".join(r1) + "\n")
(out / "R2.fastq").write_text("\n".join(r2) + "\n")
print(f"wrote {n} PE pairs → {out}")
PY
echo "Map with:"
echo "  pixi run mojo -I src src/main.mojo MojoGiraffe -gfa tests/data/toy.wl.gfa \\"
echo "    -fq1 $OUT/R1.fastq -fq2 $OUT/R2.fastq -out_gaf /tmp/ds_scale.gaf -device cpu"
