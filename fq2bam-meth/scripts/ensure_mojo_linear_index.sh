#!/usr/bin/env bash
# Prebuild Mojo linear C2T FASTA + dense-v1 k-mer pack beside a fleet FASTA.
#
# Layout (siblings of the FASTA, matching bwameth convention):
#   ${REF}.C2T.fa
#   ${REF}.mojo_linear_k${K}/
#     meta.json  kmers.bin  offsets.bin  postings.bin  contigs.txt  [ref.fa]
#
# Usage:
#   fq2bam-meth/scripts/ensure_mojo_linear_index.sh [/path/to/ref.fasta] [k]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.pixi/bin:${PATH:-}"

DEFAULT_REF="/work/genomes/linear/GRCh38/ensembl-114/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
REF="${1:-$DEFAULT_REF}"
K="${2:-${METHYLGRAPHER_LINEAR_K:-15}}"

if [[ ! -f "$REF" ]]; then
  echo "usage: $0 /path/to/ref.fasta [k]" >&2
  echo "missing: $REF" >&2
  exit 2
fi
REF="$(readlink -f "$REF")"
C2T="${REF}.C2T.fa"
CACHE="${REF}.mojo_linear_k${K}"

if [[ -f "$C2T" && -f "${CACHE}/meta.json" && -f "${CACHE}/kmers.bin" && -f "${CACHE}/postings.bin" ]]; then
  echo "Already indexed (dense-v1):"
  echo "  $C2T"
  echo "  $CACHE"
  ls -lh "$C2T" "${CACHE}/meta.json" "${CACHE}/kmers.bin" "${CACHE}/offsets.bin" "${CACHE}/postings.bin"
  exit 0
fi

if [[ ! -f "$C2T" ]]; then
  cand="/work/samples/parabricks_sample/align.linear.mojo/work/$(basename "$REF").C2T.fa"
  if [[ -f "$cand" ]]; then
    echo "Copying existing C2T FASTA → $C2T"
    cp -a "$cand" "$C2T" || cp "$cand" "$C2T"
  else
    echo "Building C2T FASTA (once) → $C2T"
    python3 - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "$REPO_ROOT/fq2bam-meth/python")
from fq2bam_meth import convert_fasta_c2t
convert_fasta_c2t(Path("$REF"), Path("$C2T"))
print("wrote", "$C2T")
PY
  fi
fi

echo "Building dense-v1 Mojo linear k=${K} pack → $CACHE"
rm -f "${CACHE}/hits.tsv"
mkdir -p "$CACHE"
python3 "$REPO_ROOT/fq2bam-meth/scripts/build_mojo_linear_pack.py" \
  --ref "$C2T" --out "$CACHE" -k "$K"

# Keep a ref.fa copy inside the pack for mapper verify without resolving meta paths.
if [[ ! -f "${CACHE}/ref.fa" ]]; then
  ln -sf "$C2T" "${CACHE}/ref.fa" 2>/dev/null || cp -a "$C2T" "${CACHE}/ref.fa" || cp "$C2T" "${CACHE}/ref.fa"
fi

[[ -f "${CACHE}/kmers.bin" && -f "${CACHE}/meta.json" ]] || {
  echo "ERROR: dense pack incomplete" >&2
  exit 4
}
echo "OK:"
ls -lh "$C2T" "${CACHE}/"*
