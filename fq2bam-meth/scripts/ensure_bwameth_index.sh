#!/usr/bin/env bash
# Build bwa-meth C2T index required by Clara ``pbrun fq2bam_meth``.
#
#   fq2bam-meth/scripts/ensure_bwameth_index.sh /path/to/ref.fasta
#
# Creates ``${REF}.bwameth.c2t`` + bwa ``.amb/.ann/.bwt/.pac/.sa``.
# Needs ``bwa`` on PATH and ``bwameth`` (pip) or a local bwameth.py.
set -euo pipefail

REF="${1:-}"
if [[ -z "$REF" || ! -f "$REF" ]]; then
  echo "usage: $0 /path/to/ref.fasta" >&2
  exit 2
fi
REF="$(readlink -f "$REF")"

if [[ -f "${REF}.bwameth.c2t" && -f "${REF}.bwameth.c2t.bwt" ]]; then
  echo "Already indexed: ${REF}.bwameth.c2t"
  exit 0
fi

if ! command -v bwa >/dev/null 2>&1; then
  echo "ERROR: bwa not on PATH" >&2
  exit 3
fi

BWAMETH=()
if command -v bwameth.py >/dev/null 2>&1; then
  BWAMETH=(bwameth.py)
elif python3 -c "import bwameth" 2>/dev/null; then
  BWAMETH=(python3 -m bwameth)
else
  echo "Installing bwameth into user site..."
  pip3 install --user 'bwameth' || pip3 install --user 'git+https://github.com/brentp/bwa-meth.git'
  BWAMETH=(python3 -c 'from bwameth import main; import sys; sys.argv=["bwameth.py","index",sys.argv[1]]; main()' )
fi

echo "Indexing (slow, once per reference): $REF"
if [[ "${#BWAMETH[@]}" -eq 1 && "${BWAMETH[0]}" == bwameth.py ]]; then
  bwameth.py index "$REF"
elif python3 -c "import bwameth" 2>/dev/null; then
  # bwameth CLI entry
  python3 - <<PY
import sys
sys.argv = ["bwameth.py", "index", r"$REF"]
import bwameth
bwameth.main()
PY
else
  "${BWAMETH[@]}" "$REF"
fi

[[ -f "${REF}.bwameth.c2t" ]] || { echo "ERROR: index missing after bwameth" >&2; exit 4; }
echo "OK: ${REF}.bwameth.c2t"
ls -lh "${REF}.bwameth.c2t"*
