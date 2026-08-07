#!/usr/bin/env bash
# scripts/run_toy_mcall.sh — smoke-test MethylCall + MergeCpG end to end
# against the tiny fixture graph in tests/data/ (see tests/data/README.md).
#
# Usage:
#   scripts/run_toy_mcall.sh [python|mojo]
#
# Defaults to the Python engine (`engine/cli.py`). Pass `mojo` to instead
# route through `src/main.mojo` (native MethylCall / MergeCpG; outputs should
# match the Python engine — see scripts/parity_compare.py).
#
# The toy graph's path is only 20 bp long, so `-minimum_identity` is set to
# 10 (below the CLI default of 20) — see tests/data/README.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$REPO_ROOT"

ENGINE="${1:-python}"
if [[ "$ENGINE" != "python" && "$ENGINE" != "mojo" ]]; then
    echo "usage: $0 [python|mojo]" >&2
    exit 1
fi

INDEX_PREFIX="$REPO_ROOT/tests/data/toy"
WORK_DIR="$(mktemp -d)"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

cp "$REPO_ROOT/tests/data/work_dir/alignment.gaf" "$WORK_DIR/alignment.gaf"

if [[ "$ENGINE" == "mojo" ]]; then
    export METHYLGRAPHER_ENGINE=mojo
else
    unset METHYLGRAPHER_ENGINE 2>/dev/null || true
fi

echo "== methylGrapher-mojo toy run (engine=$ENGINE) =="
echo "work_dir=$WORK_DIR"
echo

echo "-- MethylCall --"
"$REPO_ROOT/bin/methylGrapher" MethylCall \
    -work_dir "$WORK_DIR" \
    -index_prefix "$INDEX_PREFIX" \
    -minimum_identity 10 \
    -minimum_mapq 0 \
    -t 1

echo
echo "-- MergeCpG --"
"$REPO_ROOT/bin/methylGrapher" MergeCpG \
    -work_dir "$WORK_DIR" \
    -index_prefix "$INDEX_PREFIX"

echo
echo "== graph.methyl =="
cat "$WORK_DIR/graph.methyl"

echo
echo "== graph.cpg.tsv =="
cat "$WORK_DIR/graph.cpg.tsv"

if [[ "${KEEP_WORK_DIR:-0}" == "1" ]]; then
    trap - EXIT
    echo
    echo "KEEP_WORK_DIR=1: output kept at $WORK_DIR"
fi
