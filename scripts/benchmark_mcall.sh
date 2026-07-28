#!/usr/bin/env bash
# scripts/benchmark_mcall.sh — time & measure peak memory of MethylCall at
# `-t 8` and `-t 16`, via `/usr/bin/time -v`.
#
# Usage:
#   scripts/benchmark_mcall.sh [WORK_DIR] [INDEX_PREFIX]
#
# WORK_DIR must contain an `alignment.gaf` (as produced by `Align`, or the
# toy fixture at tests/data/work_dir/alignment.gaf, used by default).
# INDEX_PREFIX defaults to the toy fixture (tests/data/toy); point it at a
# real `{prefix}.wl.gfa` / `{prefix}.wl.node.replacement.json` pair to
# benchmark against real data.
#
# Env overrides:
#   METHYLGRAPHER_ENGINE  python (default) | mojo
#   MIN_IDENTITY          -minimum_identity value (default: 10, tuned for
#                          the tiny toy fixture; use 50 — the engine
#                          default — for real alignments)
#   MIN_MAPQ              -minimum_mapq value (default: 0; engine default
#                          for real alignments is 20)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$REPO_ROOT"

WORK_DIR_SRC="${1:-$REPO_ROOT/tests/data/work_dir}"
INDEX_PREFIX="${2:-$REPO_ROOT/tests/data/toy}"
ENGINE="${METHYLGRAPHER_ENGINE:-python}"
MIN_IDENTITY="${MIN_IDENTITY:-10}"
MIN_MAPQ="${MIN_MAPQ:-0}"

if [[ ! -f "$WORK_DIR_SRC/alignment.gaf" ]]; then
    echo "error: $WORK_DIR_SRC/alignment.gaf not found" >&2
    exit 1
fi

if ! /usr/bin/time -v true >/dev/null 2>&1; then
    echo "error: /usr/bin/time -v is not available (install GNU 'time')" >&2
    exit 1
fi

for THREADS in 8 16; do
    echo "================================================================"
    echo "MethylCall benchmark: engine=$ENGINE threads=$THREADS"
    echo "  work_dir_src=$WORK_DIR_SRC index_prefix=$INDEX_PREFIX"
    echo "  minimum_identity=$MIN_IDENTITY minimum_mapq=$MIN_MAPQ"
    echo "================================================================"

    BENCH_DIR="$(mktemp -d)"
    cp "$WORK_DIR_SRC/alignment.gaf" "$BENCH_DIR/alignment.gaf"

    METHYLGRAPHER_ENGINE="$ENGINE" /usr/bin/time -v \
        "$REPO_ROOT/bin/methylGrapher" MethylCall \
        -work_dir "$BENCH_DIR" \
        -index_prefix "$INDEX_PREFIX" \
        -minimum_identity "$MIN_IDENTITY" \
        -minimum_mapq "$MIN_MAPQ" \
        -t "$THREADS"

    rm -rf "$BENCH_DIR"
    echo
done
