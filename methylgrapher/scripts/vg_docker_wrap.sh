#!/usr/bin/env bash
# Run host-side helper code with image vg (binds this mojo-align checkout).
set -euo pipefail
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
ARGS=()
for a in "$@"; do
  ARGS+=("$a")
done
exec docker run --rm \
  -v "${REPO_ROOT}:${REPO_ROOT}" \
  -v /tmp:/tmp \
  -w "${REPO_ROOT}" \
  goliath/methylgrapher:1.70-mojo \
  vg "${ARGS[@]}"
