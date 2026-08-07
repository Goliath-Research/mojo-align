#!/usr/bin/env bash
# Run host-side helper code with image vg (binds repo + abs paths).
set -euo pipefail
ARGS=()
for a in "$@"; do
  ARGS+=("$a")
done
exec docker run --rm \
  -v /home/ubuntu/methylGrapher-mojo:/home/ubuntu/methylGrapher-mojo \
  -v /tmp:/tmp \
  -w /home/ubuntu/methylGrapher-mojo \
  epimethyl/methylgrapher:1.70-mojo \
  vg "${ARGS[@]}"
