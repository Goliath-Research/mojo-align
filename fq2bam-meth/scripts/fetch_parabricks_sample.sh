#!/usr/bin/env bash
# Fetch NVIDIA Parabricks tutorial sample into /work/samples/parabricks_sample.
#
# Layout after extract (matches NVIDIA docs + our sampleDir contract):
#   /work/samples/parabricks_sample/
#     Data/sample_1.fq.gz
#     Data/sample_2.fq.gz
#     Ref/Homo_sapiens_assembly38.fasta (+ indexes)
#     align.linear.parabricks/   (created empty)
#     align.linear.mojo/         (created empty)
#     README.md
#
# Usage:
#   fq2bam-meth/scripts/fetch_parabricks_sample.sh [--force]
set -euo pipefail

SAMPLE_DIR="${PARABRICKS_SAMPLE_DIR:-/work/samples/parabricks_sample}"
URL="${PARABRICKS_SAMPLE_URL:-https://s3.amazonaws.com/parabricks.sample/parabricks_sample.tar.gz}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

mkdir -p "$(dirname "$SAMPLE_DIR")"
PARENT="$(dirname "$SAMPLE_DIR")"
NAME="$(basename "$SAMPLE_DIR")"

if [[ -f "$SAMPLE_DIR/Data/sample_1.fq.gz" && -f "$SAMPLE_DIR/Ref/Homo_sapiens_assembly38.fasta" && "$FORCE" -eq 0 ]]; then
  echo "Already present: $SAMPLE_DIR"
else
  TMP_TGZ="${TMPDIR:-/tmp}/parabricks_sample.tar.gz"
  echo "Downloading $URL → $TMP_TGZ"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$TMP_TGZ" "$URL"
  else
    curl -L -C - -o "$TMP_TGZ" "$URL"
  fi
  echo "Extracting into $PARENT (creates $NAME/)"
  # Tar top-level dir is parabricks_sample/
  if [[ -d "$SAMPLE_DIR" && "$FORCE" -eq 1 ]]; then
    rm -rf "$SAMPLE_DIR"
  fi
  tar -xvf "$TMP_TGZ" -C "$PARENT"
  if [[ "$(basename "$SAMPLE_DIR")" != "parabricks_sample" ]]; then
    # Allow alternate SAMPLE_DIR name
    mv "$PARENT/parabricks_sample" "$SAMPLE_DIR"
  fi
fi

mkdir -p \
  "$SAMPLE_DIR/align.linear.parabricks" \
  "$SAMPLE_DIR/align.linear.mojo" \
  "$SAMPLE_DIR/work"

# Do NOT symlink sample_* → root names that the parity harness writes;
# subset outputs go under work/ so Data/ stays immutable.

cat > "$SAMPLE_DIR/README.md" << EOF
# parabricks_sample

NVIDIA Clara Parabricks tutorial inputs staged for Epimethyl linear parity:

- Source: \`$URL\`
- Align paths: \`align.linear.parabricks/\` vs \`align.linear.mojo/\`

Run:

\`\`\`bash
export PARABRICKS_SAMPLE=$SAMPLE_DIR
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh \\
  --parabricks-sample $SAMPLE_DIR \\
  --sample-dir $SAMPLE_DIR \\
  --sample-id parabricks_sample \\
  --device nvidia
\`\`\`
EOF

echo "Ready: $SAMPLE_DIR"
ls -la "$SAMPLE_DIR"
ls -la "$SAMPLE_DIR/Data" | head
ls -la "$SAMPLE_DIR/Ref" | head
