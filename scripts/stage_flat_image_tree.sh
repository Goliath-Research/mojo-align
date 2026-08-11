#!/usr/bin/env bash
# Assemble a flat /opt/methylgrapher-mojo-compatible tree from mojo-align packages.
#
# Usage:
#   scripts/stage_flat_image_tree.sh [DEST]
#
# DEST defaults to ./_flat_image. Layout:
#   DEST/{engine,src,scripts,tests,bin,pixi.toml,...}
# so MethylPipeline Dockerfile.mojo can COPY engine/ src/ scripts/ tests/ unchanged.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
DEST="${1:-$REPO_ROOT/_flat_image}"

rm -rf "$DEST"
mkdir -p "$DEST"/{engine,src,scripts,tests/data,bin}

# Mojo sources → flat src/
cp -a "$REPO_ROOT"/gpu-common/src/. "$DEST/src/"
cp -a "$REPO_ROOT"/fq2bam-meth/src/. "$DEST/src/"
cp -a "$REPO_ROOT"/giraffe/src/. "$DEST/src/"
cp -a "$REPO_ROOT"/methylgrapher/src/. "$DEST/src/"

# Engine: real science modules + load implementations (prefer real files over shims)
cp -a "$REPO_ROOT"/methylgrapher/engine/. "$DEST/engine/"
# Overwrite shims with real implementations for a self-contained flat tree
cp -a "$REPO_ROOT"/gpu-common/python/gpu_mem.py "$DEST/engine/gpu_mem.py"
cp -a "$REPO_ROOT"/fq2bam-meth/python/fq2bam_meth.py "$DEST/engine/fq2bam_meth.py"
cp -a "$REPO_ROOT"/giraffe/python/. "$DEST/engine/"
# Restore package helpers that must not be overwritten incorrectly
cp -a "$REPO_ROOT"/methylgrapher/engine/__init__.py "$DEST/engine/"
cp -a "$REPO_ROOT"/methylgrapher/engine/cli.py "$DEST/engine/"
cp -a "$REPO_ROOT"/methylgrapher/engine/alignments.py "$DEST/engine/"
cp -a "$REPO_ROOT"/methylgrapher/engine/align_backends.py "$DEST/engine/"
cp -a "$REPO_ROOT"/methylgrapher/engine/mcall.py "$DEST/engine/"
cp -a "$REPO_ROOT"/methylgrapher/engine/gfa.py "$DEST/engine/"
cp -a "$REPO_ROOT"/methylgrapher/engine/utility.py "$DEST/engine/"
cp -a "$REPO_ROOT"/methylgrapher/engine/mgmp.py "$DEST/engine/"
rm -f "$DEST/engine/_pkg_shim.py"

# Scripts (legacy /opt/.../scripts paths)
cp -a "$REPO_ROOT"/giraffe/scripts/. "$DEST/scripts/"
cp -a "$REPO_ROOT"/fq2bam-meth/scripts/. "$DEST/scripts/" 2>/dev/null || true
cp -a "$REPO_ROOT"/methylgrapher/scripts/. "$DEST/scripts/" 2>/dev/null || true

# Tests + fixtures
cp -a "$REPO_ROOT"/giraffe/tests/. "$DEST/tests/" 2>/dev/null || true
cp -a "$REPO_ROOT"/fq2bam-meth/tests/. "$DEST/tests/" 2>/dev/null || true
cp -a "$REPO_ROOT"/gpu-common/tests/. "$DEST/tests/" 2>/dev/null || true
cp -a "$REPO_ROOT"/methylgrapher/tests/. "$DEST/tests/" 2>/dev/null || true

# Launcher for flat layout (single -I src)
cat > "$DEST/bin/methylGrapher" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="${REPO_ROOT}/scripts:${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
if [[ "${METHYLGRAPHER_ENGINE:-}" == "mojo" ]]; then
  exec pixi run mojo -I src src/main.mojo "$@"
else
  exec pixi run python -m engine.cli "$@"
fi
EOF
chmod +x "$DEST/bin/methylGrapher"

# Minimal pixi/metadata for the staged tree
cp -a "$REPO_ROOT/pixi.toml" "$DEST/pixi.toml" 2>/dev/null || true
cp -a "$REPO_ROOT/pixi.lock" "$DEST/pixi.lock" 2>/dev/null || true

echo "Staged flat image tree at $DEST"
find "$DEST" -maxdepth 2 -type d | head -40
