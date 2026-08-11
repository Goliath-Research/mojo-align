# mojo-align

Mojo align monorepo for Epimethyl WGBS SamplePrep. History imported from
`methylGrapher-mojo`. Unified CLI remains `bin/methylGrapher`.

## Packages

| Package | Role | License |
|---------|------|---------|
| `gpu-common/` | Portable DeviceContext device select + seed kernels + HBM preflight | *none (internal)* |
| `fq2bam-meth/` | Mojo linear WGBS mapper (`align.linear.mojo`) | *none (internal)* |
| `giraffe/` | MojoGiraffe / GBZ stream map (`align.pangenome_wgbs.mojo`) | *none (internal)* |
| `methylgrapher/` | Science + Align orchestration + `engine/` CLI | **MIT** — see [`methylgrapher/LICENSE`](methylgrapher/LICENSE) |

Only `methylgrapher/` carries an open-source license (upstream methylGrapher MIT).
Do not attach LICENSE files to the other packages.

## Align paths + sample layout

Canonical align IDs (folder names under each sample):

| ID | Runtime |
|----|---------|
| `align.linear.parabricks` | NVIDIA Parabricks `fq2bam_meth` |
| `align.pangenome.parabricks` | Parabricks `giraffe` (BAM) |
| `align.linear.mojo` | MojoFq2bamMeth (`fq2bam-meth`) |
| `align.pangenome.vg` | `vg giraffe` via methylGrapher (`cpu_vg`) |
| `align.pangenome_wgbs.mojo` | MojoGiraffe dual-graph |

```
/work/samples/<sampleId>/
  *.fastq.gz
  align.linear.parabricks/   # BAM, BAI, metrics, {chr}-{ctx}.h5
  align.linear.mojo/         # same shape — parity vs Parabricks
  align.pangenome.parabricks/
  align.pangenome.vg/
  align.pangenome_wgbs.mojo/
```

Multiple align dirs may coexist for side-by-side parity and linear→pangenome
comparisons. MethylPipeline owns creating these folders; tools write to the
`-work_dir` they are given.

## Quick start

```bash
pixi install
./bin/methylGrapher help
METHYLGRAPHER_ENGINE=mojo ./bin/methylGrapher help
pixi run python -m pytest
```

Mojo include paths (also set by `bin/methylGrapher`):

```text
-I gpu-common/src -I fq2bam-meth/src -I giraffe/src -I methylgrapher/src
```

## Fleet image staging

MethylPipeline `Dockerfile.mojo` expects a flat `engine/` + `src/` tree. Assemble:

```bash
bash scripts/stage_flat_image_tree.sh /tmp/mojo-flat
export METHYLGRAPHER_MOJO_ROOT=/tmp/mojo-flat   # or point at this repo + stage in the build script
```

In-container paths remain `/opt/methylgrapher-mojo` + `methylGrapher` entrypoint.

## Dual CI

- **This repo (`mojo-align`)** — primary: [`ci/azure-pipelines.yml`](ci/azure-pipelines.yml)
- **`methylGrapher-mojo`** — kept during cutover for rollback / image pins until
  `METHYLGRAPHER_MOJO_ROOT` points here and the fleet image rebuilds clean.

## Migration notes

See [`MIGRATION_LOG.md`](MIGRATION_LOG.md) for the Python→Mojo science cutover history.
