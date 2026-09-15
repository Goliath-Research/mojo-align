# mojo-align

Open-source Mojo align monorepo for WGBS SamplePrep (MIT; linear path follows
**bwa-meth**, `methylgrapher/` follows **methylGrapher**). Git history includes
the former tree that was named `methylGrapher-mojo` (that repository no longer
exists). Unified CLI remains `bin/methylGrapher`.

## Packages

| Package | Role | License |
|---------|------|---------|
| `gpu-common/` | Portable DeviceContext device select + seed kernels + HBM preflight | **MIT** — see [`LICENSE`](LICENSE) |
| `fq2bam-meth/` | Mojo linear WGBS mapper (`align.linear.mojo`); published **bwa-meth** method class | **MIT** — see [`LICENSE`](LICENSE) |
| `giraffe/` | MojoGiraffe / GBZ stream map (`align.pangenome_wgbs.mojo`) | **MIT** — see [`LICENSE`](LICENSE) |
| `methylgrapher/` | Science + Align orchestration + `engine/` CLI | **MIT** — [`methylgrapher/LICENSE`](methylgrapher/LICENSE) (methylGrapher) |
| `numeric/` | Post-align DeviceContext kernels (centroid stream) | **MIT** — see [`LICENSE`](LICENSE) |

The whole monorepo is MIT open source. Linear mapping follows [bwa-meth](https://github.com/brentp/bwa-meth); `methylgrapher/` is a methylGrapher derivative.

## Align paths + sample layout

Canonical align IDs (folder names under each sample). **Before/after bakeoffs keep
Clara and `vg` as first-class arms** — Mojo is preferred science, not a deletion
of originals.

| ID | Runtime | Role |
|----|---------|------|
| `align.linear.parabricks` | NVIDIA Parabricks `fq2bam_meth` | **Before** linear baseline |
| `align.linear.mojo` | MojoFq2bamMeth (`fq2bam-meth`) | **After** portable linear |
| `align.pangenome.parabricks` | Parabricks `giraffe` (BAM) | Stock non-BS pangenome (not WGBS GAF) |
| `align.pangenome.vg` / `align.pangenome_wgbs.vg` | `vg giraffe` (`cpu_vg`) | **Before** named-coordinate GAF oracle |
| `align.pangenome_wgbs.mojo` | MojoGiraffe dual-graph | **After** preferred WGBS science |

Optional extract staging (MethylPipeline compare harness; not written by this CLI):

| ID | Tool | Role |
|----|------|------|
| `extract.methylextractor/` | MethylExtractor | Production linear extract |
| `extract.methyldackel/` | Upstream MethylDackel | Optional A/B only |

```
/work/samples/<sampleId>/
  *.fastq.gz
  align.linear.parabricks/   # BAM, BAI, metrics
  align.linear.mojo/         # same shape — parity vs Parabricks
  align.pangenome.parabricks/
  align.pangenome.vg/          # or align.pangenome_wgbs.vg
  align.pangenome_wgbs.mojo/
  extract.methylextractor/     # optional staging
  extract.methyldackel/        # optional A/B
```

Multiple align dirs may coexist for side-by-side parity and linear→pangenome
comparisons. MethylPipeline owns creating these folders; tools write to the
`-work_dir` they are given. Comparison reports land under
`/work/samples/_comparisons/<stamp>/` (see MethylPipeline
`docs/architecture/sample-prep-tooling.md`).

## Quick start

```bash
pixi install
./bin/methylGrapher help
METHYLGRAPHER_ENGINE=mojo ./bin/methylGrapher help
pixi run python -m pytest
```

## Linear parity (Clara vs Mojo)

```bash
fq2bam-meth/scripts/fetch_parabricks_sample.sh   # once → /work/samples/parabricks_sample
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh --device nvidia
```

Details: [`fq2bam-meth/docs/LINEAR_PARITY.md`](fq2bam-meth/docs/LINEAR_PARITY.md).

Mojo include paths (also set by `bin/methylGrapher`):

```text
-I gpu-common/src -I fq2bam-meth/src -I giraffe/src -I methylgrapher/src -I numeric/src
```

## Fleet image staging

MethylPipeline `Dockerfile.mojo` expects a flat `engine/` + `src/` tree. Assemble:

```bash
bash scripts/stage_flat_image_tree.sh /tmp/mojo-flat
export MOJO_ALIGN_ROOT=/tmp/mojo-flat   # or point at this repo + stage in the build script
```

In-container paths are `/opt/mojo-align` + `methylGrapher` entrypoint.

## CI

- **This repo (`mojo-align`)** is the only tree: [`ci/azure-pipelines.yml`](ci/azure-pipelines.yml)
- Env family is `MOJO_ALIGN_*`.

## Migration notes

See [`MIGRATION_LOG.md`](MIGRATION_LOG.md) for the Python→Mojo science cutover history.
