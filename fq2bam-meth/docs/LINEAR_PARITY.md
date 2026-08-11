# Linear Align parity: `align.linear.parabricks` vs `align.linear.mojo`

Prove concordance between Clara Parabricks `fq2bam_meth` and MojoFq2bamMeth
on the same inputs, using the per-path sample layout.

## Sample layout

```
/work/samples/<sampleId>/          # or /tmp/samples/...
  <sampleId>_R1.fastq.gz
  <sampleId>_R2.fastq.gz
  align.linear.parabricks/
    <sampleId>.bam
    <sampleId>.bam.bai
    <sampleId>.fq2bam_meth.log
  align.linear.mojo/
    <sampleId>.bam
    <sampleId>.bam.bai
    qc/
    <sampleId>.fq2bam_meth.log
  linear_parity_report.json
```

## NVIDIA Parabricks tutorial sample

Stage once under the fleet sample tree:

```bash
fq2bam-meth/scripts/fetch_parabricks_sample.sh
# → /work/samples/parabricks_sample/{Data,Ref,align.linear.*}
```

Inputs:

| Path | Role |
|------|------|
| `/work/samples/parabricks_sample/Data/sample_1.fq.gz` | R1 |
| `/work/samples/parabricks_sample/Data/sample_2.fq.gz` | R2 |
| `/work/samples/parabricks_sample/Ref/Homo_sapiens_assembly38.fasta` | reference |

## Run both arms + score

```bash
# Defaults: sample-dir=/work/samples/parabricks_sample, first 50k PE pairs
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh --device nvidia

# Full sample:
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh --max-pairs 0 --device nvidia

# Score only (BAMs already present)
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh --compare-only
```

Clara runs via host `pbrun` when available, otherwise Docker
(`METHYL_PARABRICKS_IMAGE`, default `nvcr.io/nvidia/clara/clara-parabricks:4.5.1-1`).

## Gates (report JSON)

| Gate | Default | Meaning |
|------|---------|---------|
| `mapped_rate` | \|Δ\| ≤ 0.02 | `samtools flagstat` mapped / total |
| `idxstats_spearman` | ≥ 0.95 | Per-contig mapped-count Spearman |

Aligned with MethylPipeline [`mojo-fq2bam-concordance-gates.md`](../../../MethylPipeline/docs/plans/mojo-fq2bam-concordance-gates.md).

## Local smoke (no Clara sample / no GPU)

```bash
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh --toy --skip-clara --device cpu
# Then copy Mojo BAM into the Clara folder only to exercise the comparator:
# (not a science parity claim — structural smoke)
```

Prefer the Parabricks sample on an NGC/GH200 host for the real gate.
