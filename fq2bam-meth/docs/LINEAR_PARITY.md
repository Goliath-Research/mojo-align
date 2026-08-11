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
| `/work/samples/parabricks_sample/Ref/Homo_sapiens_assembly38.fasta` | tutorial ref (no `.bwameth.c2t`) |
| Fleet Ensembl GRCh38 (auto-fallback) | used when Clara index is missing |

## Reference indexes (under `/work/genomes`)

Fleet linear GRCh38 (default after assembly38 bwameth fallback):

`/work/genomes/linear/GRCh38/ensembl-114/Homo_sapiens.GRCh38.dna.primary_assembly.fa`

Sibling indexes (prebuild once; do not rebuild per sample):

| Sibling | Tool |
|---------|------|
| `${REF}.bwameth.c2t` (+ `.bwt` …) | Clara `fq2bam_meth` |
| `${REF}.C2T.fa` | Mojo C→T reference |
| `${REF}.mojo_linear_k15/` | Mojo dense-v1 pack (`meta.json`, `kmers.bin`, `offsets.bin`, `postings.bin`, `ref.fa`) |

```bash
# Clara (if missing)
fq2bam-meth/scripts/ensure_bwameth_index.sh "$REF"

# Mojo linear (if missing) — slow, once
fq2bam-meth/scripts/ensure_mojo_linear_index.sh "$REF" 15
```

MojoFq2bamMeth auto-uses `${REF}.mojo_linear_k${k}/` when the ref lives under
`/work/genomes` (override with `-cache_dir` / `METHYLGRAPHER_LINEAR_CACHE_DIR`).
If the dense pack is missing, the first worker builds it on the fly under an
exclusive flock on `${cache}.lock` (other workers wait, then reuse). Disable
with `METHYLGRAPHER_LINEAR_CACHE_BUILD=0`.

After the dense-v1 pack is complete, upload the linear pin (includes Mojo
siblings under the same prefix) with the existing genomes sync:

```bash
MethylPipeline/scripts/sync_genomes_to_s3.sh --only linear/GRCh38/ensembl-114
```

Phase 0 `provision_selected_genomes.sh` / recipe `s3_sync` of
`linear/GRCh38/ensembl-114/` already downloads that whole tree to `/work`.
See [`reference-inventory-qnap.md`](../../../MethylPipeline/docs/deployment/reference-inventory-qnap.md).

## Run both arms + score

```bash
# Defaults: sample-dir=/work/samples/parabricks_sample, first 50k PE pairs
# Uses fleet indexed GRCh38 if assembly38 lacks .bwameth.c2t
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

## GATK 4 / Picard metrics (consumer parity)

Clara `fq2bam_meth` is positioned as GATK 4–compatible. Mojo must clear the
same consumer checks before cutover:

```bash
python3 fq2bam-meth/scripts/compare_gatk_picard_metrics.py \
  --clara-bam /work/samples/parabricks_sample/align.linear.parabricks/parabricks_sample.bam \
  --mojo-bam  /work/samples/parabricks_sample/align.linear.mojo/parabricks_sample.bam \
  --ref /work/genomes/linear/GRCh38/ensembl-114/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
  --out-dir /tmp/gatk_parity_parabricks_sample
```

Requires `gatk` on PATH or `GATK_JAR` / `PICARD_JAR`. Gates include
`ValidateSamFile`, alignment summary `% aligned`, insert-size metrics, and
flagstat mapped/proper-pair deltas. Mojo BAMs emit `@RG`, restore
pre-conversion SEQ, and run `samtools fixmate -m` → `sort` → `markdup`.

## Local smoke (no Clara sample / no GPU)

```bash
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh --toy --skip-clara --device cpu
# Then copy Mojo BAM into the Clara folder only to exercise the comparator:
# (not a science parity claim — structural smoke)
```

Prefer the Parabricks sample on an NGC/GH200 host for the real gate.
