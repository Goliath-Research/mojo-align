# MojoFq2bamMeth linear fixture

Tiny AT-rich linear reference + PE reads for smoke-testing portable linear
WGBS Align without Clara / large genomes.

| File | Role |
|------|------|
| `ref.fa` | 64 bp `chr1` (A/T only so C2T/G2A conversion is a no-op for mapping) |
| `R1.fastq` / `R2.fastq` | 32 bp PE exact matches |

```bash
scripts/run_toy_fq2bam_meth.sh
# or
bin/methylGrapher MojoFq2bamMeth \
  -fq1 tests/data/fq2bam_fixture/R1.fastq \
  -fq2 tests/data/fq2bam_fixture/R2.fastq \
  -ref tests/data/fq2bam_fixture/ref.fa \
  -out_bam /tmp/toy.bam -out_qc_dir /tmp/toy_qc -sample_id toy \
  -device cpu -k 8 -work_dir /tmp/toy_fq2bam
```
