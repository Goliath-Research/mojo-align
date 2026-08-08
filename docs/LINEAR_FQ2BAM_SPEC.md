# MojoFq2bamMeth — portable linear WGBS Align

Science contract for **linear** (non-pangenome) WGBS Align as a Clara
Parabricks `fq2bam_meth` substitute on NVIDIA **and** AMD ROCm.

This path emits **BAM + QC JSON** for `methyl_alignment_qc`. It is separate
from dual-graph `Align` / MojoGiraffe (GAF → MethylCall).

## Pipeline

1. Directional convert: R1 C→T, R2 G→A; reference C→T.
2. **Mojo linear mapper** (`src/linear_*.mojo`): k-mer index + portable GPU
   seed (`DeviceContext` `cuda` / `hip` / `cpu`) + seed-and-extend → SAM.
3. `samtools view|sort|index`.
4. Parabricks-shaped metrics JSON (`metrics_source=samtools+placeholders`).

BWA-MEM is **CPU fallback only** (`METHYLGRAPHER_LINEAR_MAPPER=bwa`).

## Device targets

| Device | Kernel target | API |
|--------|---------------|-----|
| `nvidia` | `nvidia:sm_90` | `DeviceContext(api="cuda")` |
| `amd` | `amdgpu:gfx942` (or `METHYLGRAPHER_AMDGPU_ARCH`) | `DeviceContext(api="hip")` |
| `cpu` | host | `DeviceContext(api="cpu")` |
| `auto` | prefer NVIDIA → AMD → CPU | same as Giraffe |

## CLI

```bash
bin/methylGrapher MojoFq2bamMeth \
  -fq1 R1.fastq.gz -fq2 R2.fastq.gz -ref GRCh38.fa \
  -out_bam sample.bam -out_qc_dir sample_qc -sample_id SAMPLE \
  -device auto -t 16
```

Works on default Python engine and `METHYLGRAPHER_ENGINE=mojo`.

## Env

| Variable | Effect |
|----------|--------|
| `METHYLGRAPHER_LINEAR_MAPPER` | `mojo` (default) \| `bwa` \| `auto` |
| `METHYLGRAPHER_ALIGN_DEVICE` | default `-device` |
| `METHYLGRAPHER_AMDGPU_ARCH` | e.g. `gfx942` |
| `METHYLGRAPHER_LINEAR_K` | k-mer size (default 15) |
| `METHYLGRAPHER_BWA_THREADS` | threads for BWA fallback / sort |

## Modules

| Path | Role |
|------|------|
| `src/linear_index.mojo` | FASTA + k-mer postings / cache |
| `src/linear_seed.mojo` | k-mer extract |
| `src/linear_gpu_kernels.mojo` | DeviceContext probe + portable seed |
| `src/linear_extend.mojo` | extend / PE flags / SAM lines |
| `src/linear_mapper.mojo` | end-to-end map → SAM |
| `engine/fq2bam_meth.py` | convert, mapper select, samtools, QC |

## Performance gates

See [`BENCHMARK_FQ2BAM_METH.md`](BENCHMARK_FQ2BAM_METH.md). Complete requires
NVIDIA ≤ Clara `fq2bam_meth` (± documented tolerance) and AMD competitive with
the NVIDIA Mojo twin.
