# MojoFq2bamMeth benchmarks (linear WGBS Align)

Science contract: directional PE → BAM + QC. See [`LINEAR_FQ2BAM_SPEC.md`](LINEAR_FQ2BAM_SPEC.md).

## Toy fixture

```bash
scripts/run_toy_fq2bam_meth.sh python cpu
scripts/benchmark_fq2bam_meth.sh
```

| Backend | Device | Fixture | Notes |
|---------|--------|---------|-------|
| Mojo linear | `cpu` / `DeviceContext(api=cpu)` | PASS | exact match PE |
| Mojo linear | `nvidia:sm_90` | operator | needs Modular NVPTX driver ≥580 or `MODULAR_NVPTX_COMPILER_PATH` |
| Mojo linear | `amdgpu:gfx942` | operator | ROCm image `1.70-mojo-rocm` |
| BWA-MEM fallback | CPU | PASS | `METHYLGRAPHER_LINEAR_MAPPER=bwa` |

## Production gates (operator)

| Gate | Criterion | Status |
|------|-----------|--------|
| Toy PE BAM + QC schema | mapped > 0, JSON keys present | **PASS** (CI/local) |
| NVIDIA subset vs Clara `pbrun fq2bam_meth` | Mojo wall ≤ Clara (±10%) | **PENDING** operator on GH200 |
| AMD MI300X twin | Mojo AMD wall ≈ NVIDIA Mojo twin; ≫ BWA CPU | **PENDING** ROCm bakeoff |
| Site flip | `METHYLGRAPHER_LINEAR_MAPPER=mojo` default | after NVIDIA+AMD gates |

Record operator runs:

```bash
scripts/benchmark_fq2bam_meth.sh /path/to/R1.fastq.gz /path/to/R2.fastq.gz /path/to/ref.fa
```

## Rollback

```bash
export METHYLGRAPHER_LINEAR_MAPPER=bwa
```
