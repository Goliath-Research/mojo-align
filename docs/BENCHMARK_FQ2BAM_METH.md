# MojoFq2bamMeth benchmarks (linear WGBS Align)

Science contract: directional PE → BAM + QC. See [`LINEAR_FQ2BAM_SPEC.md`](LINEAR_FQ2BAM_SPEC.md).

## Toy fixture

```bash
scripts/run_toy_fq2bam_meth.sh python cpu
scripts/benchmark_fq2bam_meth.sh
```

| Backend | Device | Fixture | Notes |
|---------|--------|---------|-------|
| Mojo linear | `cpu` / `DeviceContext(api=cpu)` | PASS | native index + extend; exact-match PE |
| Mojo linear | `nvidia:sm_90` | PASS (toy) / operator (subset) | DeviceContext warmup; host-fallback if driver &lt;580 |
| Mojo linear | `amdgpu:gfx942` | operator | ROCm image `1.70-mojo-rocm` |
| BWA-MEM | CPU | PASS | `METHYLGRAPHER_LINEAR_MAPPER=bwa` or automatic `bwa_fallback` |

## Production gates (operator)

| Gate | Criterion | Status |
|------|-----------|--------|
| Toy PE BAM + QC schema | mapped > 0, JSON keys, `mapper=mojo` | **PASS** (CI/local) |
| NVIDIA subset vs Clara `pbrun fq2bam_meth` | Mojo wall ≤ Clara (±10%) | **PENDING** operator on GH200 |
| AMD MI300X twin | Mojo AMD wall ≈ NVIDIA Mojo twin; ≫ BWA CPU | **PENDING** ROCm bakeoff |
| Complete status | both wall-clock gates green | **PENDING** (default mapper is already `mojo`) |

Default `METHYLGRAPHER_LINEAR_MAPPER` is already `mojo`; operator gates decide when to mark the component **Complete** in the README, not when to flip the default.

Record operator runs:

```bash
scripts/benchmark_fq2bam_meth.sh /path/to/R1.fastq.gz /path/to/R2.fastq.gz /path/to/ref.fa
```

## Rollback

```bash
export METHYLGRAPHER_LINEAR_MAPPER=bwa
```
