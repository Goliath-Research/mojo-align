# MojoFq2bamMeth benchmarks (linear WGBS Align)

Science contract: directional PE → BAM + QC. See [`LINEAR_FQ2BAM_SPEC.md`](LINEAR_FQ2BAM_SPEC.md).

## Toy fixture

```bash
scripts/run_toy_fq2bam_meth.sh python cpu
scripts/benchmark_fq2bam_meth.sh
# Clara bakeoff (Mojo wall strictly < Clara; fail-closed GPU):
scripts/benchmark_clara_fq2bam_meth.sh [R1] [R2] [REF] nvidia
```

## Parabricks sample parity (`align.linear.parabricks` vs `align.linear.mojo`)

End-to-end concordance on NVIDIA’s public `parabricks_sample` bundle (or any
FASTQ pair), writing side-by-side product dirs:

```bash
fq2bam-meth/scripts/fetch_parabricks_sample.sh   # → /work/samples/parabricks_sample
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh --device nvidia
```

See [`LINEAR_PARITY.md`](LINEAR_PARITY.md) and [`LINEAR_ENGINES.md`](LINEAR_ENGINES.md)
(parity = frozen science default; `speed` / `fm` = opt-in until promotion gates pass).

| Backend | Device | Fixture | Notes |
|---------|--------|---------|-------|
| Mojo linear | `cpu` / `DeviceContext(api=cpu)` | PASS | streaming batches + hash postings; seeds→extend |
| Mojo linear | `nvidia:sm_90` | PASS (toy) / operator (subset) | GPU seeds wired into extend; `GPU_REQUIRE=1` bakeoff |
| Mojo linear | `amdgpu:gfx942` | operator | ROCm image `1.70-mojo-rocm` |
| BWA-MEM | CPU | PASS | `LINEAR_MAPPER=bwa` only; **blocked** when `GPU_REQUIRE=1` |

## Clara baseline (B0)

Harness: [`scripts/benchmark_clara_fq2bam_meth.sh`](../scripts/benchmark_clara_fq2bam_meth.sh).

| Run | Wall | Status |
|-----|------|--------|
| Mojo toy (`GPU_REQUIRE=1`, nvidia) | ~1.8 s | recorded locally |
| Clara `pbrun fq2bam_meth` | — | **PENDING** (`pbrun` not on this host; operator on NGC/GH200) |

## Production gates (operator)

| Gate | Criterion | Status |
|------|-----------|--------|
| Toy PE BAM + QC schema | mapped > 0, JSON keys, `mapper=mojo` | **PASS** (CI/local) |
| NVIDIA subset vs Clara `pbrun fq2bam_meth` | Mojo wall **&lt; Clara** (strict) | **PENDING** operator on GH200 |
| AMD MI300X twin | Mojo AMD wall ≈ NVIDIA Mojo twin; ≫ BWA CPU | **PENDING** ROCm bakeoff |
| Complete status | both wall-clock gates green | **PENDING** (default mapper is already `mojo`) |

Default `METHYLGRAPHER_LINEAR_MAPPER` is already `mojo`; operator gates decide when to mark the component **Complete** in the README, not when to flip the default.

Record operator runs:

```bash
scripts/benchmark_clara_fq2bam_meth.sh /path/to/R1.fastq.gz /path/to/R2.fastq.gz /path/to/ref.fa nvidia
scripts/benchmark_fq2bam_meth.sh /path/to/R1.fastq.gz /path/to/R2.fastq.gz /path/to/ref.fa
```

## Rollback

```bash
export METHYLGRAPHER_LINEAR_MAPPER=bwa
```
