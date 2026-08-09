# MojoFq2bamMeth — portable linear WGBS Align

Science contract for **linear** (non-pangenome) WGBS Align as a Clara
Parabricks `fq2bam_meth` substitute on NVIDIA **and** AMD ROCm.

This path emits **BAM + QC JSON** for `methyl_alignment_qc`. It is separate
from dual-graph `Align` / MojoGiraffe (GAF → MethylCall).

## Pipeline

1. **Python orchestrator** ([`engine/fq2bam_meth.py`](../engine/fq2bam_meth.py)): directional convert (R1 C→T, R2 G→A; reference C→T), mapper select, `samtools` sort/index, Parabricks-shaped QC JSON.
2. **Native Mojo linear mapper** ([`src/linear_mapper.mojo`](../src/linear_mapper.mojo)):
   - Build/load C2T FASTA + k-mer index (`linear_index`)
   - Portable `DeviceContext` seed probe/warmup (`linear_gpu_kernels` → `cuda` / `hip` / `cpu`; targets `nvidia:sm_90` / `amdgpu:gfx942`)
   - Seed-and-extend on the Mojo index (`linear_extend`: exact / RC / k-mer vote + PE SAM flags) → SAM
3. Stream SAM → `samtools view|sort|index`.

Mapping is **native Mojo** (index + extend + SAM), not a BWA wrap. GPU seed currently warms/probes the accelerator; extend consumes the Mojo in-memory index (device-resident postings are the next kernel step).

### Fallback

- Default mapper: `mojo` (`METHYLGRAPHER_LINEAR_MAPPER` unset / `mojo` / `auto`).
- Explicit CPU path: `METHYLGRAPHER_LINEAR_MAPPER=bwa`.
- If the Mojo mapper subprocess fails, the orchestrator **automatically** falls back to streamed BWA-MEM and records `mapper=bwa_fallback` in the QC JSON (requires `bwa` on `PATH`).

## Device targets

| Device | Kernel target | API |
|--------|---------------|-----|
| `nvidia` | `nvidia:sm_90` | `DeviceContext(api="cuda")` |
| `amd` | `amdgpu:gfx942` (or `METHYLGRAPHER_AMDGPU_ARCH`) | `DeviceContext(api="hip")` |
| `cpu` | host | `DeviceContext(api="cpu")` |
| `auto` | prefer NVIDIA → AMD → CPU | via `giraffe_device.select_device` |

Unlike Giraffe GBZ (`METHYLGRAPHER_GPU_REQUIRE`), linear Align does **not** fail closed if CUDA/HIP create fails — it logs `host-fallback` and continues the Mojo CPU extend path.

## CLI

```bash
bin/methylGrapher MojoFq2bamMeth \
  -fq1 R1.fastq.gz -fq2 R2.fastq.gz -ref GRCh38.fa \
  -out_bam sample.bam -out_qc_dir sample_qc -sample_id SAMPLE \
  -device auto -t 16 -k 15
```

- Dual-ship: default Python engine and `METHYLGRAPHER_ENGINE=mojo` both call `engine.fq2bam_meth`.
- `-t` applies to BWA fallback / `samtools sort` (Mojo mapper has no thread flag).
- `-k` sets Mojo linear k-mer size (default `METHYLGRAPHER_LINEAR_K` / 15).

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
| `src/linear_index.mojo` | FASTA + k-mer postings / `{cache_dir}` |
| `src/linear_seed.mojo` | k-mer extract / seed postings |
| `src/linear_gpu_kernels.mojo` | DeviceContext probe + portable seed (reuses Giraffe device helpers) |
| `src/linear_extend.mojo` | exact/RC/k-mer extend + PE SAM flags (`0x8` = mate unmapped) |
| `src/linear_mapper.mojo` | end-to-end map → SAM (`-ref -fq1 -out_sam …`) |
| `engine/fq2bam_meth.py` | convert, invoke Mojo, BWA fallback, samtools, QC |

## Performance gates

See [`BENCHMARK_FQ2BAM_METH.md`](BENCHMARK_FQ2BAM_METH.md). Marking Complete requires NVIDIA ≤ Clara `fq2bam_meth` (± tolerance) and AMD competitive with the NVIDIA Mojo twin.
