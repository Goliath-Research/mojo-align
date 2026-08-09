# MojoFq2bamMeth — portable linear WGBS Align

Science contract for **linear** (non-pangenome) WGBS Align as a Clara
Parabricks `fq2bam_meth` substitute on NVIDIA **and** AMD ROCm.

This path emits **BAM + QC JSON** for `methyl_alignment_qc`. It is separate
from dual-graph `Align` / MojoGiraffe (GAF → MethylCall).

## Pipeline

1. **Python orchestrator** ([`engine/fq2bam_meth.py`](../engine/fq2bam_meth.py)): directional convert (R1 C→T, R2 G→A; reference C→T), mapper select, `samtools` sort/index, Parabricks-shaped QC JSON.
2. **Native Mojo linear mapper** ([`src/linear_mapper.mojo`](../src/linear_mapper.mojo)):
   - Streaming FASTQ batches (`METHYLGRAPHER_LINEAR_READ_BATCH`, default 4096) — never full production FASTQ as Mojo rows
   - Hash postings `Dict[kmer → locs]` (`linear_index`) — no `hit_table` linear scan
   - Optional fused BS convert (`-bs_r1 C2T` / `-bs_r2 G2A`) in the mapper
   - Portable GPU/host seeds (`linear_gpu_kernels`) **wired into** `extend_read_with_seeds`
   - Gapless extend + PE SAM flags → SAM
3. Stream SAM → `samtools view|sort|index`.

Mapping is **native Mojo** (index + extend + SAM), not a BWA wrap.

### Fallback

- Default mapper: `mojo` (`METHYLGRAPHER_LINEAR_MAPPER` unset / `mojo` / `auto`).
- Explicit CPU path: `METHYLGRAPHER_LINEAR_MAPPER=bwa`.
- If the Mojo mapper subprocess fails, the orchestrator falls back to streamed BWA-MEM (`mapper=bwa_fallback`) **unless** `METHYLGRAPHER_GPU_REQUIRE=1` (bakeoff fail-closed).

## Device targets

| Device | Kernel target | API |
|--------|---------------|-----|
| `nvidia` | `nvidia:sm_90` | `DeviceContext(api="cuda")` |
| `amd` | `amdgpu:gfx942` (or `METHYLGRAPHER_AMDGPU_ARCH`) | `DeviceContext(api="hip")` |
| `cpu` | host | `DeviceContext(api="cpu")` |
| `auto` | prefer NVIDIA → AMD → CPU | via `giraffe_device.select_device` |

`METHYLGRAPHER_GPU_REQUIRE=1` (Clara bakeoffs): forbids `LINEAR_MAPPER=bwa` and disables automatic BWA fallback on Mojo failure. Without it, DeviceContext create may log `host-fallback` and continue Mojo CPU extend.

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
| `METHYLGRAPHER_LINEAR_READ_BATCH` | streaming batch size (default 4096) |
| `METHYLGRAPHER_GPU_REQUIRE` | `1` = fail-closed GPU bakeoff (no BWA) |
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
