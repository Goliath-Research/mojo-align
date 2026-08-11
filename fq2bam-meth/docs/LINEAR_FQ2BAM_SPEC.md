# MojoFq2bamMeth — portable linear WGBS Align

Science contract for **linear** (non-pangenome) WGBS Align as a Clara
Parabricks `fq2bam_meth` substitute on NVIDIA **and** AMD ROCm.

This path emits **BAM + QC JSON** for `methyl_alignment_qc`. It is separate
from dual-graph `Align` / MojoGiraffe (GAF → MethylCall).

## Pipeline

1. **Python orchestrator** ([`engine/fq2bam_meth.py`](../engine/fq2bam_meth.py)): directional convert (R1 C→T, R2 G→A; reference C→T), mapper select, `samtools` sort/index, Parabricks-shaped QC JSON.
2. **Native Mojo linear mapper** ([`src/linear_mapper.mojo`](../src/linear_mapper.mojo)):
   - Streaming FASTQ batches (`METHYLGRAPHER_LINEAR_READ_BATCH`, default **16384**) — never full production FASTQ as Mojo rows
   - Fleet dense-v1 mmap pack (`kmers.bin` / `offsets.bin` / `postings.bin`) via `linear_index`; in-memory Dict only for tiny fixtures
   - Optional fused BS convert (`-bs_r1 C2T` / `-bs_r2 G2A`) in the mapper
   - Portable GPU/host seeds (`linear_gpu_kernels`) **wired into** `extend_read_with_seeds` (not discarded warmup)
   - Gapless extend + PE SAM flags → SAM
3. Stream SAM → `samtools view|sort|index`.

Mapping is **native Mojo** (index + seed → extend + SAM), not a BWA wrap.

### Fallback

- Default mapper: `mojo` (`METHYLGRAPHER_LINEAR_MAPPER` unset / `mojo` / `auto`).
- Explicit CPU path: `METHYLGRAPHER_LINEAR_MAPPER=bwa`.
- If the Mojo mapper subprocess fails, the orchestrator falls back to streamed BWA-MEM (`mapper=bwa_fallback`) **unless** `METHYLGRAPHER_GPU_REQUIRE` is explicitly `1`/`true`/`yes`/`on` (bakeoff fail-closed — no BWA).

## Device targets

| Device | Kernel target | API |
|--------|---------------|-----|
| `nvidia` | `nvidia:sm_90` | `DeviceContext(api="cuda")` |
| `amd` | `amdgpu:gfx942` (or `METHYLGRAPHER_AMDGPU_ARCH`) | `DeviceContext(api="hip")` |
| `cpu` | host | `DeviceContext(api="cpu")` |
| `auto` | prefer NVIDIA → AMD → CPU | via `giraffe_device.select_device` |

DeviceContext for nvidia/amd: empty/`1` `METHYLGRAPHER_GPU_REQUIRE` fails closed on create failure; set `=0` to allow host seed extract. BWA fallback is blocked only when REQUIRE is **explicitly** on (`1`/`true`/…).

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
| `METHYLGRAPHER_LINEAR_READ_BATCH` | streaming batch size (default 16384) |
| `METHYLGRAPHER_GPU_REQUIRE` | empty/`1` fail-closes DeviceContext for nvidia/amd; explicit `1` also blocks BWA fallback; `0` allows host seeds |
| `METHYLGRAPHER_BWA_THREADS` | threads for BWA fallback / sort |

## Modules

| Path | Role |
|------|------|
| `src/linear_index.mojo` | FASTA + dense-v1 mmap pack / `{cache_dir}` |
| `python/mojo_linear_pack.py` | Build dense-v1 pack (numpy sort → CSR bins) |
| `src/linear_seed.mojo` | k-mer extract / seed postings |
| `src/linear_gpu_kernels.mojo` | DeviceContext probe + portable seed (reuses Giraffe device helpers) |
| `src/linear_extend.mojo` | `extend_read_with_seeds` + PE SAM flags (`0x8` = mate unmapped) |
| `src/linear_mapper.mojo` | streaming map → SAM (`-ref -fq1 -out_sam …`) |
| `engine/fq2bam_meth.py` | convert, invoke Mojo, BWA fallback, samtools, QC |

## Performance gates

See [`BENCHMARK_FQ2BAM_METH.md`](BENCHMARK_FQ2BAM_METH.md). Marking Complete requires NVIDIA Mojo wall **strictly &lt; Clara** `pbrun fq2bam_meth` on the same sample/SKU, plus concordance gates (flagstat / CpG), and AMD competitive with the NVIDIA Mojo twin.
