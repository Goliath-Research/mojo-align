# methylGrapher-mojo

Mojo (Modular) cutover of [methylGrapher](https://github.com/twlab/methylGrapher) — pangenome graph methylation calling.

Original tool by Wenjin Zhang / Ting Wang Lab (WUSTL).
Mojo migration by David Izada Rodriguez / Goliath Research.

---

## Status

methylGrapher-mojo is a **cutover**, not a big-bang rewrite: a faithful Python
port of methylGrapher 0.2.0 lives in `engine/` for PrepareGenome / Main and
shared helpers, while `src/main.mojo` owns the Mojo CLI. Hot paths
(MethylCall, MergeCpG, ConversionRate, Align control plane, MojoGiraffe,
MojoFq2bamMeth mapper) are native Mojo or Mojo-orchestrated, with selective
Python interop for GAF filtering, GBZ streaming map, and linear prep/QC.
Set `METHYLGRAPHER_MCALL_ENGINE=python` to force Align / MethylCall /
MergeCpG / ConversionRate onto `engine.cli`.

| Component | Status |
|---|---|
| `engine/` (Python 0.2.0 port + patches) | 🟢 Complete — `python -m engine.cli`; Align backends; GBZ pack ensure + `quartet_map` **oracle**; MojoFq2bamMeth orchestration |
| `src/utility.mojo` | 🟢 Complete — Phred / RC / bool / subprocess + gzip text I/O |
| `src/mcall_core.mojo` | 🟢 Complete — native `alignment_path_parse`, `cs_tag_parse` |
| `src/gfa.mojo` | 🟢 Complete — segment `Dict` for MethylCall |
| `src/mcall.mojo` | 🟢 Complete — native `alignment_to_methylation` + `parallelize` MethylCall |
| `src/merge_cpg.mojo` / `conversion_rate.mojo` | 🟢 Complete — native MergeCpG + ConversionRate |
| `src/align.mojo` | 🟢 Complete — Align orchestration; map kernel via `engine.align_backends` |
| `src/giraffe_*.mojo` + `giraffe_stream_map.mojo` | 🔵 In progress — MojoGiraffe: GFA fixture native; **GBZ production** = Mojo stream map (GPU seed → locate → cluster → gapless → GAF); `quartet_map` oracle only — `docs/GIRAFFE_SPEC.md` |
| `src/linear_*.mojo` / `MojoFq2bamMeth` | 🔵 In progress — **native Mojo** linear mapper; GPU seeds feed extend; gate Mojo wall **&lt; Clara** — `docs/LINEAR_FQ2BAM_SPEC.md` |
| `src/main.mojo` | 🟢 Complete — CLI: native Align/MethylCall/MergeCpG/ConversionRate/MojoGiraffe; MojoFq2bamMeth → `engine.fq2bam_meth` (which shells `linear_mapper.mojo`); PrepareGenome/Main → `engine.cli` |
| Align backends | 🟢 `cpu_vg` / `gpu_giraffe` / `mojo_giraffe` — Parabricks GAF **NO-GO** — `docs/PHASE0_GH200_ALIGN.md` |
| `bin/methylGrapher` | 🟢 Complete — `METHYLGRAPHER_ENGINE=mojo\|python` (default: python); `MojoFq2bamMeth` dual-ship |

🟡 MVP → 🔵 In progress → 🟢 Complete

---

## Requirements

- [Mojo via `pixi`](https://docs.modular.com/mojo/manual/get-started/) — this repo pins **Mojo 1.0.0b2** (`pixi.toml` / `pixi.lock`); GPU paths use `std.gpu.host.DeviceContext` (NVIDIA driver ≥580 or `MODULAR_NVPTX_COMPILER_PATH` for CUDA create; AMD via HIP / `amdgpu:gfx942`)
- `vg` — required for `PrepareGenome` and `cpu_vg` Align; not needed for MethylCall/MergeCpG on an existing `alignment.gaf`, or fixture-scale MojoGiraffe
- `samtools` — required for `MojoFq2bamMeth`; `bwa` for explicit `LINEAR_MAPPER=bwa` or automatic `bwa_fallback`

## Install

```bash
export PATH="$HOME/.pixi/bin:$PATH"
cd methylGrapher-mojo
pixi install
```

## Run

```bash
export PATH="$HOME/.pixi/bin:$PATH"

# Python engine (default launcher)
pixi run python -m engine.cli help
bin/methylGrapher help

# Mojo CLI
pixi run mojo src/main.mojo help
METHYLGRAPHER_ENGINE=mojo bin/methylGrapher help
```

Real commands use the same argv shape as upstream methylGrapher:

```bash
bin/methylGrapher MethylCall -work_dir /work/projects/my-study/my_run \
    -index_prefix /work/projects/my-study/index/my_index \
    -minimum_identity 20 -minimum_mapq 0 -t 16

METHYLGRAPHER_ENGINE=mojo bin/methylGrapher MethylCall -work_dir ... -index_prefix ...

# Dual-graph Align (GAF → MethylCall)
METHYLGRAPHER_ENGINE=mojo bin/methylGrapher Align \
    -index_prefix ... -fq1 ... -fq2 ... -work_dir ... \
    -align_engine cpu_vg   # or gpu_giraffe | mojo_giraffe

# Native Mojo linear WGBS Align (BAM + QC; Clara fq2bam_meth substitute)
# Dual-ship: works without METHYLGRAPHER_ENGINE=mojo
bin/methylGrapher MojoFq2bamMeth \
    -fq1 R1.fastq.gz -fq2 R2.fastq.gz -ref GRCh38.fa \
    -out_bam sample.bam -out_qc_dir sample_qc -sample_id SAMPLE \
    -device auto -k 15   # cpu | nvidia | amd
```

Rollback Align/MethylCall/MergeCpG/ConversionRate to the Python engine:

```bash
METHYLGRAPHER_ENGINE=mojo METHYLGRAPHER_MCALL_ENGINE=python bin/methylGrapher MethylCall ...
```

### Toy smoke tests

```bash
scripts/run_toy_mcall.sh python   # or: mojo
scripts/parity_compare.py --a-work-dir <dir_a> --b-work-dir <dir_b>
scripts/benchmark_mcall.sh [WORK_DIR] [INDEX_PREFIX]

# MojoGiraffe — docs/GIRAFFE_SPEC.md / docs/BENCHMARK_GIRAFFE.md
# MojoFq2bamMeth — docs/LINEAR_FQ2BAM_SPEC.md / docs/BENCHMARK_FQ2BAM_METH.md
scripts/run_toy_fq2bam_meth.sh python cpu
scripts/run_toy_fq2bam_meth.sh mojo nvidia
scripts/benchmark_fq2bam_meth.sh
```

## Design Decisions vs. Original Python

| Concern | This cutover | Reason |
|---|---|---|
| PrepareGenome / Main | `engine/` via Python interop | Multiprocessing-heavy indexing stays in the 0.2.0 port |
| MethylCall hot path | Native Mojo; GAF filter via `engine.mcall.iter_alignment_batches` | Highest-value per-record work in Mojo; complex GAF bookkeeping for parity |
| Align | Mojo control plane + `engine.align_backends` | Pluggable science GAF mappers; Parabricks BAM-only (Phase 0 NO-GO) |
| MojoGiraffe (GBZ) | Mojo device gate + GPU warmup → `giraffe_stream_map` | Buffy-scale FASTQ must stream (no full-file Mojo load); `quartet_map` is oracle only |
| MojoFq2bamMeth | Native Mojo `linear_*.mojo` mapper (index + extend → SAM); Python convert/QC/`samtools` | Full linear map in Mojo, not BWA acceleration. DeviceContext seeds both NVIDIA and AMD. Auto `bwa_fallback` if Mojo subprocess fails. Clara/ROCm wall-clock gates before Complete |
| `pysam` | `vg`/`samtools` via `subprocess` | Avoids Python C-extension; Mojo uses interop for subprocess |
| `multiprocessing.Pool` | Mojo `parallelize()` on native MethylCall | Shared-memory workers on the hot loop |
| `argparse` | Manual `-key value` (Mojo + engine CLI); argparse only in `fq2bam_meth` | Matches upstream for stock commands |
| MethylCall CLI defaults | `minimum_identity=20`, `minimum_mapq=0` | Stock 0.2.0 / DS20M parity |

### Patches applied on top of methylGrapher 0.2.0 (see `MIGRATION_LOG.md`)

1. **Malformed GAF tolerance** — skip GAF lines with fewer than 12 columns.
2. **Single GFA worker only** — `gfa_worker_num=1` always (no dual-GFA when `thread > 20`).
3. **CLI quality defaults** — keep stock 20/0 (early 50/20 draft reverted).

## Architecture

```
engine/
  cli.py / mcall.py / alignments.py / align_backends.py
  gfa.py / utility.py
  giraffe_gbz_helper.py / segment_pack.py
  minimizer_index.py / zipcodes_index.py
  quartet_map.py        # GBZ oracle + ensure_pack / READY gate
  fq2bam_meth.py        # MojoFq2bamMeth orchestrator (convert, Mojo map, BWA fallback, QC)

src/
  main.mojo             # Mojo CLI dispatcher
  align.mojo / mcall.mojo / mcall_core.mojo / gfa.mojo
  merge_cpg.mojo / conversion_rate.mojo / utility.mojo
  giraffe_*.mojo        # MojoGiraffe (GBZ → giraffe_stream_map; GFA native)
  giraffe_stream_map.mojo # Production GBZ hot path
  linear_index.mojo / linear_seed.mojo / linear_gpu_kernels.mojo
  linear_extend.mojo / linear_mapper.mojo   # native linear map → SAM
  legacy_scaffold/

bin/methylGrapher       # METHYLGRAPHER_ENGINE=mojo|python

tests/                  # see tests/README.md
  data/                 # toy GFA/GAF, giraffe_fixture, fq2bam_fixture

scripts/
  run_toy_mcall.sh / parity_compare.py / benchmark_mcall.sh
  run_toy_fq2bam_meth.sh / benchmark_fq2bam_meth.sh
  build_mojo_segment_pack.py / build_mojo_gbz_cache.py
  giraffe_gaf_parity.py / giraffe_gpu_minimizer.py / gpu_seed_worker.py
  benchmark_giraffe.sh / spike_gh200_dual_graph_align.sh

docs/
  GIRAFFE_SPEC.md / BENCHMARK_GIRAFFE.md / ROCM_GIRAFFE_GATES.md
  LINEAR_FQ2BAM_SPEC.md / BENCHMARK_FQ2BAM_METH.md
  BENCHMARK_MCALL.md / PHASE0_GH200_ALIGN.md
```

## Environment knobs

| Variable | Effect |
|---|---|
| `METHYLGRAPHER_ENGINE` | Launcher: `python` (default) or `mojo` |
| `METHYLGRAPHER_MCALL_ENGINE` | `python` → Align/MethylCall/MergeCpG/ConversionRate via `engine.cli` |
| `METHYLGRAPHER_ALIGN_ENGINE` | `cpu_vg` (default) \| `gpu_giraffe` \| `mojo_giraffe` |
| `METHYLGRAPHER_GPU_GIRAFFE_FALLBACK` | `mojo` (default) \| `vg` \| `error` |
| `METHYLGRAPHER_MOJO_GIRAFFE_READY` | default **on** (`1`); set `0`/`false`/`off` to force vg for GBZ |
| `METHYLGRAPHER_GIRAFFE_DEVICE` | Giraffe device (Align backends); Mojo also honors `METHYLGRAPHER_ALIGN_DEVICE` |
| `METHYLGRAPHER_ALIGN_DEVICE` | `auto` \| `cpu` \| `nvidia` \| `amd` (Giraffe + MojoFq2bamMeth) |
| `METHYLGRAPHER_GPU_REQUIRE` | fail closed if DeviceContext cannot be created for nvidia/amd (Giraffe; default on when pinned) |
| `METHYLGRAPHER_AMDGPU_ARCH` | e.g. `gfx942` |
| `METHYLGRAPHER_LINEAR_MAPPER` | `mojo` (default) \| `bwa` \| `auto` |
| `METHYLGRAPHER_LINEAR_K` | Mojo linear k-mer size (default 15) |
| `METHYLGRAPHER_BWA_THREADS` | BWA fallback / `samtools sort` threads |
| `METHYLGRAPHER_MOJO_READ_BATCH` | streaming FASTQ batch size for `giraffe_stream_map` |
| `MODULAR_NVPTX_COMPILER_PATH` | `ptxas` path when NVIDIA driver &lt; Modular’s minimum |

## License

MIT — same as upstream methylGrapher.
