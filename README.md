# methylGrapher-mojo

Mojo (Modular) cutover of [methylGrapher](https://github.com/twlab/methylGrapher) — pangenome graph methylation calling.

Original tool by Wenjin Zhang / Ting Wang Lab (WUSTL).
Mojo migration by David Izada Rodriguez / Goliath Research.

---

## Status

methylGrapher-mojo is a **cutover**, not a big-bang rewrite: a faithful Python
port of methylGrapher 0.2.0 lives in `engine/` for PrepareGenome / Main and
shared helpers, while `src/main.mojo` owns the Mojo CLI and runs Align,
MethylCall, MergeCpG, ConversionRate, and MojoGiraffe natively (with selective
Python interop for GAF filtering, Align map backends, and parity-sensitive
merge steps). Set `METHYLGRAPHER_MCALL_ENGINE=python` to force full
`engine.cli` rollback of those native commands.

| Component | Status |
|---|---|
| `engine/` (Python 0.2.0 port + patches) | 🟢 Complete — `python -m engine.cli`; Align backends, GBZ/minimizer helpers |
| `src/utility.mojo` | 🟢 Complete — Phred / RC / bool / subprocess + gzip text I/O |
| `src/mcall_core.mojo` | 🟢 Complete — native `alignment_path_parse`, `cs_tag_parse` |
| `src/gfa.mojo` | 🟢 Complete — segment `Dict` for MethylCall |
| `src/mcall.mojo` | 🟢 Complete — native `alignment_to_methylation` + `parallelize` MethylCall |
| `src/merge_cpg.mojo` / `conversion_rate.mojo` | 🟢 Complete — native MergeCpG + ConversionRate |
| `src/align.mojo` | 🟢 Complete — Align orchestration; map kernel via `engine.align_backends` |
| `src/giraffe_*.mojo` | 🟢 Complete — MojoGiraffe (GFA or GBZ→GAF); GPU seed `nvidia:sm_90` / `amdgpu:gfx942` |
| `engine/fq2bam_meth.py` + `MojoFq2bamMeth` | 🟡 MVP — portable linear WGBS Align (Clara substitute); BWA-MEM + Parabricks-shaped QC JSON |
| `src/main.mojo` | 🟢 Complete — native: `help`/`vg_check`/`Align`/`MojoGiraffe`/`MethylCall`/`MergeCpG`/`ConversionRate`; PrepareGenome/Main → `engine.cli` |
| Align backends | 🟢 `cpu_vg` / `gpu_giraffe` / `mojo_giraffe` — `docs/GIRAFFE_SPEC.md`; Parabricks GAF **NO-GO** — `docs/PHASE0_GH200_ALIGN.md` |
| `bin/methylGrapher` | 🟢 Complete — `METHYLGRAPHER_ENGINE=mojo\|python` (default: python) |

🟡 Not started → 🔵 In progress → 🟢 Complete

---

## Requirements

- [Mojo via `pixi`](https://docs.modular.com/mojo/manual/get-started/) — this repo pins **Mojo 1.0.0b2** (`pixi.toml` / `pixi.lock`)
- `vg` (graph genome toolkit) — required for `PrepareGenome` and `cpu_vg` Align; not needed for `MethylCall`/`MergeCpG` against an existing `alignment.gaf`, or for fixture-scale `MojoGiraffe`

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

# Mojo CLI (native Align / MethylCall / MergeCpG / ConversionRate / MojoGiraffe;
# PrepareGenome / Main still dispatch to engine.cli)
pixi run mojo src/main.mojo help
METHYLGRAPHER_ENGINE=mojo bin/methylGrapher help
```

Real commands use the same argv shape as upstream methylGrapher:

```bash
bin/methylGrapher MethylCall -work_dir /work/projects/my-study/my_run \
    -index_prefix /work/projects/my-study/index/my_index \
    -minimum_identity 20 -minimum_mapq 0 -t 16

METHYLGRAPHER_ENGINE=mojo bin/methylGrapher MethylCall -work_dir ... -index_prefix ...

# Align with pluggable map backend (or METHYLGRAPHER_ALIGN_ENGINE)
METHYLGRAPHER_ENGINE=mojo bin/methylGrapher Align \
    -index_prefix ... -fq1 ... -fq2 ... -work_dir ... \
    -align_engine cpu_vg   # or gpu_giraffe | mojo_giraffe
```

Rollback native Mojo commands to the Python engine without changing the launcher:

```bash
METHYLGRAPHER_ENGINE=mojo METHYLGRAPHER_MCALL_ENGINE=python bin/methylGrapher MethylCall ...
```

### Toy smoke test

A tiny 3-segment graph + synthetic 2-read GAF lives under `tests/data/`
(see `tests/data/README.md`):

```bash
scripts/run_toy_mcall.sh python   # or: scripts/run_toy_mcall.sh mojo
```

Compare two runs for exact output parity:

```bash
scripts/parity_compare.py --a-work-dir <dir_a> --b-work-dir <dir_b>
```

Benchmark `MethylCall` at `-t 8` / `-t 16`:

```bash
scripts/benchmark_mcall.sh [WORK_DIR] [INDEX_PREFIX]
```

Mojo Giraffe toy (GFA or GBZ) — see `docs/GIRAFFE_SPEC.md` /
`docs/BENCHMARK_GIRAFFE.md`.

## Design Decisions vs. Original Python

| Concern | This cutover | Reason |
|---|---|---|
| PrepareGenome / Main | `engine/` via Python interop from Mojo CLI | Multiprocessing-heavy indexing + orchestration stays in the faithful 0.2.0 port |
| MethylCall hot path | Native Mojo (`mcall` + `mcall_core` + `gfa` + `parallelize`); GAF filter via `engine.mcall.iter_alignment_batches` | Highest-value per-record work in Mojo; complex GAF bookkeeping stays in Python for parity |
| Align | Mojo control plane (`align.mojo`) + `engine.align_backends` (`cpu_vg` / `gpu_giraffe` / `mojo_giraffe`) | Pluggable science GAF mappers; Parabricks is BAM-only (Phase 0 NO-GO) |
| MojoGiraffe | Native `src/giraffe_*.mojo` (GFA fixtures or GBZ quartet) | Production prefers GBZ + dense segment pack; `METHYLGRAPHER_MOJO_GIRAFFE_READY=1` gates site flip |
| `pysam` | `vg`/`samtools` via `subprocess` | Avoids a Python C-extension; Mojo uses interop for subprocess |
| `multiprocessing.Pool` | Mojo `parallelize()` on native MethylCall; Python path via `METHYLGRAPHER_MCALL_ENGINE=python` | Shared-memory workers on the hot loop |
| `argparse` | Manual `-key value` argv parsing | Matches upstream; no stdlib argparse in Mojo |
| MethylCall CLI defaults | `minimum_identity=20`, `minimum_mapq=0` (stock 0.2.0) | Restored for DS20M / docker parity; `mcall.py` internal kwargs still default 50/20 when called directly |

### Patches applied on top of methylGrapher 0.2.0 (see `MIGRATION_LOG.md`)

1. **Malformed GAF tolerance** — `engine/mcall.py`'s `alignment_parse()` skips
   GAF lines with fewer than 12 columns instead of raising deep inside
   alignment processing.
2. **Single GFA worker only** — `mcall_main()`/`call_parallel()` and the CLI
   always use `gfa_worker_num=1`; the original 0.2.0 dual-GFA-worker mode
   (`thread > 20`) is removed.
3. **CLI quality defaults** — `MethylCall`/`Main` keep stock 0.2.0 defaults
   `minimum_identity=20`, `minimum_mapq=0` (an early cutover draft used 50/20
   and broke DS20M parity; reverted).

## Architecture

```
engine/                 # Python engine — methylGrapher 0.2.0 port + patches
  cli.py                # PrepareGenome / Main / python-path MethylCall; `python -m engine.cli`
  mcall.py              # Methylation calling + iter_alignment_batches (filter parity)
  alignments.py         # FASTQ convert, dual-graph Align, GAF merge
  align_backends.py     # cpu_vg | gpu_giraffe | mojo_giraffe
  gfa.py / utility.py   # GFA + I/O / help / conversion-rate helpers
  giraffe_gbz_helper.py # GBZ quartet resolve + segment cache
  segment_pack.py       # Dense sequences.bin / offsets.bin
  minimizer_index.py    # mmap .shortread.withzip.min
  zipcodes_index.py     # zipcode clustering
  quartet_map.py        # MojoGiraffe ready / map entry

src/
  main.mojo             # Mojo CLI dispatcher
  align.mojo            # Align orchestration → align_backends
  mcall.mojo            # Native MethylCall + parallelize
  mcall_core.mojo       # alignment_path_parse / cs_tag_parse
  gfa.mojo / merge_cpg.mojo / conversion_rate.mojo / utility.mojo
  giraffe_*.mojo        # MojoGiraffe (GFA or GBZ → GAF)
  legacy_scaffold/      # Pre-cutover stubs (reference only)

bin/
  methylGrapher         # METHYLGRAPHER_ENGINE=mojo|python (default: python)

tests/
  test_mcall_core.mojo  # Native parser / GFA / methylation unit tests
  test_*.py             # Align backends, GBZ, minimizer, segment pack, quartet
  data/                 # Toy GFA/GAF + giraffe_fixture (GFA + gbz_toy)

scripts/
  run_toy_mcall.sh / parity_compare.py / benchmark_mcall.sh
  build_mojo_segment_pack.py / build_mojo_gbz_cache.py
  giraffe_gaf_parity.py / giraffe_gpu_minimizer.py / benchmark_giraffe.sh
  spike_gh200_dual_graph_align.sh

docs/
  GIRAFFE_SPEC.md / BENCHMARK_GIRAFFE.md / BENCHMARK_MCALL.md / PHASE0_GH200_ALIGN.md

python_reference/       # Read-only original methylGrapher 0.2.0 sources
```

## Environment knobs

| Variable | Effect |
|---|---|
| `METHYLGRAPHER_ENGINE` | Launcher: `python` (default) or `mojo` |
| `METHYLGRAPHER_MCALL_ENGINE` | `python` forces Mojo CLI to dispatch Align/MethylCall/MergeCpG/ConversionRate to `engine.cli` |
| `METHYLGRAPHER_ALIGN_ENGINE` | `cpu_vg` (default) \| `gpu_giraffe` \| `mojo_giraffe` |
| `METHYLGRAPHER_GPU_GIRAFFE_FALLBACK` | `mojo` (default) \| `vg` \| `error` |
| `METHYLGRAPHER_MOJO_GIRAFFE_READY` | `1` required for production GBZ MojoGiraffe selection |

## License

MIT — same as upstream methylGrapher.
