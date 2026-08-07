# methylGrapher-mojo

Mojo (Modular) cutover of [methylGrapher](https://github.com/twlab/methylGrapher) — pangenome graph methylation calling.

Original tool by Wenjin Zhang / Ting Wang Lab (WUSTL).
Mojo migration by David Izada Rodriguez / Goliath Research.

---

## Status

methylGrapher-mojo is being migrated in a **cutover** shape rather than a
big-bang rewrite: a faithful Python port of methylGrapher 0.2.0 lives in
`engine/` and does all of the actual work, while `src/` is a real Mojo 1.0
program that owns the CLI process, implements a couple of commands
natively, and reaches into `engine/` via Python interop for everything
else. Hot, per-record parsers move to native Mojo first; the
multiprocessing pipeline stays in Python until it's profiled and ported
deliberately.

| Component | Status |
|---|---|
| `engine/` (Python 0.2.0 port + patches) | 🟢 Complete — `Align`/`MethylCall`/`MergeCpG`/`Main`/`PrepareGenome`/`ConversionRate`/`help`/`vg_check` all runnable via `python -m engine.cli` |
| `src/utility.mojo` | 🟢 Complete — Phred / RC / bool / subprocess + gzip text I/O |
| `src/mcall_core.mojo` | 🟢 Complete — native `alignment_path_parse`, `cs_tag_parse` |
| `src/gfa.mojo` | 🟢 Complete — segment `Dict` for MethylCall |
| `src/mcall.mojo` | 🟢 Complete — native `alignment_to_methylation` + `parallelize` MethylCall driver |
| `src/merge_cpg.mojo` / `conversion_rate.mojo` | 🟢 Complete — native MergeCpG + ConversionRate |
| `src/main.mojo` | 🟢 Complete — `help`/`vg_check`/`Align`/`MethylCall`/`MergeCpG`/`ConversionRate` native; PrepareGenome/Main → `engine.cli` |
| Align backends | 🟢 `cpu_vg` / `gpu_giraffe` / `mojo_giraffe` (`engine/align_backends.py`); Mojo Giraffe GAF — `docs/GIRAFFE_SPEC.md`; Phase 0 Parabricks GAF **NO-GO** — `docs/PHASE0_GH200_ALIGN.md` |
| Mojo Giraffe | 🟢 GFA + **GBZ-native** (`-gbz/-dist/-min`); GPU seed `nvidia:sm_90`; production prefers GBZ quartet |
| `bin/methylGrapher` | 🟢 Complete — engine-switchable launcher (`METHYLGRAPHER_ENGINE=mojo`) |
| Native Mojo methylation-calling pipeline | 🟢 Complete for MethylCall hot path; Align / MergeCpG merge still `engine/` |

🟡 Not started → 🔵 In progress → 🟢 Complete

---

## Requirements

- [Mojo via `pixi`](https://docs.modular.com/mojo/manual/get-started/) — this repo pins **Mojo 1.0.0b2** (`pixi.toml` / `pixi.lock`)
- `vg` (graph genome toolkit) — required for `PrepareGenome`/`Align`; not needed to run `MethylCall`/`MergeCpG` against an existing `alignment.gaf`

## Install

```bash
export PATH="$HOME/.pixi/bin:$PATH"
cd methylGrapher-mojo
pixi install
```

## Run

```bash
export PATH="$HOME/.pixi/bin:$PATH"

# Python engine (default)
pixi run python -m engine.cli help
bin/methylGrapher help

# Mojo CLI (help/vg_check native; everything else dispatched to the same
# Python engine via interop)
pixi run mojo src/main.mojo help
METHYLGRAPHER_ENGINE=mojo bin/methylGrapher help
```

Real commands look the same regardless of engine (same argv shape as
upstream methylGrapher):

```bash
bin/methylGrapher MethylCall -work_dir /work/projects/my-study/my_run \
    -index_prefix /work/projects/my-study/index/my_index \
    -minimum_identity 50 -minimum_mapq 20 -t 16

METHYLGRAPHER_ENGINE=mojo bin/methylGrapher MethylCall -work_dir ... -index_prefix ...
```

### Toy smoke test

A tiny, self-contained 3-segment graph + synthetic 2-read GAF lives under
`tests/data/` (see `tests/data/README.md`) so `MethylCall`/`MergeCpG` can be
exercised end to end without `vg`, real FASTQ, or a `PrepareGenome` run:

```bash
scripts/run_toy_mcall.sh python   # or: scripts/run_toy_mcall.sh mojo
```

Compare two runs (e.g. Python engine vs. Mojo dispatch, or before/after a
native Mojo port of a pipeline stage) for exact output parity:

```bash
scripts/parity_compare.py --a-work-dir <dir_a> --b-work-dir <dir_b>
```

Benchmark `MethylCall` at `-t 8` / `-t 16` with `/usr/bin/time -v`:

```bash
scripts/benchmark_mcall.sh [WORK_DIR] [INDEX_PREFIX]
```

## Design Decisions vs. Original Python

| Concern | This cutover | Reason |
|---|---|---|
| Business logic (Align/MethylCall/GFA/multiprocessing) | Ported faithfully into `engine/`, called via Python interop from `src/main.mojo` | De-risks the cutover: the CLI, process entry point, and hot parsers move to Mojo first, without a big-bang rewrite of the whole (multiprocessing-heavy) pipeline |
| Per-alignment-line parsing (`alignment_path_parse`, `cs_tag_parse`) | Native Mojo (`src/mcall_core.mojo`) | Called once per GAF record — the highest-value functions to port first for future SIMD/parallel work |
| `pysam` | `vg`/`samtools` subprocess via Python `subprocess` (from both `engine/` and `src/utility.mojo`'s `system_execute`) | Avoids a Python C-extension dependency; Mojo 1.0 has no native subprocess API yet, so interop with `subprocess` is used from Mojo too |
| `multiprocessing.Pool` / `Process` | Mojo `parallelize()` in native MethylCall; Python path kept for `METHYLGRAPHER_MCALL_ENGINE=python` | Shared-memory workers; no GIL/pickle on the hot loop |
| `argparse` | Manual `-key value` argv parsing (both `engine/cli.py` and `src/main.mojo`) | Matches upstream methylGrapher's own manual parsing; no stdlib argparse in Mojo |

### Patches applied on top of methylGrapher 0.2.0 (see `MIGRATION_LOG.md`)

1. **Malformed GAF tolerance** — `engine/mcall.py`'s `alignment_parse()` skips
   GAF lines with fewer than 12 columns instead of raising deep inside
   alignment processing.
2. **Single GFA worker only** — `engine/mcall.py`'s `mcall_main()`/
   `call_parallel()` always use `gfa_worker_num=1`; the original 0.2.0
   dual-GFA-worker mode (`thread > 20`) is removed.
3. **Raised alignment-quality defaults** — `MethylCall`/`Main` default to
   `minimum_identity=50`, `minimum_mapq=20` (matching `mcall.py`'s own
   function defaults) instead of the original CLI defaults of `20`/`0`.

## Architecture

```
engine/              # Python engine — faithful methylGrapher 0.2.0 port + patches
  __init__.py
  cli.py             # CLI logic (mirrors python_reference/main.py); `python -m engine.cli`
  mcall.py           # Methylation calling (multiprocessing pipeline)
  alignments.py      # FASTQ conversion, vg giraffe invocation, GAF merge
  gfa.py             # GFA graph parsing/conversion
  mgmp.py            # Experimental worker-orchestration scaffolding (unused; kept for parity)
  utility.py         # I/O helpers, Phred tables, config parser, help text

src/
  main.mojo          # CLI dispatcher: help/vg_check native, else -> engine.cli via Python interop
  mcall_core.mojo    # Native alignment_path_parse() / cs_tag_parse()
  utility.mojo       # Native phred_to_int / reverse_complement / bool_from_str / system_execute
  legacy_scaffold/   # Pre-cutover Mojo stub scaffold (superseded; kept for reference)

bin/
  methylGrapher      # Launcher; METHYLGRAPHER_ENGINE=mojo|python (default: python)

tests/
  data/              # Toy 3-segment GFA + synthetic GAF fixture (see tests/data/README.md)

scripts/
  run_toy_mcall.sh   # MethylCall + MergeCpG smoke test against tests/data/
  parity_compare.py  # Diff graph.methyl / graph.cpg.tsv between two runs
  benchmark_mcall.sh # /usr/bin/time -v MethylCall at -t 8 and -t 16

python_reference/    # Read-only original methylGrapher 0.2.0 sources (do not edit)
```

## License

MIT — same as upstream methylGrapher.
