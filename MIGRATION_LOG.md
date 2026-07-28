# Migration Log

Tracking decisions and progress for the Python -> Mojo port.

---

## 2026-07-28 — Cutover: `engine/` Python port + Mojo 1.0 CLI/interop

**Context:** the 2026-07-28 "Initial scaffold" entry below stubbed out 7
`.mojo` modules against an early/assumed Mojo syntax. Mojo 1.0.0b2 (installed
via `pixi`) changed enough (`fn` removed, `def`-only; implicit stdlib imports
banned, `std.` prefix required; UTF-8-safe `String` indexing via
`s[byte=a:b]`; `alias` deprecated for `comptime`; `Copyable`/`Movable`
structs need explicit `^` moves on return) that the stub scaffold no longer
reflected how the language actually works. Rather than keep guessing at
Mojo's stdlib pipeline surface (GFA parsing, multiprocessing-style workers,
GAF I/O) up front, this session executed a **cutover**: port
methylGrapher 0.2.0 faithfully into a real, runnable Python package
(`engine/`), then bring up a real Mojo 1.0 CLI (`src/main.mojo`) that owns
the process entry point and implements the highest-value pieces natively,
dispatching everything else to `engine/` via verified Python interop. The
old stub scaffold moved to `src/legacy_scaffold/` (superseded, kept for
reference only).

**Completed:**
- `engine/` — Python package, faithful port of `python_reference/` (0.2.0):
  `__init__.py`, `cli.py`, `mcall.py`, `alignments.py`, `utility.py`,
  `gfa.py`, `mgmp.py`. Runnable as `python -m engine.cli <command> ...`
  with the exact same argv shape as upstream methylGrapher. Internal
  cross-module imports converted to package-relative (`from . import ...`).
  Three behavioral patches applied (see "Patches" below).
- `src/utility.mojo` — native Mojo: `phred_to_int`, `reverse_complement`,
  `bool_from_str`, `system_execute` (shells out via Python `subprocess`
  interop, since Mojo 1.0 has no native subprocess API yet).
- `src/mcall_core.mojo` — native Mojo `alignment_path_parse()` and
  `cs_tag_parse()`, verified against the Python reference implementation on
  hand-computed cases (plain paths, reverse-strand segments, tag offsets,
  insertions, deletions). These are the two hottest per-GAF-record parsers
  in `mcall.py` and the first candidates for a native port.
- `src/main.mojo` — CLI dispatcher, version `0.1.0-mojo`. `help` and
  `vg_check` are implemented natively in Mojo; `PrepareGenome`, `Align`,
  `MethylCall`, `MergeCpG`, `Main`, `ConversionRate` (and the
  never-implemented-upstream `mergegaf`) are forwarded, argv unchanged, to
  `engine.cli.main()` via `from std.python import Python` interop (repo
  root added to `sys.path` at dispatch time; `bin/methylGrapher` always
  `cd`s to the repo root first so this resolves regardless of caller cwd).
  **Interop gotcha discovered & handled:** the embedded CPython
  interpreter's `stdout` buffers independently of Mojo's `print()` — output
  from `engine.cli.main()` did not appear at all until `sys.stdout.flush()`
  was called explicitly from the Mojo side after the interop call; this is
  now done unconditionally after every dispatch.
- `bin/methylGrapher` — bash launcher; `METHYLGRAPHER_ENGINE=mojo` runs
  `pixi run mojo src/main.mojo "$@"`, otherwise `pixi run python -m
  engine.cli "$@"`. Resolves the repo root from its own path (works from
  any cwd) and prepends `$HOME/.pixi/bin` to `PATH`.
- `tests/data/` — tiny 3-segment GFA (`toy.gfa`/`toy.wl.gfa`), empty
  `toy.wl.node.replacement.json`, `toy.cpg.tsv` (generated from `toy.gfa`
  via `engine.utility.get_all_cpg_from_graph()`), and a synthetic
  post-processing-shape `work_dir/alignment.gaf` with two reads (one
  `C2T`-side, one `G2A`-side) engineered so every one of the graph's 5 CpG
  pairs gets full bidirectional coverage. See `tests/data/README.md`.
- `scripts/run_toy_mcall.sh` — `MethylCall` + `MergeCpG` against the toy
  fixture, selectable engine (`python`|`mojo`).
- `scripts/parity_compare.py` — order-independent diff of `graph.methyl` /
  `graph.cpg.tsv` between two runs; used to confirm Python-engine and
  Mojo-dispatch toy runs are byte-identical.
- `scripts/benchmark_mcall.sh` — `/usr/bin/time -v` around `MethylCall` at
  `-t 8` and `-t 16`.
- `README.md` rewritten for the cutover architecture; this log entry.

**Patches applied to the Python engine (0.2.0 -> `engine/`):**
1. `engine/mcall.py`'s `alignment_parse()` now skips GAF lines with fewer
   than 12 columns (a valid GAF record has >= 12 mandatory columns) instead
   of raising `IndexError` inside `get_best_alignment_from_same_read_pair()`.
2. `engine/mcall.py`'s `mcall_main()` and `call_parallel()` both force
   `gfa_worker_num=1` unconditionally; `engine/cli.py` no longer computes
   `gfa_worker_num=2` when `thread > 20` (0.2.0's dual-GFA-worker mode is
   removed for this cutover — single GFA worker only).
3. `engine/cli.py`'s `MethylCall`/`Main` commands keep stock 0.2.0 CLI
   defaults `minimum_identity=20`, `minimum_mapq=0` (pipeline does not pass
   these flags). An earlier cutover draft briefly used 50/20; that broke
   DS20M parity and was reverted.
4. (Housekeeping, not a behavior change to the active pipeline)
   `engine/mgmp.py` — guarded the module's leftover debug driver
   (`test_args(...)` + `sys.exit(3)`) under `if __name__ == "__main__":`;
   in upstream 0.2.0 this ran unconditionally at import time, making the
   (already-unused/experimental) module unimportable.

**Verification performed:**
- `pixi run python -m engine.cli help` and `pixi run mojo src/main.mojo
  help` both print the expected usage text.
- `pixi run mojo src/main.mojo vg_check` correctly reports `vg` missing in
  this sandbox (no `vg` binary installed here).
- `scripts/run_toy_mcall.sh python` and `scripts/run_toy_mcall.sh mojo`
  both produce the expected 10-row `graph.methyl` / 5-row `graph.cpg.tsv`
  (met=2, cov=2 for all 5 CpGs); `scripts/parity_compare.py` confirms the
  two engines' outputs are identical.
- `scripts/benchmark_mcall.sh` runs cleanly at `-t 8` and `-t 16` under
  `/usr/bin/time -v` (toy fixture only — not a meaningful throughput
  benchmark on 2 reads, but validates the harness).
- `src/mcall_core.mojo`'s `cs_tag_parse()` manually cross-checked against
  the `python_reference/mcall.py` algorithm for tag strings with leading
  insertions, trailing deletions, and internal indels.

**Known TODOs (in priority order):**
1. Port `mcall.alignment_to_methylation()` (the actual per-base methylation
   call logic) to Mojo, building on `mcall_core.alignment_path_parse()` /
   `cs_tag_parse()`.
2. Port GFA segment-sequence lookup (`gfa.GraphicalFragmentAssemblyMemory`)
   to native Mojo `Dict[String, String]`.
3. Replace the Python `multiprocessing` pipeline in `engine/mcall.py` with
   Mojo `parallelize()` once the hot loop above is native.
4. `utility.gzip` read/write support (currently only in the Python engine).
5. Benchmark real (non-toy) datasets with `scripts/benchmark_mcall.sh` once
   a `PrepareGenome`-indexed graph + real `alignment.gaf` are available.
6. Add `tests/*.mojo` unit tests for `mcall_core.mojo` (currently verified
   ad hoc; see "Verification performed" above).

---

## 2026-07-28 — Initial scaffold

**Completed:**
- Repository created: `DavidAtGoliathResearch/methylGrapher-mojo`
- `mojoproject.toml` — package manifest for `magic` toolchain
- All 7 source modules stubbed as `.mojo` files with full docstrings,
  struct definitions, and function signatures mirroring the Python originals
- `python_reference/README.md` — pointer to original Python sources
- `MIGRATION_LOG.md` (this file)

**Design decisions:**
- `samtools view` (subprocess) replaces `pysam` — samtools already on target env,
  avoids Python C-extension dependency, simpler to maintain
- `parallelize()` from Mojo `algorithm` replaces `multiprocessing.Pool`:
  shared memory, no GIL, no pickle, near-linear thread scaling
- Manual state-machine parsers replace Python `re` — more SIMD-friendly,
  no regex stdlib dependency in Mojo yet
- Python interop (`from python import Python`) used for `subprocess` until
  Mojo native subprocess API stabilises

**Known TODOs (in priority order):**
1. `alignments.mojo` — implement `alignment_main()` (FASTQ convert + vg giraffe)
2. `mcall.mojo` — replace serial loop with `parallelize()` batches (Phase 6)
3. `mcall.mojo` — SIMD-vectorised CpG scanner (Phase 4)
4. `utility.mojo` — gzip read/write via subprocess pipe
5. `gfa.mojo` — full SNV_trim logic
6. `longread.mojo` — SAM->graph coordinate mapping from CIGAR+cs tags
7. `utility.mojo` — `conversionrate` and `mergecpg` commands
8. Add unit tests under `tests/`

---

## Next session

Superseded by the 2026-07-28 "Cutover" entry above (this scaffold was
replaced; see its "Known TODOs" for the current plan).

---

## 2026-07-28 — DS20M subset parity + dual-ship

**Parity:** 20k-line GAF subset from `DPLST-051425-111148-DS20M` against
`hprc-d9-bs.wl.gfa` (~43 GB). After restoring CLI defaults 20/0,
`graph.methyl` (16993 rows) and `graph.cpg.tsv` (16776 rows) are identical
vs Docker `epimethyl/methylgrapher:1.70` (stock 0.2.0). See
`docs/BENCHMARK_MCALL.md`.

**Dual-ship (MethylPipeline):**
- `actionConfig.methylgrapher_wgbs.engine` = `python`|`mojo`
- Image `epimethyl/methylgrapher:1.70-mojo` built via
  `scripts/build_methylgrapher_mojo_image.sh`; smoke passed on 64K pages.
- Default remains `python` until cutover gate
  (`docs/plans/methylgrapher-mojo-cutover-gate.md`).

