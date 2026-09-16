# Migration Log

Tracking decisions and progress for the Python -> Mojo port.

**Current status** (implementation source of truth): see `README.md`.
Historical entries below are chronological; later dated sections supersede
earlier TODOs when they conflict.

---

## 2026-08-09 — No app-level CUDA Runtime on native Mojo path

**Problem:** `engine/gpu_h2d.py` called `libcudart` `cudaMemcpy` for index H2D after the
native-Mojo cutover — an explicit CUDA Runtime dependency in app code.

**Completed:**
- Index upload is Mojo-only: HostBuffer + `enqueue_copy` + device `copy_bytes_offset_kernel`.
- `engine/gpu_h2d.py` stub raises if imported; container entrypoint no longer prepends cudart.
- HBM preflight stays on `nvidia-smi` / `rocm-smi` (not `cudaMemGetInfo`).

## 2026-08-09 — GPU HBM preflight before index upload

**Problem:** Index residency OOM surfaced as raw `CUDA_ERROR_OUT_OF_MEMORY` after
`gpu_index_resident_gib≈37` while live Align already held ~87 GiB HBM.

**Completed:**
- `engine/gpu_mem.py` — query free/total via `nvidia-smi` / `rocm-smi` (no CUDA context).
- `require_gpu_index_capacity` before `DeviceContext` buffer alloc; clear need/free GiB error.
- Overhead gate default 2.4× science slabs (`METHYLGRAPHER_GPU_RESIDENT_OVERHEAD`); disable with `METHYLGRAPHER_GPU_MEM_PREFLIGHT=0`.

## 2026-08-09 — GPU-native stream map (HT + gapless on DeviceContext)

**Problem:** Prior “native Mojo” path only GPU-hashed seeds, then host window-reduce /
mmap HT / cluster / gapless on one CPU thread (~20 k pairs/s, GPU util 0%).

**Completed:**
- `giraffe_gpu_index.mojo` + `engine/gpu_h2d.py` — chunked H2D of `.min` HT + dense pack once per graph.
- `giraffe_gpu_map_kernels.mojo` — device window-reduce, Q1Q1 unique HT probe, cluster prune, gapless; D2H hits only; host GAF emit.
- `giraffe_stream_map` GPU session retargeted; fail-closed when `GPU_REQUIRE` and backend is not DeviceContext CUDA/HIP.
- Banner: `seed_backend=devicecontext-cuda+gpu_ht+gpu_gapless+mojo_stream`.
- Toy PE smoke: 6 GAF lines on nvidia (short-read fixture) + non-zero GPU stage timers on longer reads.
- Fleet Align caps remain stripped until Buffy ≤2 h + DS20M gates.

## 2026-08-09 — Docs resync to full native-Mojo map paths

**Synced docs to code:**
- Production GBZ Align = `giraffe_stream_map` (not Python `quartet_map` hot loop).
- Dual-graph C2T/G2A **serialized by default** on GPU/Mojo; `DUAL_GRAPH_PARALLEL=1` opt-in.
- Linear: seeds wired into `extend_read_with_seeds`; `LINEAR_READ_BATCH` default 16384;
  Complete gate = Mojo wall **strictly &lt; Clara** (not ±10%).
- `GPU_REQUIRE`: empty/`1` fail-closes DeviceContext for nvidia/amd; explicit `1` blocks BWA.
- CuPy / host-nvidia fallback refused on production stream map.
- Updated: README, `GIRAFFE_SPEC`, `LINEAR_FQ2BAM_SPEC`, `BENCHMARK_*`.

---

## 2026-08-09 — Dual-map serialize + Mojo pack/cluster + QC fail-closed

**Completed:**
- Serialize C2T/G2A by default on GPU/Mojo (`METHYLGRAPHER_DUAL_GRAPH_PARALLEL=0`)
  to avoid dual DeviceContext `CUDA_ERROR_ILLEGAL_ADDRESS`.
- `giraffe_pack.mojo` — contiguous dense-v1 via libc mmap (no per-get Python).
- `giraffe_dist.mojo` — Mojo-native bucket + abs-Δ prune (no DistIndex import).
- QC BAM: `qc_bam_fallback=error` default for mojo QC (no multi-hour vg).
- Smoke: `scripts/smoke_mojo_pack_cluster.mojo`.

## 2026-08-09 — Native Mojo GBZ stream map (hot path)

**Completed:**
- `src/giraffe_stream_map.mojo` — batched FASTQ → Mojo Giraffe `(k,w)` minimizers
  (`giraffe_minimizer.mojo`: DeviceContext pack+fwd/RC hash, Mojo window reduce;
  **no CuPy / host-nvidia-fallback**) → HT `locate_key_batches` → Mojo cluster →
  `gapless_extend_with_pack` → streaming GAF; PE `ri`/`os`/`rc`.
- `giraffe_gbz.map_gbz_native` no longer calls Python `map_fastq_to_gaf` for the
  hot loop; `engine/quartet_map.py` demoted to oracle + `ensure_pack_for_gbz`.
- Production locate no longer imports `scripts/giraffe_gpu_minimizer.py` (CuPy).
- GPU seed kernels decode k-mers from DeviceContext codes (hashes feed extend).
- Smoke: `scripts/smoke_mojo_stream_map.mojo`; tests: `tests/test_mojo_stream_map.py`.
- Docs: `GIRAFFE_SPEC.md` / README updated.

**Operator gates (unchanged):** Buffy dual-map ≤ ~2 h; DS20M MethylCall parity.

---

## 2026-08-09 — Docs resync after Mojo Giraffe + linear fixes

**Historical (superseded same day by native stream map + later docs resync):**
- Earlier draft claimed GBZ → streaming `engine.quartet_map` and linear seed as
  probe/warmup only — both false after `giraffe_stream_map` + `extend_read_with_seeds`.
- Still accurate from this entry: READY default-on; env knobs; test inventory;
  identity/mapq 20/0 help header.

---

## 2026-08-08 — MojoFq2bamMeth portable linear GPU Align

**Completed:**
- Toolchain: `std.gpu.host.DeviceContext` importable on Mojo 1.0.0b2
  (`api=cpu` works; CUDA create needs driver ≥580 / `MODULAR_NVPTX_COMPILER_PATH`).
  No Modular pin bump required.
- Native modules `src/linear_{index,seed,gpu_kernels,extend,mapper}.mojo` —
  C2T FASTA index, portable seed (`nvidia:sm_90` / `amdgpu:gfx942`),
  seed-and-extend → SAM, PE flags.
- `engine/fq2bam_meth.py` prefers Mojo linear mapper; BWA-MEM is CPU fallback
  (`METHYLGRAPHER_LINEAR_MAPPER`). Streamed BWA→samtools path retained.
- Dual-ship: `MojoFq2bamMeth` on `engine.cli` and Mojo `main.mojo`.
- QC: `samtools flagstat` + `stats` derived fields; placeholders labeled
  (`metrics_source=samtools+placeholders`).
- Fixture `tests/data/fq2bam_fixture/`, `tests/test_fq2bam_meth.py`,
  `scripts/run_toy_fq2bam_meth.sh`, `scripts/benchmark_fq2bam_meth.sh`.
- Spec / gates: `docs/LINEAR_FQ2BAM_SPEC.md`, `docs/BENCHMARK_FQ2BAM_METH.md`.

**Operator gates (block Complete):**
- NVIDIA subset wall **strictly &lt; Clara** `pbrun fq2bam_meth` (same sample/SKU).
- AMD MI300X twin competitive with NVIDIA Mojo and ≫ BWA CPU.
  *(±10% wording from early draft superseded.)*

---

## 2026-08-07 — GBZ-native Mojo Giraffe

**Completed:**
- GBZ contract + toy PrepareGenome-shaped fixtures (`tests/data/giraffe_fixture/gbz_toy/`).
- `engine/giraffe_gbz_helper.py` + `scripts/build_mojo_gbz_cache.py` (vg convert / segment cache).
- Mojo modules `giraffe_gbz` / `giraffe_minzip` / `giraffe_dist`; CLI `-gbz/-dist/-min/-zipcodes`.
- `align_backends`: prefer GBZ quartet for `gpu_giraffe`/`mojo_giraffe` (GFA size-cap no longer blocks production).
- Toy GBZ PE golden parity PASS; GPU seed `nvidia:sm_90` on GH200.
- Buffy ≤2h + DS20M MethylCall parity still operator gates after C2T/G2A caches.

---

## 2026-08-06 — Mojo GPU Giraffe GAF

**Completed:**
- Spec + golden fixture: `docs/GIRAFFE_SPEC.md`, `tests/data/giraffe_fixture/`.
- Native modules `src/giraffe_*.mojo` + `MojoGiraffe` CLI (GFA→GAF, PE tags).
- Portable GPU seed via `giraffe_device` + `scripts/giraffe_gpu_minimizer.py`
  (targets `nvidia:sm_90` / `amdgpu`; CuPy was optional then — production
  stream map later DeviceContext-only).
- `engine/align_backends.py` — `cpu_vg` | `gpu_giraffe` | `mojo_giraffe`;
  `gpu_giraffe` defaults to prefer Mojo (`FALLBACK=mojo`) with auto-vg for
  oversized/GBZ-only indexes.
- Benchmarks: `docs/BENCHMARK_GIRAFFE.md`, `scripts/benchmark_giraffe.sh`.
- Progressive: Buffy ≤2 h + GBZ-native + DS20M MethylCall parity still operator gates.

---

## 2026-08-06 — Mojo Align orchestration + GH200 Phase 0

**Completed:**
- Phase 0 spike: Parabricks 4.7 `pbrun giraffe` is **BAM-only** (no GAF /
  named-coordinates) → **NO-GO** as MethylCall science mapper on GH200.
  See `docs/PHASE0_GH200_ALIGN.md` + `scripts/spike_gh200_dual_graph_align.sh`.
- `engine/align_backends.py` — `cpu_vg` | `gpu_giraffe` (default fallback `vg`
  GAF until a true GPU→GAF tool exists; `FALLBACK=error` fails closed).
- `src/align.mojo` + `main.mojo` Align native orchestration; `-align_engine`.
- Tests: `tests/test_align_backends.py`.

**Operator:** do not treat `align_engine=gpu_giraffe` as Parabricks GAF; it is
GH200 dual-graph Align with vg GAF interim unless Phase 0 is re-opened green.

---

## 2026-08-06 — Native MethylCall hot path + parallelize

**Completed:**
- `src/gfa.mojo` — `GraphicalFragmentAssemblyMemory` loads `S` lines into
  `Dict[String, String]` (gzip-aware via `utility.open_text_read`).
- `src/mcall.mojo` — native `alignment_to_methylation()` on top of
  `mcall_core` parsers; `run_methylcall_native()` drives MethylCall with
  Mojo GFA lookup + `std.algorithm.parallelize` over fragments per batch.
  GAF filtering still uses `engine.mcall.iter_alignment_batches` (parity).
- `src/merge_cpg.mojo` / `src/conversion_rate.mojo` — native MergeCpG +
  ConversionRate; wired from `src/main.mojo` (same python env rollback).
- `src/utility.mojo` — `open_text_read` / `open_text_write` / gzip helpers.
- `src/main.mojo` — `MethylCall` / `MergeCpG` / `ConversionRate` native;
  set `METHYLGRAPHER_MCALL_ENGINE=python` for legacy `engine.cli`.
- `engine/mcall.py` — `iter_alignment_batches`, gzip-aware `alignment.gaf`
  open (`.gz` / `.gzip` siblings).
- `tests/test_mcall_core.mojo` — parser / GFA / methylation / gzip unit tests
  (`pixi run mojo -I src tests/test_mcall_core.mojo`).
- Parity: toy + DS20M subset `graph.methyl` / `graph.cpg.tsv` identical vs
  python engine; see `docs/BENCHMARK_MCALL.md` (native ~15 GiB RSS vs ~22 GiB).

**Known TODOs:**
1. Operator full-Buffy GAF (~679 GiB) wall/RSS after `:1.70-mojo` deploy.
2. Port Align / remaining GAF filter into Mojo when Align becomes the bottleneck.
3. Optional SIMD CpG scan (legacy scaffold Phase 4).

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

**Known TODOs (at time of entry; superseded by later 2026-08-06/07 work):**
1. Port `mcall.alignment_to_methylation()` (the actual per-base methylation
   call logic) to Mojo, building on `mcall_core.alignment_path_parse()` /
   `cs_tag_parse()`. → Done 2026-08-06 (`src/mcall.mojo`).
2. Port GFA segment-sequence lookup (`gfa.GraphicalFragmentAssemblyMemory`)
   to native Mojo `Dict[String, String]`. → Done 2026-08-06 (`src/gfa.mojo`).
3. Replace the Python `multiprocessing` pipeline in `engine/mcall.py` with
   Mojo `parallelize()` once the hot loop above is native. → Done for
   MethylCall hot path 2026-08-06; GAF filter still in `engine.mcall`.
4. `utility.gzip` read/write support (currently only in the Python engine).
   → Done 2026-08-06 (`src/utility.mojo` open_text_*).
5. Benchmark real (non-toy) datasets with `scripts/benchmark_mcall.sh` once
   a `PrepareGenome`-indexed graph + real `alignment.gaf` are available.
   → DS20M subset measured; see `docs/BENCHMARK_MCALL.md`.
6. Add `tests/*.mojo` unit tests for `mcall_core.mojo` (currently verified
   ad hoc; see "Verification performed" above). → Done (`tests/test_mcall_core.mojo`).

---

## 2026-07-28 — Initial scaffold

**Completed:**
- Repository created under the former name `methylGrapher-mojo` (history now lives in this repo)
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
vs Docker `goliath/methylgrapher:1.70` (stock 0.2.0). See
`docs/BENCHMARK_MCALL.md`.

**Dual-ship (MethylPipeline):**
- `actionConfig.methylgrapher_wgbs.engine` = `python`|`mojo`
- Image `goliath/methylgrapher:1.70-mojo` built via
  `scripts/build_mojo_align_image.sh`; smoke passed on 64K pages.
- Default remains `python` until cutover gate
  (`docs/plans/methylgrapher-mojo-cutover-gate.md`).

