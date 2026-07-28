# Migration Log

Tracking decisions and progress for the Python -> Mojo port.

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

- [ ] Implement `alignment_main()` in `alignments.mojo`
- [ ] Wire `parallelize()` in `mgmp.mojo`
- [ ] Write `test_gfa.mojo` with a synthetic 3-node GFA
