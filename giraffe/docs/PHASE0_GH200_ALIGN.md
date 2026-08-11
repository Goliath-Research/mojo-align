# Phase 0 — GH200 dual-graph Align spike (go/no-go)

**Host:** NVIDIA GH200 (≈96 GiB HBM), aarch64 Grace.  
**Indexes:** `/work/genomes/pangenome/GRCh38/d9-bs/1.70` C2T/G2A (methylGrapher PrepareGenome 1.70).  
**Date:** 2026-08-06.

## Question

Can **GPU Giraffe** (Parabricks `pbrun giraffe`) map BS-converted reads to d9-bs C2T/G2A indexes and emit **GAF with `--named-coordinates` semantics** acceptable to Mojo MethylCall?

## Result: **NO-GO for Parabricks as science mapper**

| Check | Result |
|-------|--------|
| `pbrun giraffe` output formats (4.7.0-1) | **`--out-bam` only** — no GAF/GAM, no `--named-coordinates` |
| Embedded `vg` in Parabricks image | **Absent** (`vg: command not found`) |
| Stock `pangenome` path | Unchanged — Parabricks BAM remains correct for linear-surjected QC |
| Fake MethylCall from surjected BAM | **Forbidden** (loses graph path / cs tags) |

Conclusion: Parabricks accelerates stock pangenome BAM, but **cannot** satisfy methylGrapher’s dual-graph **GAF** contract on 4.7.0-1.

## Escalation (per plan)

1. **Mojo Giraffe (science GAF):** `MojoGiraffe` CLI + portable GPU seed (`nvidia:sm_90`). Backend `gpu_giraffe` defaults to `FALLBACK=mojo` and prefers **GBZ quartet** (`-gbz/-dist/-min/-zipcodes`) via staged segment cache — **no GFA size-cap** on the production path. Emergency: `FALLBACK=vg` or `align_engine=cpu_vg`.
2. **Parabricks GAF:** still blocked on NVIDIA adding GAF/`named-coordinates` (this Phase 0 NO-GO remains).
3. **Do not** flip site `align_engine=gpu_giraffe` as “Parabricks GAF” or “Buffy ≤2 h done” until operator gates in MethylPipeline `docs/plans/mojo-giraffe-cutover-gate.md` pass.

## Timing probes

Run [`scripts/spike_gh200_dual_graph_align.sh`](../scripts/spike_gh200_dual_graph_align.sh):

- Confirms Parabricks CLI gap (always).
- Optional: tiny dual `vg giraffe` GAF via `epimethyl/methylgrapher:1.70-mojo` (needs image + indexes).
- Optional: Parabricks BAM wall on converted PE subset (speed reference only — not science GAF).

Full-depth Buffy dual-map wall (~2 h gate) remains an operator measurement after Mojo Align lands; CPU vg alone was ~4 h for **one** QC giraffe on this cohort historically.

## Go criteria for flipping site default

Re-open Phase 0 when **any** of:

- Parabricks (or successor) emits GAF + named-coordinates parity vs `vg giraffe -o gaf --named-coordinates`, or
- Native Mojo/CUDA Giraffe reaches science parity,

**and** dual-map + merge wall ≤ ~2 h on full Buffy on GH200.
