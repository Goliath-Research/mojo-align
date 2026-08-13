# Linear GPU engines: fm vs parity vs speed

Three Mojo GPU mappers share the orchestrator and SAM/BAM consumer path.

| Engine | Env | Module | Index | Role |
|--------|-----|--------|-------|------|
| **fm** (default) | `METHYLGRAPHER_LINEAR_ENGINE=fm` | `src/linear_gpu_fm.mojo` | BWA `.bwt/.sa/.pac` | Clara-shaped FM-index path |
| **parity** | `METHYLGRAPHER_LINEAR_ENGINE=parity` | `src/linear_gpu_locate.mojo` | dense-v1 k-mer pack | Frozen k-mer science path |
| **speed** (opt-in) | `METHYLGRAPHER_LINEAR_ENGINE=speed` | `src/linear_gpu_speed.mojo` | dense-v1 k-mer pack | K-mer consensus experiments |

## Why separate engines

**fm** is the Clara-shaped replacement: GPU FM-index seed + extend on the same
`bwameth.c2t` BWA index Clara uses. Parity stays as a frozen k-mer fallback
(`science` alias). Speed may still tune that k-mer index.

## Science gates (vs Clara)

Same report as [`LINEAR_PARITY.md`](LINEAR_PARITY.md):

| Gate | Default |
|------|---------|
| Mapped rate \|Δ\| | ≤ 0.02 |
| Primary idxstats Spearman | ≥ 0.95 |

FM cleared these on the Parabricks 100k smoke and the full ~53M-read sample
(mapped 98.33% vs Clara 99.68%, Δ 0.014; Spearman 1.00; above frozen parity
97.79%). Production default is **fm**.

## Run

```bash
# Default (FM-index)
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh --skip-clara --max-pairs 100000

# Frozen k-mer engine
METHYLGRAPHER_LINEAR_ENGINE=parity \
  fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh \
    --engine parity --skip-clara --max-pairs 100000
```

### FM knobs (do not change parity)

| Variable | Default | Meaning |
|----------|---------|---------|
| `METHYLGRAPHER_FM_SEED_LEN` | 18 | Min SMEM length |
| `METHYLGRAPHER_FM_SEED_STRIDE` | 2 | Right-endpoint spacing for unique SMEM |
| `METHYLGRAPHER_FM_MAX_OCC` | 4096 | Give-up cap while left-extending a repetitive seed |
| `METHYLGRAPHER_FM_ADJ` | 8 | ±bp locus search around unique SMEM |
| `METHYLGRAPHER_FM_MAX_DIFF` | 12 | Gapless/softclip NM budget |
| `METHYLGRAPHER_FM_MAX_SOFT` | 20 | End soft-clip cap |
| `METHYLGRAPHER_FM_RESCUE_WIN` | 512 | Mate-rescue window (bp) when exactly one end mapped |
| `METHYLGRAPHER_BAM_LEVEL` | 1 | BGZF level for native BAM emit |
| `METHYLGRAPHER_BAM_THREADS` | 32 | Parallel BGZF deflate workers |
| `METHYLGRAPHER_FM_SORT_CAP` | 67108864 | Max records for the one-shot GPU sort (`SORT_TILE=0`). ~53M fits; ~800M 30× does not. |
| `METHYLGRAPHER_FM_SORT_TILE` | `auto` (no flag) | From `nvidia-smi` **total** HBM (worker owns the GPU) + FASTQ-size `n`. One-shot when uncompressed BAM + sort keys fit in 90% of HBM after the FM index (Parabricks ~53M on 96 GiB). Otherwise SSD tiles — Clara `--low-memory` analog, no operator switch. Override only for debug (`0` / `oneshot` / N). |
| `METHYLGRAPHER_BAM_ARENA_DIR` | `{work}/bam_arena` | SSD mmap for uncompressed BAM chunks (`ram` to keep in memory) |
| `METHYLGRAPHER_LINEAR_MARKDUP` | 1 | GPU (fm) / samtools (parity) duplicate marking |

### Speed knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `METHYLGRAPHER_SPEED_SEED_STRIDE` | 3 | Seed spacing |
| `METHYLGRAPHER_SPEED_MAX_OCC` | 4096 | Locate occupancy cap |
| `METHYLGRAPHER_SPEED_VOTE_FAST` | 32 | Pass-0 rare-seed vote cap |
| `METHYLGRAPHER_SPEED_VOTE_OCC` | 256 | Pass-1 vote cap (unmapped only) |
| `METHYLGRAPHER_SPEED_MAX_DIFF` | 10 | Gapless/softclip NM budget |
| `METHYLGRAPHER_SPEED_MAX_SOFT` | 16 | End soft-clip cap |

Parity knobs (`VOTE_OCC`, `SEED_STRIDE=3`, `MAX_OCC=16384`, …) apply only to
the frozen engine.

## 100k smoke (Parabricks sample vs Clara slice)

| Engine | Mapped | \|Δ\| vs Clara | Spearman | `map_wall_s` | Gate |
|--------|--------|---------------|----------|--------------|------|
| Clara (slice) | 99.71% | — | — | — | — |
| parity (k-mer) | 97.83% | 0.019 | 1.00 | ~14.1 | **pass** |
| fm (unique SMEM + ±adj + 1bp del) | 97.97% | 0.017 | 1.00 | **~2.1** | **pass** (100k) |
| fm (native BAM + mate-rescue) | **98.24%** | **0.015** | 1.00 | kernel 0.34 / emit **0.75** / map 2.1 | **pass** (100k) |
| fm (GPU sort + markdup) | **98.24%** | **0.015** | 1.00 | kernel 0.35 / sort **0.016** / gather 0.57 / map 2.09; 246 dups; `samtools index` only | **pass** (100k) |
| fm (no Mojo `String` copies + GPU/emit overlap) | **98.24%** | **0.015** | 1.00 | kernel 0.16 / emit 0.43 / fastq 0.34 / **map 0.95** | **pass** (100k) |
| fm (Mojo bulk FASTQ / pigz fd) | **98.23%** | **0.015** | 1.00 | kernel 0.14 / emit 0.24 / fastq **0.12** / **map 0.51** | **pass** (100k) |
| fm (SSD tile merge `SORT_TILE=65536`) | **98.23%** | **0.015** | 1.00 | 4 runs (incl. leftover 3392); 246 dups; **map 1.41**; `samtools index` OK | **pass** (100k) |

## Full sample (~53.3M reads)

| Engine | Mapped | \|Δ\| vs Clara | Spearman | kernel / emit / map_wall | E2E | Gate |
|--------|--------|---------------|----------|--------------------------|-----|------|
| Clara `GPU-PBBWA mem` | 99.68% | — | — | **83.8 s** mem | ~122 s | — |
| parity (earlier full) | 97.79% | 0.019 | 1.00 | map **4351 s** | — | **pass** |
| fm (SAM on disk) | 97.96% | 0.017 | 1.00 | map 553 s | ~1063 s | **pass** |
| fm (BAM FIFO) | 98.11% | 0.016 | 1.00 | 83.4 / 232 / 583 s | 826 s | **pass** |
| fm (native BAM + mate-rescue) | **98.34%** | **0.013** | **1.00** | **88.0 / 201 / 560 s** | **779 s** | **pass** |
| fm (GPU sort + markdup) | **98.34%** | **0.013** | **1.00** | **88.0 / emit 168 / sort 2.1 / map 528 s** | **~705 s** (incl. compare; BAM+index ~9 min) | **pass** |
| fm (pigz FASTQ + parallel BGZF) | **98.34%** | **0.013** | **1.00** | kernel hidden under FASTQ 143 / emit 84 (BGZF 26) / sort 2.1 / **map 237 s** | **~380 s** script (BAM+index ~4 min) | **pass** |
| fm (bytearray FASTQ + Mojo pack + async pigz) | **98.34%** | **0.013** | **1.00** | kernel 9.4 (hidden) / emit 114 (BGZF 28) / fastq 96 / sort 2.2 / **map 222 s** | **~367 s** script | **pass** |
| fm (Mojo bulk FASTQ / pigz fd) | **98.33%** | **0.014** | **1.00** | kernel 1.9 (hidden) / emit **66** (BGZF 28) / fastq **32** / sort 2.2 / **map 103 s** | **~256 s** script | **pass** |
| fm (pack ∥ FASTQ `parallelize`) | **98.33%** | **0.014** | **1.00** | kernel **24** (exposed) / emit **64** (BGZF 28) / fastq 32 (hidden under pack) / **map 94 s** | **~249 s** script | **pass** |
| fm (2-deep GPU + event wait) | **98.33%** | **0.014** | **1.00** | kernel wait **22** / emit **64** (BGZF 28) / fastq 32 (hidden) / sort 2.2 / **map 92 s** | **~251 s** script | **pass** |
| fm (2-deep + copy-lite BGZF) | **98.33%** | **0.014** | **1.00** | kernel wait **22** / emit **60** (BGZF **25**) / fastq 32 (hidden) / sort 2.2 / **map 88 s** | **~242 s** script | **pass** |
| fm (SSD tile merge `SORT_TILE=auto` 4M) | **98.33%** | **0.014** | **1.00** | 13 runs; kernel 8.9 / emit 193 / sort 17 / **map 485 s**; 9.96M dups; index OK | **pass** (30×-safe default) |

GPU kernel is ~605k reads/s (rescue included; Clara mem ~636k). Mojo
`parallelize` runs BAM pack and FASTQ parse on two workers (third arena so
they do not alias; Python `append_raw` stays on the main thread). A 2-deep
GPU pipeline (second buffer set + `DeviceEvent` after D2H) queues batch N+1
before waiting for N. Event wait is event-scoped (not a full device sync),
but per-batch GPU work still exceeds the CPU overlap window, so **~22 s GPU
wait** stays exposed. Copy-lite BGZF (`write_from_addr`, one copy per 65 KiB
block, deeper zlib in-flight) cut gather **28 s → 25 s**. Full-sample **map
94 s → 92 s → 88 s** vs Clara mem **83.8 s**. Leftover vs Clara is that GPU
bubble plus gather/BGZF; a second host `parallelize` in `linear_gpu_fm.mojo`
OOMs the kernel compile (~436 GiB). Default engine is **fm**.

Mate-rescue recovered the good one-end-mapped mates. Remaining Δ vs Clara is mostly 2–4 bp indels and low-quality extras we are not chasing.

Round-trip unit: `fq2bam-meth/tests/fm_roundtrip.mojo` (unique 32-mer BWT+SA).
FASTQ unit: `fq2bam-meth/tests/test_linear_fastq.mojo`.
