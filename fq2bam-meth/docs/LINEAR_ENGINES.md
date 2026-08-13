# Linear GPU engines: parity vs speed vs fm

Three Mojo GPU mappers share the orchestrator and SAM/BAM consumer path.
They are **not** interchangeable until promotion gates pass.

| Engine | Env | Module | Index | Role |
|--------|-----|--------|-------|------|
| **parity** (default) | `METHYLGRAPHER_LINEAR_ENGINE=parity` | `src/linear_gpu_locate.mojo` | dense-v1 k-mer pack | Frozen science path vs Clara |
| **speed** (opt-in) | `METHYLGRAPHER_LINEAR_ENGINE=speed` | `src/linear_gpu_speed.mojo` | dense-v1 k-mer pack | K-mer consensus experiments |
| **fm** (opt-in) | `METHYLGRAPHER_LINEAR_ENGINE=fm` | `src/linear_gpu_fm.mojo` | BWA `.bwt/.sa/.pac` | BWA-MEM-style FM-index path |

## Why separate engines

Parity exists to stay a valid Clara `fq2bam_meth` replacement on the k-mer
path. Speed may tune that index. **fm** is the real Clara-shaped replacement:
GPU FM-index seed + extend on the same `bwameth.c2t` BWA index Clara uses.

None becomes the default until promotion gates pass.

## Science gates (vs Clara)

Same report as [`LINEAR_PARITY.md`](LINEAR_PARITY.md):

| Gate | Default |
|------|---------|
| Mapped rate \|Δ\| | ≤ 0.02 |
| Primary idxstats Spearman | ≥ 0.95 |

## Promotion (fm or speed → default)

An opt-in engine may replace parity as the default **only if all** hold on
the Parabricks sample (100k smoke **and** full sample):

1. Clara gates above **pass**
2. Mojo mapped rate **≥** frozen parity mapped rate (match or improve)
3. Idxstats Spearman vs Clara **≥** 0.95
4. BAM still GATK/Picard consumable (`@RG`, original SEQ, coordinate sort)

Until then: production / bakeoff stay on **parity**.

## Run

```bash
# Science default (k-mer)
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh --skip-clara --max-pairs 100000

# FM-index engine (opt-in; uses ${REF}.bwameth.c2t.{bwt,sa,pac})
METHYLGRAPHER_LINEAR_ENGINE=fm \
  fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh \
    --engine fm --sample-dir /tmp/parity_100k_fm --skip-clara --max-pairs 100000
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
| `METHYLGRAPHER_FM_SORT_CAP` | 67108864 | Max records for GPU sort/markdup host tables |
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

GPU kernel is ~605k reads/s (rescue included; Clara mem ~636k). Mojo
`parallelize` runs BAM pack and FASTQ parse on two workers (third arena so
they do not alias; Python `append_raw` stays on the main thread). Full-sample
**map 103 s → 94 s**. FASTQ (~32 s) is hidden under pack; that made per-batch
CPU shorter than the GPU kernel, so **~24 s of GPU wait is now visible**.
Remaining wall is that GPU bubble plus final gather/BGZF (~28 s). Next cut is
a 2-deep GPU pipeline (keep the next kernel in flight during emit) toward
Clara's ~84 s mem / ~122 s E2E. Default stays **parity**.

Mate-rescue recovered the good one-end-mapped mates. Remaining Δ vs Clara is mostly 2–4 bp indels and low-quality extras we are not chasing.

Round-trip unit: `fq2bam-meth/tests/fm_roundtrip.mojo` (unique 32-mer BWT+SA).
FASTQ unit: `fq2bam-meth/tests/test_linear_fastq.mojo`.
