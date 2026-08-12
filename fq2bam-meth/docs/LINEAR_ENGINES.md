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
| fm (native BAM + mate-rescue) | **98.24%** | **0.015** | 1.00 | kernel 0.34 / emit 1.46 / map 2.8 | **pass** (100k) |

## Full sample (~53.3M reads, BAM FIFO, no on-disk SAM)

| Engine | Mapped | \|Δ\| vs Clara | Spearman | kernel / emit / map_wall | E2E | Gate |
|--------|--------|---------------|----------|--------------------------|-----|------|
| Clara `GPU-PBBWA mem` | 99.68% | — | — | **83.8 s** mem | — | — |
| parity (earlier full) | 97.79% | 0.019 | 1.00 | map **4351 s** | — | **pass** |
| fm (SAM on disk) | 97.96% | 0.017 | 1.00 | map 553 s | ~1063 s | **pass** |
| fm (BAM FIFO) | **98.11%** | **0.016** | **1.00** | **83.4 / 232 / 583 s** | **826 s** | **pass** |

GPU kernel on the full sample is **639k reads/s**, matching Clara mem (~636k). `map_wall` is still emit-bound (Mojo SAM strings → FIFO → `samtools view -u`). Next speed lever is binary BAM emit.

Same BWT, not the same aligner: Clara is full BWA-MEM (bidirectional SMEM, reseed, chain, banded affine SW, **mate rescue**). We map each read independently (unique SMEM + gapless/softclip/±8 + 1bp del). On the 100k slice, **87%** of Clara-mapped / FM-unmapped mates already had the other mate placed by FM, and **92%** of those sit within 500 bp of that mate — pairing + SW, not a better index.

FM stays opt-in until promotion is explicitly flipped (gates pass; mapped ≥ parity).

Round-trip unit: `fq2bam-meth/tests/fm_roundtrip.mojo` (unique 32-mer BWT+SA).
