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
| `METHYLGRAPHER_FM_SEED_LEN` | 19 | Exact seed length (BWA-MEM min) |
| `METHYLGRAPHER_FM_SEED_STRIDE` | 5 | Seed start spacing |
| `METHYLGRAPHER_FM_MAX_OCC` | 256 | Max SA interval size to extend |
| `METHYLGRAPHER_FM_MAX_DIFF` | 10 | Gapless/softclip NM budget |
| `METHYLGRAPHER_FM_MAX_SOFT` | 16 | End soft-clip cap |

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

## 100k smoke (2026-08-12, Parabricks sample vs Clara slice)

| Engine | Mapped | \|Δ\| vs Clara | Spearman | `map_wall_s` | Gate |
|--------|--------|---------------|----------|--------------|------|
| Clara (slice) | 99.71% | — | — | — | — |
| parity (k-mer) | 97.83% | 0.019 | 1.00 | ~14.1 | **pass** |
| fm (v1 exact-seed + gapless) | ~52–53% | ~0.47 | ≥0.99 | **~1.8–2.6** | fail |

FM v1 is wired and fast (no 50 GiB posting walks) but **not** promotion-ready:
needs SMEM/reseed/chain + banded affine extend (and/or host indel path) before
default switch. Artifacts: `/tmp/parity_100k_fm/linear_parity_report.json`.

Round-trip unit: `fq2bam-meth/tests/fm_roundtrip.mojo` (unique 32-mer BWT+SA).
