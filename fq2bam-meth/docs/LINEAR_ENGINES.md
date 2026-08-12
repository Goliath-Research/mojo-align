# Linear GPU engines: parity vs speed

Two Mojo GPU mappers share the orchestrator, dense-v1 pack, and SAM/BAM
consumer path. They are **not** interchangeable until the speed engine
matches or improves quality.

| Engine | Env | Module | Role |
|--------|-----|--------|------|
| **parity** (default) | `METHYLGRAPHER_LINEAR_ENGINE=parity` | `src/linear_gpu_locate.mojo` | Frozen science path vs Clara |
| **speed** (opt-in) | `METHYLGRAPHER_LINEAR_ENGINE=speed` | `src/linear_gpu_speed.mojo` | Wall-clock experiments; any algorithm |

## Why two engines

Parity exists to stay a valid Clara `fq2bam_meth` replacement: mapped-rate
and idxstats gates, GATK-consumable BAM. Tuning that kernel for speed has
repeatedly broken the 0.02 mapped-rate gate.

Speed may use any strategy (unique-seed consensus + rare/heavy vote today;
FM-index / BWA-MEM shaped later). It does **not** become the default until
promotion gates pass.

## Science gates (both engines, vs Clara)

Same report as [`LINEAR_PARITY.md`](LINEAR_PARITY.md):

| Gate | Default |
|------|---------|
| Mapped rate \|Δ\| | ≤ 0.02 |
| Primary idxstats Spearman | ≥ 0.95 |

## Promotion (speed → default)

Speed may replace parity as the default **only if all** hold on the
Parabricks sample (50k smoke **and** full sample):

1. Clara gates above **pass**
2. Mojo mapped rate **≥** frozen parity mapped rate (match or improve)
3. Idxstats Spearman vs Clara **≥** parity’s Spearman (or still ≥ 0.95)
4. BAM still GATK/Picard consumable (`@RG`, original SEQ, coordinate sort)

Until then: production / bakeoff / `parity_linear_parabricks_vs_mojo.sh`
stay on **parity**.

## Run

```bash
# Science default (unchanged)
fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh --skip-clara --max-pairs 50000

# Speed engine (opt-in; expect quality to lag until it catches up)
METHYLGRAPHER_LINEAR_ENGINE=speed \
  fq2bam-meth/scripts/parity_linear_parabricks_vs_mojo.sh \
    --engine speed --sample-dir /tmp/parity_50k_speed --skip-clara --max-pairs 50000
```

Speed-only knobs (do not change parity):

| Variable | Default | Meaning |
|----------|---------|---------|
| `METHYLGRAPHER_SPEED_SEED_STRIDE` | 3 | Seed spacing |
| `METHYLGRAPHER_SPEED_MAX_OCC` | 4096 | Locate occupancy cap |
| `METHYLGRAPHER_SPEED_VOTE_FAST` | 32 | Pass-0 rare-seed vote cap |
| `METHYLGRAPHER_SPEED_VOTE_OCC` | 256 | Pass-1 vote cap (unmapped only) |
| `METHYLGRAPHER_SPEED_MAX_DIFF` | 8 | Gapless/softclip NM budget |
| `METHYLGRAPHER_SPEED_MAX_SOFT` | 12 | End soft-clip cap |

Parity knobs (`VOTE_OCC`, `SEED_STRIDE=3`, `MAX_OCC=16384`, …) apply only to
the frozen engine.
