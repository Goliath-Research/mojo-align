# MethylCall benchmark (20k-line DS20M GAF subset)

Index: `hprc-d9-bs.wl.gfa` (~43 GB, 60 118 570 segments). Host: Grace/ARM64 64K pages.

Engine forces `gfa_worker_num=1`. Stock 0.2.0 also uses one GFA worker at `-t` 8/16; the dual-GFA cliff is `-t > 20`.

## Prior (Python engine dispatch)

| engine | -t | wall (s) | peak RSS (GiB) |
|---|---:|---:|---:|
| docker 0.2.0 | 8 | 174.2 | 22.12 |
| engine (single GFA) | 8 | 167.3 | 22.12 |
| docker 0.2.0 | 16 | 175.9 | 22.12 |
| engine (single GFA) | 16 | 165.7 | 22.12 |

Parity: `graph.methyl` / `graph.cpg.tsv` identical vs docker 0.2.0 on this subset (CLI defaults identity=20, mapq=0).

## Native Mojo hot path (2026-08-06)

Native path: Mojo `Dict` GFA + `alignment_to_methylation` + `parallelize()`; GAF filter still via `engine.mcall.iter_alignment_batches`.

| engine | -t | wall (s) | peak RSS (GiB) | notes |
|---|---:|---:|---:|---|
| engine (python multiprocessing) | 8 | 174.4 | 22.12 | `/tmp/mg-native-subset/py_work` |
| native Mojo | 1 | 127.0 | 15.21 | GFA load dominates subset |
| native Mojo | 8 | 134.4 | 15.61 | Parallelize safe; subset too small to beat `-t 1` |

Parity: `graph.methyl` identical vs python engine on this subset (16 993 rows) — `scripts/parity_compare.py --skip-cpg`.

### Full Buffy sample

`/work/samples/HBCST-052125-87293/methylgrapher_work/alignment.gaf` is ~679 GiB. Full-sample wall/RSS should be re-measured by operators after deploying `:1.70-mojo`; expect a larger relative win once GFA load is amortized over billions of GAF lines. Keep `engine: python` / `:1.70` as one-release rollback.
