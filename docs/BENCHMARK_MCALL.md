# MethylCall benchmark (20k-line DS20M GAF subset)

Index: `hprc-d9-bs.wl.gfa` (~43 GB). Host: Grace/ARM64 64K pages.

Engine forces `gfa_worker_num=1`. Stock 0.2.0 also uses one GFA worker at `-t` 8/16; the dual-GFA cliff is `-t > 20`.

| engine | -t | wall (s) | peak RSS (GiB) |
|---|---:|---:|---:|
| docker 0.2.0 | 8 | 174.2 | 22.12 |
| engine (single GFA) | 8 | 167.3 | 22.12 |
| docker 0.2.0 | 16 | 175.9 | 22.12 |
| engine (single GFA) | 16 | 165.7 | 22.12 |

Parity: `graph.methyl` / `graph.cpg.tsv` identical vs docker 0.2.0 on this subset (CLI defaults identity=20, mapq=0).
