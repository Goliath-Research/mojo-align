# methylGrapher-mojo

Mojo (Modular) port of [methylGrapher](https://github.com/twlab/methylGrapher) — pangenome graph methylation calling.

Original tool by Wenjin Zhang / Ting Wang Lab (WUSTL).  
Mojo migration by David Izada Rodriguez / Goliath Research.

---

## Status

| Module | Status |
|---|---|
| `utility.mojo` | 🟡 Scaffold |
| `alignments.mojo` | 🟡 Scaffold |
| `gfa.mojo` | 🟡 Scaffold |
| `mcall.mojo` | 🟡 Scaffold |
| `longread.mojo` | 🟡 Scaffold |
| `mgmp.mojo` | 🟡 Scaffold |
| `main.mojo` | 🟡 Scaffold |

🟡 Scaffold → 🔵 In Progress → 🟢 Complete

---

## Requirements

- [Modular / Magic](https://docs.modular.com/magic/) — Mojo package manager
- `samtools` (for BAM/GAF I/O, already on target environment)
- `vg` (graph genome toolkit)

## Install

```bash
magic install
```

## Run

```bash
magic run mojo src/main.mojo help
```

## Design Decisions vs. Original Python

| Python | Mojo replacement | Reason |
|---|---|---|
| `pysam` | `samtools view` subprocess | samtools already installed; avoids Python C-ext dependency |
| `multiprocessing.Pool` | `parallelize()` from `algorithm` | No GIL, shared memory, zero pickle overhead |
| `dict` | `Dict[K, V]` from `collections` | Native Mojo, same semantics |
| `re` regex | Manual state machines | Full control, SIMD-friendly for hot paths |
| `argparse` | Manual `sys.argv` parse (see `main.mojo`) | No stdlib argparse yet in Mojo |

## Architecture

```
src/
  main.mojo        # CLI dispatcher (mirrors main.py)
  utility.mojo     # I/O helpers, Phred tables, GFA converter, config parser
  alignments.mojo  # GAF alignment structs + samtools-based BAM reader
  gfa.mojo         # GFA graph parser, CpG extraction
  mcall.mojo       # Methylation calling engine (hot path)
  longread.mojo    # ONT / PacBio MM-tag base modification path
  mgmp.mojo        # Parallel worker orchestration
python_reference/  # Read-only original Python sources
```

## License

MIT — same as upstream methylGrapher.
