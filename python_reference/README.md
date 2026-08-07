# Python Reference Sources

This directory contains the **original Python sources** from
[twlab/methylGrapher](https://github.com/twlab/methylGrapher) (0.2.0) for
reference during the Mojo migration.

Do **not** edit these files — they are read-only reference material.

Active code lives in:

- `../engine/` — faithful runnable port of these sources (+ cutover patches)
- `../src/*.mojo` — Mojo CLI and native ports of hot paths / Giraffe

| Reference file | Active counterpart |
|---|---|
| `main.py` | `engine/cli.py` + `src/main.mojo` |
| `utility.py` | `engine/utility.py` + `src/utility.mojo` / `merge_cpg.mojo` / `conversion_rate.mojo` |
| `alignments.py` | `engine/alignments.py` + `src/align.mojo` / `engine/align_backends.py` |
| `gfa.py` | `engine/gfa.py` + `src/gfa.mojo` |
| `mcall.py` | `engine/mcall.py` + `src/mcall.mojo` / `mcall_core.mojo` |
| `longread.py` | `src/legacy_scaffold/longread.mojo` (not on the active cutover path) |
| `mgmp.py` | `engine/mgmp.py` (experimental; unused by the CLI) |
