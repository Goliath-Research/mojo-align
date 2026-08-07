# Toy fixture data

A tiny, hand-designed 3-segment pangenome used for smoke-testing `MethylCall`
and `MergeCpG` end to end (see `scripts/run_toy_mcall.sh` and
`scripts/parity_compare.py`), without requiring `vg`, real FASTQ/BAM data, or
a `PrepareGenome` run.

## Graph (`toy.gfa`)

```
H  VN:Z:1.0
S  1  ACGTACGTAC  SN:Z:GRCh38.chr1  SO:i:1000  SR:i:0
S  2  GTACGTACGT  SN:Z:GRCh38.chr1  SO:i:1010  SR:i:0
S  3  TTTTAAAACC
L  1  +  2  +  0M
L  1  +  3  +  0M
```

- Segments `1` and `2` are linked (`>1>2`) and mapped to `hg38 chr1:1000-1019`
  (used by the toy alignment). Concatenated path sequence is the 20 bp
  `ACGTACGTACGTACGTACGT` (a repeating `ACGT` motif).
- Segment `3` is an unused alternate branch off segment `1`, included only so
  the graph has the requested 3 segments / a simple bubble.
- `toy.wl.gfa` is the "with lambda" / bisulfite-index input `mcall_main`
  expects at `{index_prefix}.wl.gfa`; here it's identical to `toy.gfa` since
  no lambda spike-in or SNV trimming is needed for this toy graph.
- `toy.wl.node.replacement.json` is `{}` — this graph has no SNV bubbles for
  `gfa.get_replacement_SNV()` to collapse.
- `toy.cpg.tsv` was generated from `toy.gfa` via
  `engine.utility.get_all_cpg_from_graph()` and lists the graph's 5 CpG
  positions: 4 within-segment (`C0`-`C3`) and 1 that spans the segment 1/2
  junction (`E0`).

## Alignment (`work_dir/alignment.gaf`)

Two synthetic single-end "reads", already in the post-`tmp_gaf_processing`
GAF shape that `mcall.alignment_parse()` reads directly from
`{work_dir}/alignment.gaf` (12 mandatory GAF columns + `AS`/`bq`/`cs`/`os`/`rc`
tags, matching real `vg giraffe`+methylGrapher output):

- `read1` (`rc:Z:CT`, i.e. from a `C2T`-converted alignment) matches the
  reference exactly, so it calls every `C`-context cytosine on the `+`
  strand as methylated.
- `read2` (`rc:Z:GA`, i.e. from a `G2A`-converted alignment) also matches the
  reference exactly, calling every `G`-context cytosine (`-` strand) as
  methylated.

Together the two reads cover **both sides of all 5 CpG pairs**, so after
`MethylCall` + `MergeCpG` every CpG in `graph.cpg.tsv` reports `met=2, cov=2`.
Because the toy path is only 20 bp long, `run_toy_mcall.sh` passes
`-minimum_identity 10 -minimum_mapq 0` so the short matches clear the
alignment-identity filter (CLI defaults are `minimum_identity=20`,
`minimum_mapq=0`, matching stock methylGrapher 0.2.0).
