# Tests

Unit and integration tests for methylGrapher-mojo.

## Structure

```
tests/
  test_utility.mojo       # Phred tables, reverse_complement, config parser
  test_gfa.mojo           # GFA parsing, CpG extraction
  test_alignments.mojo    # SAM/GAF parsing, samtools bridge
  test_mcall.mojo         # Methylation calling logic
  test_longread.mojo      # MM/ML tag parsing
```

## Run

```bash
magic run mojo test/
```

Test data (small synthetic GFA + GAF) will live in `tests/data/`.
