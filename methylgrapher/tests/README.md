# Tests

Unit and integration tests for methylGrapher-mojo.

## Structure

```
tests/
  test_mcall_core.mojo          # Native parsers, GFA Dict, methylation, gzip
  test_align_backends.py        # cpu_vg / gpu_giraffe / mojo_giraffe selection
  test_align_backends_ready.py  # MOJO_ALIGN_GIRAFFE_READY gating
  test_giraffe_gbz_helper.py    # GBZ quartet resolve / segment cache
  test_giraffe_gaf_parity.py    # Mojo GAF vs golden fixture
  test_fq2bam_meth.py           # Linear WGBS convert / QC / end-to-end map
  test_quartet_stream.py        # streaming batch / MOJO_READ_BATCH
  test_mojo_stream_map.py       # production Mojo giraffe_stream_map + oracle golden
  test_gpu_minimizer_batch.py   # GPU/host minimizer batch helper (oracle / legacy)
  test_minimizer_index.py       # .min mmap helper
  test_segment_pack.py          # dense sequences.bin / offsets.bin
  test_quartet_map.py           # quartet_map oracle / MojoGiraffe READY gate
  probe_devicecontext_cuda.mojo # std.gpu DeviceContext / kernel probe
  probe_gpu.mojo                # legacy probe (may lag std.gpu import style)
  data/                         # Toy fixtures (see data/README.md, fq2bam_fixture/)
```

## Run

```bash
# Mojo unit tests (need -I src for imports)
pixi run mojo -I src tests/test_mcall_core.mojo

# Python tests
pixi run python -m pytest tests/ -q

# End-to-end MethylCall / MergeCpG on the toy fixture
scripts/run_toy_mcall.sh python
scripts/run_toy_mcall.sh mojo

# Linear WGBS Align (MojoFq2bamMeth)
scripts/run_toy_fq2bam_meth.sh python cpu
pixi run python -m pytest tests/test_fq2bam_meth.py -q
```

Giraffe GAF parity against the golden PE fixture:

```bash
pixi run mojo -I src src/main.mojo MojoGiraffe \
  -gfa tests/data/toy.wl.gfa \
  -fq1 tests/data/giraffe_fixture/R1.fastq \
  -fq2 tests/data/giraffe_fixture/R2.fastq \
  -out_gaf /tmp/mojo_pe.gaf -device cpu -k 5
python3 scripts/giraffe_gaf_parity.py --mojo /tmp/mojo_pe.gaf \
  --golden tests/data/giraffe_fixture/golden.gaf --require-extra-tags
```
