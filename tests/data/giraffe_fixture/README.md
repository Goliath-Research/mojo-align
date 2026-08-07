# Giraffe fixtures

Toy GFA: `../toy.wl.gfa`.
FASTQs are exact segment matches. `golden.gaf` is the MethylCall-shaped PE contract
(path / `cs:Z` / `ri` / `os` / `rc`) for Mojo Giraffe CPU/GPU parity.

```bash
pixi run mojo -I src src/main.mojo MojoGiraffe \
  -gfa tests/data/toy.wl.gfa \
  -fq1 tests/data/giraffe_fixture/R1.fastq \
  -fq2 tests/data/giraffe_fixture/R2.fastq \
  -out_gaf /tmp/mojo_pe.gaf -device cpu -k 5
python3 scripts/giraffe_gaf_parity.py --mojo /tmp/mojo_pe.gaf \
  --golden tests/data/giraffe_fixture/golden.gaf --require-extra-tags
```
