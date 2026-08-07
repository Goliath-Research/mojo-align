# Mojo Giraffe stages (short-read PE → GAF)

Science contract for `pangenome_wgbs`: emit **GAF** with **named-coordinates** semantics compatible with Mojo MethylCall (same tags methylGrapher expects from `vg giraffe -o gaf -M 2 --named-coordinates`).

## Index inputs (vg 1.70 / d9-bs)

| File | Role |
|------|------|
| `{prefix}.giraffe.gbz` | Graph + haplotypes (**production**, C2T/G2A) |
| `{prefix}.dist` | Distance index |
| `{prefix}.shortread.withzip.min` | Minimizer index |
| `{prefix}.shortread.zipcodes` | Zipcode clustering annotations |
| `{prefix}.wl.gfa` / fixture GFA | Development / fixture-scale only |

### GBZ production contract

- **No companion GFA required** for Align when the four-file quartet exists under `index_prefix`.
- Mojo opens GBZ via staged helper ([`engine/giraffe_gbz_helper.py`](../engine/giraffe_gbz_helper.py)): stream `vg convert -f gfa` (S-lines only) or durable `{gbz}.mojo_segments/` cache (`scripts/build_mojo_gbz_cache.py`).
- Minimizer/zip/dist files are accepted as inputs; seed locate currently builds postings from decoded segments (native `.min` decode is a follow-on).
- PE / `-M 2` multimapping: emit up to two primary hits with `ri`/`os`/`rc` tags MethylCall understands.
- Named coordinates: path column uses segment ids as emitted by GBZ→segment decode (same ids MethylCall resolves via PrepareGenome node maps).

## Pipeline stages

1. **Index load** — GBZ segment decode (cache/mmap) or GFA fixture load.
2. **Minimizer seed** — k-mer / minimizer locate (GPU-portable via `giraffe_device`).
3. **Hit collect / cluster** — distance/zipcode heuristics (`giraffe_dist` staged).
4. **Extend** — seed-and-extend (`giraffe_extend`).
5. **Pair** — PE tags / fragment filter.
6. **GAF emit** — path, MAPQ, `cs:Z:`.

## MethylCall-consumed GAF fields

- Query name, path (named coordinates), matches / MAPQ, `cs:Z:` (+ PE `ri`/`os`/`rc` when paired)

## Module layout (`src/giraffe_*.mojo`)

| Module | Responsibility |
|--------|----------------|
| `giraffe_gbz.mojo` | GBZ open / cache / helper map entry |
| `giraffe_minzip.mojo` | Minimizer postings from segments |
| `giraffe_dist.mojo` | Seed clustering (`.dist` stand-in) |
| `giraffe_index.mojo` | GFA graph + minimizer index |
| `giraffe_seed.mojo` | K-mer extraction |
| `giraffe_extend.mojo` | Graph walk extend |
| `giraffe_gaf_emit.mojo` | GAF formatting |
| `giraffe_device.mojo` | cpu / nvidia / amd select |
| `giraffe_gpu_kernels.mojo` | Portable seed kernel surface (`nvidia:sm_90`) |
| `giraffe_mapper.mojo` | `MojoGiraffe` CLI (`-gbz` or `-gfa`) |

## Fixtures

- GFA toy: `tests/data/toy.wl.gfa` + `tests/data/giraffe_fixture/`
- GBZ toy (PrepareGenome-shaped): `tests/data/giraffe_fixture/gbz_toy/toy.wl.C2T.*`

## Vendors

- **NVIDIA GH200**: `nvidia:sm_90`
- **AMD**: `amdgpu` when ROCm present
