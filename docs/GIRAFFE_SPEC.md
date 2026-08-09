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
- Dense segment pack (`sequences.bin` + `offsets.bin` under `{gbz}.mojo_segments/`) from [`scripts/build_mojo_segment_pack.py`](../scripts/build_mojo_segment_pack.py) (or legacy jsonl / `vg convert` for tiny fixtures).
- Production GBZ map path (Buffy-scale FASTQ):
  1. Mojo `giraffe_gbz.map_gbz_native` — device select / `require_device_or_raise`, DeviceContext warmup, ensure pack
  2. Streaming Python [`engine/quartet_map.py`](../engine/quartet_map.py) `map_fastq_to_gaf` — minimizer locate, zip/dist cluster, gapless extend, GAF emit  
  Mojo must **not** `_parse_fastq` whole production FASTQs (OOM / exit 137).
- Minimizer mmap: [`engine/minimizer_index.py`](../engine/minimizer_index.py); zip/dist: [`engine/zipcodes_index.py`](../engine/zipcodes_index.py).
- Selection: `METHYLGRAPHER_MOJO_GIRAFFE_READY` defaults to **on** (`1`); set `0` / `false` / `off` to force vg.
- PE tags: primary pair gets `ri` / `os` / `rc` for MethylCall. Full vg `-M 2` multimapping for PE is not yet mirrored (SE can return up to two scored hits in `quartet_map`).
- Named coordinates: path column uses segment ids from the dense pack (same ids MethylCall resolves via PrepareGenome node maps).

### GFA fixture path

Toy / development: native Mojo `giraffe_mapper` + `giraffe_index` / `extend_exact` (loads FASTQ in Mojo — fixture-scale only).

## Pipeline stages (production GBZ)

1. **Index / pack** — ensure `{gbz}.mojo_segments/` dense pack.
2. **Device gate + GPU warmup** — `giraffe_device` + `giraffe_gpu_kernels` (`nvidia:sm_90` / `amdgpu:gfx942`). Driver &lt;580 needs `MODULAR_NVPTX_COMPILER_PATH=ptxas`. `METHYLGRAPHER_GPU_REQUIRE` (default on for nvidia/amd pins) fails closed if DeviceContext cannot be created.
3. **Streaming locate / cluster / extend / GAF** — `engine.quartet_map` (CuPy/host batch seeds when GPU; never materialize full FASTQ in Mojo).
4. **Pair tags** — PE `ri`/`os`/`rc` on the primary pair.

## MethylCall-consumed GAF fields

- Query name, path (named coordinates), matches / MAPQ, `cs:Z:` (+ PE `ri`/`os`/`rc` when paired)

## Module layout

| Module | Responsibility |
|--------|----------------|
| `giraffe_gbz.mojo` | GBZ entry: device gate, GPU warmup, call streaming `quartet_map` |
| `giraffe_mapper.mojo` | `MojoGiraffe` CLI (`-gbz` or `-gfa`) |
| `giraffe_device.mojo` | `cpu` / `nvidia` / `amd` select + `require_device_or_raise` |
| `giraffe_gpu_kernels.mojo` | Portable seed kernel surface (`nvidia:sm_90` / `amdgpu:gfx942`) |
| `giraffe_index.mojo` / `giraffe_seed.mojo` / `giraffe_extend.mojo` / `giraffe_gaf_emit.mojo` | GFA fixture-scale native path |
| `giraffe_minzip.mojo` / `giraffe_dist.mojo` | Mojo-side locate/cluster helpers (GBZ production uses Python equivalents in `quartet_map`) |
| `engine/quartet_map.py` | Production streaming locate → cluster → gapless → GAF |

## Fixtures

- GFA toy: `tests/data/toy.wl.gfa` + `tests/data/giraffe_fixture/`
- GBZ toy (PrepareGenome-shaped): `tests/data/giraffe_fixture/gbz_toy/toy.wl.C2T.*`

## Vendors

- **NVIDIA GH200**: `nvidia:sm_90`
- **AMD Instinct**: `amdgpu:gfx942` (override with `METHYLGRAPHER_AMDGPU_ARCH`)
