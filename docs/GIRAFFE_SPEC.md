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
  1. Mojo `giraffe_gbz.map_gbz_native` — device select / `require_device_or_raise`, DeviceContext warmup, ensure dense pack
  2. Mojo [`giraffe_stream_map.mojo`](../src/giraffe_stream_map.mojo) — batched FASTQ stream → Mojo Giraffe `(k,w)` minimizers ([`giraffe_minimizer.mojo`](../src/giraffe_minimizer.mojo): DeviceContext pack+fwd/RC hash on NVIDIA/AMD, Mojo host window reduce; **never CuPy**) → HT locate via thin `MinimizerIndex.locate_key_batches` → Mojo zip/dist cluster → `gapless_extend_with_pack` → streaming GAF emit  
  Mojo must **not** load whole production FASTQs into memory (OOM / exit 137).
- Python [`engine/quartet_map.py`](../engine/quartet_map.py) is the **oracle** only (`map_fastq_to_gaf` for parity tests; `ensure_pack_for_gbz` for pack resolve).
- Native Mojo gapless over dense pack: [`src/giraffe_gapless.mojo`](../src/giraffe_gapless.mojo) / [`src/giraffe_pack.mojo`](../src/giraffe_pack.mojo).
- Dual-graph Align overlaps C2T∥G2A (`METHYLGRAPHER_DUAL_GRAPH_PARALLEL=1` default); MojoGiraffe streams GAF via `/dev/fd/3` (no temp+cat).
- Selection: `METHYLGRAPHER_MOJO_GIRAFFE_READY` defaults to **on** (`1`); set `0` / `false` / `off` to force vg.
- PE tags: primary pair gets `ri` / `os` / `rc` for MethylCall; up to two scored hits per mate (`-M 2` style) when gapless returns them.
- PE emit order (MethylCall): **primary R1 → optional secondary R1 → primary R2 → optional secondary R2**. The first GAF row for a query must carry `ri`/`os`/`rc` (do not emit untagged secondaries before tagged primaries).
- Named coordinates: path column uses segment ids from the dense pack (same ids MethylCall resolves via PrepareGenome node maps).

### GFA fixture path

Toy / development: native Mojo `giraffe_mapper` + `giraffe_index` / `extend_exact` (loads FASTQ in Mojo — fixture-scale only).

## Pipeline stages (production GBZ)

1. **Index / pack** — ensure `{gbz}.mojo_segments/` dense pack.
2. **Device gate + GPU warmup** — `giraffe_device` + `giraffe_gpu_kernels` (`nvidia:sm_90` / `amdgpu:gfx942`). Driver &lt;580 needs `MODULAR_NVPTX_COMPILER_PATH=ptxas`. `METHYLGRAPHER_GPU_REQUIRE` (default on for nvidia/amd pins) fails closed if DeviceContext cannot be created.
3. **Streaming locate / cluster / extend / GAF** — `giraffe_stream_map` (`METHYLGRAPHER_PROFILE_STAGES` stage prints).
4. **Pair tags** — PE `ri`/`os`/`rc` on the primary pair.

## MethylCall-consumed GAF fields

- Query name, path (named coordinates), matches / MAPQ, `cs:Z:` (+ PE `ri`/`os`/`rc` when paired)

## Module layout

| Module | Responsibility |
|--------|----------------|
| `giraffe_gbz.mojo` | GBZ entry: device gate, GPU warmup, ensure pack → `giraffe_stream_map` |
| `giraffe_stream_map.mojo` | **Production** batched FASTQ → seed/locate/cluster/extend/GAF |
| `giraffe_mapper.mojo` | `MojoGiraffe` CLI (`-gbz` or `-gfa`) |
| `giraffe_device.mojo` | `cpu` / `nvidia` / `amd` select + `require_device_or_raise` |
| `giraffe_gpu_kernels.mojo` | Portable seed kernels; GPU codes → k-mers for extend |
| `giraffe_index.mojo` / `giraffe_seed.mojo` / `giraffe_extend.mojo` / `giraffe_gaf_emit.mojo` | GFA fixture-scale native path |
| `giraffe_gapless.mojo` / `giraffe_pack.mojo` | Native Mojo gapless (+ multi-node) over dense pack |
| `giraffe_minzip.mojo` / `giraffe_dist.mojo` | Mojo-side batch locate/cluster helpers |
| `engine/quartet_map.py` | Oracle + `ensure_pack_for_gbz` / `mojo_giraffe_ready` |
| `engine/stage_timer.py` | Stage wall breakdown for oracle bakeoffs |

## Fixtures

- GFA toy: `tests/data/toy.wl.gfa` + `tests/data/giraffe_fixture/`
- GBZ toy (PrepareGenome-shaped): `tests/data/giraffe_fixture/gbz_toy/toy.wl.C2T.*`

## Vendors

- **NVIDIA GH200**: `nvidia:sm_90`
- **AMD Instinct**: `amdgpu:gfx942` (override with `METHYLGRAPHER_AMDGPU_ARCH`)
