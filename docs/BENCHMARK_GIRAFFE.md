# Giraffe mapper benchmarks (Mojo GBZ vs vg)

Science contract: dual-graph GAF + named-coordinates → Mojo MethylCall.
See [`GIRAFFE_SPEC.md`](GIRAFFE_SPEC.md).

## Stage profile (A0)

```bash
python3 scripts/profile_giraffe_stages.py --device cpu \
  --out /tmp/giraffe_stage_profile.json
# Buffy / subset: set FQ1/FQ2/GBZ/MIN and METHYLGRAPHER_PROFILE_JSON
```

Toy GBZ (2026-08-08, GH200 host; cold pack build excluded from STAGE_TIMER):

| Stage | Share (toy) | Notes |
|-------|-------------|-------|
| `seed_locate` | ~4% | in-process GPU/host minimizer + `.min` locate (no JSON IPC) |
| `cluster_extend` | ~75% | zip/dist cluster + gapless (+ multi-node heuristic) |
| `gaf_emit` | ~16% | streamed GAF lines |
| `fastq_batch` | ~5% | streaming PE batches |

`vg giraffe` full-Buffy dual-map baseline remains **~6.2 h** (operator). Mojo gate: dual-map ≤ **~2 h** + DS20M MethylCall parity.

## Toy GBZ fixture

```bash
export VG_PATH=vg   # or scripts/vg_docker_wrap.sh
pixi run mojo -I src src/main.mojo MojoGiraffe \
  -gbz tests/data/giraffe_fixture/gbz_toy/toy.giraffe.gbz \
  -fq1 tests/data/giraffe_fixture/R1.fastq \
  -fq2 tests/data/giraffe_fixture/R2.fastq \
  -out_gaf /tmp/mojo_gbz.gaf -device nvidia -k 5
python3 scripts/giraffe_gaf_parity.py --mojo /tmp/mojo_gbz.gaf \
  --golden tests/data/giraffe_fixture/golden.gaf --require-extra-tags
```

| Backend | Index | Device | Fixture wall | GAF parity |
|---------|-------|--------|--------------|------------|
| MojoGiraffe `-gbz` | toy.giraffe.gbz | cpu / nvidia:sm_90 | ~10 s (incl. vg convert) | **PASS** vs golden |
| vg giraffe 1.70 | same toy GBZ | Grace | toy often `*` paths | use golden for MethylCall tags |
## DS20M / Buffy progressive gates

| Milestone | Gate | Status |
|-----------|------|--------|
| Toy GBZ PE vs golden | path / cs / ri / os / rc | **PASS** |
| DS-scale (500 PE) on toy GBZ | GAF lines land | **PASS** (protocol smoke) |
| Buffy-subset seed+extend (known `vg`-mapped C2T reads) | quartet_map | **PASS** 13/13 (~0.1 s) |
| DS20M `graph.methyl` vs `cpu_vg` | `parity_compare.py` | **PENDING** operator |
| Full Buffy dual-map ≤ ~2 h | wall vs ~6.2 h `vg` baseline | **PENDING** operator (C2T∥G2A parallel + stream GAF wired) |
| Production `gpu_giraffe` → Mojo GBZ | `READY=1` + dense pack + quartet | **WIRED** (default on; opt out with `READY=0`) |

Build production dense segment packs (preferred — from companion GFA):

```bash
python scripts/build_mojo_segment_pack.py \
  --gfa /var/tmp/methylgrapher-index/hprc-d9-bs.wl.gfa \
  --gbz /work/genomes/pangenome/GRCh38/d9-bs/1.70/hprc-d9-bs.wl.C2T.giraffe.gbz \
  --out /work/cache/mojo_segments/hprc-d9-bs.wl.C2T.giraffe.gbz.mojo_segments \
  --also-link-g2a
export METHYLGRAPHER_MOJO_GIRAFFE_READY=1   # only after Buffy ≤2h + MethylCall parity
```

## NVIDIA vs AMD bakeoff

| Vendor | Device API | Toy GBZ | Notes |
|--------|------------|---------|-------|
| NVIDIA GH200 | `nvidia:sm_90` seed helper | PASS | CuPy optional |
| AMD Instinct | `amdgpu:gfx942` (MI300X) / HIP host kernels | bakeoff | See `docs/ROCM_GIRAFFE_GATES.md`; set `METHYLGRAPHER_AMDGPU_ARCH` if needed |

## Rollback

```bash
export METHYLGRAPHER_ALIGN_ENGINE=cpu_vg
# or
export METHYLGRAPHER_GPU_GIRAFFE_FALLBACK=vg
```
