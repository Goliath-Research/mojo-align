# Giraffe mapper benchmarks (Mojo GBZ vs vg)

Science contract: dual-graph GAF + named-coordinates → Mojo MethylCall.
See [`GIRAFFE_SPEC.md`](GIRAFFE_SPEC.md).

Production hot path is native Mojo [`giraffe_stream_map.mojo`](../src/giraffe_stream_map.mojo).
Python [`engine/quartet_map.py`](../engine/quartet_map.py) is the **oracle** (parity / pack ensure), not Align.

## Stage profile

**Oracle** (Python `quartet_map` — for A/B vs Mojo):

```bash
python3 scripts/profile_giraffe_stages.py --device cpu \
  --out /tmp/giraffe_stage_profile.json
```

**Production Mojo stream map** — set `METHYLGRAPHER_PROFILE_STAGES=1` and run
`MojoGiraffe -gbz …`; stage lines print as `mojo_stream stages_s …`.

Toy GBZ oracle profile (2026-08-08, GH200; cold pack build excluded):

| Stage | Share (toy) | Notes |
|-------|-------------|-------|
| `seed_locate` | ~4% | minimizer + `.min` locate |
| `cluster_extend` | ~75% | zip/dist cluster + gapless |
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
| MojoGiraffe `-gbz` (`giraffe_stream_map`) | toy.giraffe.gbz | cpu / nvidia:sm_90 | ~10 s (incl. pack) | **PASS** vs golden |
| vg giraffe 1.70 | same toy GBZ | Grace | toy often `*` paths | use golden for MethylCall tags |

## DS20M / Buffy progressive gates

| Milestone | Gate | Status |
|-----------|------|--------|
| Toy GBZ PE vs golden | path / cs / ri / os / rc | **PASS** (`giraffe_stream_map`) |
| DS-scale (500 PE) on toy GBZ | GAF lines land | **PASS** (protocol smoke) |
| Buffy-subset seed+extend | oracle `quartet_map` 13/13; production = stream_map | **PASS** stream_map 13/13 (pack-walk fixture; emit-time named-coords) |
| DS20M `graph.methyl` vs `cpu_vg` | `parity_compare.py` | **PASS** 20k-line subset: python vs mojo MethylCall identical (`graph.methyl` 18996, `graph.cpg.tsv` 18837) |
| Full Buffy dual-map ≤ ~2 h | wall vs ~6.2 h `vg` baseline | **NOT MET (2026-08-14)** Mojo C2T started (~10k pairs/s stage estimate ⇒ ≫2 h dual); crashed `R2 ended early` at ~87 s — GAF emit still dominant; keep `cpu_vg` default |
| Production `gpu_giraffe` → Mojo GBZ | READY default-on + dense pack + quartet | **WIRED** (opt out with `READY=0`) |

Build production dense segment packs (preferred — from companion GFA):

```bash
python scripts/build_mojo_segment_pack.py \
  --gfa /var/tmp/methylgrapher-index/hprc-d9-bs.wl.gfa \
  --gbz /work/genomes/pangenome/GRCh38/d9-bs/1.70/hprc-d9-bs.wl.C2T.giraffe.gbz \
  --out /work/cache/mojo_segments/hprc-d9-bs.wl.C2T.giraffe.gbz.mojo_segments \
  --also-link-g2a
# READY defaults on; opt out until gates pass if desired:
# export METHYLGRAPHER_MOJO_GIRAFFE_READY=0
```

## NVIDIA vs AMD bakeoff

| Vendor | Device API | Toy GBZ | Notes |
|--------|------------|---------|-------|
| NVIDIA GH200 | `nvidia:sm_90` DeviceContext | PASS | CuPy / host-nvidia fallback **refused** on stream map |
| AMD Instinct | `amdgpu:gfx942` (MI300X) / HIP | bakeoff | See `docs/ROCM_GIRAFFE_GATES.md`; set `METHYLGRAPHER_AMDGPU_ARCH` if needed |

## Rollback

```bash
export METHYLGRAPHER_ALIGN_ENGINE=cpu_vg
# or
export METHYLGRAPHER_GPU_GIRAFFE_FALLBACK=vg
# or temporarily:
export METHYLGRAPHER_MOJO_GIRAFFE_READY=0
```
