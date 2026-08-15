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
Requires `MODULAR_NVPTX_COMPILER_PATH=/usr/bin/ptxas` on driver &lt;580 and
`METHYLGRAPHER_GPU_HBM_FRACTION` (e.g. `0.90`) for the capacity preflight.

Toy GBZ oracle profile (2026-08-08, GH200; cold pack build excluded):

| Stage | Share (toy) | Notes |
|-------|-------------|-------|
| `seed_locate` | ~4% | minimizer + `.min` locate |
| `cluster_extend` | ~75% | zip/dist cluster + gapless |
| `gaf_emit` | ~16% | streamed GAF lines |
| `fastq_batch` | ~5% | streaming PE batches |

### Production Buffy-scale subset (2026-08-15, GH200, 65 536 PE)

HPRC d9-bs C2T GBZ + dense pack; `METHYLGRAPHER_MOJO_READ_BATCH=8192`; banner
`devicecontext-cuda+gpu_ht+gpu_gapless+mojo_stream`.

| Stage (steady batch) | Wall / 8192 pairs | Notes |
|----------------------|-------------------|-------|
| `gpu_seed` + `locate` | ~1–2 ms | Device kernels; buffer reuse |
| GPU gapless (in `sync_prefetch`) | ~1–2 ms true GPU | Prefetch overlaps FASTQ with sync |
| `sync_prefetch` | ~**90–100 ms** | Dominated by MG FASTQ header parse (`os:Z` originals) |
| `host_hits` | ~5 ms | PE tag assembly (shrunk vs prior List copies) |
| `gaf_write_ov` / emit | ~2–5 ms | Already fixed 2026-08-14 |

**Attribution:** remaining wall is **FASTQ / MG-header parse**, not DeviceContext map kernels. Kernel work this pass (reuse, always-on `DIST_CAP` prune, gapless early-exit, sync∥prefetch, chunked `_ascii_span`, scan-based MG header parse) keeps GPU ≪ I/O.

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
| Full Buffy dual-map ≤ ~2 h | wall vs ~6.2 h `vg` baseline | **NOT MET (2026-08-15 kernel pass)** Prior emit fix ~78k pairs/s (~5.4 h dual). This pass: GPU seed/locate/gapless ~2 ms/8192; steady wall still ~**75–90k pairs/s** (MG FASTQ header parse ~0.1 s/batch). Need ~210k pairs/s for serial ≤2 h. Multi-GPU `DUAL_GRAPH_PARALLEL=1` only when `nvidia-smi -L` ≥2 (one process/GPU). Keep `cpu_vg` library default. |
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
| AMD Instinct | `amdgpu:gfx942` (MI300X) / HIP | bakeoff **PENDING** | See `docs/ROCM_GIRAFFE_GATES.md`; set `METHYLGRAPHER_AMDGPU_ARCH` if needed. Same Mojo DeviceContext sources as NVIDIA — no second kernel dialect. Buffy wall twin of GH200 when MI300X available. |

## Comparison arms (vg retained)

`align_engine=cpu_vg` (`vg giraffe`) remains a first-class **before** arm (`align.pangenome_wgbs.vg`). MojoGiraffe is preferred science for production WGBS procedures; do **not** retire `cpu_vg` or flip fleet defaults solely from this bakeoff. MethylPipeline matrix: `docs/architecture/sample-prep-tooling.md` / plan `comparison-arms-bakeoff`.

| Gate (2026-08-15) | Criterion | Status |
|-------------------|-----------|--------|
| vg vs Mojo dual-map wall | Report hours on agreed subset; ≤2h aspirational | **NOT MET** — ~75–90k pairs/s after kernel pass; see stage table above |
| Named-coordinate GAF science | MethylCall path for both arms | Available via engine overlay |
| Multi-GPU dual-parallel | `METHYLGRAPHER_DUAL_GRAPH_PARALLEL=1` or `auto` with ≥2 GPUs; one process per GPU | Documented; this host has 1× GH200 → serial |
| Production site / library default flip | — | **Not done** (by design; keep `cpu_vg`) |

## Rollback

```bash
export METHYLGRAPHER_ALIGN_ENGINE=cpu_vg
# or
export METHYLGRAPHER_GPU_GIRAFFE_FALLBACK=vg
# or temporarily:
export METHYLGRAPHER_MOJO_GIRAFFE_READY=0
```
