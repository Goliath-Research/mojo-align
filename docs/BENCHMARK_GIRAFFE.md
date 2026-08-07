# Giraffe mapper benchmarks (Mojo GBZ vs vg)

Science contract: dual-graph GAF + named-coordinates → Mojo MethylCall.
See [`GIRAFFE_SPEC.md`](GIRAFFE_SPEC.md).

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
| Buffy-subset / DS20M `graph.methyl` vs `cpu_vg` | `parity_compare.py` | **PENDING** operator (build C2T/G2A caches first) |
| Full Buffy dual-map ≤ ~2 h | wall on NVMe | **PENDING** measurement |
| Production `gpu_giraffe` → Mojo GBZ (no GFA size-cap) | quartet present | **SHIPPED** in `align_backends` |

Build production segment caches (one-time per strand GBZ):

```bash
python scripts/build_mojo_gbz_cache.py \
  --gbz /work/genomes/pangenome/GRCh38/d9-bs/1.70/hprc-d9-bs.wl.C2T.giraffe.gbz
python scripts/build_mojo_gbz_cache.py \
  --gbz /work/genomes/pangenome/GRCh38/d9-bs/1.70/hprc-d9-bs.wl.G2A.giraffe.gbz
```

## NVIDIA vs AMD bakeoff

| Vendor | Device API | Toy GBZ | Notes |
|--------|------------|---------|-------|
| NVIDIA GH200 | `nvidia:sm_90` seed helper | PASS | CuPy optional |
| AMD | `amdgpu` / host fallback | ready | fill on ROCm node |

## Rollback

```bash
export METHYLGRAPHER_ALIGN_ENGINE=cpu_vg
# or
export METHYLGRAPHER_GPU_GIRAFFE_FALLBACK=vg
```
