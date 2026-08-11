# ROCm Mojo Giraffe gates (AMD Instinct)

Companion to MethylPipeline [`docs/plans/mojo-multi-gpu-dual-align.plan.md`](../../MethylPipeline/docs/plans/mojo-multi-gpu-dual-align.plan.md) Phase 1.

## Image

```bash
# From MethylPipeline (stages this repo into workers/docker/methylgrapher)
METHYLGRAPHER_MOJO_GPU_VARIANT=rocm \
METHYLGRAPHER_MOJO_IMAGE_TAG=1.70-mojo-rocm \
  bash scripts/build_methylgrapher_mojo_image.sh

# CUDA twin (Lambda / NGC hosts)
METHYLGRAPHER_MOJO_GPU_VARIANT=cuda \
METHYLGRAPHER_MOJO_IMAGE_TAG=1.70-mojo-cuda \
  bash scripts/build_methylgrapher_mojo_image.sh
```

Alias `:1.70-mojo` may point at the CUDA build for backward compatibility.

Site pin:

```json
"actionConfig": {
  "methylgrapher_wgbs": {
    "engine": "mojo",
    "align_engine": "gpu_giraffe",
    "giraffe_device": "auto",
    "align_device": "auto",
    "image": "epimethyl/methylgrapher:1.70-mojo-rocm"
  }
}
```

## Host prerequisites

- ROCm driver + `rocm-smi`
- Container runtime with AMD GPU device access (ROCm container toolkit / `--device=/dev/kfd --device=/dev/dri`)
- Shared NFS: genomes, `/work/cache/mojo_segments`, sample dirs

## Measurement gates

| Gate | How |
|------|-----|
| Device probe | `docker run --rm … methylGrapher MojoGiraffe … -device amd` logs `kernel_target=amdgpu` (or `amdgpu:gfx942`) |
| Toy GBZ PE | Same golden as NVIDIA (`docs/BENCHMARK_GIRAFFE.md`) |
| Known-mapped Buffy C2T | 13/13 style fixture |
| Full Buffy wall | ≤ target vs GH200 baseline (~2 h goal) |
| Parity | DS20M / subset `graph.methyl` vs NVIDIA Mojo (and optional `cpu_vg`) |

## Rollback

`giraffe_device=cpu` or `gpu_giraffe_fallback=vg` or stock `:1.70` + `engine=python`.
