# numeric

Portable DeviceContext kernels for MethylPipeline post-align science (centroid
stream first). Device select and HBM preflight come from `gpu-common/`.

| Module | Role |
|--------|------|
| `src/centroid_kernels.mojo` | DeviceContext scatter-add, bin histogram, device probe |
| `python/centroid_kernels.py` | Host reference + Python API used by methylutils |

Include path: `-I gpu-common/src -I numeric/src`.

`gpu_backend=mojo` in MethylPipeline loads `python/centroid_kernels.py` from
`MOJO_ALIGN_ROOT` (or `/opt/mojo-align` after `stage_flat_image_tree.sh`).

Device probe (DeviceContext via gpu-common):

```bash
pixi run numeric-probe
```
