# Toy PrepareGenome-shaped GBZ indexes

Built with `vg autoindex -w giraffe` from `../toy.wl.gfa` (vg 1.70).

| File | Role |
|------|------|
| `toy.giraffe.gbz` / `toy.wl.C2T.giraffe.gbz` | GBZ |
| `toy.dist` / `toy.wl.C2T.dist` | distance index |
| `toy.shortread.withzip.min` | minimizer |
| `toy.shortread.zipcodes` | zipcodes |
| `toy.wl.G2A.*` | copy of C2T for dual-graph wiring tests |

Rebuild:

```bash
docker run --rm -v "$PWD/../..:/data" -w /data goliath/methylgrapher:1.70-mojo \
  bash -lc 'vg autoindex -p giraffe_fixture/gbz_toy/toy -w giraffe -g giraffe_fixture/gbz_toy/toy.wl.gfa -t 4'
```
