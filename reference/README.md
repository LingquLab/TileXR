# Reference Code

This directory contains helper scripts for reference-only source trees.

Downloaded code under `reference/shmem/` and `reference/ascend-transformer-boost/` is intentionally ignored by git. It is kept only for historical comparison, experiments, and upstream review; current TileXR build targets must not include or link it.

Use:

```bash
bash reference/download_shmem.sh
bash reference/download_ascend_transformer_boost.sh
```
