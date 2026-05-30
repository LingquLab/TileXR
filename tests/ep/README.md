# TileXR EP Dispatch Tests

This tree tests the standalone TileXR EP module under `src/ep`. It is independent from `src/mc2`; the MVP route does not use HCCL window helpers, shmem, or UDMA.

## Source-Only Tests

From `tests/ep`:

```bash
source ../../scripts/common_env.sh
bash build.sh source-only
./install/bin/test_tilexr_ep_layout
./install/bin/test_tilexr_ep_api_sources
./install/bin/test_tilexr_ep_host_validation
./install/bin/test_tilexr_ep_kernel_sources
```

`source-only` mode builds and installs the source-layout, API source, host validation, and kernel source tests without building the hardware demo.

## Full Hardware Demo

From `tests/ep`:

```bash
source ../../scripts/common_env.sh
bash build.sh full
bash demo/run_tilexr_ep_dispatch_demo.sh 2
```

`full` mode builds and installs `tile-comm`, `tilexr-ep`, and `libtilexr_ep_dispatch_kernel.so` under the repository `install` directory, then builds the EP demo.

## Remote Blue Verification

```bash
bash demo/deploy_and_run_blue.sh
```

The remote verification script syncs the complete repository into `/home/d00520898/tilexr_ep_dispatch_verify/TileXR` on blue, initializes submodules, sources `scripts/common_env.sh`, builds the full EP artifacts, and runs the two-rank dispatch demo.

## Future UDMA Backend

Route 2 is intentionally deferred. The MVP route uses peer memory and `SyncCollectives` only.
