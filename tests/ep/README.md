# TileXR EP Dispatch/Combine Tests

This tree tests the standalone TileXR EP module under `src/ep`. It is independent from `examples/mc2`; the same-node route uses TileXR IPC peer-memory windows and `SyncCollectives`, while cross-node dispatch/combine use TileXR-registered UDMA workspaces.

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

`full` mode builds and installs `tile-comm`, `tilexr-ep`, `libtilexr_ep_dispatch_kernel.so`, and `libtilexr_ep_combine_kernel.so` under the repository `install` directory, then builds the EP demo.

## Remote Verification

```bash
TILEXR_EP_REMOTE=<ssh-target> \
TILEXR_EP_REMOTE_BASE=<remote-scratch-dir> \
bash demo/deploy_and_run_remote.sh
```

The remote verification script syncs the complete repository into `${TILEXR_EP_REMOTE_BASE}/TileXR` on `${TILEXR_EP_REMOTE}`, initializes submodules, sources `scripts/common_env.sh`, builds the full EP artifacts, and runs the two-rank dispatch demo.

## Cross-Node UDMA

Cross-node dispatch/combine require a workspace allocated by the caller and registered with `TileXRUDMARegister`. The demo allocates a cache-line-aligned workspace, registers it before dispatch/combine, and validates both dispatch and combine outputs after all ranks synchronize.
