# TileXR Build Verification

**Updated:** 2026-05-29

This checklist reflects the current TileXR codebase. The core runtime builds `libtile-comm.so` without a compile-time or link-time shmem dependency.

## Environment

```bash
cd /path/to/TileXR
source scripts/common_env.sh
npu-smi info
```

Expected:

- CANN 9.1.0 environment is visible through `ASCEND_HOME_PATH`.
- `scripts/common_env.sh` detects architecture and SOC information.
- NPU driver version is 25.5.0 or later.

## Build Core Runtime

```bash
cmake -S . -B /tmp/tilexr-build -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build --target tile-comm -j"$(nproc)"
cmake --install /tmp/tilexr-build
```

Expected:

```bash
test -f install/lib/libtile-comm.so
```

Check dynamic dependencies:

```bash
ldd install/lib/libtile-comm.so | grep -E "ascendcl|runtime|ascend_hal|profapi"
ldd install/lib/libtile-comm.so | grep -i shmem || true
```

Expected:

- CANN runtime libraries are resolved.
- The shmem grep prints nothing for the current TileXR UDMA implementation.

## Build UDMA Tests

```bash
cd tests/udma
bash build.sh
```

Expected host-side artifacts:

```bash
test -x install/bin/test_tilexr_no_shmem_dependency
test -x install/bin/test_tilexr_udma_transport_layout
test -x install/bin/test_tilexr_udma_registry
test -x install/bin/test_tilexr_udma
```

If `bisheng` is available, the AICore demo artifacts should also exist:

```bash
test -x install/bin/tilexr_udma_demo
test -f install/lib/libtilexr_udma_demo_kernel.so
```

If `bisheng` is unavailable, `build.sh` may skip only the demo target while still building host-only tests.

## Run Host Checks

```bash
cd /path/to/TileXR/tests/udma
./install/bin/test_tilexr_no_shmem_dependency
./install/bin/test_tilexr_udma_transport_layout
./install/bin/test_tilexr_udma_registry

source ../../scripts/common_env.sh
export LD_LIBRARY_PATH="$PWD/install/lib:../../install/lib:${LD_LIBRARY_PATH:-}"
RANK=0 RANK_SIZE=1 ./install/bin/test_tilexr_udma
```

Expected:

- `TileXR comm sources have no shmem dependency`
- `TileXR UDMA transport layout checks passed`
- `TileXR UDMA registry checks passed`
- `test_tilexr_udma` exits with `Failed: 0`

## Run UDMA Data-Plane Demos

Run only on A5 / Ascend950 / 950 hardware:

```bash
cd /path/to/TileXR/tests/udma
bash demo/run_tilexr_udma_demo.sh 0 2 16 2 0
bash demo/run_tilexr_udma_demo.sh 1 2 16 2 0
```

Expected:

- every rank exits 0;
- every rank log includes `TileXR UDMA demo success`;
- every rank log includes `TileXRUDMARegister success`;
- no rank log includes `DATA MISMATCH`, signal mismatch text, or `ERROR`.

Quick log check:

```bash
latest=$(ls -td logs/tilexr_udma_demo_* | head -n1)
grep -R "TileXR UDMA demo success" "$latest"
grep -R "TileXRUDMARegister success" "$latest"
grep -R "DATA MISMATCH\\|expected non-local signals\\|TileXR UDMA demo failed\\|ERROR" "$latest" || true
```

The final grep should print nothing for a clean run.

## Common Failures

| Symptom | Likely Cause | Action |
| --- | --- | --- |
| Missing CANN headers | `common_env.sh` not sourced or CANN path mismatch | Source the environment and confirm CANN 9.1.0 layout |
| Cannot find `ascend_hal` | `devlib` path missing | Use the current top-level CMake configuration |
| Demo target skipped | `bisheng` unavailable | Install/compiler configure `bisheng`, or run host-only tests |
| UDMA disabled in demo | Unsupported hardware or HCCP/RA runtime unavailable | Use A5 / Ascend950 / 950 and check CANN driver/runtime libraries |
| shmem appears in `ldd libtile-comm.so` | Unexpected dependency regression | Inspect `src/comm/CMakeLists.txt` and source includes |

## Report Template

```text
Machine:
NPU model:
NPU count:
CANN version:
Driver version:
TileXR commit:

Core build:
ldd shmem check:
UDMA host tests:
UDMA all-gather demo:
UDMA put-signal demo:
Log directory:

Errors or warnings:
```
