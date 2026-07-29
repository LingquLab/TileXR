# TileXR Build Verification

**Updated:** 2026-07-28

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
- NPU driver version is 25.1.rc1 or later.

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
readelf -d install/lib/libtile-comm.so | grep -E "RPATH|RUNPATH" || true
```

Expected:

- CANN runtime libraries are resolved.
- The shmem grep prints nothing; A5 SDMA dynamically invokes the CANN built-in
  query and does not add a shmem or custom OPP link dependency.
- Any RPATH/RUNPATH output does not contain a CANN `devlib` directory.

## Build And Run SDMA Checks

```bash
cd /path/to/TileXR
bash tests/sdma/build.sh "$ASCEND_HOME_PATH" Ascend950
bash tests/sdma/run_tests.sh "$ASCEND_HOME_PATH"
```

Use the default `Ascend910B` target to compile-check the preserved PTO path:

```bash
bash tests/sdma/build.sh "$ASCEND_HOME_PATH"
```

On A5 / Ascend950 hardware, run the direct data-plane matrix documented in
[SDMA_TRANSPORT.md](SDMA_TRANSPORT.md). The A5 kernel must retain the repository's
`-O2` compile option, and no custom OPP environment setting is required.

## Build UDMA Tests

```bash
cd tests/udma
bash build.sh
```

Expected host-side artifacts:

```bash
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
./install/bin/test_tilexr_udma_transport_layout
./install/bin/test_tilexr_udma_registry

source ../../scripts/common_env.sh
export LD_LIBRARY_PATH="$PWD/install/lib:../../install/lib:${LD_LIBRARY_PATH:-}"
RANK=0 RANK_SIZE=1 ./install/bin/test_tilexr_udma
```

Expected:

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
| Cannot find `ascend_hal` | Driver HAL path missing | Add `/usr/local/Ascend/driver/lib64/driver` to `LD_LIBRARY_PATH`; do not use CANN `devlib` |
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
SDMA unit/build checks:
SDMA A5 data-plane matrix:
Log directory:

Errors or warnings:
```
