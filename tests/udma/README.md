# TileXR UDMA Test Guide

This directory contains tests and demos for TileXR UDMA registered-memory communication.

UDMA means UnifiedBus DMA. It is based on Huawei's UnifiedBus interconnect. In TileXR, the UDMA test path currently targets A5 hardware, identified in this repository as `Ascend950` / `950`. It is not expected to work on 910B, 310P, or other non-A5 devices.

## What This Tests

Current recommended validation focuses on TileXR-owned APIs:

- host registration of ordinary device memory with `TileXRUDMARegister`;
- device-side registered-memory access through `tilexr_udma.h`;
- all-gather style UDMA put;
- UDMA put with signal;
- registry layout and remote-address calculation.
- TileXR `src/comm` sources do not include or link shmem.

There is no standalone shmem API test in this tree. TileXR UDMA validation should go through TileXR APIs and the registered-memory demo.

## Files

```text
tests/udma/
├── CMakeLists.txt
├── build.sh
├── run_tests.sh
├── unit/
│   ├── test_tilexr_udma_registry.cpp
│   └── test_tilexr_udma_transport_layout.cpp
├── integration/
│   └── test_tilexr_udma.cpp            # TileXR communicator smoke test
└── demo/
    ├── tilexr_udma_demo.cpp            # host demo, no shmem.h dependency
    ├── tilexr_udma_demo_kernel.cpp     # AICore UDMA put / put-signal demo
    ├── run_tilexr_udma_demo.sh
    ├── README.md
    └── ASCEND_VERIFICATION.md
```

## Hardware Requirement

Use an A5 / Ascend950 / 950 machine for runtime validation.

```bash
npu-smi info
```

Expected:

- the machine has real Ascend950-class devices;
- at least 2 devices are available for the communication demo;
- CANN and the Ascend driver are installed and visible to `scripts/common_env.sh`.

Non-A5 devices may still compile parts of the test tree, but they are not valid targets for UDMA runtime validation.

## Build

Build TileXR first:

```bash
cd /path/to/TileXR
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build-udma -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build-udma --target tile-comm -j"$(nproc)"
cmake --install /tmp/tilexr-build-udma
```

Then build UDMA tests and the demo:

```bash
cd /path/to/TileXR/tests/udma
bash build.sh
```

Expected useful artifacts:

```bash
test -x install/bin/test_tilexr_udma_registry
test -x install/bin/test_tilexr_udma_transport_layout
test -x install/bin/test_tilexr_udma
test -x install/bin/tilexr_udma_demo
test -f install/lib/libtilexr_udma_demo_kernel.so
```

If `bisheng` is unavailable, the host-only tests may still build while the AICore demo is skipped.

## Recommended Validation

### 1. Host-Only Unit Tests

These are host-only checks for registered-region metadata and UDMA info layout.

```bash
cd /path/to/TileXR/tests/udma
./install/bin/test_tilexr_udma_transport_layout
./install/bin/test_tilexr_udma_registry
```

Expected:

```text
TileXR UDMA transport layout checks passed
TileXR UDMA registry checks passed
```

### 2. TileXR Communicator Smoke Test

This confirms TileXR communicator setup and `CommArgs` access. It does not prove data-plane UDMA transfer.

```bash
cd /path/to/TileXR/tests/udma
source ../../scripts/common_env.sh
export LD_LIBRARY_PATH="$PWD/install/lib:../../install/lib:${LD_LIBRARY_PATH:-}"
RANK=0 RANK_SIZE=1 ./install/bin/test_tilexr_udma
```

Expected:

- exit code 0;
- `TileXRCommInitRankLocal should succeed`;
- `TileXRGetCommArgsHost should succeed`;
- `Failed: 0`.

### 3. UDMA Demo: Put All-Gather

Run on an A5 / Ascend950 machine with at least two usable devices:

```bash
cd /path/to/TileXR/tests/udma
bash demo/run_tilexr_udma_demo.sh 0 2 16 2 0
```

Expected:

- every rank exits with code 0;
- every rank log contains `TileXR UDMA demo success`;
- every rank log contains `TileXRUDMARegister success`;
- result samples include `seg0=1000 seg1=1001`;
- no log contains `DATA MISMATCH`, `ERROR`, or `TileXR UDMA demo failed`.

### 4. UDMA Demo: Put With Signal

```bash
cd /path/to/TileXR/tests/udma
bash demo/run_tilexr_udma_demo.sh 1 2 16 2 0
```

Expected:

- every rank exits with code 0;
- every rank log contains `TileXR UDMA demo success`;
- signal values are printed;
- non-local signal entries equal `1000`;
- no log contains `expected non-local signals`, `DATA MISMATCH`, `ERROR`, or `TileXR UDMA demo failed`.

## Useful Log Checks

```bash
latest=$(ls -td logs/tilexr_udma_demo_* | head -n1)
grep -R "TileXR UDMA demo success" "$latest"
grep -R "TileXRUDMARegister success" "$latest"
grep -R "DATA MISMATCH\\|expected non-local signals\\|TileXR UDMA demo failed\\|ERROR" "$latest" || true
```

## Notes

- Demo host code must not include `shmem.h` or call shmem host APIs.
- TileXR comm initialization must not include or link shmem.
- Device kernels should use TileXR wrappers from `tilexr_udma.h`.
- UDMA buffers in the demo are ordinary `aclrtMalloc` allocations registered with `TileXRUDMARegister`.
- Current runtime validation is hardware-dependent. A CANN-only environment without real A5 devices can verify compilation, but not UDMA transfer.
