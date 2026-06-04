# UDMA Test Infrastructure Status

**Updated:** 2026-05-29
**Status:** TileXR UDMA test infrastructure is present; A5 / Ascend950 hardware is required for data-plane validation

## Summary

The current UDMA acceptance path validates TileXR-owned registered-memory communication. It no longer depends on standalone shmem tests.

The test tree verifies:

- `src/comm` does not include or link shmem;
- UDMA layout structures match the device-side expectations;
- registered-memory metadata and remote address calculations are correct;
- a TileXR communicator can expose `CommArgs`;
- AICore demo kernels can issue UDMA put and put-with-signal on A5 / Ascend950 / 950 hardware.

## Test Files

```text
tests/udma/
|-- build.sh
|-- run_tests.sh
|-- unit/
|   |-- test_tilexr_udma_transport_layout.cpp
|   `-- test_tilexr_udma_registry.cpp
|-- integration/
|   `-- test_tilexr_udma.cpp
`-- demo/
    |-- tilexr_udma_demo.cpp
    |-- tilexr_udma_demo_kernel.cpp
    |-- run_tilexr_udma_demo.sh
    |-- README.md
    `-- ASCEND_VERIFICATION.md
```

## Host-Only Checks

```bash
cd /path/to/TileXR/tests/udma
bash build.sh
./install/bin/test_tilexr_udma_transport_layout
./install/bin/test_tilexr_udma_registry
```

Expected:

```text
TileXR UDMA transport layout checks passed
TileXR UDMA registry checks passed
```

## Communicator Smoke Test

```bash
cd /path/to/TileXR/tests/udma
source ../../scripts/common_env.sh
export LD_LIBRARY_PATH="$PWD/install/lib:../../install/lib:${LD_LIBRARY_PATH:-}"
RANK=0 RANK_SIZE=1 ./install/bin/test_tilexr_udma
```

Expected:

- exit code 0;
- communicator initialization succeeds;
- `TileXRGetCommArgsHost` succeeds;
- output reports `Failed: 0`.

This is a smoke test only. It does not prove UDMA data-plane transfer.

## A5 / Ascend950 Runtime Demos

All-gather style put:

```bash
cd /path/to/TileXR/tests/udma
bash demo/run_tilexr_udma_demo.sh 0 2 16 2 0
```

Put with signal:

```bash
bash demo/run_tilexr_udma_demo.sh 1 2 16 2 0
```

Expected:

- every rank log contains `TileXR UDMA demo success`;
- every rank log contains `TileXRUDMARegister success`;
- all-gather samples include per-rank segments such as `seg0=1000 seg1=1001`;
- put-signal logs show non-local signal entries equal to `1000`;
- no log contains `DATA MISMATCH`, `expected non-local signals`, `TileXR UDMA demo failed`, or `ERROR`.

## Hardware Scope

UDMA runtime validation is only valid on A5 / Ascend950 / 950 hardware.

910B, 310P, and other non-A5 devices may still be useful for build checks or communicator smoke checks, but they must not be reported as successful UDMA runtime validation targets.

## References

- [tests/udma/README.md](../tests/udma/README.md)
- [tests/udma/QUICKSTART.md](../tests/udma/QUICKSTART.md)
- [tests/udma/demo/ASCEND_VERIFICATION.md](../tests/udma/demo/ASCEND_VERIFICATION.md)
