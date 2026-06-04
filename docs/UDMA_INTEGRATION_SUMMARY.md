# TileXR UDMA Integration Summary

**Updated:** 2026-05-30
**Status:** Implemented in TileXR comm runtime; A5 / Ascend950 runtime validation required

## Overview

TileXR now contains a TileXR-owned registered-memory UDMA path. The current implementation does not depend on shmem host APIs or link shmem into `libtile-comm.so`.

The implementation is aligned with the TileXR design direction: use device-visible metadata and AICore-side wrappers so kernels can issue fine-grained, data-driven communication rather than waiting for coarse host-controlled collective phases.

## Current Architecture

### Host Runtime

- `TileXRComm::InitUDMA()` attempts UDMA initialization for multi-rank communicators.
- `TileXRUDMATransport` dynamically loads HCCP/RA runtime symbols from CANN libraries:
  - `libhccl.so`
  - `libhccl_v2.so`
  - `libra.so`
  - `libtsdclient.so`
- The transport opens the device-side TSD process, resolves EID routes, creates contexts, channels, CQs, and QPs, then uploads a `TileXR::UDMAInfo` image to device memory.
- Failure to initialize UDMA disables the UDMA capability bit but does not break normal communicator initialization.

### Registered Memory

- Host code registers ordinary `aclrtMalloc` device memory through `TileXRUDMARegister`.
- Registration exchanges remote region descriptors across ranks through the existing TileXR socket exchange.
- `CommArgs::udmaRegistryPtr` points to a device-side registry with per-rank base pointers and sizes.
- The current registry supports one registered region per rank.

### Device API

`src/include/tilexr_udma.h` exposes device-side wrappers:

- `UDMAPutNbi`
- `UDMAGetNbi`
- `UDMAPutSignalNbi`
- `UDMAQuiet`

Wrappers check `CommArgs::extraFlag`, `udmaInfoPtr`, and `udmaRegistryPtr` before issuing UDMA work. Invalid or unavailable UDMA state turns the wrapper into a no-op.

## Key Files

```text
src/include/tilexr_api.h
src/include/comm_args.h
src/include/tilexr_udma.h
src/include/tilexr_udma_reg.h
src/include/tilexr_udma_types.h
src/comm/tilexr_comm.cpp
src/comm/comm_wrap.cpp
src/comm/udma/tilexr_hccp_loader.*
src/comm/udma/tilexr_udma_transport.*
src/comm/udma/tilexr_udma_layout.*
tests/udma/
```

## Public API Additions

```cpp
typedef uint32_t TileXRUDMAMemHandle;

int TileXRUDMARegister(TileXRCommPtr comm, GM_ADDR localPtr, size_t bytes,
                       TileXRUDMAMemHandle *handle);
int TileXRUDMAUnregister(TileXRCommPtr comm, TileXRUDMAMemHandle handle);
int TileXRGetUDMARegistryDev(TileXRCommPtr comm, GM_ADDR &registryPtr);
```

`CommArgs` contains:

```cpp
GM_ADDR udmaInfoPtr;
GM_ADDR udmaRegistryPtr;
```

The capability bit is `ExtraFlag::UDMA`.

## What Changed From The Earlier shmem Design

Early UDMA design notes used a patched shmem library to expose UDMA queue information. The current code no longer follows that path:

- `src/comm` does not include `shmem.h`.
- `tile-comm` does not link `libshmem.so` or `libaclshmem.so`.
- UDMA metadata is built by TileXR from HCCP/RA contexts and uploaded as `TileXR::UDMAInfo`.
- UDMA tests include a no-shmem dependency check to keep this boundary explicit.

The ignored `reference/shmem/` checkout can be created with `reference/download_shmem.sh` for reference, experiments, and upstream comparison, but it is not part of the current TileXR UDMA acceptance flow.

## Validation Scope

Host-only checks:

```bash
cd tests/udma
bash build.sh
./install/bin/test_tilexr_no_shmem_dependency
./install/bin/test_tilexr_udma_transport_layout
./install/bin/test_tilexr_udma_registry
```

Communicator smoke test:

```bash
source ../../scripts/common_env.sh
export LD_LIBRARY_PATH="$PWD/install/lib:../../install/lib:${LD_LIBRARY_PATH:-}"
RANK=0 RANK_SIZE=1 ./install/bin/test_tilexr_udma
```

A5 / Ascend950 data-plane demos:

```bash
bash demo/run_tilexr_udma_demo.sh 0 2 16 2 0
bash demo/run_tilexr_udma_demo.sh 1 2 16 2 0
```

Successful demo logs should include `TileXR UDMA demo success` and `TileXRUDMARegister success`, and should not include `DATA MISMATCH`, signal mismatch messages, or `ERROR`.

## Constraints

- Runtime data-plane validation requires A5 / Ascend950 / 950 hardware.
- Non-A5 devices can be used only for compilation or smoke checks, not for UDMA success claims.
- The demo kernel build requires `bisheng`; if unavailable, host-only UDMA tests can still build.
- `TileXRUDMARegister` is not supported in `InitThread` mode in the current implementation.
- The current registered-memory API supports one active registered region.

## Design Alignment

Implemented now:

- AICore-visible transport metadata.
- Registered-memory put/get/signal wrappers.
- Graceful capability detection.
- TileXR-owned runtime setup and DFX-friendly test logs.

Still design/roadmap:

- Full dynamic semantic selection among MTE, UDMA/RDMA, notify, and data-as-flag per tile.
- Best-effort CMO scheduling.
- CCU/MS offload of control loops.
- Broader tile-level profiling and replay tooling.

## Relationship To SDMA

UDMA remains the registered-memory remote transport. SDMA is a separate local
GM-to-GM transport for same-device copies. Both expose device-visible metadata
through `CommArgs`, but SDMA does not require memory registration and is enabled
only when `TILEXR_ENABLE_SDMA=1`.
