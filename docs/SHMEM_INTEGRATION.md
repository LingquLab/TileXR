# shmem Integration Status

**Updated:** 2026-05-30

This repository still contains `3rdparty/shmem`, but the current TileXR core communication and UDMA path do not use shmem as a runtime dependency.

## Current Status

- `src/comm` does not include `shmem.h`.
- `libtile-comm.so` does not link `libshmem.so` or `libaclshmem.so`.
- `TileXRComm::InitUDMA()` uses TileXR-owned HCCP/RA transport code under `src/comm/udma/`.
- UDMA validation goes through TileXR APIs, especially `TileXRUDMARegister` and device wrappers in `tilexr_udma.h`.
- `tests/udma/unit/test_tilexr_no_shmem_dependency.cpp` is the guardrail that checks this boundary.

## Why The Submodule Still Exists

`3rdparty/shmem` is kept for reference, experiments, upstream comparison, and examples. It is not required to build the current `tile-comm` target.

Do not add a shmem include or link dependency to `src/comm` unless the project explicitly chooses to revive the old shmem-backed UDMA design.

## Historical Design Note

An earlier UDMA proposal used a patched shmem API:

```cpp
int aclshmemx_get_udma_info(void **udma_info_ptr, size_t *udma_info_size);
```

That proposal expected shmem to initialize UDMA queue resources and expose an opaque device pointer to TileXR. It also expected `tile-comm` to link shmem.

The current implementation supersedes that approach. TileXR now builds `TileXR::UDMAInfo` itself from HCCP/RA queue and memory registration metadata.

## Current UDMA Flow

```text
TileXRComm::InitUDMA
  -> TileXRHccpLoader dynamically loads CANN/HCCP runtime symbols
  -> TileXRUDMATransport opens device process and RA contexts
  -> local and remote queue metadata is exchanged with TileXRSockExchange
  -> TileXR builds and uploads device-side UDMAInfo
  -> TileXRUDMARegister exchanges registered memory regions
  -> kernels use tilexr_udma.h wrappers
```

## Verification

Build `tile-comm` and check that shmem is not linked:

```bash
cd /path/to/TileXR
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build --target tile-comm -j"$(nproc)"
cmake --install /tmp/tilexr-build
ldd install/lib/libtile-comm.so | grep -i shmem || true
```

Expected: the final command prints nothing.

Run the source-level guard:

```bash
cd /path/to/TileXR/tests/udma
bash build.sh
./install/bin/test_tilexr_no_shmem_dependency
```

Expected:

```text
TileXR comm sources have no shmem dependency
```

## When To Look At shmem

Use the shmem submodule only when:

- comparing TileXR UDMA behavior with shmem examples;
- experimenting with alternative UDMA initialization strategies;
- updating historical design notes under `docs/superpowers/`;
- checking whether an upstream shmem API can replace TileXR-owned transport code in a future design.

The production README and UDMA acceptance guides should continue to describe the TileXR-owned path unless the implementation changes.

## SDMA Note

TileXR SDMA transport is also TileXR-owned and does not revive the old
shmem-backed design. The SDMA host manager may use CANN's
`aclnnShmemSdmaStarsQuery*` symbols through PTO `SdmaWorkspaceManager` to obtain
STARS queue metadata, but TileXR does not include or link shmem and does not use
shmem data-copy APIs.
