# TileXR Infra / Collectives Split Structure

## Goal

Keep the existing TileXR communication substrate usable as a small infra-only runtime, while making collective communication an optional component that users can build, install, link, and include only when needed.

## Target Layout

```text
src/
  include/
    tilexr_api.h                 # Infra-only public C API; existing compatibility header
    tilexr_collectives.h         # Optional collectives public C API
    tilexr_types.h               # Shared datatypes and status codes
    comm_args.h                  # Shared device-visible communication metadata
    tilexr_sync.h                # Shared device-side sync primitives
    tilexr_udma*.h               # Infra / UDMA public headers

  comm/                          # TileXR-infra implementation, existing path kept
    CMakeLists.txt               # Builds libtile-comm.so only
    tilexr_comm.*
    comm_wrap.cpp
    tilexr_internal.*
    ccl_kernel_args.h            # Rename later to kernel_launch_args.h if shared more broadly
    udma/
    tools/socket/

  collectives/                   # Optional TileXR-collectives component
    CMakeLists.txt               # Builds libtilexr-collectives.so
    host/
      tilexr_collectives.cpp     # Public C wrappers: TileXRAllGather, TileXRAllToAll
      collective_launcher.*      # Runtime kernel registration and LoadMTE-like launch
      collective_blockdim.*      # AllGather / AllToAll blockDim rules
      collective_validate.*      # Buffer/count/datatype/rank checks
    kernels/
      CMakeLists.txt             # Builds collectives CCE object blob
      tilexr_lccl_op.cpp
      common/
        collectives.h
        datacopy_gm2gm.h
        ipc_queue.h
      allgather/
        allgather.h
        allgather_hierarchy_double_ring.h
      alltoall/
        all2all_hierarchy.h
        all2all_hierarchy_small.h
    cmake/
      TileXRCollectivesKernels.cmake

tests/
  comm/                          # Infra-only tests
  udma/                          # Infra / UDMA tests
  memory/                        # Peer-memory demos
  collectives/                   # Optional collectives tests and tools
    CMakeLists.txt
    README.md
    unit/
      test_tilexr_collectives_api.cpp
      test_tilexr_collectives_no_infra_pollution.cpp
    integration/
      test_tilexr_collectives_correctness.cpp
    tilexr-tests/
      tilexr_collective_perf.cpp
    run_collectives_correctness.sh
```

## Library and Header Boundaries

### TileXR-infra

Target:

```text
tile-comm -> install/lib/libtile-comm.so
```

Public headers:

```text
src/include/tilexr_api.h
src/include/tilexr_types.h
src/include/comm_args.h
src/include/tilexr_sync.h
src/include/tilexr_udma*.h
```

Responsibilities:

- Communicator lifecycle: unique id, rank init, destroy.
- IPC memory setup, peer memory mapping, socket exchange.
- Device-visible `CommArgs`.
- UDMA registration and UDMA metadata.
- Generic infra helpers such as `TileXRGetCommArgsHost`, `TileXRGetCommArgsDev`.

Must not contain:

- `TileXRAllGather`, `TileXRAllToAll`, or future collective symbols.
- Collective CCE binary embedding.
- Collective kernel registration.
- Collective perf/test tools.

### TileXR-collectives

Target:

```text
tilexr-collectives -> install/lib/libtilexr-collectives.so
```

Public header:

```text
src/include/tilexr_collectives.h
```

Responsibilities:

- Public collective APIs:

```cpp
int TileXRAllGather(void *sendBuf, void *recvBuf, int64_t sendCount,
                    TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                    aclrtStream stream);

int TileXRAllToAll(void *sendBuf, void *recvBuf, int64_t sendCount,
                   TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                   aclrtStream stream);
```

- Collective parameter validation.
- Collective blockDim selection.
- Collective CCE binary registration.
- Collective kernel launch.
- Correctness and performance tooling.

Dependencies:

- Links to `tile-comm`.
- Uses public infra APIs and shared headers.
- Does not require HCCL, hcomm, shmem, ops-transformer, or mki.

## CMake Shape

Top-level:

```cmake
option(TILEXR_BUILD_COLLECTIVES "Build optional TileXR collectives library and tests" OFF)

add_subdirectory(src/comm)

if(TILEXR_BUILD_COLLECTIVES)
    add_subdirectory(src/collectives)
endif()
```

Infra remains buildable exactly as today:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build --target tile-comm -j"$(nproc)"
```

Collectives build is explicit:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$PWD/install" -DTILEXR_BUILD_COLLECTIVES=ON
cmake --build build --target tilexr-collectives -j"$(nproc)"
```

Tests:

```bash
cmake -S tests/collectives -B /tmp/tilexr-collectives-tests \
  -DCMAKE_INSTALL_PREFIX="$PWD/tests/collectives/install"
cmake --build /tmp/tilexr-collectives-tests -j"$(nproc)"
```

## Minimal Infra Extension for Collectives

Collectives should not access `TileXRComm` private members directly. Add one infra-neutral public C helper if needed:

```cpp
int TileXRCommNextMagic(TileXRCommPtr comm, int64_t *magic);
```

This belongs in `tilexr_api.h` because it is not collective-specific; it allocates a communicator epoch usable by any optional extension that launches device kernels sharing `CommArgs` flags.

If the team does not want this public, expose it through an installed-but-internal header such as:

```text
src/include/tilexr_extension_api.h
```

The preferred option is the explicit C helper because `tilexr-collectives` can then link against infra without including `src/comm/tilexr_comm.h`.

## Installation Result

Infra-only install:

```text
install/include/tilexr_api.h
install/include/tilexr_types.h
install/include/comm_args.h
install/lib/libtile-comm.so
```

With collectives:

```text
install/include/tilexr_collectives.h
install/lib/libtilexr-collectives.so
install/lib/libtile-comm.so
```

Applications choose explicitly:

```cpp
#include "tilexr_api.h"          // Infra only
#include "tilexr_collectives.h"  // Optional collectives
```

Link examples:

```bash
# Infra only
-ltile-comm

# Collectives
-ltilexr-collectives -ltile-comm
```

## Verification Gates

Infra-only:

```bash
nm -D install/lib/libtile-comm.so | grep -E "TileXRAllGather|TileXRAllToAll"
```

Expected: no output.

```bash
ldd install/lib/libtile-comm.so | grep -Ei "hcomm|hccl|shmem|aclshmem|ops-transformer|mki"
```

Expected: no output.

Collectives:

```bash
nm -D install/lib/libtilexr-collectives.so | grep -E "TileXRAllGather|TileXRAllToAll"
```

Expected: both symbols are present.

```bash
ldd install/lib/libtilexr-collectives.so | grep libtile-comm
```

Expected: `libtile-comm.so` is present.

## Migration From Previous Plan

- Do not add collective APIs to `tilexr_api.h`; add them to `tilexr_collectives.h`.
- Do not add `src/collectives/*.cpp` to `src/comm/CMakeLists.txt`; build a separate collectives target.
- Do not embed collective CCE blobs in `tile-comm`; embed them in `tilexr-collectives`.
- Do not register collective kernels in `TileXRComm::InitCommon`; register them lazily in `tilexr-collectives` before the first collective launch.
- Keep all nccl-tests style tools under `tests/collectives`, linked against `libtilexr-collectives.so`.
