# TileXR SDMA Transport Design

Date: 2026-05-30

## Summary

Add a first-class TileXR SDMA local transport, parallel to the existing UDMA
transport but scoped to same-device GM-to-GM copies. The feature is explicitly
enabled by environment variable, exposes minimal host query APIs, and provides a
device-side raw GM pointer API for kernels that want to issue SDMA copies
directly.

The design is based on the SDMA puncture test performed on `blue`, where a
kernel used PTO SDMA intrinsics to copy local GM memory successfully at 64 B,
1 KB, 4 KB, 64 KB, and 1 MB. The production feature keeps the same data path but
wraps it in TileXR-owned runtime, compatibility, tests, and DFX.

## Goals

- Provide a formal TileXR SDMA capability for local on-card GM-to-GM copies.
- Keep the feature independent of shmem as a TileXR build/runtime dependency.
- Support both CANN 9.0.0 and CANN 9.1.0.
- Keep default TileXR initialization cost unchanged unless SDMA is explicitly
  enabled.
- Expose simple device-side raw pointer copy primitives.
- Add acceptance tests that run against both CANN versions.

## Non-Goals

- No cross-card SDMA transport.
- No remote registered-memory semantics.
- No integration into collective or operator auto-selection in the first
  implementation.
- No production performance tuning or full parameter stress matrix in the first
  implementation.
- No hardcoded MMIO register addresses in TileXR code.

## Architecture

Add a new SDMA transport beside the existing UDMA transport:

```text
src/comm/sdma/
  tilexr_sdma_transport.h
  tilexr_sdma_transport.cpp

src/include/
  tilexr_sdma.h
  tilexr_sdma_types.h
  tilexr_sdma_compat.h
```

`TileXRSDMATransport` is responsible for host-side initialization and lifetime.
It owns a PTO `pto::comm::sdma::SdmaWorkspaceManager`, calls `Init()` when
enabled, and exposes the resulting device-visible workspace pointer.

`CommArgs` gains:

```cpp
GM_ADDR sdmaWorkspacePtr = nullptr;
```

`ExtraFlag` gains:

```cpp
static constexpr uint32_t SDMA = 1 << 11;
```

The exact bit can change during implementation if bit 11 is unavailable, but it
must not overlap existing flags.

`TileXRComm::InitSDMA()` runs from `Init()` and `InitThread()` after device setup
and before `SyncCommArgs()`. It is gated by `TILEXR_ENABLE_SDMA=1`. If the env
var is absent, SDMA is skipped with an INFO log. If initialization fails, TileXR
continues without setting `ExtraFlag::SDMA`.

## Host API

Add minimal query APIs:

```cpp
int TileXRSDMAAvailable(TileXRCommPtr comm, bool* available);
int TileXRGetSDMAWorkspaceDev(TileXRCommPtr comm, GM_ADDR* workspace);
```

Semantics:

- `available=true` means SDMA was explicitly enabled and initialized.
- `workspace` is read-only from the user perspective and remains owned by the
  communicator.
- Disabled or unavailable SDMA returns success with `available=false` and
  `workspace=nullptr` when arguments are valid.
- Invalid arguments return the existing TileXR parameter-check error.

## Device API

Expose a raw local GM pointer API:

```cpp
__aicore__ inline bool SDMAEnabled(const __gm__ CommArgs* args);

__aicore__ inline uint64_t SDMACopyNbi(
    const __gm__ CommArgs* args,
    __gm__ uint8_t* dst,
    __gm__ uint8_t* src,
    uint64_t bytes,
    uint32_t channelGroupIdx = TILEXR_SDMA_AUTO_CHANNEL_GROUP);

__aicore__ inline bool SDMAWait(
    const __gm__ CommArgs* args,
    uint64_t eventHandle);
```

The required first-version API is the byte-count raw pointer API above. A typed
template overload is allowed only as a trivial header-only convenience wrapper,
and it is not part of the acceptance criteria:

```cpp
template <typename T>
__aicore__ inline uint64_t SDMACopyNbi(
    const __gm__ CommArgs* args,
    __gm__ T* dst,
    __gm__ T* src,
    uint64_t elemCount,
    uint32_t channelGroupIdx = TILEXR_SDMA_AUTO_CHANNEL_GROUP);
```

Behavior:

- If SDMA is disabled or arguments are invalid, `SDMACopyNbi` returns event
  handle `0`.
- `SDMAWait(args, 0)` returns `true`.
- The API does not validate whether `dst` and `src` belong to TileXR buffers.
  They are raw same-device GM pointers.
- The first implementation uses `queue_num = 1`, `block_bytes = 1 MB`, and
  `comm_block_offset = 0`.

## Data Flow

1. Host code initializes `TileXRComm`.
2. `TileXRComm::InitSDMA()` checks `TILEXR_ENABLE_SDMA`.
3. When enabled, `TileXRSDMATransport::Init()` creates and initializes
   `SdmaWorkspaceManager`.
4. The manager creates STARS streams and runs the AICPU query path that fills
   the workspace with SDMA queue metadata.
5. TileXR stores the workspace in `commArgs_.sdmaWorkspacePtr` and sets
   `ExtraFlag::SDMA`.
6. `SyncCommArgs()` uploads the updated `CommArgs`.
7. A kernel calls `SDMACopyNbi(args, dst, src, bytes)`.
8. The wrapper builds an SDMA session using `args->sdmaWorkspacePtr`, posts a
   PTO SDMA put, and returns the event handle.
9. The kernel calls `SDMAWait(args, eventHandle)`, which completes the deferred
   doorbell/flag SQE and waits for completion.

## CANN Compatibility

Support CANN 9.0.0 and 9.1.0 through `tilexr_sdma_compat.h`.

Responsibilities of the compatibility layer:

- Be the only TileXR public header that directly includes PTO SDMA intrinsics.
- Define `PTO_COMM_NOT_SUPPORTED` before including PTO headers to avoid the
  observed include-order issue where the public async wrapper references SDMA
  intrinsics before they are declared.
- Provide TileXR wrapper functions for session build, post put, and wait.
- Define the scratch tile type:

```cpp
pto::Tile<pto::TileType::Vec, uint8_t, 1, 256>
```

- Compile to no-op wrappers when PTO SDMA headers are unavailable.

Host-side `TileXRSDMATransport` includes
`pto/npu/comm/async/sdma/sdma_workspace_manager.hpp` and relies on the manager
to dynamically resolve runtime and opapi symbols, including:

- `aclnnShmemSdmaStarsQuery`
- `aclnnShmemSdmaStarsQueryGetWorkspaceSize`

TileXR may use this query path to obtain SDMA STARS metadata, but must not add a
direct shmem library dependency or use shmem data-copy APIs.

Build paths must include both common CANN layouts:

- `${ASCEND_HOME_PATH}/${ARCH}-linux/include`
- `${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc`

Linking may include `${ASCEND_HOME_PATH}/${ARCH}-linux/devlib` only as a
link-time fallback. Runtime RPATH/RUNPATH must not point at devlib because that
can load the stub `libascend_hal.so` and cause `aclInit` failures.

## Error Handling And Fallback

`TileXRComm::InitSDMA()` is best-effort:

- Env disabled: skip and return success.
- PTO/CANN unavailable: log warning, leave SDMA disabled, return success.
- `SdmaWorkspaceManager::Init()` failure: log warning, leave SDMA disabled,
  return success.
- Null workspace after successful init: log warning, leave SDMA disabled, return
  success.

Use a process-level failure cache similar to UDMA so repeated communicators do
not repeatedly create streams or spam logs after the first initialization
failure.

Device wrappers must be defensive and avoid device-side traps for normal
unavailable states. Invalid state becomes a no-op event handle `0`.

## DFX

Extend `PrintDFX()` with SDMA fields:

- whether `TILEXR_ENABLE_SDMA` was set;
- whether SDMA initialized;
- last initialization status;
- `sdmaWorkspacePtr`;
- whether `ExtraFlag::SDMA` is set.

Logs should distinguish:

- env not enabled;
- PTO SDMA headers not available at build time;
- opapi/runtime symbols not available at runtime;
- `SdmaWorkspaceManager::Init()` failure;
- null workspace;
- suspected HAL/RPATH issue when `aclInit` or the manager fails early.

## Tests And Acceptance

First-version acceptance is the B-tier test scope.

Source and unit guards:

- `src/comm` must not include or link shmem or aclshmem.
- `tilexr_sdma_compat.h` is the only TileXR header that directly includes PTO
  SDMA intrinsics.
- `CommArgs` layout/capability bit tests cover `sdmaWorkspacePtr` and
  `ExtraFlag::SDMA`.

Disabled fallback:

- Without `TILEXR_ENABLE_SDMA=1`, `TileXRSDMAAvailable` reports false.
- `ExtraFlag::SDMA` is not set.
- Existing communicator initialization remains successful.

Build/runtime checks:

- Build `tile-comm` and SDMA demo with CANN 9.0.0.
- Build `tile-comm` and SDMA demo with CANN 9.1.0.
- Check that runtime `libascend_hal.so` resolves to the driver path, not CANN
  devlib.

Data-plane demo:

```text
tests/sdma/demo/run_tilexr_sdma_demo.sh <cann_home> <device_id>
```

For each CANN version, the demo must test:

- 64 B
- 4 KB
- 1 MB

The demo must report debug fields for:

- SDMA enabled;
- session built;
- event handle non-zero for non-zero copy;
- wait completed;
- data matched.

## Documentation

Add `docs/SDMA_TRANSPORT.md` with:

- feature overview;
- enablement with `TILEXR_ENABLE_SDMA=1`;
- supported CANN versions;
- API examples;
- test commands for both CANN versions;
- runtime RPATH/HAL warning.

Update existing UDMA/shmem docs only where necessary to say SDMA is also
TileXR-owned and does not revive shmem dependencies.

## TODO For Later Stress And Performance Work

The following are explicitly out of scope for the first implementation, but the
design should leave room for them:

- Multi-block and multi-`channelGroupIdx` allocation policy.
- STARS channel resource sharing across multiple kernels and streams.
- Parameter matrix tests for bytes, `block_bytes`, `queue_num`, sync ID, and
  `channelGroupIdx`.
- Long-loop stability tests and error injection.
- Multi-stream concurrency tests.
- Performance statistics: bandwidth, latency, size breakpoints, and MTE/SDMA
  comparisons.
- Higher-level TileXR comm-buffer offset wrappers.
- Collective/operator auto-selection between MTE, SDMA, UDMA, and existing
  paths.

## Open Decisions Resolved

- Scope: first-class SDMA transport layer.
- CANN support: both 9.0.0 and 9.1.0.
- Device API: raw GM pointer API.
- Initialization: explicit `TILEXR_ENABLE_SDMA=1`.
- Host API: minimal availability/workspace query.
- CANN compatibility: centralized shim over PTO SDMA APIs.
- First acceptance tier: B, with C-tier stress/performance tests documented as
  TODO.
