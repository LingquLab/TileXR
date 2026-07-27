# TileXR UDMA Context Refactor Design

## Summary

Refactor the host-side UDMA integration so that UDMA state is owned by a
dedicated `TileXRUDMAContext` under `src/comm/udma`, while preserving the
existing public registered-memory API and device-side UDMA wrappers.

UDMA is a user-visible registered-memory capability, not a collective backend
selection. Users still initialize a normal `TileXRComm`, call
`TileXRUDMARegister` for device memory that kernels may access through UDMA,
and use `tilexr_udma.h` device helpers from kernels. The refactor only moves
lifecycle and state ownership out of `TileXRComm`.

## Current Problem

`TileXRComm` currently owns three different layers at once:

- communicator setup, IPC memory, socket exchange, and `CommArgs` upload;
- UDMA transport initialization through `TileXRUDMATransport`;
- registered-memory bookkeeping, registry device allocation, rollback, and
  public UDMA query behavior.

This makes `TileXRComm` carry UDMA-specific fields such as `udmaInfoDev_`,
`udmaRegistryDev_`, `udmaRegisteredPtr_`, `udmaRegistry_`, and
`udmaTransport_`. The transport class is already a clean boundary for HCCP/RA
runtime setup, but the registered-memory context still leaks into the main
communicator.

The desired shape is:

- `TileXRUDMATransport` owns low-level HCCP/RA transport resources.
- `TileXRUDMAContext` owns UDMA feature state for one communicator.
- `TileXRComm` delegates UDMA work and only mirrors the resulting pointers into
  `CommArgs`.

## Goals

- Keep `TileXRUDMARegister`, `TileXRUDMAUnregister`,
  `TileXRGetUDMARegistryDev`, and `TileXRGetUDMARegistryHost` behavior
  compatible.
- Keep installed UDMA headers public:
  - `src/include/tilexr_udma.h`
  - `src/include/tilexr_udma_reg.h`
  - `src/include/tilexr_udma_types.h`
- Move UDMA lifecycle state from `TileXRComm` to `TileXRUDMAContext`.
- Keep UDMA initialization graceful: unsupported hardware or runtime failures
  disable UDMA without failing ordinary communicator initialization.
- Preserve the current process-level "skip after init failure" behavior.
- Preserve the current `InitThread` limitation for `TileXRUDMARegister`.
- Preserve the current single active registered region and handle value `0`.
- Keep `CommArgs::extraFlag`, `CommArgs::udmaInfoPtr`, and
  `CommArgs::udmaRegistryPtr` as the device ABI.
- Keep the implementation independent from shmem.

## Non-goals

- Do not convert UDMA into a collective algorithm or backend mode.
- Do not remove or hide the public UDMA registration API.
- Do not redesign `tilexr_udma.h`, `UDMAInfo`, queue layout, or HCCP route
  setup.
- Do not add multi-region registration or non-zero handles in this refactor.
- Do not change collective APIs, collective mode selection, or SDMA behavior.
- Do not add new dependencies on shmem.

## Proposed Structure

Add:

```text
src/comm/udma/tilexr_udma_context.h
src/comm/udma/tilexr_udma_context.cpp
```

Recommended host-facing internal interface:

```cpp
namespace TileXR {

struct TileXRUDMACommArgsState {
    bool available = false;
    GM_ADDR infoDev = nullptr;
    GM_ADDR registryDev = nullptr;
};

using TileXRUDMACommArgsUpdateFn =
    int (*)(const TileXRUDMACommArgsState& state, void* userData);

struct TileXRUDMAContextOptions {
    int rank = 0;
    int rankSize = 0;
    int devId = 0;
    bool threadMode = false;
    TileXRSockExchange* exchange = nullptr;
    TileXRUDMACommArgsUpdateFn updateCommArgs = nullptr;
    void* updateCommArgsUserData = nullptr;
};

class TileXRUDMAContext {
public:
    TileXRUDMAContext();
    ~TileXRUDMAContext();
    TileXRUDMAContext(const TileXRUDMAContext&) = delete;
    TileXRUDMAContext& operator=(const TileXRUDMAContext&) = delete;

    int Init(const TileXRUDMAContextOptions& options);
    void Shutdown();

    bool IsAvailable() const;
    TileXRUDMACommArgsState GetCommArgsState() const;

    int RegisterMemory(GM_ADDR localPtr, size_t bytes, TileXRUDMAMemHandle* handle);
    int UnregisterMemory(TileXRUDMAMemHandle handle);

    GM_ADDR GetRegistryDev() const;
    const TileXRUDMARegistry* GetRegistryHost() const;

private:
    void FreeRegistry();
};

} // namespace TileXR
```

The callback is provided by `TileXRComm`. It updates host `commArgs_` and calls
`UpdateCommArgsDev()` when `commArgsPtr_` has already been allocated. During
normal `Init()`, `commArgsPtr_` is still null, so the callback only updates the
host copy that `SyncCommArgs()` later uploads.

`TileXRUDMAContext` should be internal to `src/comm`. It must not be installed
or included by public headers.

## Responsibility Split

### `TileXRUDMATransport`

Keep this class focused on low-level transport resources:

- dynamic HCCP/RA symbol loading;
- device open and TSD process setup;
- EID route discovery;
- context, channel, CQ, and QP creation;
- queue import and memory import;
- device `TileXR::UDMAInfo` image allocation and refresh;
- per-pointer transport registration and unregistration.

No public registered-memory registry policy should move into this class.

### `TileXRUDMAContext`

Own communicator-level UDMA state:

- decide whether initialization should be skipped for single-rank communicators;
- consult and update the process-level "UDMA unavailable" cache;
- allocate and shut down `TileXRUDMATransport`;
- expose `udmaInfoPtr` and `udmaRegistryPtr` as a small state object for
  `CommArgs`;
- validate public registration arguments;
- reject registration in `InitThread` mode;
- use socket allgather to exchange `TileXRUDMARegionDesc`;
- build the host `TileXRUDMARegistry`;
- allocate, upload, and free the device registry image;
- unregister the previous active pointer when replacing the active region;
- invoke the comm-args update callback for UDMA pointer transitions;
- implement rollback when registration or comm-args update fails.

The class is still a host runtime class, not an installed device API.

### `TileXRComm`

Keep only orchestration:

- construct `TileXRUDMAContext` during `InitUDMA`;
- call `TileXRUDMAContext::Init`;
- provide a private `ApplyUDMACommArgsState` callback that mirrors the context
  state into `commArgs_`;
- delegate public UDMA registration and query methods;
- call `TileXRUDMAContext::Shutdown` from the destructor.

`ApplyUDMACommArgsState` should save the previous host values, apply the new
state, call `UpdateCommArgsDev()` only when `commArgsPtr_ != nullptr`, and
restore the previous host values if the device update fails.

After refactor, `TileXRComm` should not have fields for:

- `udmaInfoDev_`
- `udmaRegistryDev_`
- `udmaRegisteredPtr_`
- `udmaRegistry_`
- `std::unique_ptr<TileXRUDMATransport>`

It should have only:

```cpp
std::unique_ptr<TileXRUDMAContext> udmaContext_;
```

## Lifecycle

### Init

1. `TileXRComm::InitUDMA()` creates `TileXRUDMAContext`.
2. `TileXRUDMAContext::Init(options)` handles single-rank skip, process-level
   skip, transport creation, and transport initialization.
3. If available, the context records `transport.GetUDMAInfoDev()`.
4. The context asks `TileXRComm` to apply `GetCommArgsState()`:
   - set `commArgs_.udmaInfoPtr` to `infoDev`;
   - set `ExtraFlag::UDMA` only when `available && infoDev != nullptr`;
   - keep `commArgs_.udmaRegistryPtr` null until registration.
5. Ordinary communicator initialization continues even when UDMA is unavailable.

### Register

1. Public C API calls `TileXRComm::RegisterUDMAMemory`.
2. `TileXRComm` delegates to `udmaContext_->RegisterMemory`.
3. Context validates initialized availability, arguments, and thread mode.
4. Context calls `TileXRUDMATransport::RegisterMemory(localPtr, bytes)`.
5. Context allgathers `TileXRUDMARegionDesc` across ranks.
6. Context builds a complete `TileXRUDMARegistry`.
7. Context allocates and uploads the device registry.
8. Context preserves the previous active registration and previous registry.
9. Context asks `TileXRComm` to apply the next comm-args state.
10. If the update succeeds, context commits the new active registration and
    frees the old registration and registry.
11. If the update fails, context unregisters the new pointer, frees the new
    registry, restores the previous active state, and asks `TileXRComm` to
    restore the previous comm-args state.

### Unregister

1. Only handle `0` is valid in the current design.
2. Context preserves the active registration and registry.
3. Context asks `TileXRComm` to clear `commArgs_.udmaRegistryPtr`.
4. If the update succeeds, context unregisters the active pointer and frees the
   registry.
5. If the update fails, context keeps the previous active state and returns the
   update error.
6. UDMA capability remains initialized; only registered-memory state is removed.

### Shutdown

1. `TileXRComm` calls `udmaContext_->Shutdown()` before releasing
   communicator-owned device state.
2. Context unregisters active memory if needed.
3. Context frees the device registry.
4. Context shuts down and releases `TileXRUDMATransport`.
5. Context clears host pointers and availability state.

## Error Handling

Keep current return classes:

- not initialized: `TILEXR_ERROR_NOT_INITIALIZED`
- invalid pointer, zero bytes, or null handle: `TILEXR_ERROR_PARA_CHECK_FAIL`
- UDMA unavailable: `TILEXR_ERROR_NOT_FOUND`
- unsupported thread-mode registration: preserve current
  `TILEXR_ERROR_INTERNAL`
- transport registration failure: `TILEXR_ERROR_INTERNAL`
- allgather failure: return the allgather error
- registry allocation or copy failure: `TILEXR_ERROR_INTERNAL`
- invalid unregister handle: `TILEXR_ERROR_NOT_FOUND`

`Init()` should return `TILEXR_SUCCESS` when UDMA is unavailable but ordinary
communicator setup can continue.

Rollback rules:

- If transport registration succeeds but region exchange fails, unregister the
  new pointer before returning.
- If registry allocation or upload fails, unregister the new pointer before
  returning.
- When replacing an existing region, do not destroy the old active region until
  the new device registry has been uploaded and the comm-args update succeeds.
- If comm-args update fails after staging a replacement, unregister the new
  pointer, free the new registry, and restore the previous host/device registry
  state before returning the error.

## Public API Impact

No public API change is required.

`src/include/tilexr_api.h` may continue to expose the UDMA registration calls.
`tilexr_udma_reg.h` stays public because user code needs
`TileXRUDMAMemHandle`, `TileXRUDMARegistry`, and registry query helpers for the
existing API surface.

`tilexr_udma_context.h` must not be included from installed public headers.

## Device ABI Impact

No device ABI change is required.

The device-facing contract remains:

```cpp
CommArgs::extraFlag & ExtraFlag::UDMA
CommArgs::udmaInfoPtr
CommArgs::udmaRegistryPtr
```

`tilexr_udma.h` should continue to treat missing UDMA capability, null
`udmaInfoPtr`, or null `udmaRegistryPtr` as a no-op or invalid state according
to the current wrapper behavior.

## Test Plan

### Source guards

Add or extend a host-only source guard test that asserts:

- `src/comm/tilexr_comm.h` contains `TileXRUDMAContext`;
- `src/comm/tilexr_comm.h` does not contain `udmaInfoDev_`,
  `udmaRegistryDev_`, `udmaRegisteredPtr_`, `udmaRegistry_`, or
  `std::unique_ptr<TileXRUDMATransport>`;
- `src/comm/tilexr_comm.cpp` does not directly include
  `udma/tilexr_udma_transport.h`;
- public headers do not include `tilexr_udma_context.h`;
- no new shmem include or link dependency is introduced under `src/comm`.

### Unit tests

Keep existing host-only tests:

- `tests/udma/unit/test_tilexr_udma_registry.cpp`
- `tests/udma/unit/test_tilexr_udma_transport_layout.cpp`

Add a context-level host test if the implementation introduces test seams for
device allocation and exchange:

- register rejects null pointer, zero bytes, and null handle;
- unavailable context returns `TILEXR_ERROR_NOT_FOUND`;
- thread-mode context rejects registration;
- invalid unregister handle returns `TILEXR_ERROR_NOT_FOUND`;
- failed exchange unregisters the just-registered pointer;
- failed registry upload unregisters the just-registered pointer;
- failed `CommArgs` update restores the previous registry state.

If hardware-bound transport setup makes a direct unit test too invasive, split
the registry build/upload/replace logic into small internal helpers that can be
tested without creating HCCP/RA resources.

### Integration checks

Reuse the existing UDMA acceptance path:

```bash
cd tests/udma
bash build.sh
./install/bin/test_tilexr_udma_transport_layout
./install/bin/test_tilexr_udma_registry
```

On supported hardware, keep the data-plane demo as the runtime gate:

```bash
cd tests/udma
bash demo/run_tilexr_udma_demo.sh 0 2 16 2 0
bash demo/run_tilexr_udma_demo.sh 1 2 16 2 0
```

Non-supported devices can only prove build and host-side behavior, not UDMA
data-plane success.

## Implementation Steps

1. Add `TileXRUDMAContext` header/source under `src/comm/udma`.
2. Move the process-level UDMA unavailable mutex/cache from
   `tilexr_comm.cpp` into `tilexr_udma_context.cpp`.
3. Move `FreeUDMARegistry` logic into `TileXRUDMAContext::FreeRegistry`.
4. Move `RegisterUDMAMemory` and `UnregisterUDMAMemory` stateful logic into
   `TileXRUDMAContext::RegisterMemory` and `UnregisterMemory`.
5. Update `TileXRComm::InitUDMA` to create the context and apply
   `TileXRUDMACommArgsState` through an internal callback.
6. Add `TileXRComm::ApplyUDMACommArgsState` and pass it to the context.
7. Update `TileXRComm` UDMA public methods to delegate to the context.
8. Replace UDMA fields in `TileXRComm` with `udmaContext_`.
9. Update CMake source lists to compile `tilexr_udma_context.cpp`.
10. Add source guard coverage for the new ownership boundary.
11. Run host-only UDMA tests and, when available, supported-hardware data-plane
    demos.

## Acceptance Criteria

- The branch changes only UDMA design/code/test files needed for this refactor.
- Existing public UDMA APIs and headers remain source-compatible.
- `TileXRComm` no longer owns UDMA transport, registry, or registered-pointer
  fields directly.
- `TileXRUDMAContext` owns all communicator-level UDMA state and rollback.
- `CommArgs` UDMA pointer semantics are unchanged.
- UDMA unavailable behavior remains graceful for normal communicator init.
- `InitThread` UDMA registration remains unsupported.
- No shmem dependency is introduced.
- Host-only UDMA unit tests pass.
- Supported-hardware UDMA demo remains the data-plane validation gate.
