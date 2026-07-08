# TileXR CCU API Isolation Design

## Context

The `codex/direct-ccu-rebased` branch currently implements direct CCU support under `src/comm/ccu`, but the public and
core communication surfaces are still polluted by CCU-specific declarations:

- `src/include/tilexr_api.h` declares direct CCU handles, constants, option structs, report structs, task structs, and
  entry points.
- `src/comm/comm_wrap.cpp` contains both generic TileXR C API wrappers and direct CCU public API bridge code.
- `src/comm/tilexr_comm.h` directly includes multiple CCU internal headers and exposes CCU-specific methods and state
  on `TileXRComm`.

This makes direct CCU look like part of the baseline TileXR communication API. The desired model is that CCU remains
available in the same `tile-comm` library, but it is an internal communication backend. Users should only enable the
backend during communicator setup and select it through normal collective communication controls. They should not call
CCU resource, repository, task-prepare, submit, or readback APIs directly.

## Goals

- Remove all CCU-specific public declarations from `tilexr_api.h`.
- Remove the direct CCU public C API bridge from `comm_wrap.cpp`.
- Hide CCU runtime state and orchestration behind a CCU-owned context instead of exposing it directly on
  `TileXRComm`.
- Model CCU as an internal C++ backend class rather than a user-facing C API.
- Add a generic collective backend selection surface so users can request `AUTO`, `AIV`, `UDMA`, or `CCU` collective
  execution without seeing backend resource details.
- Keep the build and link model simple: direct CCU stays in `libtile-comm.so`; no separate shared library is introduced
  in this change.
- Preserve current direct CCU behavior for internal probes and hardware validation through CCU-owned test hooks or
  internal C++ helpers.

## Non-Goals

- Do not split CCU into a new library or optional package target.
- Do not redesign the CCU lower-layer resource allocation, repository install, or runtime launch algorithms.
- Do not promote direct CCU prepare/submit/readback operations as public user APIs in this pass.
- Do not introduce a generic backend manager abstraction before there is more than one optional collective backend.
- Do not change non-CCU TileXR communication, UDMA, SDMA, collectives, or EP behavior.

## Public API Layout

`src/include/tilexr_api.h` remains the baseline TileXR C API header. It should contain only generic communication,
UDMA, SDMA, DFX, and common lifecycle declarations. It must not contain these strings:

- `CCU`
- `Ccu`
- `DirectCcu`
- `TILEXR_DIRECT_CCU`

No installed `tilexr_ccu_api.h` is introduced. A CCU-specific public header would be premature because the intended
user flow does not include direct CCU task preparation or submission.

The only user-visible direction for CCU should be through generic runtime configuration surfaces:

- communicator initialization can enable optional backends without exposing CCU resource structs or task descriptors;
- collective calls can select a communication mode/backend using the collective API surface, not a direct CCU API.

To keep `tilexr_api.h` free of CCU symbols, communicator initialization should enable the CCU backend through one of
these generic mechanisms:

- an environment/config string such as a backend allowlist; or
- a future generic init-options API whose header names do not encode CCU-specific task concepts.

Typed CCU selection belongs in the collective API layer. A future collective options surface can live in
`tilexr_collectives.h`, for example:

```cpp
enum TileXRCollectiveBackend {
    TILEXR_COLLECTIVE_BACKEND_AUTO = 0,
    TILEXR_COLLECTIVE_BACKEND_AIV = 1,
    TILEXR_COLLECTIVE_BACKEND_UDMA = 2,
    TILEXR_COLLECTIVE_BACKEND_CCU = 3,
};

struct TileXRCollectiveOptions {
    /* Null or zero-initialized options select AUTO. */
    TileXRCollectiveBackend backend = TILEXR_COLLECTIVE_BACKEND_AUTO;
};

int TileXRAllGatherEx(
    void* sendBuf,
    void* recvBuf,
    int64_t sendCount,
    TileXR::TileXRDataType dataType,
    TileXRCommPtr comm,
    aclrtStream stream,
    const TileXRCollectiveOptions* options);
```

The exact naming can change during implementation, but the public surface must expose only backend selection and must
not expose repository, SQE, XN, CKE, instruction, or task-preparation details.

If the implementation needs a temporary internal seam for smoke probes, it should live under `src/comm/ccu` or
`tests/ccu`, should not be installed, and should not be documented as user API.

## Source Layout

The generic wrapper file, `src/comm/comm_wrap.cpp`, keeps only baseline TileXR API implementations. Direct CCU
prepare/submit/readback C wrappers are removed from this file instead of moved to an installed CCU header.

The CCU source tree owns internal C++ entry points for backend use and validation. These entry points should be
organized around a class, not a broad C API facade.

Because CCU is currently the only new backend in this change that needs a dedicated runtime context, `TileXRComm` can
own a single CCU backend context directly. Do not add a generic `TileXRCommBackends` manager in this pass. `AIV` maps
to the existing AIV/default collective path, and `UDMA` maps to existing UDMA-capable collective paths when available.
If a later backend needs the same lifecycle pattern as CCU, that will be the right time to introduce a manager.

## `TileXRComm` Boundary

`TileXRComm` should stop exposing CCU internals as public methods and direct member fields. A CCU-owned backend class
should hold the direct CCU runtime state currently stored on `TileXRComm`, including:

- `TileXRCcuDirectRuntime`
- cached basic info and reports
- lower-layer template, snapshot, plan, routes, and plan reports
- direct CCU allgather round state
- prepare/install/readback helper methods used by collectives or internal validation

The preferred implementation is `src/comm/ccu/tilexr_ccu_backend.{h,cpp}` with a class named `TileXRCcuBackend`.
`TileXRComm` owns this context as an opaque `std::unique_ptr` and exposes only narrow internal accessors needed by
communicator initialization and collective dispatch, for example:

- initialize or disable the CCU backend according to communicator configuration;
- query whether the CCU backend is available;
- dispatch a collective operation through the CCU backend when the collective layer selects that mode;
- pass rank, rank size, device id, uid, and socket-exchange facilities needed by the backend.

If a temporary accessor is needed during migration, it must be clearly CCU-scoped and not added to `tilexr_api.h`.

The backend class should expose cohesive methods rather than mirroring the old public C functions one-for-one. Example
shape:

```cpp
class TileXRCcuBackend {
public:
    int Init(const TileXRCcuBackendOptions& options);
    void Shutdown();
    bool Available() const;

    int PrepareCollective(const TileXRCcuCollectiveRequest& request, TileXRCcuCollectivePlan* plan);
    int SubmitCollective(const TileXRCcuCollectivePlan& plan, void* stream);

    int PrepareMemoryCopyForTest(const TileXRCcuMemoryCopyRequest& request, TileXRCcuPreparedTasks* tasks);
    int ReadInstructionsForTest(...);
};
```

The exact method names can differ, but the boundary should express backend lifecycle and collective execution rather
than a user-facing direct CCU API.

## CCU Internal Structure

`TileXRCcuBackend` is the facade used by `TileXRComm` and the collective dispatch layer. Internally it should separate
control-plane setup from collective execution:

- `TileXRCcuRuntimeSession`: owns runtime availability, basic info refresh, driver adapter creation, RA/HCCP state,
  lower-layer transport exchange, and shutdown.
- `TileXRCcuCollectivePlanner`: turns a typed collective request into CCU resource allocation, lower-layer install
  plan, repository image, launch package, and submit task plan.
- `TileXRCcuExecutor`: submits prepared CCU tasks to a stream, handles synchronization/reporting policy, and maps
  runtime failures to TileXR error codes.

This split keeps the existing low-level implementation reusable while preventing `TileXRComm` from knowing about
resource repositories, SQE layout, XN/CKE allocation, or diagnostic readback.

## Backend Selection And Fallback

Collective dispatch must distinguish "enabled" from "selected". If `TileXRCollectiveOptions` is `nullptr` or
zero-initialized, the selected backend is `AUTO`.

- `AUTO`: choose the best available supported backend without surfacing backend-specific errors for skipped candidates.
  The default policy is CCU when the communicator enabled and initialized CCU and the requested collective is supported,
  then UDMA when UDMA is initialized and supports the requested collective, then the existing AIV path.
- `AIV`: always use the existing AIV/default collective path and do not touch CCU backend state. It also must not require
  UDMA.
- `UDMA`: require a UDMA-backed collective path. If UDMA was not initialized, return `TILEXR_ERROR_NOT_INITIALIZED`.
  If UDMA is initialized but the requested collective, datatype, topology, rank count, or hardware state is unsupported,
  return `TILEXR_ERROR_NOT_SUPPORT`. Do not silently fall back.
- `CCU`: require CCU. If the communicator did not enable or initialize the backend, return
  `TILEXR_ERROR_NOT_INITIALIZED`. If CCU is initialized but the requested collective, datatype, topology, rank count,
  or hardware state is unsupported, return `TILEXR_ERROR_NOT_SUPPORT`. Do not silently fall back.

The old collective entry points keep their current behavior and are equivalent to `AUTO` unless a later compatibility
decision says otherwise. New `*Ex` entry points can carry explicit `TileXRCollectiveOptions`.

## Test Hooks

Diagnostic-only behavior such as memory-copy task preparation and instruction readback should not be normal backend
API. These helpers should be available only through one of these mechanisms:

- `#if defined(TILEXR_CCU_TESTING)` declarations inside CCU implementation files; or
- private test helpers under `tests/ccu`.

Production backend methods should be limited to lifecycle, capability query, collective planning, and collective
submission.

## Build And Install

`src/comm/CMakeLists.txt` should add the new CCU backend/context sources to `tile-comm`.

Install headers should not include a CCU-specific public API header.

No separate library or install component is introduced.

## Tests

Update the existing CCU tests so they encode the new boundary:

- The baseline public header test asserts `tilexr_api.h` contains no CCU symbols.
- The old CCU public C API compile probe is removed or converted into an internal C++ backend probe.
- The smoke probe uses internal CCU backend/test hooks or the eventual generic collective-mode selection path; it does
  not include an installed CCU API header.
- Existing checks that private hcomm/hccl symbols and runtime launch structs do not leak to installed public headers
  are retained against `tilexr_api.h` and other installed headers.
- A source-guard asserts `comm_wrap.cpp` does not define direct CCU prepare/submit/readback wrappers.
- CCU implementation tests that previously looked for CCU methods directly on `TileXRComm` should be adjusted to the
  new backend/context boundary.
- Collective API tests should verify default-unset `AUTO`, explicit `AUTO`, forced `AIV`, forced `UDMA`, and forced
  `CCU` fallback/error behavior using fake backend states before relying on hardware smoke runs.

## Acceptance Criteria

- `rg -n "CCU|Ccu|DirectCcu|TILEXR_DIRECT_CCU" src/include/tilexr_api.h` returns no matches.
- No installed public header named `tilexr_ccu_api.h` is added.
- The old direct CCU public API compile probe is removed or no longer treats CCU as external user API.
- `comm_wrap.cpp` contains no direct CCU API bridge implementation.
- CCU state is no longer directly stored as many fields on `TileXRComm`; it is owned by the CCU backend/context class.
- There is no generic backend manager abstraction in this pass; `TileXRComm` owns only the CCU backend context.
- Installed public headers do not expose `TileXRDirectCcu*`, `PrepareDirectCcu`, `SubmitPrepared`, `Repository`, `SQE`,
  `XN`, `CKE`, or CCU task descriptors. `tilexr_collectives.h` may expose high-level backend enum values for `AUTO`,
  `AIV`, `UDMA`, and `CCU`.
- Unset collective options select `AUTO`.
- Forced `UDMA` and forced `CCU` collective modes do not silently fall back when the selected backend is unavailable or
  unsupported.
- The focused CCU unit/source-guard tests pass.
