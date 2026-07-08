# CCU API Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor direct CCU from a public task API into an internal collective backend selected through `AUTO`, `AIV`, `UDMA`, or `CCU` collective options.

**Architecture:** Keep `tilexr_api.h` free of CCU symbols. Add high-level backend selection to `tilexr_collectives.h`, route collective calls through a small dispatcher, and make CCU an internal `TileXRCcuBackend` owned opaquely by `TileXRComm`. Inside CCU, separate runtime/session concerns from planning and task execution.

**Tech Stack:** C++14, C ABI-compatible exported functions, CMake, existing TileXR collectives tests, Python source-guard tests, Ascend runtime stubs where already used by this branch.

## Global Constraints

- Do not install or introduce a public `tilexr_ccu_api.h`.
- `src/include/tilexr_api.h` must not contain `CCU`, `Ccu`, `DirectCcu`, or `TILEXR_DIRECT_CCU`.
- Public headers must not expose `TileXRDirectCcu*`, `PrepareDirectCcu`, `SubmitPrepared`, `Repository`, `SQE`, `XN`, `CKE`, or CCU task descriptors.
- `tilexr_collectives.h` may expose high-level backend enum values for `AUTO`, `AIV`, `UDMA`, and `CCU`.
- Null or zero-initialized collective options select `AUTO`.
- Forced `UDMA` and forced `CCU` must not silently fall back when unavailable or unsupported.
- Do not add a generic backend manager in this pass; `TileXRComm` owns only the CCU backend context.
- Preserve existing AIV collective behavior for old entry points by making old entry points call the new `*Ex` path with `AUTO`.

---

## File Structure

- `src/include/tilexr_collectives.h`: public high-level backend enum, collective options, and `*Ex` collective APIs.
- `src/collectives/host/tilexr_collectives.cpp`: shared validation, `*Ex` implementations, old API forwarding, and backend dispatch.
- `src/collectives/host/collective_backend.h`: internal backend-selection helpers for `AUTO`, `AIV`, `UDMA`, and `CCU`.
- `src/collectives/host/collective_backend.cpp`: dispatch implementation and fake-state hooks for focused unit tests.
- `src/comm/tilexr_comm.h`: remove direct CCU includes/state; forward-declare `TileXRCcuBackend`; expose narrow internal CCU backend accessors.
- `src/comm/tilexr_comm.cpp`: delegate CCU lifecycle and direct CCU helper logic to `TileXRCcuBackend`.
- `src/comm/comm_wrap.cpp`: remove direct CCU public C API bridge and all direct CCU public constants/helpers.
- `src/comm/ccu/tilexr_ccu_backend.h`: internal CCU backend facade used by `TileXRComm` and collective dispatch.
- `src/comm/ccu/tilexr_ccu_backend.cpp`: backend facade implementation, initially moving behavior from `TileXRComm`.
- `src/comm/ccu/tilexr_ccu_runtime_session.h/.cpp`: runtime availability, driver adapter, basic info, RA/HCCP exchange, shutdown.
- `src/comm/ccu/tilexr_ccu_collective_planner.h/.cpp`: convert typed collective requests to existing CCU install/submit plans.
- `src/comm/ccu/tilexr_ccu_executor.h/.cpp`: submit prepared CCU tasks and map runtime failures to TileXR error codes.
- `src/comm/CMakeLists.txt`: add new CCU/backend sources; keep CCU linked into `tile-comm`.
- `tests/collectives/unit/test_tilexr_collective_backend_options.cpp`: compile/runtime checks for public options and dispatch fallback semantics.
- `tests/ccu/test_tilexr_ccu_public_comm_api.py`: invert old public API tests into source guards for no public CCU task API.
- `tests/ccu/test_tilexr_ccu_backend_boundary.py`: source guards for `TileXRCcuBackend` ownership and no direct CCU fields on `TileXRComm`.
- Existing CCU tests under `tests/ccu`: update references from `TileXRComm` direct CCU methods to backend facade or `TILEXR_CCU_TESTING` hooks.

---

### Task 1: Public Collective Backend Options And `*Ex` API

**Files:**
- Modify: `src/include/tilexr_collectives.h`
- Modify: `src/collectives/host/tilexr_collectives.cpp`
- Create: `tests/collectives/unit/test_tilexr_collective_backend_options.cpp`
- Modify: `tests/collectives/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `enum TileXRCollectiveBackend`
  - `struct TileXRCollectiveOptions`
  - `TileXRAllGatherEx`, `TileXRAllToAllEx`, `TileXRAllReduceEx`, `TileXRReduceScatterEx`, `TileXRBroadcastEx`
  - `TileXRProfileProbeEx`
- Consumes:
  - Existing `TileXRAllGather`, `TileXRAllToAll`, `TileXRAllReduce`, `TileXRReduceScatter`, `TileXRBroadcast`, `TileXRProfileProbe`

- [ ] **Step 1: Write the failing public header test**

Create `tests/collectives/unit/test_tilexr_collective_backend_options.cpp`:

```cpp
#include "tilexr_collectives.h"

#include <cstdint>

namespace {

static_assert(TILEXR_COLLECTIVE_BACKEND_AUTO == 0, "AUTO must be zero for zero-initialized options");
static_assert(TILEXR_COLLECTIVE_BACKEND_AIV == 1, "AIV enum value changed");
static_assert(TILEXR_COLLECTIVE_BACKEND_UDMA == 2, "UDMA enum value changed");
static_assert(TILEXR_COLLECTIVE_BACKEND_CCU == 3, "CCU enum value changed");

int CheckFunctionPointers()
{
    TileXRCollectiveOptions options {};
    if (options.backend != TILEXR_COLLECTIVE_BACKEND_AUTO) {
        return 1;
    }

    auto allGather = &TileXRAllGatherEx;
    auto allToAll = &TileXRAllToAllEx;
    auto allReduce = &TileXRAllReduceEx;
    auto reduceScatter = &TileXRReduceScatterEx;
    auto broadcast = &TileXRBroadcastEx;
    auto profileProbe = &TileXRProfileProbeEx;

    (void)allGather;
    (void)allToAll;
    (void)allReduce;
    (void)reduceScatter;
    (void)broadcast;
    (void)profileProbe;
    return 0;
}

} // namespace

int main()
{
    return CheckFunctionPointers();
}
```

- [ ] **Step 2: Register the failing test target**

In `tests/collectives/CMakeLists.txt`, add near the other unit executables:

```cmake
add_executable(test_tilexr_collective_backend_options
    unit/test_tilexr_collective_backend_options.cpp
)
```

After the existing `target_link_libraries(test_tilexr_collectives_header_compile ...)` block, add:

```cmake
target_link_libraries(test_tilexr_collective_backend_options PRIVATE ${TILEXR_COLLECTIVES_TEST_TARGET})
```

Add `test_tilexr_collective_backend_options` to the `foreach(_tilexr_collectives_link_target ...)` list immediately
after `test_tilexr_collectives_header_compile`.

Add this test registration immediately after `add_test(NAME test_tilexr_collectives_header_compile ...)`:

```cmake
add_test(NAME test_tilexr_collective_backend_options COMMAND test_tilexr_collective_backend_options)
```

Add `test_tilexr_collective_backend_options` to the `install(TARGETS ...)` list immediately after
`test_tilexr_collectives_header_compile`.

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collective_backend_options -j$(nproc)
```

Expected: compile failure mentioning `TILEXR_COLLECTIVE_BACKEND_AUTO` or `TileXRAllGatherEx` is not declared.

- [ ] **Step 4: Add public enum, options, and `*Ex` declarations**

In `src/include/tilexr_collectives.h`, add inside `extern "C"` before the function declarations:

```cpp
enum TileXRCollectiveBackend {
    TILEXR_COLLECTIVE_BACKEND_AUTO = 0,
    TILEXR_COLLECTIVE_BACKEND_AIV = 1,
    TILEXR_COLLECTIVE_BACKEND_UDMA = 2,
    TILEXR_COLLECTIVE_BACKEND_CCU = 3,
};

struct TileXRCollectiveOptions {
    TileXRCollectiveBackend backend;
};
```

Then declare the `*Ex` variants:

```cpp
int TileXRAllGatherEx(void *sendBuf, void *recvBuf, int64_t sendCount,
                      TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                      aclrtStream stream, const TileXRCollectiveOptions *options);
int TileXRAllToAllEx(void *sendBuf, void *recvBuf, int64_t sendCount,
                     TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                     aclrtStream stream, const TileXRCollectiveOptions *options);
int TileXRAllReduceEx(void *sendBuf, void *recvBuf, int64_t count,
                      TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                      TileXRCommPtr comm, aclrtStream stream, const TileXRCollectiveOptions *options);
int TileXRReduceScatterEx(void *sendBuf, void *recvBuf, int64_t recvCount,
                          TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                          TileXRCommPtr comm, aclrtStream stream,
                          const TileXRCollectiveOptions *options);
int TileXRBroadcastEx(void *buf, int64_t count,
                      TileXR::TileXRDataType dataType, int root,
                      TileXRCommPtr comm, aclrtStream stream, const TileXRCollectiveOptions *options);
int TileXRProfileProbeEx(void *sendBuf, void *recvBuf, int64_t count,
                         TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                         aclrtStream stream, const TileXRCollectiveOptions *options);
```

- [ ] **Step 5: Add minimal `*Ex` implementations and old API forwarding**

In `src/collectives/host/tilexr_collectives.cpp`, add helper:

```cpp
TileXRCollectiveBackend SelectedBackend(const TileXRCollectiveOptions *options)
{
    return options == nullptr ? TILEXR_COLLECTIVE_BACKEND_AUTO : options->backend;
}
```

Rename each existing function body to the corresponding `*Ex` function and accept `const TileXRCollectiveOptions *options`. At the top of each `*Ex` after validation, call:

```cpp
const TileXRCollectiveBackend backend = SelectedBackend(options);
(void)backend;
```

Then make the old function forward to `nullptr` options:

```cpp
int TileXRAllGather(void *sendBuf, void *recvBuf, int64_t sendCount,
                    TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                    aclrtStream stream)
{
    return TileXRAllGatherEx(sendBuf, recvBuf, sendCount, dataType, comm, stream, nullptr);
}
```

Repeat the same forwarding pattern for all existing collective entry points.

- [ ] **Step 6: Run test to verify it passes**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collective_backend_options -j$(nproc)
ctest --test-dir build -R test_tilexr_collective_backend_options --output-on-failure
```

Expected: target builds and the test passes.

- [ ] **Step 7: Commit**

```bash
git add src/include/tilexr_collectives.h src/collectives/host/tilexr_collectives.cpp tests/collectives/CMakeLists.txt tests/collectives/unit/test_tilexr_collective_backend_options.cpp
git commit -m "feat: add collective backend options"
```

---

### Task 2: Backend Dispatch Semantics For `AUTO`, `AIV`, `UDMA`, And `CCU`

**Files:**
- Create: `src/collectives/host/collective_backend.h`
- Create: `src/collectives/host/collective_backend.cpp`
- Modify: `src/collectives/host/tilexr_collectives.cpp`
- Modify: `src/collectives/CMakeLists.txt`
- Modify: `tests/collectives/CMakeLists.txt`
- Modify: `tests/collectives/unit/test_tilexr_collective_backend_options.cpp`

**Interfaces:**
- Consumes:
  - `TileXRCollectiveBackend`
  - `TileXRCollectiveOptions`
  - Existing AIV launch path in `tilexr_collectives.cpp`
- Produces:
  - `TileXRCollectives::Host::CollectiveRequest`
  - `TileXRCollectives::Host::DispatchCollective`
  - `TileXRCollectives::Host::SetBackendTestState`
  - `TileXRCollectives::Host::ResetBackendTestState`

- [ ] **Step 1: Extend test with fake backend-state cases**

Add this include after `#include "tilexr_collectives.h"`:

```cpp
#include "collective_backend.h"
```

Append this function before `main()`:

```cpp

int CheckBackendDispatch()
{
    using TileXRCollectives::Host::BackendTestState;
    using TileXRCollectives::Host::CollectiveRequest;
    using TileXRCollectives::Host::DispatchCollective;
    using TileXRCollectives::Host::ResetBackendTestState;
    using TileXRCollectives::Host::SetBackendTestState;

    CollectiveRequest request {};
    request.type = TileXR::TileXRType::ALL_GATHER;
    request.sendBuf = reinterpret_cast<void*>(0x1000);
    request.recvBuf = reinterpret_cast<void*>(0x2000);
    request.count = 1;
    request.dataType = TileXR::TILEXR_DATA_TYPE_INT32;
    request.comm = reinterpret_cast<TileXRCommPtr>(0x3000);
    request.stream = nullptr;

    BackendTestState state {};
    state.aivReturn = TileXR::TILEXR_SUCCESS;
    state.udmaInitialized = false;
    state.ccuInitialized = false;
    SetBackendTestState(state);
    if (DispatchCollective(request, TILEXR_COLLECTIVE_BACKEND_AUTO) != TileXR::TILEXR_SUCCESS) {
        return 2;
    }

    state.udmaInitialized = true;
    state.udmaSupported = true;
    state.udmaReturn = TileXR::TILEXR_SUCCESS;
    SetBackendTestState(state);
    if (DispatchCollective(request, TILEXR_COLLECTIVE_BACKEND_UDMA) != TileXR::TILEXR_SUCCESS) {
        return 3;
    }

    state.udmaInitialized = false;
    SetBackendTestState(state);
    if (DispatchCollective(request, TILEXR_COLLECTIVE_BACKEND_UDMA) != TileXR::TILEXR_ERROR_NOT_INITIALIZED) {
        return 4;
    }

    state.ccuInitialized = false;
    SetBackendTestState(state);
    if (DispatchCollective(request, TILEXR_COLLECTIVE_BACKEND_CCU) != TileXR::TILEXR_ERROR_NOT_INITIALIZED) {
        return 5;
    }

    ResetBackendTestState();
    return 0;
}
```

Change `main()` to:

```cpp
int main()
{
    const int pointerRet = CheckFunctionPointers();
    if (pointerRet != 0) {
        return pointerRet;
    }
    return CheckBackendDispatch();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collective_backend_options -j$(nproc)
```

Expected: compile failure mentioning `collective_backend.h` is missing.

- [ ] **Step 3: Add backend dispatch interfaces**

Create `src/collectives/host/collective_backend.h`:

```cpp
#ifndef TILEXR_COLLECTIVES_HOST_COLLECTIVE_BACKEND_H
#define TILEXR_COLLECTIVES_HOST_COLLECTIVE_BACKEND_H

#include <cstdint>

#include "acl/acl_base.h"
#include "tilexr_collectives.h"
#include "tilexr_types.h"

namespace TileXRCollectives {
namespace Host {

struct CollectiveRequest {
    TileXR::TileXRType type = TileXR::TileXRType::ALL_GATHER;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    int64_t count = 0;
    TileXR::TileXRDataType dataType = TileXR::TILEXR_DATA_TYPE_RESERVED;
    TileXR::TileXRReduceOp reduceOp = TileXR::TILEXR_REDUCE_RESERVED;
    int root = 0;
    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
};

struct BackendTestState {
    bool enabled = false;
    bool udmaInitialized = false;
    bool udmaSupported = false;
    int udmaReturn = TileXR::TILEXR_ERROR_NOT_SUPPORT;
    bool ccuInitialized = false;
    bool ccuSupported = false;
    int ccuReturn = TileXR::TILEXR_ERROR_NOT_SUPPORT;
    int aivReturn = TileXR::TILEXR_SUCCESS;
};

int DispatchCollective(const CollectiveRequest &request, TileXRCollectiveBackend backend);
void SetBackendTestState(const BackendTestState &state);
void ResetBackendTestState();

} // namespace Host
} // namespace TileXRCollectives

#endif // TILEXR_COLLECTIVES_HOST_COLLECTIVE_BACKEND_H
```

- [ ] **Step 4: Add minimal dispatch implementation**

Create `src/collectives/host/collective_backend.cpp`:

```cpp
#include "collective_backend.h"

namespace TileXRCollectives {
namespace Host {
namespace {

BackendTestState g_testState {};

int DispatchAiv(const CollectiveRequest&)
{
    return g_testState.enabled ? g_testState.aivReturn : TileXR::TILEXR_SUCCESS;
}

int DispatchUdma(const CollectiveRequest&)
{
    if (!g_testState.enabled || !g_testState.udmaInitialized) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    return g_testState.udmaSupported ? g_testState.udmaReturn : TileXR::TILEXR_ERROR_NOT_SUPPORT;
}

int DispatchCcu(const CollectiveRequest&)
{
    if (!g_testState.enabled || !g_testState.ccuInitialized) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    return g_testState.ccuSupported ? g_testState.ccuReturn : TileXR::TILEXR_ERROR_NOT_SUPPORT;
}

} // namespace

int DispatchCollective(const CollectiveRequest &request, TileXRCollectiveBackend backend)
{
    if (request.comm == nullptr || request.sendBuf == nullptr || request.recvBuf == nullptr || request.count <= 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    switch (backend) {
        case TILEXR_COLLECTIVE_BACKEND_AIV:
            return DispatchAiv(request);
        case TILEXR_COLLECTIVE_BACKEND_UDMA:
            return DispatchUdma(request);
        case TILEXR_COLLECTIVE_BACKEND_CCU:
            return DispatchCcu(request);
        case TILEXR_COLLECTIVE_BACKEND_AUTO:
        default:
            if (g_testState.enabled && g_testState.ccuInitialized && g_testState.ccuSupported) {
                return DispatchCcu(request);
            }
            if (g_testState.enabled && g_testState.udmaInitialized && g_testState.udmaSupported) {
                return DispatchUdma(request);
            }
            return DispatchAiv(request);
    }
}

void SetBackendTestState(const BackendTestState &state)
{
    g_testState = state;
    g_testState.enabled = true;
}

void ResetBackendTestState()
{
    g_testState = BackendTestState {};
}

} // namespace Host
} // namespace TileXRCollectives
```

- [ ] **Step 5: Build/link the dispatch source**

In `src/collectives/CMakeLists.txt`, add `host/collective_backend.cpp` and `host/collective_backend.h` to the `tilexr-collectives` source list.

In `tests/collectives/CMakeLists.txt`, add a private include directory for the new test:

```cmake
target_include_directories(test_tilexr_collective_backend_options PRIVATE
    ${TILEXR_ROOT}/src/collectives/host
)
```

- [ ] **Step 6: Route `*Ex` calls through dispatch without changing AIV behavior**

In `tilexr_collectives.cpp`, include `collective_backend.h`.

For each `*Ex`, after existing validation and loopback handling, build a `CollectiveRequest` and call `DispatchCollective`. In this task, keep AIV launch by returning the existing launch result for `AIV` and `AUTO`; forced `UDMA`/`CCU` semantics are exercised by tests through fake dispatch state.

Use this local helper to keep the old AIV launch path explicit:

```cpp
bool UsesForcedNonAivBackend(TileXRCollectiveBackend backend)
{
    return backend == TILEXR_COLLECTIVE_BACKEND_UDMA || backend == TILEXR_COLLECTIVE_BACKEND_CCU;
}
```

In `TileXRAllGatherEx`, before the AIV launch:

```cpp
if (UsesForcedNonAivBackend(backend)) {
    TileXRCollectives::Host::CollectiveRequest request {};
    request.type = TileXR::TileXRType::ALL_GATHER;
    request.sendBuf = sendBuf;
    request.recvBuf = recvBuf;
    request.count = sendCount;
    request.dataType = dataType;
    request.comm = comm;
    request.stream = stream;
    return TileXRCollectives::Host::DispatchCollective(request, backend);
}
```

Repeat this pattern for the other collectives:

```cpp
// TileXRAllToAllEx
request.type = TileXR::TileXRType::ALL2ALL;
request.count = sendCount;

// TileXRAllReduceEx
request.type = TileXR::TileXRType::ALL_REDUCE;
request.count = count;
request.reduceOp = op;

// TileXRReduceScatterEx
request.type = TileXR::TileXRType::REDUCE_SCATTER;
request.count = recvCount;
request.reduceOp = op;

// TileXRBroadcastEx
request.type = TileXR::TileXRType::BROADCAST;
request.sendBuf = buf;
request.recvBuf = buf;
request.count = count;
request.root = root;

// TileXRProfileProbeEx
request.type = TileXR::TileXRType::PROFILE_PROBE;
request.count = count;
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collective_backend_options -j$(nproc)
ctest --test-dir build -R test_tilexr_collective_backend_options --output-on-failure
```

Expected: test passes.

- [ ] **Step 8: Commit**

```bash
git add src/collectives/host/collective_backend.h src/collectives/host/collective_backend.cpp src/collectives/host/tilexr_collectives.cpp src/collectives/CMakeLists.txt tests/collectives/CMakeLists.txt tests/collectives/unit/test_tilexr_collective_backend_options.cpp
git commit -m "feat: route collective backend selection"
```

---

### Task 3: Remove Direct CCU Public API From Installed Headers And `comm_wrap.cpp`

**Files:**
- Modify: `src/include/tilexr_api.h`
- Modify: `src/comm/comm_wrap.cpp`
- Modify: `tests/ccu/test_tilexr_ccu_public_comm_api.py`
- Delete: `tests/ccu/ccu_public_direct_api_compile_probe.c`
- Modify: `tests/ccu/test_tilexr_ccu_public_api_compile_probe.py`

**Interfaces:**
- Consumes:
  - Existing generic TileXR C API declarations.
- Produces:
  - A clean `tilexr_api.h`.
  - No direct CCU public wrapper implementation in `comm_wrap.cpp`.

- [ ] **Step 1: Rewrite public source-guard test**

Replace `tests/ccu/test_tilexr_ccu_public_comm_api.py` with:

```python
#!/usr/bin/env python3
#
# Copyright (c) 2026 TileXR Project
#

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_HEADERS = [
    REPO_ROOT / "src" / "include" / "tilexr_api.h",
    REPO_ROOT / "src" / "include" / "tilexr_types.h",
    REPO_ROOT / "src" / "include" / "tilexr_collectives.h",
]
CORE_API_HEADER = REPO_ROOT / "src" / "include" / "tilexr_api.h"
COMM_WRAP = REPO_ROOT / "src" / "comm" / "comm_wrap.cpp"


class TileXRCcuPublicCommApiTest(unittest.TestCase):
    def test_core_api_header_has_no_ccu_symbols(self):
        header = CORE_API_HEADER.read_text(encoding="utf-8")
        for needle in ["CCU", "Ccu", "DirectCcu", "TILEXR_DIRECT_CCU"]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, header)

    def test_installed_public_headers_do_not_expose_low_level_ccu_model(self):
        forbidden = [
            "TileXRDirectCcu",
            "PrepareDirectCcu",
            "SubmitPrepared",
            "Repository",
            "SQE",
            " XN",
            " CKE",
            "TaskInfo",
            "rtCCULaunch",
            "rtCcuTaskInfo_t",
            "hcomm",
            "hccl",
        ]
        for path in PUBLIC_HEADERS:
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                with self.subTest(path=path.name, needle=needle):
                    self.assertNotIn(needle, text)

    def test_collectives_header_only_exposes_high_level_backend_names(self):
        text = (REPO_ROOT / "src" / "include" / "tilexr_collectives.h").read_text(encoding="utf-8")
        for needle in [
            "TILEXR_COLLECTIVE_BACKEND_AUTO",
            "TILEXR_COLLECTIVE_BACKEND_AIV",
            "TILEXR_COLLECTIVE_BACKEND_UDMA",
            "TILEXR_COLLECTIVE_BACKEND_CCU",
        ]:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_comm_wrap_has_no_direct_ccu_public_bridge(self):
        wrapper = COMM_WRAP.read_text(encoding="utf-8")
        for needle in [
            "TileXRCommInitRankDirectCcuWithDomain",
            "TileXRCommPrepareDirectCcu",
            "TileXRCommPrepareDirectCcuMemoryCopy",
            "TileXRDirectCcuGetPreparedTask",
            "TileXRDirectCcuSubmitPrepared",
            "TileXRCommReadDirectCcuInstructions",
            "TileXRDirectCcuDestroyPrepared",
            "TileXRDirectCcuPreparedTasks",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, wrapper)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 tests/ccu/test_tilexr_ccu_public_comm_api.py
```

Expected: failures showing CCU symbols in `tilexr_api.h` and `comm_wrap.cpp`.

- [ ] **Step 3: Remove CCU declarations from `tilexr_api.h`**

Delete these declaration groups from `src/include/tilexr_api.h`:

```cpp
typedef void *TileXRDirectCcuPreparedTasksPtr;
#define TILEXR_DIRECT_CCU_REPORT_MESSAGE_BYTES 2048
...
int TileXRCommInitRankDirectCcuWithDomain(...);
...
int TileXRDirectCcuDestroyPrepared(TileXRDirectCcuPreparedTasksPtr prepared);
```

Keep `TileXRCommPtr`, UDMA/SDMA APIs, DFX APIs, and generic comm lifecycle APIs unchanged.

- [ ] **Step 4: Remove direct CCU bridge from `comm_wrap.cpp`**

Delete the anonymous-namespace direct CCU helpers and public wrapper functions:

```cpp
TileXRDirectCcuPreparedTasks
CopyDirectCcuMessage
FillPublicPrepareReport
FillPublicSubmitReport
FillPublicInstructionReadbackReport
RepositoryInstallWindowFromPublic
RepositoryInstallDataLenModeFromPublic
RepositoryMemoryAllocModeFromPublic
InstallOrderFromPublic
MakeDirectCcuOptions
PreparedHandle
MemoryCopyDirectionFromPublic
TileXRCommInitRankDirectCcuWithDomain
TileXRCommPrepareDirectCcu
TileXRCommPrepareDirectCcuMemoryCopy
TileXRDirectCcuGetPreparedTask
TileXRDirectCcuSubmitPrepared
TileXRDirectCcuSubmitPreparedTask
TileXRCommReadDirectCcuInstructions
TileXRDirectCcuCreatePreparedForTest
TileXRDirectCcuDestroyPrepared
```

After removal, `comm_wrap.cpp` should include no CCU internal headers and no direct CCU constants.

- [ ] **Step 5: Remove external direct CCU compile probe**

Delete `tests/ccu/ccu_public_direct_api_compile_probe.c`.

Replace `tests/ccu/test_tilexr_ccu_public_api_compile_probe.py` with a guard that asserts the deleted file is gone:

```python
#!/usr/bin/env python3
#
# Copyright (c) 2026 TileXR Project
#

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class TileXRCcuPublicApiCompileProbeTest(unittest.TestCase):
    def test_external_direct_ccu_public_probe_removed(self):
        self.assertFalse((REPO_ROOT / "tests" / "ccu" / "ccu_public_direct_api_compile_probe.c").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: Run source guards**

Run:

```bash
python3 tests/ccu/test_tilexr_ccu_public_comm_api.py
python3 tests/ccu/test_tilexr_ccu_public_api_compile_probe.py
```

Expected: both pass.

- [ ] **Step 7: Commit**

```bash
git add src/include/tilexr_api.h src/comm/comm_wrap.cpp tests/ccu/test_tilexr_ccu_public_comm_api.py tests/ccu/test_tilexr_ccu_public_api_compile_probe.py
git rm tests/ccu/ccu_public_direct_api_compile_probe.c
git commit -m "refactor: remove public direct CCU API"
```

---

### Task 4: Introduce `TileXRCcuBackend` Facade And Move CCU State Out Of `TileXRComm`

**Files:**
- Create: `src/comm/ccu/tilexr_ccu_backend.h`
- Create: `src/comm/ccu/tilexr_ccu_backend.cpp`
- Modify: `src/comm/tilexr_comm.h`
- Modify: `src/comm/tilexr_comm.cpp`
- Modify: `src/comm/CMakeLists.txt`
- Create: `tests/ccu/test_tilexr_ccu_backend_boundary.py`

**Interfaces:**
- Produces:
  - `class TileXRCcuBackend`
  - `struct TileXRCcuBackendOptions`
  - `TileXRComm::GetCcuBackendForCollectives()`
  - `TileXRComm::EnableCcuBackendForTest()`
- Consumes:
  - Existing CCU implementation headers under `src/comm/ccu`

- [ ] **Step 1: Write backend-boundary source guard**

Create `tests/ccu/test_tilexr_ccu_backend_boundary.py`:

```python
#!/usr/bin/env python3
#
# Copyright (c) 2026 TileXR Project
#

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMM_HEADER = REPO_ROOT / "src" / "comm" / "tilexr_comm.h"
BACKEND_HEADER = REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_backend.h"
BACKEND_SOURCE = REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_backend.cpp"


class TileXRCcuBackendBoundaryTest(unittest.TestCase):
    def test_backend_files_exist(self):
        self.assertTrue(BACKEND_HEADER.exists())
        self.assertTrue(BACKEND_SOURCE.exists())

    def test_tilexr_comm_header_owns_only_opaque_backend(self):
        header = COMM_HEADER.read_text(encoding="utf-8")
        self.assertIn("class TileXRCcuBackend;", header)
        self.assertIn("std::unique_ptr<TileXRCcuBackend> ccuBackend_", header)
        for needle in [
            "tilexr_ccu_direct_orchestrator.h",
            "tilexr_ccu_direct_runtime.h",
            "tilexr_ccu_lower_layer_plan_builder.h",
            "TileXRCcuDirectRuntime",
            "directCcuBasicInfo_",
            "directCcuLowerLayerPlan_",
            "directCcuVerifiedEndpointRoutes_",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, header)

    def test_backend_header_exposes_facade_not_public_c_api(self):
        header = BACKEND_HEADER.read_text(encoding="utf-8")
        self.assertIn("class TileXRCcuBackend", header)
        self.assertIn("struct TileXRCcuBackendOptions", header)
        self.assertIn("PrepareCollective", header)
        self.assertIn("SubmitCollective", header)
        for needle in [
            "TileXRDirectCcuPreparedTasksPtr",
            "TileXRCommPrepareDirectCcu",
            "TileXRDirectCcuSubmitPrepared",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, header)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 tests/ccu/test_tilexr_ccu_backend_boundary.py
```

Expected: failure because backend files do not exist and `TileXRComm` still exposes direct CCU state.

- [ ] **Step 3: Add backend facade header**

Create `src/comm/ccu/tilexr_ccu_backend.h`:

```cpp
#ifndef TILEXR_CCU_BACKEND_H
#define TILEXR_CCU_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "acl/acl_base.h"
#include "tilexr_types.h"

namespace TileXR {

class TileXRComm;
class TileXRCcuRuntimeSession;
class TileXRCcuCollectivePlanner;
class TileXRCcuExecutor;

struct TileXRCcuBackendOptions {
    int rank = 0;
    int rankSize = 0;
    int devId = 0;
    std::string uid;
    TileXRComm *comm = nullptr;
};

struct TileXRCcuCollectiveRequest {
    TileXRType type = TileXRType::ALL_GATHER;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    int64_t count = 0;
    TileXRDataType dataType = TILEXR_DATA_TYPE_RESERVED;
    TileXRReduceOp reduceOp = TILEXR_REDUCE_RESERVED;
    int root = 0;
    aclrtStream stream = nullptr;
};

struct TileXRCcuCollectivePlan {
    bool ready = false;
};

class TileXRCcuBackend {
public:
    TileXRCcuBackend();
    ~TileXRCcuBackend();

    TileXRCcuBackend(const TileXRCcuBackend&) = delete;
    TileXRCcuBackend& operator=(const TileXRCcuBackend&) = delete;

    int Init(const TileXRCcuBackendOptions &options);
    void Shutdown();
    bool Available() const;
    bool Supports(const TileXRCcuCollectiveRequest &request) const;
    int PrepareCollective(const TileXRCcuCollectiveRequest &request, TileXRCcuCollectivePlan *plan);
    int SubmitCollective(const TileXRCcuCollectivePlan &plan, aclrtStream stream);

private:
    TileXRCcuBackendOptions options_;
    bool initialized_ = false;
};

} // namespace TileXR

#endif // TILEXR_CCU_BACKEND_H
```

- [ ] **Step 4: Add minimal backend source**

Create `src/comm/ccu/tilexr_ccu_backend.cpp`:

```cpp
#include "ccu/tilexr_ccu_backend.h"

namespace TileXR {

TileXRCcuBackend::TileXRCcuBackend() = default;
TileXRCcuBackend::~TileXRCcuBackend()
{
    Shutdown();
}

int TileXRCcuBackend::Init(const TileXRCcuBackendOptions &options)
{
    options_ = options;
    initialized_ = true;
    return TILEXR_SUCCESS;
}

void TileXRCcuBackend::Shutdown()
{
    initialized_ = false;
}

bool TileXRCcuBackend::Available() const
{
    return initialized_;
}

bool TileXRCcuBackend::Supports(const TileXRCcuCollectiveRequest &request) const
{
    return initialized_ && request.type == TileXRType::ALL_GATHER;
}

int TileXRCcuBackend::PrepareCollective(const TileXRCcuCollectiveRequest &request, TileXRCcuCollectivePlan *plan)
{
    if (plan == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (!initialized_) {
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    if (!Supports(request)) {
        return TILEXR_ERROR_NOT_SUPPORT;
    }
    *plan = TileXRCcuCollectivePlan {};
    plan->ready = true;
    return TILEXR_SUCCESS;
}

int TileXRCcuBackend::SubmitCollective(const TileXRCcuCollectivePlan &plan, aclrtStream)
{
    if (!initialized_) {
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    return plan.ready ? TILEXR_SUCCESS : TILEXR_ERROR_PARA_CHECK_FAIL;
}

} // namespace TileXR
```

- [ ] **Step 5: Change `TileXRComm` ownership to opaque backend**

In `src/comm/tilexr_comm.h`, remove CCU private includes:

```cpp
#include "ccu/tilexr_ccu_direct_orchestrator.h"
#include "ccu/tilexr_ccu_direct_runtime.h"
#include "ccu/tilexr_ccu_lower_layer_plan_builder.h"
```

Add forward declaration in namespace `TileXR`:

```cpp
class TileXRCcuBackend;
```

Replace direct CCU public/private method declarations with:

```cpp
int InitCcuBackend();
TileXRCcuBackend *GetCcuBackendForCollectives();
const TileXRCcuBackend *GetCcuBackendForCollectives() const;
```

Replace all direct CCU member fields with:

```cpp
std::unique_ptr<TileXRCcuBackend> ccuBackend_;
```

- [ ] **Step 6: Wire `TileXRComm` implementation**

In `src/comm/tilexr_comm.cpp`, include:

```cpp
#include "ccu/tilexr_ccu_backend.h"
```

Add:

```cpp
int TileXRComm::InitCcuBackend()
{
    if (ccuBackend_ == nullptr) {
        ccuBackend_.reset(new (std::nothrow) TileXRCcuBackend());
        if (ccuBackend_ == nullptr) {
            return TILEXR_ERROR_INTERNAL;
        }
    }
    TileXRCcuBackendOptions options {};
    options.rank = rank_;
    options.rankSize = rankSize_;
    options.devId = devId_;
    options.uid = uid_;
    options.comm = this;
    return ccuBackend_->Init(options);
}

TileXRCcuBackend *TileXRComm::GetCcuBackendForCollectives()
{
    return ccuBackend_.get();
}

const TileXRCcuBackend *TileXRComm::GetCcuBackendForCollectives() const
{
    return ccuBackend_.get();
}
```

For this task, delete the old direct CCU methods from `TileXRComm` or move their bodies into `tilexr_ccu_backend.cpp` behind private helper functions. Keep `Init()` and `InitThread()` compiling by replacing direct CCU runtime init calls with `InitCcuBackend()` only when the new generic config says CCU is enabled.

- [ ] **Step 7: Add source to build**

In `src/comm/CMakeLists.txt`, add:

```cmake
        ccu/tilexr_ccu_backend.h
        ccu/tilexr_ccu_backend.cpp
```

- [ ] **Step 8: Run source guard and build**

Run:

```bash
python3 tests/ccu/test_tilexr_ccu_backend_boundary.py
source scripts/common_env.sh
cmake --build build --target tile-comm -j$(nproc)
```

Expected: source guard passes and `tile-comm` builds.

- [ ] **Step 9: Commit**

```bash
git add src/comm/ccu/tilexr_ccu_backend.h src/comm/ccu/tilexr_ccu_backend.cpp src/comm/tilexr_comm.h src/comm/tilexr_comm.cpp src/comm/CMakeLists.txt tests/ccu/test_tilexr_ccu_backend_boundary.py
git commit -m "refactor: introduce internal CCU backend"
```

---

### Task 5: Split CCU Backend Internals Into Runtime Session, Planner, And Executor

**Files:**
- Create: `src/comm/ccu/tilexr_ccu_runtime_session.h`
- Create: `src/comm/ccu/tilexr_ccu_runtime_session.cpp`
- Create: `src/comm/ccu/tilexr_ccu_collective_planner.h`
- Create: `src/comm/ccu/tilexr_ccu_collective_planner.cpp`
- Create: `src/comm/ccu/tilexr_ccu_executor.h`
- Create: `src/comm/ccu/tilexr_ccu_executor.cpp`
- Modify: `src/comm/ccu/tilexr_ccu_backend.h`
- Modify: `src/comm/ccu/tilexr_ccu_backend.cpp`
- Modify: `src/comm/CMakeLists.txt`
- Modify: `tests/ccu/test_tilexr_ccu_backend_boundary.py`

**Interfaces:**
- Consumes:
  - `TileXRCcuBackend`
  - existing direct CCU lower-level files.
- Produces:
  - `TileXRCcuRuntimeSession`
  - `TileXRCcuCollectivePlanner`
  - `TileXRCcuExecutor`

- [ ] **Step 1: Extend boundary test**

Append to `test_tilexr_ccu_backend_boundary.py`:

```python
    def test_backend_internals_are_split(self):
        expected = [
            REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_runtime_session.h",
            REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_runtime_session.cpp",
            REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_collective_planner.h",
            REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_collective_planner.cpp",
            REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_executor.h",
            REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_executor.cpp",
        ]
        for path in expected:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists())
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 tests/ccu/test_tilexr_ccu_backend_boundary.py
```

Expected: failures for missing split files.

- [ ] **Step 3: Add runtime session**

Create `src/comm/ccu/tilexr_ccu_runtime_session.h`:

```cpp
#ifndef TILEXR_CCU_RUNTIME_SESSION_H
#define TILEXR_CCU_RUNTIME_SESSION_H

#include "ccu/tilexr_ccu_backend.h"

namespace TileXR {

class TileXRCcuRuntimeSession {
public:
    int Init(const TileXRCcuBackendOptions &options);
    void Shutdown();
    bool Available() const;

private:
    bool initialized_ = false;
};

} // namespace TileXR

#endif // TILEXR_CCU_RUNTIME_SESSION_H
```

Create `src/comm/ccu/tilexr_ccu_runtime_session.cpp`:

```cpp
#include "ccu/tilexr_ccu_runtime_session.h"

namespace TileXR {

int TileXRCcuRuntimeSession::Init(const TileXRCcuBackendOptions&)
{
    initialized_ = true;
    return TILEXR_SUCCESS;
}

void TileXRCcuRuntimeSession::Shutdown()
{
    initialized_ = false;
}

bool TileXRCcuRuntimeSession::Available() const
{
    return initialized_;
}

} // namespace TileXR
```

- [ ] **Step 4: Add collective planner**

Create `src/comm/ccu/tilexr_ccu_collective_planner.h`:

```cpp
#ifndef TILEXR_CCU_COLLECTIVE_PLANNER_H
#define TILEXR_CCU_COLLECTIVE_PLANNER_H

#include "ccu/tilexr_ccu_backend.h"

namespace TileXR {

class TileXRCcuRuntimeSession;

class TileXRCcuCollectivePlanner {
public:
    bool Supports(const TileXRCcuRuntimeSession &session, const TileXRCcuCollectiveRequest &request) const;
    int Prepare(const TileXRCcuRuntimeSession &session,
                const TileXRCcuCollectiveRequest &request,
                TileXRCcuCollectivePlan *plan) const;
};

} // namespace TileXR

#endif // TILEXR_CCU_COLLECTIVE_PLANNER_H
```

Create `src/comm/ccu/tilexr_ccu_collective_planner.cpp`:

```cpp
#include "ccu/tilexr_ccu_collective_planner.h"

#include "ccu/tilexr_ccu_runtime_session.h"

namespace TileXR {

bool TileXRCcuCollectivePlanner::Supports(
    const TileXRCcuRuntimeSession &session,
    const TileXRCcuCollectiveRequest &request) const
{
    return session.Available() && request.type == TileXRType::ALL_GATHER;
}

int TileXRCcuCollectivePlanner::Prepare(
    const TileXRCcuRuntimeSession &session,
    const TileXRCcuCollectiveRequest &request,
    TileXRCcuCollectivePlan *plan) const
{
    if (plan == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (!session.Available()) {
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    if (!Supports(session, request)) {
        return TILEXR_ERROR_NOT_SUPPORT;
    }
    *plan = TileXRCcuCollectivePlan {};
    plan->ready = true;
    return TILEXR_SUCCESS;
}

} // namespace TileXR
```

- [ ] **Step 5: Add executor**

Create `src/comm/ccu/tilexr_ccu_executor.h`:

```cpp
#ifndef TILEXR_CCU_EXECUTOR_H
#define TILEXR_CCU_EXECUTOR_H

#include "acl/acl_base.h"
#include "ccu/tilexr_ccu_backend.h"

namespace TileXR {

class TileXRCcuRuntimeSession;

class TileXRCcuExecutor {
public:
    int Submit(const TileXRCcuRuntimeSession &session, const TileXRCcuCollectivePlan &plan, aclrtStream stream) const;
};

} // namespace TileXR

#endif // TILEXR_CCU_EXECUTOR_H
```

Create `src/comm/ccu/tilexr_ccu_executor.cpp`:

```cpp
#include "ccu/tilexr_ccu_executor.h"

#include "ccu/tilexr_ccu_runtime_session.h"

namespace TileXR {

int TileXRCcuExecutor::Submit(
    const TileXRCcuRuntimeSession &session,
    const TileXRCcuCollectivePlan &plan,
    aclrtStream)
{
    if (!session.Available()) {
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    return plan.ready ? TILEXR_SUCCESS : TILEXR_ERROR_PARA_CHECK_FAIL;
}

} // namespace TileXR
```

- [ ] **Step 6: Refactor backend facade to delegate**

In `tilexr_ccu_backend.h`, replace the `bool initialized_` member with:

```cpp
std::unique_ptr<TileXRCcuRuntimeSession> runtimeSession_;
std::unique_ptr<TileXRCcuCollectivePlanner> planner_;
std::unique_ptr<TileXRCcuExecutor> executor_;
```

In `tilexr_ccu_backend.cpp`, include the three new headers and change the methods:

```cpp
TileXRCcuBackend::TileXRCcuBackend()
    : runtimeSession_(new (std::nothrow) TileXRCcuRuntimeSession()),
      planner_(new (std::nothrow) TileXRCcuCollectivePlanner()),
      executor_(new (std::nothrow) TileXRCcuExecutor())
{
}

int TileXRCcuBackend::Init(const TileXRCcuBackendOptions &options)
{
    if (runtimeSession_ == nullptr || planner_ == nullptr || executor_ == nullptr) {
        return TILEXR_ERROR_INTERNAL;
    }
    options_ = options;
    return runtimeSession_->Init(options);
}

void TileXRCcuBackend::Shutdown()
{
    if (runtimeSession_ != nullptr) {
        runtimeSession_->Shutdown();
    }
}

bool TileXRCcuBackend::Available() const
{
    return runtimeSession_ != nullptr && runtimeSession_->Available();
}

bool TileXRCcuBackend::Supports(const TileXRCcuCollectiveRequest &request) const
{
    return runtimeSession_ != nullptr && planner_ != nullptr && planner_->Supports(*runtimeSession_, request);
}

int TileXRCcuBackend::PrepareCollective(const TileXRCcuCollectiveRequest &request, TileXRCcuCollectivePlan *plan)
{
    if (runtimeSession_ == nullptr || planner_ == nullptr) {
        return TILEXR_ERROR_INTERNAL;
    }
    return planner_->Prepare(*runtimeSession_, request, plan);
}

int TileXRCcuBackend::SubmitCollective(const TileXRCcuCollectivePlan &plan, aclrtStream stream)
{
    if (runtimeSession_ == nullptr || executor_ == nullptr) {
        return TILEXR_ERROR_INTERNAL;
    }
    return executor_->Submit(*runtimeSession_, plan, stream);
}
```

- [ ] **Step 7: Add files to CMake and run build**

In `src/comm/CMakeLists.txt`, add all six new files.

Run:

```bash
python3 tests/ccu/test_tilexr_ccu_backend_boundary.py
source scripts/common_env.sh
cmake --build build --target tile-comm -j$(nproc)
```

Expected: source guard passes and `tile-comm` builds.

- [ ] **Step 8: Move existing CCU logic into split classes**

Move existing code from `TileXRComm` and current CCU files into the split classes with these ownership rules:

```text
TileXRCcuRuntimeSession:
  InitDirectCcuRuntime
  RefreshDirectCcuBasicInfo
  RegisterCcuResourceRmaBuffer
  ExportRemoteCcuRmaBuffers
  lower-layer transport exchange

TileXRCcuCollectivePlanner:
  PrepareDirectCcuInstallAttempt
  PrepareDirectCcuMemoryCopyInstallAttempt only under TILEXR_CCU_TESTING
  FillDirectCcuLowerLayerPlanFromAllocation
  lower-layer install plan generation

TileXRCcuExecutor:
  TileXRCcuSubmitPreparedTasks
  runtime submit/report mapping
  instruction readback only under TILEXR_CCU_TESTING
```

After each moved method, run:

```bash
source scripts/common_env.sh
cmake --build build --target tile-comm -j$(nproc)
python3 tests/ccu/test_tilexr_ccu_backend_boundary.py
```

Expected: build and source guard continue to pass.

- [ ] **Step 9: Commit**

```bash
git add src/comm/ccu/tilexr_ccu_runtime_session.h src/comm/ccu/tilexr_ccu_runtime_session.cpp src/comm/ccu/tilexr_ccu_collective_planner.h src/comm/ccu/tilexr_ccu_collective_planner.cpp src/comm/ccu/tilexr_ccu_executor.h src/comm/ccu/tilexr_ccu_executor.cpp src/comm/ccu/tilexr_ccu_backend.h src/comm/ccu/tilexr_ccu_backend.cpp src/comm/CMakeLists.txt tests/ccu/test_tilexr_ccu_backend_boundary.py
git commit -m "refactor: split CCU backend internals"
```

---

### Task 6: Connect Forced CCU/UDMA Dispatch To Real Backends And Verify Guards

**Files:**
- Modify: `src/collectives/host/collective_backend.cpp`
- Modify: `src/collectives/host/tilexr_collectives.cpp`
- Modify: `src/comm/tilexr_comm.h`
- Modify: `src/comm/tilexr_comm.cpp`
- Modify: `tests/collectives/unit/test_tilexr_collective_backend_options.cpp`
- Modify: `tests/ccu/test_tilexr_ccu_public_comm_api.py`

**Interfaces:**
- Consumes:
  - `TileXRComm::GetCcuBackendForCollectives()`
  - `TileXRCcuBackend::PrepareCollective`
  - `TileXRCcuBackend::SubmitCollective`
- Produces:
  - Real forced `CCU` dispatch path.
  - Real forced `UDMA` error path until UDMA-backed collectives are implemented.

- [ ] **Step 1: Add test assertions for forced modes**

In `tests/collectives/unit/test_tilexr_collective_backend_options.cpp`, add cases:

```cpp
    state.ccuInitialized = true;
    state.ccuSupported = false;
    SetBackendTestState(state);
    if (DispatchCollective(request, TILEXR_COLLECTIVE_BACKEND_CCU) != TileXR::TILEXR_ERROR_NOT_SUPPORT) {
        return 6;
    }

    state.ccuSupported = true;
    state.ccuReturn = TileXR::TILEXR_SUCCESS;
    SetBackendTestState(state);
    if (DispatchCollective(request, TILEXR_COLLECTIVE_BACKEND_CCU) != TileXR::TILEXR_SUCCESS) {
        return 7;
    }
```

- [ ] **Step 2: Run test**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collective_backend_options -j$(nproc)
ctest --test-dir build -R test_tilexr_collective_backend_options --output-on-failure
```

Expected: test passes in fake-state mode.

- [ ] **Step 3: Implement real CCU dispatch when fake state is disabled**

In `collective_backend.cpp`, include `tilexr_comm.h` and `ccu/tilexr_ccu_backend.h`.

Update `DispatchCcu`:

```cpp
int DispatchCcu(const CollectiveRequest &request)
{
    if (g_testState.enabled) {
        if (!g_testState.ccuInitialized) {
            return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
        }
        return g_testState.ccuSupported ? g_testState.ccuReturn : TileXR::TILEXR_ERROR_NOT_SUPPORT;
    }

    auto *comm = static_cast<TileXR::TileXRComm*>(request.comm);
    if (comm == nullptr || comm->GetCcuBackendForCollectives() == nullptr) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    TileXR::TileXRCcuBackend *backend = comm->GetCcuBackendForCollectives();
    TileXR::TileXRCcuCollectiveRequest ccuRequest {};
    ccuRequest.type = request.type;
    ccuRequest.sendBuf = request.sendBuf;
    ccuRequest.recvBuf = request.recvBuf;
    ccuRequest.count = request.count;
    ccuRequest.dataType = request.dataType;
    ccuRequest.reduceOp = request.reduceOp;
    ccuRequest.root = request.root;
    ccuRequest.stream = request.stream;

    TileXR::TileXRCcuCollectivePlan plan {};
    const int prepareRet = backend->PrepareCollective(ccuRequest, &plan);
    if (prepareRet != TileXR::TILEXR_SUCCESS) {
        return prepareRet;
    }
    return backend->SubmitCollective(plan, request.stream);
}
```

- [ ] **Step 4: Keep UDMA forced mode explicit**

Until a UDMA-backed collective path exists, keep `DispatchUdma` returning:

```cpp
return TileXR::TILEXR_ERROR_NOT_SUPPORT;
```

when UDMA is initialized but no matching collective backend exists. Do not route to AIV.

- [ ] **Step 5: Run focused tests and build**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target tilexr-collectives test_tilexr_collective_backend_options -j$(nproc)
ctest --test-dir build -R test_tilexr_collective_backend_options --output-on-failure
python3 tests/ccu/test_tilexr_ccu_public_comm_api.py
python3 tests/ccu/test_tilexr_ccu_backend_boundary.py
```

Expected: build passes and all listed tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/collectives/host/collective_backend.cpp src/collectives/host/tilexr_collectives.cpp src/comm/tilexr_comm.h src/comm/tilexr_comm.cpp tests/collectives/unit/test_tilexr_collective_backend_options.cpp tests/ccu/test_tilexr_ccu_public_comm_api.py
git commit -m "feat: connect CCU collective backend dispatch"
```

---

### Task 7: Final Verification

**Files:**
- No new files.
- Verify all files touched in Tasks 1-6.

**Interfaces:**
- Consumes all previous task outputs.
- Produces verified refactor state.

- [ ] **Step 1: Run public surface guards**

Run:

```bash
rg -n "CCU|Ccu|DirectCcu|TILEXR_DIRECT_CCU" src/include/tilexr_api.h
```

Expected: no output and exit code `1`.

Run:

```bash
python3 tests/ccu/test_tilexr_ccu_public_comm_api.py
python3 tests/ccu/test_tilexr_ccu_public_api_compile_probe.py
python3 tests/ccu/test_tilexr_ccu_backend_boundary.py
```

Expected: all pass.

- [ ] **Step 2: Run collective backend tests**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collective_backend_options test_tilexr_collectives_header_compile test_tilexr_collectives_api -j$(nproc)
ctest --test-dir build -R "test_tilexr_collective_backend_options|test_tilexr_collectives_header_compile|test_tilexr_collectives_api" --output-on-failure
```

Expected: all listed tests pass.

- [ ] **Step 3: Build core libraries**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target tile-comm tilexr-collectives -j$(nproc)
```

Expected: both targets build.

- [ ] **Step 4: Inspect remaining CCU leakage**

Run:

```bash
rg -n "TileXRDirectCcu|PrepareDirectCcu|SubmitPrepared|TILEXR_DIRECT_CCU|TileXRCommInitRankDirectCcu" src tests
```

Expected: matches only in intentionally retained internal CCU implementation tests guarded by `TILEXR_CCU_TESTING`, or no matches after test migration. No matches in installed public headers or `comm_wrap.cpp`.

- [ ] **Step 5: Confirm no verification-only changes remain**

Run:

```bash
git status --short
```

Expected: no output. If this command lists files, inspect them with `git diff` and either commit the intentional fix
with the exact files shown by `git status --short`, or revert generated artifacts that are not source changes.
