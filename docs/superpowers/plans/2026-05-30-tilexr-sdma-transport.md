# TileXR SDMA Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-class TileXR SDMA local GM-to-GM transport with explicit enablement, host query APIs, device raw pointer wrappers, tests, docs, and CANN 9.0/9.1 validation.

**Architecture:** Add `src/comm/sdma` beside the existing UDMA transport and expose SDMA metadata through `CommArgs`. Host code owns PTO `SdmaWorkspaceManager`; device code talks through `tilexr_sdma.h` and a `tilexr_sdma_compat.h` shim that isolates CANN PTO differences. The feature is disabled by default and only initializes when `TILEXR_ENABLE_SDMA=1`.

**Tech Stack:** C++14 host runtime, Ascend C/BiSheng device kernels, CMake, ACL runtime, CANN PTO SDMA headers, existing TileXR C API and test style.

---

## File Structure

Create or modify these files:

- `src/include/tilexr_sdma_types.h`: SDMA constants and small status enum used by host/tests/device headers.
- `src/include/tilexr_sdma_compat.h`: only TileXR public header that includes PTO SDMA intrinsics.
- `src/include/tilexr_sdma.h`: device-side public raw pointer SDMA API.
- `src/include/comm_args.h`: add `ExtraFlag::SDMA` and `CommArgs::sdmaWorkspacePtr`.
- `src/include/tilexr_api.h`: add `TileXRSDMAAvailable` and `TileXRGetSDMAWorkspaceDev`.
- `src/comm/sdma/tilexr_sdma_transport.h`: host transport interface and state.
- `src/comm/sdma/tilexr_sdma_transport.cpp`: env gating, PTO workspace manager initialization, status tracking.
- `src/comm/tilexr_comm.h`: own `TileXRSDMATransport` and expose internal SDMA query helpers.
- `src/comm/tilexr_comm.cpp`: call `InitSDMA()`, synchronize `CommArgs`, print DFX.
- `src/comm/comm_wrap.cpp`: implement public host query APIs.
- `src/comm/CMakeLists.txt`: compile SDMA transport and install SDMA headers.
- `tests/sdma/CMakeLists.txt`: SDMA unit/integration/demo build.
- `tests/sdma/build.sh`: build root `tile-comm` and SDMA tests against a selected CANN install.
- `tests/sdma/run_tests.sh`: run unit tests and optional integration tests.
- `tests/sdma/unit/*.cpp`: source guards, metadata/API tests, transport disabled tests.
- `tests/sdma/integration/test_tilexr_sdma_disabled_comm.cpp`: communicator disabled fallback test.
- `tests/sdma/demo/tilexr_sdma_demo.cpp`: host data-plane demo.
- `tests/sdma/demo/tilexr_sdma_demo_kernel.cpp`: AI Core SDMA copy demo kernel.
- `tests/sdma/demo/run_tilexr_sdma_demo.sh`: run demo for selected CANN/device/sizes.
- `docs/SDMA_TRANSPORT.md`: user-facing feature doc and validation guide.

Keep `3rdparty/shmem` as reference-only. Do not add shmem includes or shmem link dependencies to `src/comm`.

---

### Task 1: SDMA Metadata, Test Harness, And Install Surface

**Files:**
- Create: `src/include/tilexr_sdma_types.h`
- Modify: `src/include/comm_args.h`
- Modify: `src/comm/CMakeLists.txt`
- Create: `tests/sdma/CMakeLists.txt`
- Create: `tests/sdma/build.sh`
- Create: `tests/sdma/run_tests.sh`
- Create: `tests/sdma/unit/test_tilexr_sdma_metadata.cpp`

- [ ] **Step 1: Write the failing metadata test and SDMA test harness**

Create `tests/sdma/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(TileXR_SDMA_Tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(ARCH $ENV{ARCH})
if(NOT ARCH)
    set(ARCH $ENV{TILEXR_OS_ARCH})
endif()
if(NOT ARCH)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(ARCH STREQUAL "arm64")
        set(ARCH "aarch64")
    endif()
endif()

set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
if(NOT ASCEND_DRIVER_PATH)
    set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver")
endif()

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

find_library(TILEXR_LIB tile-comm
    HINTS "${TILEXR_ROOT}/install/lib" "${TILEXR_ROOT}/build/src/comm"
    REQUIRED)

include_directories(
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${TILEXR_ROOT}/src/include
)

link_directories(
    ${ASCEND_DRIVER_PATH}/lib64/driver
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
    ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)

add_executable(test_tilexr_sdma_metadata
    unit/test_tilexr_sdma_metadata.cpp
)
target_include_directories(test_tilexr_sdma_metadata PRIVATE
    ${TILEXR_ROOT}/src/include
)

set(INSTALL_TARGETS
    test_tilexr_sdma_metadata
)

install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
```

Create `tests/sdma/build.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TILEXR_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CANN_HOME="${1:-${ASCEND_HOME_PATH:-}}"

if [ -z "${CANN_HOME}" ]; then
    source "${TILEXR_ROOT}/scripts/common_env.sh"
else
    export ASCEND_HOME_PATH="${CANN_HOME}"
    export ASCEND_TOOLKIT_HOME="${CANN_HOME}"
    export ASCEND_OPP_PATH="${CANN_HOME}/opp"
    if [ -f "${CANN_HOME}/set_env.sh" ]; then
        set +u
        source "${CANN_HOME}/set_env.sh"
        set -u
    fi
fi

export ARCH="${ARCH:-${TILEXR_OS_ARCH:-$(uname -m)}}"
if [ "${ARCH}" = "arm64" ]; then
    export ARCH="aarch64"
fi
export ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"

ROOT_BUILD="${TILEXR_ROOT}/build-sdma-tests"
ROOT_INSTALL="${TILEXR_ROOT}/install"
TEST_BUILD="${SCRIPT_DIR}/build"
TEST_INSTALL="${SCRIPT_DIR}/install"

rm -rf "${ROOT_BUILD}" "${TEST_BUILD}" "${TEST_INSTALL}"
mkdir -p "${ROOT_BUILD}" "${TEST_BUILD}" "${TEST_INSTALL}"

cmake -S "${TILEXR_ROOT}" -B "${ROOT_BUILD}" \
    -DCMAKE_INSTALL_PREFIX="${ROOT_INSTALL}" \
    -DTILEXR_BUILD_TESTS=OFF
cmake --build "${ROOT_BUILD}" --target install -j"$(nproc)"

cmake -S "${SCRIPT_DIR}" -B "${TEST_BUILD}" \
    -DCMAKE_INSTALL_PREFIX="${TEST_INSTALL}"
cmake --build "${TEST_BUILD}" --target install -j"$(nproc)"

echo "SDMA tests installed to ${TEST_INSTALL}/bin"
```

Create `tests/sdma/run_tests.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALL_DIR="${SCRIPT_DIR}/install"
TILEXR_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TILEXR_ROOT}/install/lib:${LD_LIBRARY_PATH:-}"

"${INSTALL_DIR}/bin/test_tilexr_sdma_metadata"

echo "TileXR SDMA unit tests passed"
```

Create `tests/sdma/unit/test_tilexr_sdma_metadata.cpp`:

```cpp
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "comm_args.h"
#include "tilexr_sdma_types.h"

namespace {

int g_failures = 0;

#define CHECK_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " #expr << std::endl; \
            ++g_failures; \
        } \
    } while (0)

#define CHECK_EQ(lhs, rhs) \
    do { \
        auto lhsValue = (lhs); \
        auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs \
                      << " (" << lhsValue << " vs " << rhsValue << ")" << std::endl; \
            ++g_failures; \
        } \
    } while (0)

void TestSdmaFlagDoesNotOverlapExistingFlags()
{
    constexpr uint32_t sdma = TileXR::ExtraFlag::SDMA;
    CHECK_EQ(sdma, static_cast<uint32_t>(1U << 11));
    CHECK_EQ(sdma & TileXR::ExtraFlag::UDMA, 0U);
    CHECK_EQ(sdma & TileXR::ExtraFlag::RDMA, 0U);
    CHECK_EQ(sdma & TileXR::ExtraFlag::TOPO_PCIE, 0U);
}

void TestCommArgsHasSdmaWorkspace()
{
    TileXR::CommArgs args {};
    CHECK_TRUE(args.sdmaWorkspacePtr == nullptr);
    CHECK_TRUE(offsetof(TileXR::CommArgs, sdmaWorkspacePtr) > offsetof(TileXR::CommArgs, udmaRegistryPtr));
}

void TestSdmaConstants()
{
    CHECK_EQ(TileXR::TILEXR_SDMA_DEFAULT_BLOCK_BYTES, static_cast<uint64_t>(1024 * 1024));
    CHECK_EQ(TileXR::TILEXR_SDMA_DEFAULT_QUEUE_NUM, 1U);
    CHECK_EQ(TileXR::TILEXR_SDMA_SCRATCH_BYTES, 256U);
}

} // namespace

int main()
{
    TestSdmaFlagDoesNotOverlapExistingFlags();
    TestCommArgsHasSdmaWorkspace();
    TestSdmaConstants();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA metadata checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA metadata checks passed" << std::endl;
    return 0;
}
```

Make scripts executable:

```bash
chmod +x tests/sdma/build.sh tests/sdma/run_tests.sh
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
bash tests/sdma/build.sh
```

Expected: build fails because `tilexr_sdma_types.h`, `TileXR::ExtraFlag::SDMA`, and `CommArgs::sdmaWorkspacePtr` do not exist.

- [ ] **Step 3: Add SDMA metadata**

Create `src/include/tilexr_sdma_types.h`:

```cpp
/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_SDMA_TYPES_H
#define TILEXR_SDMA_TYPES_H

#include <cstdint>

namespace TileXR {

constexpr uint32_t TILEXR_SDMA_SCRATCH_BYTES = 256U;
constexpr uint64_t TILEXR_SDMA_DEFAULT_BLOCK_BYTES = 1024ULL * 1024ULL;
constexpr uint64_t TILEXR_SDMA_DEFAULT_COMM_BLOCK_OFFSET = 0ULL;
constexpr uint32_t TILEXR_SDMA_DEFAULT_QUEUE_NUM = 1U;
constexpr uint32_t TILEXR_SDMA_AUTO_CHANNEL_GROUP = 0xFFFFFFFFU;
constexpr int32_t TILEXR_SDMA_DEMO_MAGIC = 0x53444D41; // "SDMA"

enum class SDMAInitStatus : int32_t {
    DISABLED_BY_ENV = 0,
    INITIALIZED = 1,
    PTO_UNAVAILABLE = 2,
    INIT_FAILED = 3,
    NULL_WORKSPACE = 4,
};

} // namespace TileXR

#endif // TILEXR_SDMA_TYPES_H
```

Modify `src/include/comm_args.h`:

```cpp
#include "tilexr_sdma_types.h"
```

Add the flag after `UDMA`:

```cpp
static constexpr uint32_t SDMA = 1 << 11;
```

Add the pointer after `udmaRegistryPtr`:

```cpp
GM_ADDR sdmaWorkspacePtr = nullptr;  // device-side SDMA workspace; nullptr 表示 SDMA 不可用
```

Modify `src/comm/CMakeLists.txt` install headers list:

```cmake
${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_sdma_types.h
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
bash tests/sdma/build.sh
bash tests/sdma/run_tests.sh
```

Expected:

```text
TileXR SDMA metadata checks passed
TileXR SDMA unit tests passed
```

- [ ] **Step 5: Commit**

```bash
git add src/include/comm_args.h src/include/tilexr_sdma_types.h src/comm/CMakeLists.txt tests/sdma
git commit -m "feat: add sdma metadata surface"
```

---

### Task 2: Host Query API Declarations And Null-Argument Behavior

**Files:**
- Modify: `src/include/tilexr_api.h`
- Modify: `src/comm/comm_wrap.cpp`
- Modify: `tests/sdma/CMakeLists.txt`
- Create: `tests/sdma/unit/test_tilexr_sdma_api_invalid.cpp`

- [ ] **Step 1: Write the failing host API test**

Create `tests/sdma/unit/test_tilexr_sdma_api_invalid.cpp`:

```cpp
#include <iostream>

#include "tilexr_api.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

#define CHECK_EQ(lhs, rhs) \
    do { \
        auto lhsValue = (lhs); \
        auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs \
                      << " (" << lhsValue << " vs " << rhsValue << ")" << std::endl; \
            ++g_failures; \
        } \
    } while (0)

void TestAvailableRejectsInvalidArgs()
{
    bool available = true;
    CHECK_EQ(TileXRSDMAAvailable(nullptr, &available), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CHECK_EQ(TileXRSDMAAvailable(reinterpret_cast<TileXRCommPtr>(0x1), nullptr),
             TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

void TestWorkspaceRejectsInvalidArgs()
{
    GM_ADDR workspace = reinterpret_cast<GM_ADDR>(0x1234);
    CHECK_EQ(TileXRGetSDMAWorkspaceDev(nullptr, &workspace), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CHECK_EQ(TileXRGetSDMAWorkspaceDev(reinterpret_cast<TileXRCommPtr>(0x1), nullptr),
             TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

int main()
{
    TestAvailableRejectsInvalidArgs();
    TestWorkspaceRejectsInvalidArgs();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA API invalid-argument checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA API invalid-argument checks passed" << std::endl;
    return 0;
}
```

Modify `tests/sdma/CMakeLists.txt`:

```cmake
add_executable(test_tilexr_sdma_api_invalid
    unit/test_tilexr_sdma_api_invalid.cpp
)
target_link_libraries(test_tilexr_sdma_api_invalid
    ${TILEXR_LIB}
    ascendcl
    runtime
    ascend_hal
)
list(APPEND INSTALL_TARGETS test_tilexr_sdma_api_invalid)
```

Modify `tests/sdma/run_tests.sh`:

```bash
"${INSTALL_DIR}/bin/test_tilexr_sdma_api_invalid"
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
bash tests/sdma/build.sh
```

Expected: compile fails because `TileXRSDMAAvailable` and `TileXRGetSDMAWorkspaceDev` are not declared.

- [ ] **Step 3: Add API declarations and null-argument wrapper implementation**

Modify `src/include/tilexr_api.h` after `TileXRGetUDMARegistryDev`:

```cpp
int TileXRSDMAAvailable(TileXRCommPtr comm, bool *available);

int TileXRGetSDMAWorkspaceDev(TileXRCommPtr comm, GM_ADDR *workspace);
```

Modify `src/comm/comm_wrap.cpp` after `TileXRGetUDMARegistryDev`:

```cpp
int TileXRSDMAAvailable(TileXRCommPtr comm, bool *available)
{
    if (comm == nullptr || available == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRSDMAAvailable invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *available = false;
    return TILEXR_SUCCESS;
}

int TileXRGetSDMAWorkspaceDev(TileXRCommPtr comm, GM_ADDR *workspace)
{
    if (comm == nullptr || workspace == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRGetSDMAWorkspaceDev invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *workspace = nullptr;
    return TILEXR_SUCCESS;
}
```

This temporary non-null behavior is replaced in Task 4 after `TileXRComm` owns the SDMA transport state.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
bash tests/sdma/build.sh
bash tests/sdma/run_tests.sh
```

Expected includes:

```text
TileXR SDMA API invalid-argument checks passed
TileXR SDMA unit tests passed
```

- [ ] **Step 5: Commit**

```bash
git add src/include/tilexr_api.h src/comm/comm_wrap.cpp tests/sdma
git commit -m "feat: add sdma host query api surface"
```

---

### Task 3: Host SDMA Transport Disabled Path

**Files:**
- Create: `src/comm/sdma/tilexr_sdma_transport.h`
- Create: `src/comm/sdma/tilexr_sdma_transport.cpp`
- Modify: `src/comm/CMakeLists.txt`
- Modify: `tests/sdma/CMakeLists.txt`
- Create: `tests/sdma/unit/test_tilexr_sdma_transport_disabled.cpp`

- [ ] **Step 1: Write the failing disabled-transport test**

Create `tests/sdma/unit/test_tilexr_sdma_transport_disabled.cpp`:

```cpp
#include <cstdlib>
#include <iostream>

#include "sdma/tilexr_sdma_transport.h"
#include "tilexr_sdma_types.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

#define CHECK_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " #expr << std::endl; \
            ++g_failures; \
        } \
    } while (0)

#define CHECK_EQ(lhs, rhs) \
    do { \
        auto lhsValue = (lhs); \
        auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs \
                      << " (" << static_cast<int>(lhsValue) << " vs " << static_cast<int>(rhsValue) << ")" \
                      << std::endl; \
            ++g_failures; \
        } \
    } while (0)

void TestEnvDisabledSkipsInitialization()
{
    unsetenv("TILEXR_ENABLE_SDMA");
    TileXR::TileXRSDMATransport transport;
    TileXR::TileXRSDMATransportOptions options {};
    options.devId = 0;
    CHECK_EQ(transport.Init(options), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!transport.IsAvailable());
    CHECK_TRUE(transport.GetWorkspaceDev() == nullptr);
    CHECK_EQ(transport.GetLastStatus(), TileXR::SDMAInitStatus::DISABLED_BY_ENV);
}

void TestEnvZeroSkipsInitialization()
{
    setenv("TILEXR_ENABLE_SDMA", "0", 1);
    TileXR::TileXRSDMATransport transport;
    TileXR::TileXRSDMATransportOptions options {};
    options.devId = 0;
    CHECK_EQ(transport.Init(options), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!transport.IsAvailable());
    CHECK_TRUE(transport.GetWorkspaceDev() == nullptr);
    CHECK_EQ(transport.GetLastStatus(), TileXR::SDMAInitStatus::DISABLED_BY_ENV);
    unsetenv("TILEXR_ENABLE_SDMA");
}

} // namespace

int main()
{
    TestEnvDisabledSkipsInitialization();
    TestEnvZeroSkipsInitialization();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA transport disabled checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA transport disabled checks passed" << std::endl;
    return 0;
}
```

Modify `tests/sdma/CMakeLists.txt`:

```cmake
add_executable(test_tilexr_sdma_transport_disabled
    unit/test_tilexr_sdma_transport_disabled.cpp
    ${TILEXR_ROOT}/src/comm/sdma/tilexr_sdma_transport.cpp
)
target_include_directories(test_tilexr_sdma_transport_disabled PRIVATE
    ${TILEXR_ROOT}/src/comm
    ${TILEXR_ROOT}/src/include
)
target_link_libraries(test_tilexr_sdma_transport_disabled
    ascendcl
    runtime
    ascend_hal
)
list(APPEND INSTALL_TARGETS test_tilexr_sdma_transport_disabled)
```

Modify `tests/sdma/run_tests.sh`:

```bash
"${INSTALL_DIR}/bin/test_tilexr_sdma_transport_disabled"
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
bash tests/sdma/build.sh
```

Expected: compile fails because `sdma/tilexr_sdma_transport.h` does not exist.

- [ ] **Step 3: Implement disabled-path transport**

Create `src/comm/sdma/tilexr_sdma_transport.h`:

```cpp
/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_SDMA_TRANSPORT_H
#define TILEXR_SDMA_TRANSPORT_H

#include "comm_args.h"
#include "tilexr_sdma_types.h"

namespace TileXR {

struct TileXRSDMATransportOptions {
    int devId = 0;
};

class TileXRSDMATransport {
public:
    TileXRSDMATransport() = default;
    ~TileXRSDMATransport();
    TileXRSDMATransport(const TileXRSDMATransport&) = delete;
    TileXRSDMATransport& operator=(const TileXRSDMATransport&) = delete;

    int Init(const TileXRSDMATransportOptions& options);
    void Shutdown();

    bool IsAvailable() const;
    GM_ADDR GetWorkspaceDev() const;
    SDMAInitStatus GetLastStatus() const;

private:
    static bool EnvEnabled();

    TileXRSDMATransportOptions options_ {};
    bool available_ = false;
    GM_ADDR workspaceDev_ = nullptr;
    SDMAInitStatus lastStatus_ = SDMAInitStatus::DISABLED_BY_ENV;
};

} // namespace TileXR

#endif // TILEXR_SDMA_TRANSPORT_H
```

Create `src/comm/sdma/tilexr_sdma_transport.cpp`:

```cpp
/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "sdma/tilexr_sdma_transport.h"

#include <cstdlib>
#include <string>

#include "tilexr_log.h"
#include "tilexr_types.h"

namespace TileXR {

TileXRSDMATransport::~TileXRSDMATransport()
{
    Shutdown();
}

bool TileXRSDMATransport::EnvEnabled()
{
    const char* value = std::getenv("TILEXR_ENABLE_SDMA");
    if (value == nullptr) {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

int TileXRSDMATransport::Init(const TileXRSDMATransportOptions& options)
{
    options_ = options;
    available_ = false;
    workspaceDev_ = nullptr;

    if (!EnvEnabled()) {
        lastStatus_ = SDMAInitStatus::DISABLED_BY_ENV;
        TILEXR_LOG(INFO) << "TileXR SDMA disabled; set TILEXR_ENABLE_SDMA=1 to enable";
        return TILEXR_SUCCESS;
    }

    lastStatus_ = SDMAInitStatus::PTO_UNAVAILABLE;
    TILEXR_LOG(WARN) << "TileXR SDMA PTO support is not compiled in yet";
    return TILEXR_SUCCESS;
}

void TileXRSDMATransport::Shutdown()
{
    available_ = false;
    workspaceDev_ = nullptr;
}

bool TileXRSDMATransport::IsAvailable() const
{
    return available_;
}

GM_ADDR TileXRSDMATransport::GetWorkspaceDev() const
{
    return workspaceDev_;
}

SDMAInitStatus TileXRSDMATransport::GetLastStatus() const
{
    return lastStatus_;
}

} // namespace TileXR
```

Modify `src/comm/CMakeLists.txt` source list:

```cmake
sdma/tilexr_sdma_transport.h
sdma/tilexr_sdma_transport.cpp
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
bash tests/sdma/build.sh
bash tests/sdma/run_tests.sh
```

Expected includes:

```text
TileXR SDMA transport disabled checks passed
TileXR SDMA unit tests passed
```

- [ ] **Step 5: Commit**

```bash
git add src/comm/CMakeLists.txt src/comm/sdma tests/sdma
git commit -m "feat: add disabled sdma transport"
```

---

### Task 4: Wire SDMA Transport Into TileXRComm And Public Queries

**Files:**
- Modify: `src/comm/tilexr_comm.h`
- Modify: `src/comm/tilexr_comm.cpp`
- Modify: `src/comm/comm_wrap.cpp`
- Modify: `tests/sdma/CMakeLists.txt`
- Create: `tests/sdma/unit/test_tilexr_sdma_comm_wiring.cpp`
- Create: `tests/sdma/integration/test_tilexr_sdma_disabled_comm.cpp`

- [ ] **Step 1: Write the failing comm wiring source test and disabled communicator integration test**

Create `tests/sdma/unit/test_tilexr_sdma_comm_wiring.cpp`:

```cpp
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int g_failures = 0;

std::string RepoPath(const std::string& path)
{
#ifdef TILEXR_SOURCE_ROOT
    return std::string(TILEXR_SOURCE_ROOT) + "/" + path;
#else
    return path;
#endif
}

std::string ReadFile(const std::string& path)
{
    std::ifstream input(RepoPath(path).c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << RepoPath(path) << std::endl;
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckContains(const std::string& path, const std::string& needle)
{
    const auto text = ReadFile(path);
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected text not found in " << path << ": " << needle << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    CheckContains("src/comm/tilexr_comm.h", "std::unique_ptr<TileXRSDMATransport> sdmaTransport_");
    CheckContains("src/comm/tilexr_comm.cpp", "int TileXRComm::InitSDMA()");
    CheckContains("src/comm/tilexr_comm.cpp", "commArgs_.sdmaWorkspacePtr = sdmaWorkspaceDev_");
    CheckContains("src/comm/tilexr_comm.cpp", "commArgs_.extraFlag |= ExtraFlag::SDMA");
    CheckContains("src/comm/comm_wrap.cpp", "*available = c->IsSDMAAvailable()");
    CheckContains("src/comm/comm_wrap.cpp", "*workspace = c->GetSDMAWorkspacePtr()");

    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA comm wiring checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA comm wiring checks passed" << std::endl;
    return 0;
}
```

Create `tests/sdma/integration/test_tilexr_sdma_disabled_comm.cpp`:

```cpp
#include <cstdlib>
#include <iostream>

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include "tilexr_api.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

#define CHECK_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " #expr << std::endl; \
            ++g_failures; \
        } \
    } while (0)

#define CHECK_EQ(lhs, rhs) \
    do { \
        auto lhsValue = (lhs); \
        auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs \
                      << " (" << lhsValue << " vs " << rhsValue << ")" << std::endl; \
            ++g_failures; \
        } \
    } while (0)

} // namespace

int main()
{
    unsetenv("TILEXR_ENABLE_SDMA");

    CHECK_EQ(aclInit(nullptr), ACL_SUCCESS);
    CHECK_EQ(aclrtSetDevice(0), ACL_SUCCESS);

    TileXRCommPtr comm = nullptr;
    CHECK_EQ(TileXRCommInitRankLocal(1, 0, &comm), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(comm != nullptr);

    bool available = true;
    GM_ADDR workspace = reinterpret_cast<GM_ADDR>(0x1234);
    CHECK_EQ(TileXRSDMAAvailable(comm, &available), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(TileXRGetSDMAWorkspaceDev(comm, &workspace), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!available);
    CHECK_TRUE(workspace == nullptr);

    TileXR::CommArgs* args = nullptr;
    CHECK_EQ(TileXRGetCommArgsHost(comm, args), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(args != nullptr);
    CHECK_TRUE((args->extraFlag & TileXR::ExtraFlag::SDMA) == 0);
    CHECK_TRUE(args->sdmaWorkspacePtr == nullptr);

    CHECK_EQ(TileXRCommDestroy(comm), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    CHECK_EQ(aclFinalize(), ACL_SUCCESS);

    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA disabled communicator checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA disabled communicator checks passed" << std::endl;
    return 0;
}
```

Modify `tests/sdma/CMakeLists.txt`:

```cmake
add_executable(test_tilexr_sdma_comm_wiring
    unit/test_tilexr_sdma_comm_wiring.cpp
)
target_compile_definitions(test_tilexr_sdma_comm_wiring PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)
list(APPEND INSTALL_TARGETS test_tilexr_sdma_comm_wiring)

add_executable(test_tilexr_sdma_disabled_comm
    integration/test_tilexr_sdma_disabled_comm.cpp
)
target_link_libraries(test_tilexr_sdma_disabled_comm
    ${TILEXR_LIB}
    ascendcl
    runtime
    ascend_hal
)
list(APPEND INSTALL_TARGETS test_tilexr_sdma_disabled_comm)
```

Modify `tests/sdma/run_tests.sh` to run this only when hardware is available:

```bash
"${INSTALL_DIR}/bin/test_tilexr_sdma_comm_wiring"

if command -v npu-smi >/dev/null 2>&1; then
    "${INSTALL_DIR}/bin/test_tilexr_sdma_disabled_comm"
else
    echo "Skip test_tilexr_sdma_disabled_comm: npu-smi not found"
fi
```

- [ ] **Step 2: Run the failing integration test**

Run:

```bash
bash tests/sdma/build.sh
bash tests/sdma/run_tests.sh
```

Expected: `test_tilexr_sdma_comm_wiring` fails because `TileXRComm` does not own SDMA state yet and public SDMA queries still return hardcoded disabled values. `test_tilexr_sdma_disabled_comm` may already pass before implementation because the temporary hardcoded query behavior is also disabled; use the source wiring test as the red test for this task.

- [ ] **Step 3: Add TileXRComm SDMA state and initialization**

Modify `src/comm/tilexr_comm.h`:

```cpp
class TileXRSDMATransport;
```

Add public methods:

```cpp
bool IsSDMAAvailable() const;
GM_ADDR GetSDMAWorkspacePtr() const;
SDMAInitStatus GetSDMAInitStatus() const;
```

Add private methods:

```cpp
int InitSDMA();
void ResetSDMAState();
```

Add private fields near UDMA fields:

```cpp
GM_ADDR sdmaWorkspaceDev_ = nullptr;
SDMAInitStatus sdmaInitStatus_ = SDMAInitStatus::DISABLED_BY_ENV;
std::unique_ptr<TileXRSDMATransport> sdmaTransport_;
```

Include the SDMA types header:

```cpp
#include "../include/tilexr_sdma_types.h"
```

Modify `src/comm/tilexr_comm.cpp` includes:

```cpp
#include "sdma/tilexr_sdma_transport.h"
```

Add process-level lock/cache near UDMA globals:

```cpp
static std::mutex g_sdmaMtx;
static bool g_sdmaUnavailable = false;
```

Add methods after `InitUDMA()`:

```cpp
int TileXRComm::InitSDMA()
{
    {
        lock_guard<mutex> lock(g_sdmaMtx);
        if (g_sdmaUnavailable) {
            TILEXR_LOG(INFO) << "InitSDMA skipped after previous SDMA init failure";
            sdmaInitStatus_ = SDMAInitStatus::PTO_UNAVAILABLE;
            return TILEXR_SUCCESS;
        }
    }

    sdmaTransport_.reset(new (nothrow) TileXRSDMATransport());
    if (sdmaTransport_ == nullptr) {
        TILEXR_LOG(WARN) << "TileXRSDMATransport allocation failed, SDMA disabled";
        sdmaInitStatus_ = SDMAInitStatus::INIT_FAILED;
        return TILEXR_SUCCESS;
    }

    TileXRSDMATransportOptions options {};
    options.devId = devId_;
    int ret = sdmaTransport_->Init(options);
    sdmaInitStatus_ = sdmaTransport_->GetLastStatus();
    if (ret != TILEXR_SUCCESS || !sdmaTransport_->IsAvailable()) {
        if (sdmaInitStatus_ != SDMAInitStatus::DISABLED_BY_ENV) {
            TILEXR_LOG(WARN) << "TileXR SDMA init unavailable, status " << static_cast<int>(sdmaInitStatus_);
            lock_guard<mutex> lock(g_sdmaMtx);
            g_sdmaUnavailable = true;
        }
        sdmaTransport_.reset();
        sdmaWorkspaceDev_ = nullptr;
        commArgs_.sdmaWorkspacePtr = nullptr;
        return TILEXR_SUCCESS;
    }

    sdmaWorkspaceDev_ = sdmaTransport_->GetWorkspaceDev();
    if (sdmaWorkspaceDev_ == nullptr) {
        TILEXR_LOG(WARN) << "TileXR SDMA workspace is null, SDMA disabled";
        sdmaInitStatus_ = SDMAInitStatus::NULL_WORKSPACE;
        sdmaTransport_.reset();
        return TILEXR_SUCCESS;
    }

    commArgs_.sdmaWorkspacePtr = sdmaWorkspaceDev_;
    commArgs_.extraFlag |= ExtraFlag::SDMA;
    sdmaInitStatus_ = SDMAInitStatus::INITIALIZED;
    TILEXR_LOG(INFO) << "InitSDMA success, workspace " << static_cast<void*>(sdmaWorkspaceDev_);
    return TILEXR_SUCCESS;
}

bool TileXRComm::IsSDMAAvailable() const
{
    return (commArgs_.extraFlag & ExtraFlag::SDMA) != 0 && commArgs_.sdmaWorkspacePtr != nullptr;
}

GM_ADDR TileXRComm::GetSDMAWorkspacePtr() const
{
    return sdmaWorkspaceDev_;
}

SDMAInitStatus TileXRComm::GetSDMAInitStatus() const
{
    return sdmaInitStatus_;
}
```

Call `InitSDMA()` in both `Init()` and `InitThread()` before `SyncCommArgs()`:

```cpp
ret = InitSDMA();
if (ret != TILEXR_SUCCESS) {
    return ret;
}
```

For `InitThread()`, declare `int ret = InitSDMA();` before checking it.

Extend `PrintDFX()` before the closing brace:

```cpp
ss << "\n  sdma: {"
   << " enabled: " << (((commArgs_.extraFlag & ExtraFlag::SDMA) != 0) ? "true" : "false")
   << ", status: " << static_cast<int>(sdmaInitStatus_)
   << ", workspace: " << static_cast<void*>(commArgs_.sdmaWorkspacePtr)
   << " }";
```

- [ ] **Step 4: Replace public API hardcoded behavior**

Modify `src/comm/comm_wrap.cpp`:

```cpp
int TileXRSDMAAvailable(TileXRCommPtr comm, bool *available)
{
    if (comm == nullptr || available == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRSDMAAvailable invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    *available = c->IsSDMAAvailable();
    return TILEXR_SUCCESS;
}

int TileXRGetSDMAWorkspaceDev(TileXRCommPtr comm, GM_ADDR *workspace)
{
    if (comm == nullptr || workspace == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRGetSDMAWorkspaceDev invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    *workspace = c->GetSDMAWorkspacePtr();
    return TILEXR_SUCCESS;
}
```

- [ ] **Step 5: Run tests**

Run:

```bash
bash tests/sdma/build.sh
bash tests/sdma/run_tests.sh
```

Expected includes:

```text
TileXR SDMA disabled communicator checks passed
TileXR SDMA unit tests passed
```

- [ ] **Step 6: Commit**

```bash
git add src/comm/tilexr_comm.h src/comm/tilexr_comm.cpp src/comm/comm_wrap.cpp tests/sdma
git commit -m "feat: wire sdma transport into comm"
```

---

### Task 5: PTO Workspace Manager Initialization And Runtime HAL Guard

**Files:**
- Create: `src/include/tilexr_sdma_compat.h`
- Modify: `src/comm/sdma/tilexr_sdma_transport.h`
- Modify: `src/comm/sdma/tilexr_sdma_transport.cpp`
- Modify: `src/comm/CMakeLists.txt`
- Modify: `tests/sdma/CMakeLists.txt`
- Create: `tests/sdma/unit/test_tilexr_sdma_source_guard.cpp`
- Create: `tests/sdma/check_runtime_deps.sh`

- [ ] **Step 1: Write source and runtime dependency guard tests**

Create `tests/sdma/unit/test_tilexr_sdma_source_guard.cpp`:

```cpp
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

std::string RepoPath(const std::string& path)
{
#ifdef TILEXR_SOURCE_ROOT
    return std::string(TILEXR_SOURCE_ROOT) + "/" + path;
#else
    return path;
#endif
}

std::string ReadFile(const std::string& path)
{
    std::ifstream input(RepoPath(path).c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << RepoPath(path) << std::endl;
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckNoNeedle(const std::string& path, const std::string& text, const std::string& needle)
{
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "unexpected dependency in " << path << ": " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void CheckNeedle(const std::string& path, const std::string& text, const std::string& needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected text not found in " << path << ": " << needle << std::endl;
        ++g_failures;
    }
}

void TestCommSourcesDoNotUseShmem()
{
    const std::vector<std::string> paths = {
        "src/comm/CMakeLists.txt",
        "src/comm/tilexr_comm.cpp",
        "src/comm/comm_wrap.cpp",
        "src/comm/tilexr_comm.h",
        "src/comm/sdma/tilexr_sdma_transport.cpp",
        "src/comm/sdma/tilexr_sdma_transport.h",
    };
    const std::vector<std::string> forbidden = {
        "shmem.h",
        "libshmem",
        "aclshmem",
        "ACLSHMEM",
    };
    for (const auto& path : paths) {
        const auto text = ReadFile(path);
        for (const auto& needle : forbidden) {
            CheckNoNeedle(path, text, needle);
        }
    }
}

void TestOnlyCompatIncludesSdmaIntrinsics()
{
    const auto compat = ReadFile("src/include/tilexr_sdma_compat.h");
    CheckNeedle("src/include/tilexr_sdma_compat.h", compat,
                "pto/npu/comm/async/sdma/sdma_async_intrin.hpp");
    const auto transport = ReadFile("src/comm/sdma/tilexr_sdma_transport.cpp");
    CheckNeedle("src/comm/sdma/tilexr_sdma_transport.cpp", transport,
                "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp");
    const std::vector<std::string> disallowed = {
        "src/include/tilexr_sdma.h",
        "src/include/comm_args.h",
        "src/comm/tilexr_comm.cpp",
    };
    for (const auto& path : disallowed) {
        const auto text = ReadFile(path);
        CheckNoNeedle(path, text, "pto/npu/comm/async/sdma/");
    }
}

} // namespace

int main()
{
    TestCommSourcesDoNotUseShmem();
    TestOnlyCompatIncludesSdmaIntrinsics();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA source guard checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA source guard checks passed" << std::endl;
    return 0;
}
```

Create `tests/sdma/check_runtime_deps.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

lib="${1:-install/lib/libtile-comm.so}"

if [ ! -f "${lib}" ]; then
    echo "ERROR: ${lib} not found"
    exit 1
fi

deps=$(ldd "${lib}")
echo "${deps}"

if echo "${deps}" | grep -E 'libascend_hal.so => .*devlib' >/dev/null; then
    echo "ERROR: libascend_hal.so resolved from CANN devlib; runtime must use driver HAL"
    exit 1
fi

if echo "${deps}" | grep -i 'shmem' >/dev/null; then
    echo "ERROR: tile-comm links shmem unexpectedly"
    exit 1
fi

echo "TileXR SDMA runtime dependency check passed"
```

Make it executable:

```bash
chmod +x tests/sdma/check_runtime_deps.sh
```

Modify `tests/sdma/CMakeLists.txt`:

```cmake
add_executable(test_tilexr_sdma_source_guard
    unit/test_tilexr_sdma_source_guard.cpp
)
target_compile_definitions(test_tilexr_sdma_source_guard PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)
list(APPEND INSTALL_TARGETS test_tilexr_sdma_source_guard)
```

Modify `tests/sdma/run_tests.sh`:

```bash
"${INSTALL_DIR}/bin/test_tilexr_sdma_source_guard"
bash "${SCRIPT_DIR}/check_runtime_deps.sh" "${TILEXR_ROOT}/install/lib/libtile-comm.so"
```

- [ ] **Step 2: Run the failing source guard**

Run:

```bash
bash tests/sdma/build.sh
bash tests/sdma/run_tests.sh
```

Expected: source guard fails because `tilexr_sdma_compat.h` does not exist yet and the transport does not include the PTO workspace manager header yet.

- [ ] **Step 3: Add CMake detection for PTO SDMA headers**

Modify `src/comm/CMakeLists.txt` before `add_library`:

```cmake
include(CheckIncludeFileCXX)
set(CMAKE_REQUIRED_INCLUDES
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
        ${ASCEND_DRIVER_PATH}/kernel/inc)
check_include_file_cxx("pto/npu/comm/async/sdma/sdma_workspace_manager.hpp" TILEXR_HAVE_PTO_SDMA_WORKSPACE)
check_include_file_cxx("pto/npu/comm/async/sdma/sdma_async_intrin.hpp" TILEXR_HAVE_PTO_SDMA_INTRIN)
if(TILEXR_HAVE_PTO_SDMA_WORKSPACE AND TILEXR_HAVE_PTO_SDMA_INTRIN)
    set(TILEXR_HAVE_PTO_SDMA ON)
else()
    set(TILEXR_HAVE_PTO_SDMA OFF)
endif()
message(STATUS "TILEXR_HAVE_PTO_SDMA: ${TILEXR_HAVE_PTO_SDMA}")
```

Add compile definition to `tile-comm`:

```cmake
if(TILEXR_HAVE_PTO_SDMA)
    target_compile_definitions(tile-comm PRIVATE TILEXR_HAVE_PTO_SDMA=1)
else()
    target_compile_definitions(tile-comm PRIVATE TILEXR_HAVE_PTO_SDMA=0)
endif()
```

Keep driver link directory before CANN devlib:

```cmake
target_link_directories(tile-comm
        PRIVATE
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
        ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)
```

Do not add devlib to any install RPATH.

- [ ] **Step 4: Add initial compatibility header and real PTO workspace manager support**

Create `src/include/tilexr_sdma_compat.h` as the initial compatibility boundary header. Task 6 fills in the device intrinsics:

```cpp
/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_SDMA_COMPAT_H
#define TILEXR_SDMA_COMPAT_H

#include "comm_args.h"
#include "tilexr_sdma_types.h"

#if TILEXR_ASCENDC_AICORE_COMPILE && defined(TILEXR_HAVE_PTO_SDMA) && TILEXR_HAVE_PTO_SDMA
#ifndef PTO_COMM_NOT_SUPPORTED
#define PTO_COMM_NOT_SUPPORTED 1
#endif
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"
#endif

#endif // TILEXR_SDMA_COMPAT_H
```

Modify `src/comm/sdma/tilexr_sdma_transport.h`:

```cpp
#include <memory>
```

Add a private implementation type and field:

```cpp
struct Impl;
std::unique_ptr<Impl> impl_;
```

Modify `src/comm/sdma/tilexr_sdma_transport.cpp`:

```cpp
#if TILEXR_HAVE_PTO_SDMA
#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"
#endif
```

Add this implementation holder after `namespace TileXR {`:

```cpp
struct TileXRSDMATransport::Impl {
#if TILEXR_HAVE_PTO_SDMA
    pto::comm::sdma::SdmaWorkspaceManager workspaceManager;
#endif
};
```

Replace the enabled branch in `Init()`:

```cpp
#if TILEXR_HAVE_PTO_SDMA
    impl_.reset(new (std::nothrow) Impl());
    if (impl_ == nullptr) {
        lastStatus_ = SDMAInitStatus::INIT_FAILED;
        TILEXR_LOG(WARN) << "TileXR SDMA workspace manager allocation failed";
        return TILEXR_SUCCESS;
    }
    if (!impl_->workspaceManager.Init()) {
        lastStatus_ = SDMAInitStatus::INIT_FAILED;
        TILEXR_LOG(WARN) << "TileXR SDMA workspace manager init failed";
        impl_.reset();
        return TILEXR_SUCCESS;
    }
    workspaceDev_ = static_cast<GM_ADDR>(impl_->workspaceManager.GetWorkspaceAddr());
    if (workspaceDev_ == nullptr) {
        lastStatus_ = SDMAInitStatus::NULL_WORKSPACE;
        TILEXR_LOG(WARN) << "TileXR SDMA workspace manager returned null workspace";
        impl_->workspaceManager.Finalize();
        impl_.reset();
        return TILEXR_SUCCESS;
    }
    available_ = true;
    lastStatus_ = SDMAInitStatus::INITIALIZED;
    TILEXR_LOG(INFO) << "TileXR SDMA initialized on dev " << options_.devId
                     << ", workspace " << static_cast<void*>(workspaceDev_);
    return TILEXR_SUCCESS;
#else
    lastStatus_ = SDMAInitStatus::PTO_UNAVAILABLE;
    TILEXR_LOG(WARN) << "TileXR SDMA PTO headers unavailable at build time";
    return TILEXR_SUCCESS;
#endif
```

Modify `Shutdown()`:

```cpp
#if TILEXR_HAVE_PTO_SDMA
    if (impl_ != nullptr) {
        impl_->workspaceManager.Finalize();
        impl_.reset();
    }
#endif
    available_ = false;
    workspaceDev_ = nullptr;
```

- [ ] **Step 5: Run tests and runtime dependency guard**

Run:

```bash
bash tests/sdma/build.sh
bash tests/sdma/run_tests.sh
```

Expected includes:

```text
TileXR SDMA source guard checks passed
TileXR SDMA runtime dependency check passed
TileXR SDMA unit tests passed
```

- [ ] **Step 6: Commit**

```bash
git add src/include/tilexr_sdma_compat.h src/comm/CMakeLists.txt src/comm/sdma tests/sdma
git commit -m "feat: initialize sdma workspace manager"
```

---

### Task 6: Device Compatibility Shim And Public SDMA Wrapper

**Files:**
- Modify: `src/include/tilexr_sdma_compat.h`
- Create: `src/include/tilexr_sdma.h`
- Modify: `src/comm/CMakeLists.txt`
- Modify: `tests/sdma/CMakeLists.txt`
- Create: `tests/sdma/unit/test_tilexr_sdma_header_compile.cpp`

- [ ] **Step 1: Write failing header compile test**

Create `tests/sdma/unit/test_tilexr_sdma_header_compile.cpp`:

```cpp
#include <iostream>

#include "tilexr_sdma.h"
#include "tilexr_sdma_types.h"

int main()
{
    if (TileXR::TILEXR_SDMA_AUTO_CHANNEL_GROUP != 0xFFFFFFFFU) {
        std::cerr << "unexpected TILEXR_SDMA_AUTO_CHANNEL_GROUP" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA header compile check passed" << std::endl;
    return 0;
}
```

Modify `tests/sdma/CMakeLists.txt`:

```cmake
add_executable(test_tilexr_sdma_header_compile
    unit/test_tilexr_sdma_header_compile.cpp
)
target_include_directories(test_tilexr_sdma_header_compile PRIVATE
    ${TILEXR_ROOT}/src/include
)
list(APPEND INSTALL_TARGETS test_tilexr_sdma_header_compile)
```

Modify `tests/sdma/run_tests.sh`:

```bash
"${INSTALL_DIR}/bin/test_tilexr_sdma_header_compile"
```

- [ ] **Step 2: Run the failing header test**

Run:

```bash
bash tests/sdma/build.sh
```

Expected: compile fails because `tilexr_sdma.h` does not exist.

- [ ] **Step 3: Replace the initial compatibility header with the device shim**

Replace `src/include/tilexr_sdma_compat.h` with:

```cpp
/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_SDMA_COMPAT_H
#define TILEXR_SDMA_COMPAT_H

#include "comm_args.h"
#include "tilexr_sdma_types.h"

#if TILEXR_ASCENDC_AICORE_COMPILE && defined(TILEXR_HAVE_PTO_SDMA) && TILEXR_HAVE_PTO_SDMA

#ifndef PTO_COMM_NOT_SUPPORTED
#define PTO_COMM_NOT_SUPPORTED 1
#endif

#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"

namespace TileXR {
namespace detail {

using SDMAScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, TILEXR_SDMA_SCRATCH_BYTES>;

__aicore__ inline bool TileXRSDMABuildSession(
    SDMAScratchTile& scratch,
    __gm__ uint8_t* workspace,
    pto::comm::sdma::SdmaSession& session,
    uint32_t channelGroupIdx)
{
    pto::TASSIGN(scratch, 0);
    pto::comm::sdma::SdmaBaseConfig config {
        TILEXR_SDMA_DEFAULT_BLOCK_BYTES,
        TILEXR_SDMA_DEFAULT_COMM_BLOCK_OFFSET,
        TILEXR_SDMA_DEFAULT_QUEUE_NUM
    };
    return pto::comm::sdma::BuildSdmaSession(scratch, workspace, session, 0, config, channelGroupIdx);
}

__aicore__ inline uint64_t TileXRSDMAPostPut(
    __gm__ uint8_t* dst,
    __gm__ uint8_t* src,
    uint64_t bytes,
    const pto::comm::sdma::SdmaSession& session)
{
    return pto::comm::sdma::__sdma_put_async(dst, src, bytes, session.execCtx);
}

__aicore__ inline bool TileXRSDMAWait(uint64_t eventHandle, const pto::comm::sdma::SdmaSession& session)
{
    return pto::comm::sdma::detail::SdmaWaitEvent(eventHandle, session);
}

} // namespace detail
} // namespace TileXR

#endif // TILEXR_ASCENDC_AICORE_COMPILE && TILEXR_HAVE_PTO_SDMA

#endif // TILEXR_SDMA_COMPAT_H
```

- [ ] **Step 4: Add public device wrapper**

Create `src/include/tilexr_sdma.h`:

```cpp
/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_SDMA_H
#define TILEXR_SDMA_H

#include "comm_args.h"
#include "tilexr_sdma_types.h"

#if TILEXR_ASCENDC_AICORE_COMPILE && defined(TILEXR_HAVE_PTO_SDMA) && TILEXR_HAVE_PTO_SDMA
#include "tilexr_sdma_compat.h"
#endif

namespace TileXR {

#if TILEXR_ASCENDC_AICORE_COMPILE

__aicore__ inline bool SDMAEnabled(const __gm__ CommArgs* args)
{
    return args != nullptr && ((args->extraFlag & ExtraFlag::SDMA) != 0) && args->sdmaWorkspacePtr != nullptr;
}

__aicore__ inline uint32_t SDMAResolveChannelGroup(uint32_t channelGroupIdx)
{
    if (channelGroupIdx == TILEXR_SDMA_AUTO_CHANNEL_GROUP) {
        return static_cast<uint32_t>(AscendC::GetBlockIdx());
    }
    return channelGroupIdx;
}

__aicore__ inline uint64_t SDMACopyNbi(
    const __gm__ CommArgs* args,
    __gm__ uint8_t* dst,
    __gm__ uint8_t* src,
    uint64_t bytes,
    uint32_t channelGroupIdx = TILEXR_SDMA_AUTO_CHANNEL_GROUP)
{
#if defined(TILEXR_HAVE_PTO_SDMA) && TILEXR_HAVE_PTO_SDMA
    if (!SDMAEnabled(args) || dst == nullptr || src == nullptr || bytes == 0) {
        return 0;
    }
    detail::SDMAScratchTile scratch;
    pto::comm::sdma::SdmaSession session {};
    const uint32_t resolvedGroup = SDMAResolveChannelGroup(channelGroupIdx);
    const bool built = detail::TileXRSDMABuildSession(
        scratch, reinterpret_cast<__gm__ uint8_t*>(args->sdmaWorkspacePtr), session, resolvedGroup);
    if (!built || !session.valid) {
        return 0;
    }
    return detail::TileXRSDMAPostPut(dst, src, bytes, session);
#else
    (void)args;
    (void)dst;
    (void)src;
    (void)bytes;
    (void)channelGroupIdx;
    return 0;
#endif
}

__aicore__ inline bool SDMAWait(
    const __gm__ CommArgs* args,
    uint64_t eventHandle,
    uint32_t channelGroupIdx = TILEXR_SDMA_AUTO_CHANNEL_GROUP)
{
#if defined(TILEXR_HAVE_PTO_SDMA) && TILEXR_HAVE_PTO_SDMA
    if (eventHandle == 0) {
        return true;
    }
    if (!SDMAEnabled(args)) {
        return false;
    }
    detail::SDMAScratchTile scratch;
    pto::comm::sdma::SdmaSession session {};
    const uint32_t resolvedGroup = SDMAResolveChannelGroup(channelGroupIdx);
    const bool built = detail::TileXRSDMABuildSession(
        scratch, reinterpret_cast<__gm__ uint8_t*>(args->sdmaWorkspacePtr), session, resolvedGroup);
    if (!built || !session.valid) {
        return false;
    }
    return detail::TileXRSDMAWait(eventHandle, session);
#else
    (void)args;
    (void)eventHandle;
    (void)channelGroupIdx;
    return eventHandle == 0;
#endif
}

#endif // TILEXR_ASCENDC_AICORE_COMPILE

} // namespace TileXR

#endif // TILEXR_SDMA_H
```

Modify `src/comm/CMakeLists.txt` install headers:

```cmake
${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_sdma.h
${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_sdma_compat.h
```

- [ ] **Step 5: Run tests**

Run:

```bash
bash tests/sdma/build.sh
bash tests/sdma/run_tests.sh
```

Expected includes:

```text
TileXR SDMA header compile check passed
TileXR SDMA source guard checks passed
```

- [ ] **Step 6: Commit**

```bash
git add src/include/tilexr_sdma.h src/include/tilexr_sdma_compat.h src/comm/CMakeLists.txt tests/sdma
git commit -m "feat: add sdma device wrapper"
```

---

### Task 7: SDMA Data-Plane Demo

**Files:**
- Modify: `tests/sdma/CMakeLists.txt`
- Create: `tests/sdma/demo/tilexr_sdma_demo_kernel.cpp`
- Create: `tests/sdma/demo/tilexr_sdma_demo.cpp`
- Create: `tests/sdma/demo/run_tilexr_sdma_demo.sh`

- [ ] **Step 1: Write the demo kernel**

Create `tests/sdma/demo/tilexr_sdma_demo_kernel.cpp`:

```cpp
/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_sdma.h"

extern "C" __global__ __aicore__ void tilexr_sdma_copy_kernel(
    GM_ADDR commArgsGM,
    GM_ADDR dstGM,
    GM_ADDR srcGM,
    GM_ADDR debugGM,
    uint32_t bytes)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto dst = reinterpret_cast<__gm__ uint8_t*>(dstGM);
    auto src = reinterpret_cast<__gm__ uint8_t*>(srcGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    if ASCEND_IS_AIV {
        if (debug != nullptr) {
            debug[0] = TileXR::TILEXR_SDMA_DEMO_MAGIC;
            debug[1] = static_cast<int32_t>(AscendC::GetBlockIdx());
            debug[2] = static_cast<int32_t>(bytes);
            debug[3] = TileXR::SDMAEnabled(args) ? 1 : 0;
            debug[4] = 0;
            debug[5] = 0;
        }
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        uint64_t event = TileXR::SDMACopyNbi(args, dst, src, static_cast<uint64_t>(bytes), 0);
        if (debug != nullptr) {
            debug[4] = event == 0 ? 0 : 1;
        }
        bool waitOk = TileXR::SDMAWait(args, event, 0);
        if (debug != nullptr) {
            debug[5] = waitOk ? 1 : 0;
        }
    }
}

extern "C" void launch_tilexr_sdma_copy(
    uint32_t blockDim,
    void* stream,
    GM_ADDR commArgs,
    GM_ADDR dst,
    GM_ADDR src,
    GM_ADDR debug,
    uint32_t bytes)
{
    tilexr_sdma_copy_kernel<<<blockDim, nullptr, stream>>>(commArgs, dst, src, debug, bytes);
}
```

- [ ] **Step 2: Write the host demo**

Create `tests/sdma/demo/tilexr_sdma_demo.cpp`:

```cpp
/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "tilexr_api.h"
#include "tilexr_sdma_types.h"
#include "tilexr_types.h"

extern "C" void launch_tilexr_sdma_copy(
    uint32_t blockDim,
    void* stream,
    GM_ADDR commArgs,
    GM_ADDR dst,
    GM_ADDR src,
    GM_ADDR debug,
    uint32_t bytes);

namespace {

constexpr size_t kDebugWords = 16;

bool CheckAcl(const char* step, aclError ret)
{
    if (ret == ACL_SUCCESS) {
        std::cout << step << " success" << std::endl;
        return true;
    }
    std::cerr << "ERROR: " << step << " failed, ret=" << ret << std::endl;
    return false;
}

bool CheckTileXR(const char* step, int ret)
{
    if (ret == TileXR::TILEXR_SUCCESS) {
        std::cout << step << " success" << std::endl;
        return true;
    }
    std::cerr << "ERROR: " << step << " failed, ret=" << ret << std::endl;
    return false;
}

} // namespace

int main(int argc, char** argv)
{
    uint32_t bytes = argc > 1 ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 0)) : 4096U;
    if (bytes == 0 || (bytes % 64U) != 0U) {
        std::cerr << "bytes must be non-zero and 64-byte aligned" << std::endl;
        return 1;
    }

    setenv("TILEXR_ENABLE_SDMA", "1", 1);

    if (!CheckAcl("aclInit", aclInit(nullptr))) {
        return 2;
    }
    if (!CheckAcl("aclrtSetDevice", aclrtSetDevice(0))) {
        aclFinalize();
        return 2;
    }

    aclrtStream stream = nullptr;
    if (!CheckAcl("aclrtCreateStream", aclrtCreateStream(&stream))) {
        aclrtResetDevice(0);
        aclFinalize();
        return 2;
    }

    TileXRCommPtr comm = nullptr;
    if (!CheckTileXR("TileXRCommInitRankLocal", TileXRCommInitRankLocal(1, 0, &comm))) {
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 3;
    }

    bool sdmaAvailable = false;
    GM_ADDR sdmaWorkspace = nullptr;
    if (!CheckTileXR("TileXRSDMAAvailable", TileXRSDMAAvailable(comm, &sdmaAvailable)) ||
        !CheckTileXR("TileXRGetSDMAWorkspaceDev", TileXRGetSDMAWorkspaceDev(comm, &sdmaWorkspace))) {
        TileXRCommDestroy(comm);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 3;
    }
    std::cout << "SDMA available=" << (sdmaAvailable ? "true" : "false")
              << " workspace=" << static_cast<void*>(sdmaWorkspace) << std::endl;
    if (!sdmaAvailable || sdmaWorkspace == nullptr) {
        std::cerr << "ERROR: SDMA unavailable" << std::endl;
        TileXRCommDestroy(comm);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 4;
    }

    GM_ADDR commArgsDev = nullptr;
    if (!CheckTileXR("TileXRGetCommArgsDev", TileXRGetCommArgsDev(comm, commArgsDev))) {
        TileXRCommDestroy(comm);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 3;
    }

    void* srcDev = nullptr;
    void* dstDev = nullptr;
    void* debugDev = nullptr;
    CheckAcl("aclrtMalloc src", aclrtMalloc(&srcDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CheckAcl("aclrtMalloc dst", aclrtMalloc(&dstDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CheckAcl("aclrtMalloc debug", aclrtMalloc(&debugDev, kDebugWords * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<uint8_t> src(bytes);
    std::vector<uint8_t> dst(bytes, 0);
    std::vector<uint8_t> zero(bytes, 0);
    std::vector<int32_t> debug(kDebugWords, 0);
    for (uint32_t i = 0; i < bytes; ++i) {
        src[i] = static_cast<uint8_t>((i * 17U + 3U) & 0xffU);
    }

    CheckAcl("copy src H2D", aclrtMemcpy(srcDev, bytes, src.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CheckAcl("copy dst H2D", aclrtMemcpy(dstDev, bytes, zero.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CheckAcl("copy debug H2D", aclrtMemcpy(debugDev, debug.size() * sizeof(int32_t), debug.data(),
                                           debug.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));

    launch_tilexr_sdma_copy(1, stream, commArgsDev, static_cast<GM_ADDR>(dstDev), static_cast<GM_ADDR>(srcDev),
                            static_cast<GM_ADDR>(debugDev), bytes);
    aclError syncRet = aclrtSynchronizeStream(stream);
    std::cout << "aclrtSynchronizeStream ret=" << syncRet << std::endl;
    if (syncRet != ACL_SUCCESS) {
        return 5;
    }

    CheckAcl("copy dst D2H", aclrtMemcpy(dst.data(), dst.size(), dstDev, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CheckAcl("copy debug D2H", aclrtMemcpy(debug.data(), debug.size() * sizeof(int32_t), debugDev,
                                           debug.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << "debug:";
    for (int32_t word : debug) {
        std::cout << " " << word;
    }
    std::cout << std::endl;

    bool match = true;
    for (uint32_t i = 0; i < bytes; ++i) {
        if (dst[i] != src[i]) {
            std::cerr << "ERROR: mismatch at byte " << i << std::endl;
            match = false;
            break;
        }
    }

    aclrtFree(srcDev);
    aclrtFree(dstDev);
    aclrtFree(debugDev);
    TileXRCommDestroy(comm);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    if (!match || debug[0] != TileXR::TILEXR_SDMA_DEMO_MAGIC || debug[3] != 1 || debug[4] != 1 || debug[5] != 1) {
        std::cerr << "ERROR: SDMA demo failed" << std::endl;
        return 6;
    }

    std::cout << "PASS TileXR SDMA copied " << bytes << " bytes correctly" << std::endl;
    return 0;
}
```

- [ ] **Step 3: Add demo build rules**

Modify `tests/sdma/CMakeLists.txt`:

```cmake
option(BUILD_TILEXR_SDMA_DEMO "Build TileXR SDMA demo with Ascend C kernel" ON)
set(TILEXR_SDMA_DEMO_SOC_TYPE "Ascend910B" CACHE STRING "SOC type used for TileXR SDMA demo kernel")

if(BUILD_TILEXR_SDMA_DEMO)
    find_program(BISHENG_EXECUTABLE bisheng)
    if(NOT BISHENG_EXECUTABLE)
        message(WARNING "bisheng not found; skip tilexr_sdma_demo")
    else()
        if(TILEXR_SDMA_DEMO_SOC_TYPE STREQUAL "Ascend950")
            set(TILEXR_SDMA_NPU_ARCH "dav-3510")
            set(TILEXR_SDMA_AICORE_ARCH "--cce-aicore-arch=dav-c310-vec")
            set(TILEXR_SDMA_CATLASS_ARCH "3510")
        else()
            set(TILEXR_SDMA_NPU_ARCH "dav-2201")
            set(TILEXR_SDMA_AICORE_ARCH "--cce-aicore-arch=dav-c220-vec")
            set(TILEXR_SDMA_CATLASS_ARCH "2201")
        endif()

        execute_process(
            COMMAND ${BISHENG_EXECUTABLE} -v
            OUTPUT_VARIABLE BISHENG_VERSION_OUTPUT
            ERROR_VARIABLE BISHENG_VERSION_OUTPUT
            RESULT_VARIABLE BISHENG_VERSION_RESULT
        )
        if(BISHENG_VERSION_RESULT EQUAL 0 AND BISHENG_VERSION_OUTPUT MATCHES "([0-9]{8})")
            set(TILEXR_SDMA_BISHENG_DATE "${CMAKE_MATCH_1}")
        else()
            set(TILEXR_SDMA_BISHENG_DATE "0")
        endif()
        if(TILEXR_SDMA_BISHENG_DATE GREATER_EQUAL 20250428)
            set(TILEXR_SDMA_KERNEL_COMPILE_OPTIONS
                -xasc
                -Xhost-start -ftrapv -Xhost-end
                --npu-arch=${TILEXR_SDMA_NPU_ARCH}
                --cce-auto-infer-kernel-type=false
            )
            set(TILEXR_SDMA_KERNEL_LINK_OPTIONS --cce-fatobj-link)
        else()
            set(TILEXR_SDMA_KERNEL_COMPILE_OPTIONS
                -xcce
                -Xhost-start -ftrapv -Xhost-end
                -mllvm -cce-aicore-stack-size=0x8000
                -mllvm -cce-aicore-function-stack-size=0x8000
                -mllvm -cce-aicore-record-overflow=true
                -mllvm -cce-aicore-addr-transform
                -mllvm -cce-aicore-dcci-insert-for-scalar=false
                ${TILEXR_SDMA_AICORE_ARCH}
            )
            set(TILEXR_SDMA_KERNEL_LINK_OPTIONS --cce-fatobj-link ${TILEXR_SDMA_AICORE_ARCH})
        endif()

        set(TILEXR_SDMA_DEMO_KERNEL_SO "${CMAKE_CURRENT_BINARY_DIR}/libtilexr_sdma_demo_kernel.so")
        set(TILEXR_SDMA_DEMO_KERNEL_INCLUDES
            -I${ASCEND_HOME_PATH}/compiler/tikcpp
            -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
            -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
            -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/impl
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/interface
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/include/
            -I${ASCEND_DRIVER_PATH}/kernel/inc
            -I${TILEXR_ROOT}/src/include
            -DTILEXR_HAVE_PTO_SDMA=1
        )
        add_custom_command(
            OUTPUT "${TILEXR_SDMA_DEMO_KERNEL_SO}"
            COMMAND ${BISHENG_EXECUTABLE}
                ${TILEXR_SDMA_KERNEL_COMPILE_OPTIONS}
                -std=gnu++17
                -fPIC
                -shared
                ${TILEXR_SDMA_KERNEL_LINK_OPTIONS}
                -DCATLASS_ARCH=${TILEXR_SDMA_CATLASS_ARCH}
                ${TILEXR_SDMA_DEMO_KERNEL_INCLUDES}
                "${CMAKE_CURRENT_SOURCE_DIR}/demo/tilexr_sdma_demo_kernel.cpp"
                -L${ASCEND_DRIVER_PATH}/lib64/driver
                -L${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
                -L${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
                -lruntime
                -lascendcl
                -lstdc++
                -lm
                -ldl
                -lnnopbase
                -lpthread
                -o "${TILEXR_SDMA_DEMO_KERNEL_SO}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/demo/tilexr_sdma_demo_kernel.cpp"
                "${TILEXR_ROOT}/src/include/tilexr_sdma.h"
                "${TILEXR_ROOT}/src/include/tilexr_sdma_compat.h"
            VERBATIM
            COMMENT "Building TileXR SDMA demo kernel with bisheng"
        )
        add_custom_target(tilexr_sdma_demo_kernel ALL DEPENDS "${TILEXR_SDMA_DEMO_KERNEL_SO}")

        add_executable(tilexr_sdma_demo
            demo/tilexr_sdma_demo.cpp
        )
        add_dependencies(tilexr_sdma_demo tilexr_sdma_demo_kernel)
        target_link_directories(tilexr_sdma_demo PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        target_link_libraries(tilexr_sdma_demo
            "${TILEXR_SDMA_DEMO_KERNEL_SO}"
            ${TILEXR_LIB}
            ascendcl
            runtime
            ascend_hal
        )
        list(APPEND INSTALL_TARGETS tilexr_sdma_demo)
        install(FILES "${TILEXR_SDMA_DEMO_KERNEL_SO}" DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        message(STATUS "TileXR SDMA demo enabled with ${BISHENG_EXECUTABLE}, SOC=${TILEXR_SDMA_DEMO_SOC_TYPE}")
    endif()
endif()
```

Modify `tests/sdma/build.sh` to disable demo when `bisheng` is missing:

```bash
if command -v bisheng >/dev/null 2>&1; then
    DEMO_OPTION="-DBUILD_TILEXR_SDMA_DEMO=ON"
else
    echo "WARN: bisheng not found; TileXR SDMA demo target will be skipped."
    DEMO_OPTION="-DBUILD_TILEXR_SDMA_DEMO=OFF"
fi

cmake -S "${SCRIPT_DIR}" -B "${TEST_BUILD}" \
    -DCMAKE_INSTALL_PREFIX="${TEST_INSTALL}" \
    ${DEMO_OPTION}
```

- [ ] **Step 4: Add demo run script**

Create `tests/sdma/demo/run_tilexr_sdma_demo.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SDMA_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
TILEXR_ROOT=$(cd "${SDMA_DIR}/../.." && pwd)
INSTALL_DIR="${SDMA_DIR}/install"

CANN_HOME="${1:-${ASCEND_HOME_PATH:-}}"
DEVICE_ID="${2:-0}"
shift $(( $# > 0 ? 1 : 0 )) || true
shift $(( $# > 0 ? 1 : 0 )) || true
SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=(64 4096 1048576)
fi

if [ -z "${CANN_HOME}" ]; then
    echo "ERROR: CANN_HOME argument or ASCEND_HOME_PATH is required"
    exit 1
fi

if [ -f "${CANN_HOME}/set_env.sh" ]; then
    set +u
    source "${CANN_HOME}/set_env.sh"
    set -u
fi

export TILEXR_ENABLE_SDMA=1
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TILEXR_ROOT}/install/lib:${CANN_HOME}/lib64:${CANN_HOME}/$(uname -m)-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

bin="${INSTALL_DIR}/bin/tilexr_sdma_demo"
if [ ! -x "${bin}" ]; then
    echo "ERROR: ${bin} not found. Run: cd ${SDMA_DIR} && bash build.sh ${CANN_HOME}"
    exit 1
fi

echo "=========================================="
echo "  TileXR SDMA Demo"
echo "=========================================="
echo "CANN_HOME: ${CANN_HOME}"
echo "DEVICE_ID: ${DEVICE_ID}"
echo "Sizes: ${SIZES[*]}"
echo "Binary: ${bin}"
echo "=========================================="

for bytes in "${SIZES[@]}"; do
    echo "---- bytes=${bytes} ----"
    "${bin}" "${bytes}"
done
```

Make it executable:

```bash
chmod +x tests/sdma/demo/run_tilexr_sdma_demo.sh
```

- [ ] **Step 5: Build and run demo on one CANN install**

Run on `blue` or another Ascend host:

```bash
bash tests/sdma/build.sh /home/gsn3/Ascend/cann-9.0.0
bash tests/sdma/demo/run_tilexr_sdma_demo.sh /home/gsn3/Ascend/cann-9.0.0 0 64 4096 1048576
```

Expected for each size:

```text
SDMA available=true
aclrtSynchronizeStream ret=0
PASS TileXR SDMA copied <bytes> bytes correctly
```

- [ ] **Step 6: Commit**

```bash
git add tests/sdma
git commit -m "test: add sdma data-plane demo"
```

---

### Task 8: Documentation And Two-CANN Validation

**Files:**
- Create: `docs/SDMA_TRANSPORT.md`
- Modify: `docs/SHMEM_INTEGRATION.md`
- Modify: `docs/UDMA_INTEGRATION_SUMMARY.md`

- [ ] **Step 1: Write SDMA user documentation**

Create `docs/SDMA_TRANSPORT.md`:

```markdown
# TileXR SDMA Transport

TileXR SDMA transport provides a first-class local on-card GM-to-GM copy path.
It is separate from UDMA: SDMA is local to one device, while UDMA targets
registered remote memory on supported A5/Ascend950 systems.

## Enablement

SDMA is disabled by default. Enable it explicitly:

```bash
export TILEXR_ENABLE_SDMA=1
```

When disabled, `TileXRComm` initialization does not create STARS streams and
`TileXRSDMAAvailable` reports `false`.

## Host API

```cpp
bool available = false;
GM_ADDR workspace = nullptr;
TileXRSDMAAvailable(comm, &available);
TileXRGetSDMAWorkspaceDev(comm, &workspace);
```

The workspace pointer is owned by `TileXRComm`; callers must not free it.

## Device API

```cpp
#include "tilexr_sdma.h"

uint64_t event = TileXR::SDMACopyNbi(args, dst, src, bytes, 0);
bool ok = TileXR::SDMAWait(args, event, 0);
```

The API accepts raw same-device GM pointers. It does not register memory and
does not validate whether pointers belong to TileXR communication buffers.

## CANN Compatibility

The implementation targets CANN 9.0.0 and CANN 9.1.0. TileXR isolates PTO SDMA
header differences in `tilexr_sdma_compat.h`.

Runtime must load `libascend_hal.so` from the driver path, typically:

```text
/usr/local/Ascend/driver/lib64/driver/libascend_hal.so
```

Do not put `${ASCEND_HOME_PATH}/${ARCH}-linux/devlib` into runtime RPATH. The
devlib HAL can cause `aclInit` failures such as `500000` with `init soc version
failed`.

## Build And Test

Build tests against a selected CANN install:

```bash
bash tests/sdma/build.sh /path/to/cann
bash tests/sdma/run_tests.sh
```

Run data-plane demo:

```bash
bash tests/sdma/demo/run_tilexr_sdma_demo.sh /path/to/cann 0 64 4096 1048576
```

Expected demo success line:

```text
PASS TileXR SDMA copied <bytes> bytes correctly
```

## Acceptance

For release validation, run the unit tests and demo against both CANN versions:

```bash
bash tests/sdma/build.sh "${TILEXR_CANN_90_HOME}"
bash tests/sdma/run_tests.sh
bash tests/sdma/demo/run_tilexr_sdma_demo.sh "${TILEXR_CANN_90_HOME}" 0 64 4096 1048576

bash tests/sdma/build.sh "${TILEXR_CANN_91_HOME}"
bash tests/sdma/run_tests.sh
bash tests/sdma/demo/run_tilexr_sdma_demo.sh "${TILEXR_CANN_91_HOME}" 0 64 4096 1048576
```

## Deferred Stress And Performance Scope

Deferred validation scope includes multi-block channel-group assignment, multi-stream
concurrency, long-loop stability, parameter matrix tests for queue settings,
and MTE/SDMA bandwidth and latency comparisons.
```

- [ ] **Step 2: Update existing docs with SDMA boundary notes**

Append to `docs/SHMEM_INTEGRATION.md`:

```markdown

## SDMA Note

TileXR SDMA transport is also TileXR-owned and does not revive the old
shmem-backed design. The SDMA host manager may use CANN's
`aclnnShmemSdmaStarsQuery*` symbols through PTO `SdmaWorkspaceManager` to obtain
STARS queue metadata, but TileXR does not include or link shmem and does not use
shmem data-copy APIs.
```

Append to `docs/UDMA_INTEGRATION_SUMMARY.md`:

```markdown

## Relationship To SDMA

UDMA remains the registered-memory remote transport. SDMA is a separate local
GM-to-GM transport for same-device copies. Both expose device-visible metadata
through `CommArgs`, but SDMA does not require memory registration and is enabled
only when `TILEXR_ENABLE_SDMA=1`.
```

- [ ] **Step 3: Run docs and source tests**

Run:

```bash
bash tests/sdma/build.sh
bash tests/sdma/run_tests.sh
```

Expected:

```text
TileXR SDMA unit tests passed
```

- [ ] **Step 4: Run CANN 9.0 and 9.1 acceptance on hardware**

On a host with both CANN versions installed:

```bash
export TILEXR_CANN_90_HOME=/home/gsn3/Ascend/cann-9.0.0
export TILEXR_CANN_91_HOME=/path/to/cann-9.1.0

bash tests/sdma/build.sh "${TILEXR_CANN_90_HOME}"
bash tests/sdma/run_tests.sh
bash tests/sdma/demo/run_tilexr_sdma_demo.sh "${TILEXR_CANN_90_HOME}" 0 64 4096 1048576

bash tests/sdma/build.sh "${TILEXR_CANN_91_HOME}"
bash tests/sdma/run_tests.sh
bash tests/sdma/demo/run_tilexr_sdma_demo.sh "${TILEXR_CANN_91_HOME}" 0 64 4096 1048576
```

Expected for each CANN version:

```text
TileXR SDMA unit tests passed
PASS TileXR SDMA copied 64 bytes correctly
PASS TileXR SDMA copied 4096 bytes correctly
PASS TileXR SDMA copied 1048576 bytes correctly
```

- [ ] **Step 5: Commit**

```bash
git add docs/SDMA_TRANSPORT.md docs/SHMEM_INTEGRATION.md docs/UDMA_INTEGRATION_SUMMARY.md tests/sdma
git commit -m "docs: add sdma transport guide"
```

---

## Final Verification Checklist

Run this before declaring the implementation complete:

```bash
git status --short
bash tests/sdma/build.sh "${TILEXR_CANN_90_HOME}"
bash tests/sdma/run_tests.sh
bash tests/sdma/demo/run_tilexr_sdma_demo.sh "${TILEXR_CANN_90_HOME}" 0 64 4096 1048576
bash tests/sdma/build.sh "${TILEXR_CANN_91_HOME}"
bash tests/sdma/run_tests.sh
bash tests/sdma/demo/run_tilexr_sdma_demo.sh "${TILEXR_CANN_91_HOME}" 0 64 4096 1048576
```

Expected:

- no unexpected shmem dependency in source guards;
- no runtime `libascend_hal.so` from CANN devlib;
- disabled fallback passes without `TILEXR_ENABLE_SDMA`;
- SDMA demo passes for 64 B, 4 KB, and 1 MB on both CANN versions.

If CANN 9.1.0 is not installed on the hardware host yet, install it first and set `TILEXR_CANN_91_HOME` to that install root before running final acceptance.
