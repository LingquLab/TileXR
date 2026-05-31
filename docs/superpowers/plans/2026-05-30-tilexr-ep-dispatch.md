# TileXR EP Dispatch MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `src/ep` TileXR EP dispatch MVP that routes `[BS,H]` token rows to expert-owner ranks through TileXR peer-memory windows.

**Architecture:** EP is an optional TileXR runtime module with a public C API in `src/include/tilexr_ep.h`, host validation and layout helpers in `src/ep/host`, shared POD layout definitions in `src/ep/common`, and one Ascend C kernel in `src/ep/kernels`. The MVP uses `CommArgs::peerMems[] + TileXR::IPC_DATA_OFFSET` as the receive window, `SyncCollectives` for clear/ready synchronization, and `TileXRCommNextMagic` to avoid stale flags. The first kernel launch uses `blockDim = 1` for deterministic fixed-slot counters; UDMA and multi-AIV routing stay out of this route.

**Tech Stack:** C++14, CMake 3.16, TileXR `tile-comm`, Ascend C with `bisheng`, ACL/runtime libraries, simple C++ unit tests, and shell deployment scripts.

---

## File Structure

- Create `src/include/tilexr_ep.h`: public EP dispatch C API.
- Create `src/ep/CMakeLists.txt`: optional `tilexr-ep` library, EP kernel shared object, install rules.
- Create `src/ep/common/ep_window.h`: kernel-safe POD structs and constants for receive-window layout.
- Create `src/ep/host/ep_layout.h`: host layout, dtype, overflow, and expert mapping declarations.
- Create `src/ep/host/ep_layout.cpp`: host layout, dtype, overflow, and expert mapping implementation.
- Create `src/ep/host/ep_dispatch_host.h`: internal host parameter/context structs and validation declarations.
- Create `src/ep/host/ep_dispatch_host.cpp`: internal validation and comm-args preparation.
- Create `src/ep/host/ep_launch_context.cpp`: TileXR communicator host/device argument lookup.
- Create `src/ep/host/tilexr_ep_dispatch.cpp`: public `TileXRMoeEpDispatch` entry point.
- Create `src/ep/host/ep_kernel_launch.h`: host-to-kernel launch declaration.
- Create `src/ep/host/ep_kernel_launch.cpp`: `TileXRCommNextMagic` call and direct EP kernel launcher call.
- Create `src/ep/kernels/tilexr_ep_dispatch_kernel.cpp`: Ascend C peer-memory dispatch kernel and `launch_tilexr_ep_dispatch_kernel`.
- Modify `CMakeLists.txt`: add `TILEXR_BUILD_EP` and `add_subdirectory(src/ep)`.
- Create `tests/ep/CMakeLists.txt`: unit tests and optional EP demo target.
- Create `tests/ep/build.sh`: source-only and full EP build helper.
- Create `tests/ep/unit/test_tilexr_ep_layout.cpp`: mapping and window-layout unit tests.
- Create `tests/ep/unit/test_tilexr_ep_api_sources.cpp`: public API, build placement, and banned dependency source checks.
- Create `tests/ep/unit/test_tilexr_ep_host_validation.cpp`: internal host validation tests that do not require a device.
- Create `tests/ep/unit/test_tilexr_ep_kernel_sources.cpp`: kernel source checks for peer-memory and `SyncCollectives`.
- Create `tests/ep/demo/tilexr_ep_dispatch_demo.cpp`: two-rank deterministic FP16-bit-pattern dispatch demo.
- Create `tests/ep/demo/run_tilexr_ep_dispatch_demo.sh`: local multi-process demo runner.
- Create `tests/ep/demo/deploy_and_run_blue.sh`: complete remote deployment to `ssh blue:/home/d00520898/tilexr_ep_dispatch_verify/TileXR`.
- Create `tests/ep/README.md`: build, run, and remote verification notes.

## Task 1: Window Layout Helpers and Unit Tests

**Files:**
- Create: `src/ep/common/ep_window.h`
- Create: `src/ep/host/ep_layout.h`
- Create: `src/ep/host/ep_layout.cpp`
- Create: `tests/ep/CMakeLists.txt`
- Create: `tests/ep/unit/test_tilexr_ep_layout.cpp`

- [ ] **Step 1: Write the failing layout test**

Create `tests/ep/unit/test_tilexr_ep_layout.cpp` with:

```cpp
#include <cstdint>
#include <iostream>

#include "ep_layout.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

void CheckInt64(const char *label, int64_t actual, int64_t expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void CheckInt(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void CheckBool(const char *label, bool actual, bool expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void TestExpertMapping()
{
    CheckInt("expert 0 dst", TileXREp::TileXREpDstRank(0, 4), 0);
    CheckInt("expert 0 local", TileXREp::TileXREpLocalExpert(0, 4), 0);
    CheckInt("expert 3 dst", TileXREp::TileXREpDstRank(3, 4), 0);
    CheckInt("expert 3 local", TileXREp::TileXREpLocalExpert(3, 4), 3);
    CheckInt("expert 4 dst", TileXREp::TileXREpDstRank(4, 4), 1);
    CheckInt("expert 4 local", TileXREp::TileXREpLocalExpert(4, 4), 0);
    CheckInt("expert 7 dst", TileXREp::TileXREpDstRank(7, 4), 1);
    CheckInt("expert 7 local", TileXREp::TileXREpLocalExpert(7, 4), 3);
    CheckInt("negative expert dst", TileXREp::TileXREpDstRank(-1, 4), TileXR::TILEXR_INVALID_VALUE);
    CheckInt("zero local expert dst", TileXREp::TileXREpDstRank(1, 0), TileXR::TILEXR_INVALID_VALUE);
}

void TestDataTypes()
{
    CheckBool("fp16 supported", TileXREp::TileXREpIsSupportedDataType(TileXR::TILEXR_DATA_TYPE_FP16), true);
    CheckBool("bf16 supported", TileXREp::TileXREpIsSupportedDataType(TileXR::TILEXR_DATA_TYPE_BFP16), true);
    CheckBool("fp32 unsupported", TileXREp::TileXREpIsSupportedDataType(TileXR::TILEXR_DATA_TYPE_FP32), false);
    CheckInt64("fp16 bytes", TileXREp::TileXREpDataTypeSize(TileXR::TILEXR_DATA_TYPE_FP16), 2);
    CheckInt64("bf16 bytes", TileXREp::TileXREpDataTypeSize(TileXR::TILEXR_DATA_TYPE_BFP16), 2);
    CheckInt64("int32 bytes invalid", TileXREp::TileXREpDataTypeSize(TileXR::TILEXR_DATA_TYPE_INT32),
        TileXR::TILEXR_INVALID_VALUE);
}

void TestWindowConfig()
{
    TileXREp::EpWindowConfig config {};
    const int ret = TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 8, TileXR::TILEXR_DATA_TYPE_FP16, &config);
    CheckInt("valid config ret", ret, TileXR::TILEXR_SUCCESS);
    CheckInt64("local experts", config.localExpertNum, 4);
    CheckInt64("dtype bytes", config.dtypeBytes, 2);
    CheckInt64("max routes", config.maxRoutesPerSrc, 8);
    CheckInt64("row bytes", config.rowBytes, 16);
    CheckInt64("payload bytes", config.payloadBytesPerSlot, 128);
    CheckInt64("assist bytes", config.assistBytesPerSlot, 128);
    CheckInt64("slot bytes", config.slotBytes, 320);
    CheckInt64("total bytes", config.totalBytes, 704);
}

void TestRejectsInvalidConfig()
{
    TileXREp::EpWindowConfig config {};
    CheckInt("null out", TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 8, TileXR::TILEXR_DATA_TYPE_FP16, nullptr),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("non-divisible experts", TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 7, TileXR::TILEXR_DATA_TYPE_FP16, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("unsupported dtype", TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 8, TileXR::TILEXR_DATA_TYPE_FP32, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("oversized window", TileXREp::TileXREpBuildWindowConfig(
        2, 1024 * 1024, 64, 8, 8, TileXR::TILEXR_DATA_TYPE_FP16, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

int main()
{
    TestExpertMapping();
    TestDataTypes();
    TestWindowConfig();
    TestRejectsInvalidConfig();
    return g_failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Add the first `tests/ep` CMake target**

Create `tests/ep/CMakeLists.txt` with:

```cmake
cmake_minimum_required(VERSION 3.16)
project(TileXR_EP_Tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(GNUInstallDirs)
enable_testing()

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(TILEXR_INSTALL_PREFIX "${TILEXR_ROOT}/install" CACHE PATH "TileXR install prefix")
set(TILEXR_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}" CACHE STRING "TileXR install library directory")
set(TILEXR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}" CACHE STRING "TileXR install include directory")
option(BUILD_TILEXR_EP_DEMO "Build TileXR EP dispatch hardware demo" OFF)

set(_tilexr_default_ascend_home_path "$ENV{ASCEND_HOME_PATH}")
if(NOT _tilexr_default_ascend_home_path)
    set(_tilexr_default_ascend_home_path "/usr/local/Ascend/ascend-toolkit/latest")
endif()
set(ASCEND_HOME_PATH "${_tilexr_default_ascend_home_path}" CACHE PATH "Ascend toolkit path")

set(_tilexr_default_arch "$ENV{ARCH}")
if(NOT _tilexr_default_arch)
    set(_tilexr_default_arch "$ENV{TILEXR_OS_ARCH}")
endif()
if(NOT _tilexr_default_arch)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE _tilexr_default_arch OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_tilexr_default_arch STREQUAL "arm64")
        set(_tilexr_default_arch "aarch64")
    endif()
endif()
set(ARCH "${_tilexr_default_arch}" CACHE STRING "Target OS architecture")

set(_tilexr_default_ascend_driver_path "$ENV{ASCEND_DRIVER_PATH}")
if(NOT _tilexr_default_ascend_driver_path)
    set(_tilexr_default_ascend_driver_path "/usr/local/Ascend/driver")
endif()
set(ASCEND_DRIVER_PATH "${_tilexr_default_ascend_driver_path}" CACHE PATH "Ascend driver path")

set(TILEXR_EP_TEST_INCLUDE_DIRS
    ${TILEXR_ROOT}/src/include
    ${TILEXR_ROOT}/src/ep/common
    ${TILEXR_ROOT}/src/ep/host
)
foreach(_tilexr_ep_test_include_dir
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include
)
    if(EXISTS "${_tilexr_ep_test_include_dir}")
        list(APPEND TILEXR_EP_TEST_INCLUDE_DIRS "${_tilexr_ep_test_include_dir}")
    endif()
endforeach()

add_executable(test_tilexr_ep_layout
    unit/test_tilexr_ep_layout.cpp
    ${TILEXR_ROOT}/src/ep/host/ep_layout.cpp
)
target_include_directories(test_tilexr_ep_layout PRIVATE ${TILEXR_EP_TEST_INCLUDE_DIRS})
add_test(NAME test_tilexr_ep_layout COMMAND test_tilexr_ep_layout)

install(TARGETS test_tilexr_ep_layout RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
```

- [ ] **Step 3: Run the test to verify it fails before implementation**

Run:

```bash
cmake -S tests/ep -B tests/ep/build -DBUILD_TILEXR_EP_DEMO=OFF
cmake --build tests/ep/build --target test_tilexr_ep_layout -j"$(nproc)"
```

Expected: configure or compile fails because `src/ep/host/ep_layout.cpp` and `ep_layout.h` do not exist.

- [ ] **Step 4: Add shared window POD definitions**

Create `src/ep/common/ep_window.h` with:

```cpp
#ifndef TILEXR_EP_COMMON_EP_WINDOW_H
#define TILEXR_EP_COMMON_EP_WINDOW_H

#include <cstdint>

namespace TileXREp {

constexpr int64_t kEpWindowAlignmentBytes = 32;
constexpr int64_t kEpAssistTupleInts = 4;
constexpr int64_t kEpWindowHeaderBytes = 64;
constexpr int64_t kEpSrcSlotHeaderBytes = 64;
constexpr int32_t kEpStepWindowCleared = 71;
constexpr int32_t kEpStepDispatchReady = 72;
constexpr uint32_t kEpWindowMagic = 0x54584550U;

struct EpWindowHeader {
    uint32_t magic;
    int32_t rankSize;
    int64_t maxRoutesPerSrc;
    int64_t rowBytes;
    int64_t slotBytes;
    int64_t totalBytes;
    int64_t reserved0;
    int64_t reserved1;
    int64_t reserved2;
};

struct EpSrcSlotHeader {
    int32_t count;
    int32_t srcRank;
    int64_t payloadBytes;
    int64_t assistBytes;
    int64_t reserved0;
    int64_t reserved1;
    int64_t reserved2;
    int64_t reserved3;
    int64_t reserved4;
};

struct EpAssistTuple {
    int32_t srcRank;
    int32_t tokenId;
    int32_t topKId;
    int32_t expertId;
};

static_assert(sizeof(EpWindowHeader) == kEpWindowHeaderBytes, "unexpected EpWindowHeader size");
static_assert(sizeof(EpSrcSlotHeader) == kEpSrcSlotHeaderBytes, "unexpected EpSrcSlotHeader size");
static_assert(sizeof(EpAssistTuple) == kEpAssistTupleInts * static_cast<int64_t>(sizeof(int32_t)),
    "unexpected EpAssistTuple size");

} // namespace TileXREp

#endif // TILEXR_EP_COMMON_EP_WINDOW_H
```

- [ ] **Step 5: Add host layout declarations**

Create `src/ep/host/ep_layout.h` with:

```cpp
#ifndef TILEXR_EP_HOST_EP_LAYOUT_H
#define TILEXR_EP_HOST_EP_LAYOUT_H

#include <cstdint>

#include "comm_args.h"
#include "ep_window.h"
#include "tilexr_types.h"

namespace TileXREp {

struct EpWindowConfig {
    int64_t rankSize = 0;
    int64_t bs = 0;
    int64_t h = 0;
    int64_t topK = 0;
    int64_t moeExpertNum = 0;
    int64_t localExpertNum = 0;
    int64_t dtypeBytes = 0;
    int64_t maxRoutesPerSrc = 0;
    int64_t rowBytes = 0;
    int64_t payloadBytesPerSlot = 0;
    int64_t assistBytesPerSlot = 0;
    int64_t slotBytes = 0;
    int64_t totalBytes = 0;
};

int64_t TileXREpAlignUp(int64_t value, int64_t alignment);
int64_t TileXREpDataTypeSize(TileXR::TileXRDataType dtype);
bool TileXREpIsSupportedDataType(TileXR::TileXRDataType dtype);
int TileXREpBuildWindowConfig(int64_t rankSize, int64_t bs, int64_t h, int64_t topK,
    int64_t moeExpertNum, TileXR::TileXRDataType dtype, EpWindowConfig *out);
int TileXREpDstRank(int32_t expertId, int64_t localExpertNum);
int TileXREpLocalExpert(int32_t expertId, int64_t localExpertNum);

} // namespace TileXREp

#endif // TILEXR_EP_HOST_EP_LAYOUT_H
```

- [ ] **Step 6: Implement host layout helpers**

Create `src/ep/host/ep_layout.cpp` with:

```cpp
#include "ep_layout.h"

#include <limits>

namespace TileXREp {
namespace {

bool MulInt64(int64_t lhs, int64_t rhs, int64_t *out)
{
    if (out == nullptr || lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool AddInt64(int64_t lhs, int64_t rhs, int64_t *out)
{
    if (out == nullptr || lhs < 0 || rhs < 0) {
        return false;
    }
    if (rhs > std::numeric_limits<int64_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

} // namespace

int64_t TileXREpAlignUp(int64_t value, int64_t alignment)
{
    if (value < 0 || alignment <= 0) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    const int64_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    int64_t result = 0;
    return AddInt64(value, alignment - remainder, &result) ? result : TileXR::TILEXR_INVALID_VALUE;
}

int64_t TileXREpDataTypeSize(TileXR::TileXRDataType dtype)
{
    switch (dtype) {
        case TileXR::TILEXR_DATA_TYPE_FP16:
        case TileXR::TILEXR_DATA_TYPE_BFP16:
            return 2;
        default:
            return TileXR::TILEXR_INVALID_VALUE;
    }
}

bool TileXREpIsSupportedDataType(TileXR::TileXRDataType dtype)
{
    return TileXREpDataTypeSize(dtype) > 0;
}

int TileXREpBuildWindowConfig(int64_t rankSize, int64_t bs, int64_t h, int64_t topK,
    int64_t moeExpertNum, TileXR::TileXRDataType dtype, EpWindowConfig *out)
{
    if (out == nullptr || rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE ||
        bs <= 0 || h <= 0 || topK <= 0 || moeExpertNum <= 0 || moeExpertNum % rankSize != 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    const int64_t dtypeBytes = TileXREpDataTypeSize(dtype);
    if (dtypeBytes <= 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    int64_t maxRoutes = 0;
    int64_t rowBytes = 0;
    int64_t payloadBytes = 0;
    int64_t assistRecords = 0;
    int64_t assistBytes = 0;
    int64_t slotBytesRaw = 0;
    int64_t slotBytes = 0;
    int64_t slotsBytes = 0;
    int64_t totalBytes = 0;

    if (!MulInt64(bs, topK, &maxRoutes) ||
        !MulInt64(h, dtypeBytes, &rowBytes) ||
        !MulInt64(maxRoutes, rowBytes, &payloadBytes) ||
        !MulInt64(maxRoutes, kEpAssistTupleInts, &assistRecords) ||
        !MulInt64(assistRecords, static_cast<int64_t>(sizeof(int32_t)), &assistBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    payloadBytes = TileXREpAlignUp(payloadBytes, kEpWindowAlignmentBytes);
    assistBytes = TileXREpAlignUp(assistBytes, kEpWindowAlignmentBytes);
    if (payloadBytes <= 0 || assistBytes <= 0 ||
        !AddInt64(kEpSrcSlotHeaderBytes, payloadBytes, &slotBytesRaw) ||
        !AddInt64(slotBytesRaw, assistBytes, &slotBytesRaw)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    slotBytes = TileXREpAlignUp(slotBytesRaw, kEpWindowAlignmentBytes);
    if (slotBytes <= 0 || !MulInt64(rankSize, slotBytes, &slotsBytes) ||
        !AddInt64(kEpWindowHeaderBytes, slotsBytes, &totalBytes) ||
        totalBytes > TileXR::IPC_BUFF_MAX_SIZE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    EpWindowConfig config {};
    config.rankSize = rankSize;
    config.bs = bs;
    config.h = h;
    config.topK = topK;
    config.moeExpertNum = moeExpertNum;
    config.localExpertNum = moeExpertNum / rankSize;
    config.dtypeBytes = dtypeBytes;
    config.maxRoutesPerSrc = maxRoutes;
    config.rowBytes = rowBytes;
    config.payloadBytesPerSlot = payloadBytes;
    config.assistBytesPerSlot = assistBytes;
    config.slotBytes = slotBytes;
    config.totalBytes = totalBytes;
    *out = config;
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpDstRank(int32_t expertId, int64_t localExpertNum)
{
    if (expertId < 0 || localExpertNum <= 0) {
        return static_cast<int>(TileXR::TILEXR_INVALID_VALUE);
    }
    return static_cast<int>(expertId / localExpertNum);
}

int TileXREpLocalExpert(int32_t expertId, int64_t localExpertNum)
{
    if (expertId < 0 || localExpertNum <= 0) {
        return static_cast<int>(TileXR::TILEXR_INVALID_VALUE);
    }
    return static_cast<int>(expertId % localExpertNum);
}

} // namespace TileXREp
```

- [ ] **Step 7: Run the layout test and commit**

Run:

```bash
cmake -S tests/ep -B tests/ep/build -DBUILD_TILEXR_EP_DEMO=OFF
cmake --build tests/ep/build --target test_tilexr_ep_layout -j"$(nproc)"
tests/ep/build/test_tilexr_ep_layout
```

Expected: all commands succeed and `test_tilexr_ep_layout` exits `0`.

Commit:

```bash
git add src/ep/common/ep_window.h src/ep/host/ep_layout.h src/ep/host/ep_layout.cpp tests/ep/CMakeLists.txt tests/ep/unit/test_tilexr_ep_layout.cpp
git commit -m "feat: add ep window layout helpers"
```

## Task 2: Public API, Build Placement, and Source Guards

**Files:**
- Create: `src/include/tilexr_ep.h`
- Create: `tests/ep/unit/test_tilexr_ep_api_sources.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/ep/CMakeLists.txt`

- [ ] **Step 1: Write the failing source guard test**

Create `tests/ep/unit/test_tilexr_ep_api_sources.cpp` with:

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
    const std::string fullPath = RepoPath(path);
    std::ifstream input(fullPath.c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << fullPath << std::endl;
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckContains(const std::string& path, const std::string& text, const std::string& needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected " << path << " to contain: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckDoesNotContain(const std::string& path, const std::string& text, const std::string& needle)
{
    const std::string::size_type pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "expected " << path << " not to contain: " << needle << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void TestPublicHeader()
{
    const std::string path = "src/include/tilexr_ep.h";
    const std::string text = ReadFile(path);
    CheckContains(path, text, "#ifdef __cplusplus");
    CheckContains(path, text, "extern \"C\"");
    CheckContains(path, text, "int TileXRMoeEpDispatch(");
    CheckContains(path, text, "TileXRCommPtr comm");
    CheckContains(path, text, "TileXR::TileXRDataType dtype");
    CheckContains(path, text, "aclrtStream stream");
}

void TestBuildPlacement()
{
    const std::string rootPath = "CMakeLists.txt";
    const std::string rootText = ReadFile(rootPath);
    CheckContains(rootPath, rootText, "option(TILEXR_BUILD_EP \"Build TileXR EP communication library\" OFF)");
    CheckContains(rootPath, rootText, "add_subdirectory(src/ep)");

    const std::string epPath = "src/ep/CMakeLists.txt";
    const std::string epText = ReadFile(epPath);
    CheckContains(epPath, epText, "add_library(tilexr-ep SHARED");
    CheckContains(epPath, epText, "tile-comm");
    CheckContains(epPath, epText, "tilexr_ep.h");
    CheckContains(epPath, epText, "install(TARGETS tilexr-ep");
}

void TestNoForbiddenDependencies()
{
    const std::vector<std::string> paths = {
        "src/include/tilexr_ep.h",
        "src/ep/CMakeLists.txt",
        "src/ep/host/ep_layout.h",
        "src/ep/host/ep_layout.cpp"
    };
    const std::vector<std::string> forbidden = {
        "src/mc2",
        "3rdparty/ops-transformer",
        "GetHcclContext",
        "TileXRUDMARegister",
        "UDMAPut",
        "shmem"
    };
    for (const std::string& path : paths) {
        const std::string text = ReadFile(path);
        for (const std::string& needle : forbidden) {
            CheckDoesNotContain(path, text, needle);
        }
    }
}

} // namespace

int main()
{
    TestPublicHeader();
    TestBuildPlacement();
    TestNoForbiddenDependencies();
    return g_failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Register the source guard test**

Append to `tests/ep/CMakeLists.txt`:

```cmake
add_executable(test_tilexr_ep_api_sources
    unit/test_tilexr_ep_api_sources.cpp
)
target_compile_definitions(test_tilexr_ep_api_sources PRIVATE TILEXR_SOURCE_ROOT="${TILEXR_ROOT}")
add_test(NAME test_tilexr_ep_api_sources COMMAND test_tilexr_ep_api_sources)
install(TARGETS test_tilexr_ep_api_sources RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
```

- [ ] **Step 3: Run the source guard to verify it fails**

Run:

```bash
cmake -S tests/ep -B tests/ep/build -DBUILD_TILEXR_EP_DEMO=OFF
cmake --build tests/ep/build --target test_tilexr_ep_api_sources -j"$(nproc)"
tests/ep/build/test_tilexr_ep_api_sources
```

Expected: executable runs and reports missing `src/include/tilexr_ep.h` and `src/ep/CMakeLists.txt`.

- [ ] **Step 4: Add the public header**

Create `src/include/tilexr_ep.h` with:

```cpp
#ifndef TILEXR_EP_H
#define TILEXR_EP_H

#ifdef __cplusplus

#include <cstdint>

#include "acl/acl_base.h"
#include "tilexr_api.h"
#include "tilexr_types.h"

// This public API is C++ header-compatible because it reuses TileXR namespace datatypes.
extern "C" {

int TileXRMoeEpDispatch(void *x, int32_t *expertIds, TileXRCommPtr comm,
    int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
    void *expandXOut, int64_t *expertTokenNumsOut, int32_t *epRecvCountsOut,
    int32_t *assistInfoForCombineOut, TileXR::TileXRDataType dtype, aclrtStream stream);

}

#endif

#endif // TILEXR_EP_H
```

- [ ] **Step 5: Add the top-level EP build switch**

Modify root `CMakeLists.txt` near the existing options:

```cmake
option(TILEXR_BUILD_COLLECTIVES "Build optional TileXR collectives library" OFF)
option(TILEXR_BUILD_EP "Build TileXR EP communication library" OFF)
option(TILEXR_BUILD_TESTS "Build TileXR tests" OFF)
```

Modify root `CMakeLists.txt` after `add_subdirectory(src/comm)`:

```cmake
add_subdirectory(src/comm)
if(TILEXR_BUILD_EP)
    add_subdirectory(src/ep)
endif()
```

- [ ] **Step 6: Add initial `src/ep/CMakeLists.txt`**

Create `src/ep/CMakeLists.txt` with:

```cmake
include(GNUInstallDirs)

add_library(tilexr-ep SHARED
    host/ep_layout.cpp
)

target_include_directories(tilexr-ep
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/host
)

target_link_libraries(tilexr-ep
    PRIVATE
    tile-comm
)

install(TARGETS tilexr-ep LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_ep.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

- [ ] **Step 7: Run tests and commit**

Run:

```bash
cmake -S tests/ep -B tests/ep/build -DBUILD_TILEXR_EP_DEMO=OFF
cmake --build tests/ep/build --target test_tilexr_ep_api_sources -j"$(nproc)"
tests/ep/build/test_tilexr_ep_api_sources
```

Expected: source guard exits `0`.

Commit:

```bash
git add CMakeLists.txt src/include/tilexr_ep.h src/ep/CMakeLists.txt tests/ep/CMakeLists.txt tests/ep/unit/test_tilexr_ep_api_sources.cpp
git commit -m "feat: add ep public api scaffold"
```

## Task 3: Host Validation and Public Dispatch Entry

**Files:**
- Create: `src/ep/host/ep_dispatch_host.h`
- Create: `src/ep/host/ep_dispatch_host.cpp`
- Create: `src/ep/host/ep_launch_context.cpp`
- Create: `src/ep/host/ep_kernel_launch.h`
- Create: `src/ep/host/ep_kernel_launch.cpp`
- Create: `src/ep/host/tilexr_ep_dispatch.cpp`
- Create: `tests/ep/unit/test_tilexr_ep_host_validation.cpp`
- Modify: `src/ep/CMakeLists.txt`
- Modify: `tests/ep/CMakeLists.txt`

- [ ] **Step 1: Write the failing host validation test**

Create `tests/ep/unit/test_tilexr_ep_host_validation.cpp` with:

```cpp
#include <cstdint>
#include <iostream>

#include "ep_dispatch_host.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

void CheckInt(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

TileXREp::EpDispatchParams ValidParams()
{
    static uint16_t x[32] = {};
    static int32_t expertIds[8] = {};
    static uint16_t expandXOut[64] = {};
    static int64_t expertTokenNumsOut[4] = {};
    static int32_t epRecvCountsOut[2] = {};
    static int32_t assistInfoForCombineOut[32] = {};

    TileXREp::EpDispatchParams params {};
    params.x = x;
    params.expertIds = expertIds;
    params.comm = reinterpret_cast<TileXRCommPtr>(0x1000);
    params.bs = 4;
    params.h = 8;
    params.topK = 2;
    params.moeExpertNum = 8;
    params.expandXOut = expandXOut;
    params.expertTokenNumsOut = expertTokenNumsOut;
    params.epRecvCountsOut = epRecvCountsOut;
    params.assistInfoForCombineOut = assistInfoForCombineOut;
    params.dtype = TileXR::TILEXR_DATA_TYPE_FP16;
    params.stream = reinterpret_cast<aclrtStream>(0x2000);
    return params;
}

TileXR::CommArgs ValidCommArgs()
{
    TileXR::CommArgs args {};
    args.rank = 0;
    args.rankSize = 2;
    args.peerMems[0] = reinterpret_cast<GM_ADDR>(0x10000000);
    args.peerMems[1] = reinterpret_cast<GM_ADDR>(0x20000000);
    return args;
}

void TestBasicValidation()
{
    TileXREp::EpDispatchParams params = ValidParams();
    CheckInt("valid basic params", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_SUCCESS);

    params = ValidParams();
    params.x = nullptr;
    CheckInt("null x", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidParams();
    params.comm = nullptr;
    CheckInt("null comm", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidParams();
    params.bs = 0;
    CheckInt("zero bs", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidParams();
    params.dtype = TileXR::TILEXR_DATA_TYPE_FP32;
    CheckInt("unsupported dtype", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

void TestCommValidation()
{
    TileXREp::EpDispatchParams params = ValidParams();
    TileXR::CommArgs args = ValidCommArgs();
    TileXREp::EpWindowConfig config {};
    CheckInt("valid config", TileXREp::TileXREpValidateDispatchConfig(params, args, &config),
        TileXR::TILEXR_SUCCESS);

    args = ValidCommArgs();
    args.rankSize = 0;
    CheckInt("invalid rank size", TileXREp::TileXREpValidateDispatchConfig(params, args, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    args = ValidCommArgs();
    args.peerMems[1] = nullptr;
    CheckInt("missing peer mem", TileXREp::TileXREpValidateDispatchConfig(params, args, &config),
        TileXR::TILEXR_ERROR_NOT_INITIALIZED);

    params = ValidParams();
    params.moeExpertNum = 7;
    args = ValidCommArgs();
    CheckInt("non-divisible experts", TileXREp::TileXREpValidateDispatchConfig(params, args, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

int main()
{
    TestBasicValidation();
    TestCommValidation();
    return g_failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Register the host validation test**

Append to `tests/ep/CMakeLists.txt`:

```cmake
add_executable(test_tilexr_ep_host_validation
    unit/test_tilexr_ep_host_validation.cpp
    ${TILEXR_ROOT}/src/ep/host/ep_layout.cpp
    ${TILEXR_ROOT}/src/ep/host/ep_dispatch_host.cpp
)
target_include_directories(test_tilexr_ep_host_validation PRIVATE ${TILEXR_EP_TEST_INCLUDE_DIRS})
add_test(NAME test_tilexr_ep_host_validation COMMAND test_tilexr_ep_host_validation)
install(TARGETS test_tilexr_ep_host_validation RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
```

- [ ] **Step 3: Run the host validation test to verify it fails**

Run:

```bash
cmake -S tests/ep -B tests/ep/build -DBUILD_TILEXR_EP_DEMO=OFF
cmake --build tests/ep/build --target test_tilexr_ep_host_validation -j"$(nproc)"
```

Expected: compile fails because `ep_dispatch_host.h` and `ep_dispatch_host.cpp` do not exist.

- [ ] **Step 4: Add host validation declarations**

Create `src/ep/host/ep_dispatch_host.h` with:

```cpp
#ifndef TILEXR_EP_HOST_EP_DISPATCH_HOST_H
#define TILEXR_EP_HOST_EP_DISPATCH_HOST_H

#include <cstdint>

#include "acl/acl_base.h"
#include "ep_layout.h"
#include "tilexr_api.h"

namespace TileXREp {

struct EpDispatchParams {
    void *x = nullptr;
    int32_t *expertIds = nullptr;
    TileXRCommPtr comm = nullptr;
    int64_t bs = 0;
    int64_t h = 0;
    int64_t topK = 0;
    int64_t moeExpertNum = 0;
    void *expandXOut = nullptr;
    int64_t *expertTokenNumsOut = nullptr;
    int32_t *epRecvCountsOut = nullptr;
    int32_t *assistInfoForCombineOut = nullptr;
    TileXR::TileXRDataType dtype = TileXR::TILEXR_DATA_TYPE_RESERVED;
    aclrtStream stream = nullptr;
};

struct EpHostLaunchContext {
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    EpWindowConfig window {};
};

int TileXREpValidateBasicDispatchParams(const EpDispatchParams &params);
int TileXREpValidateDispatchConfig(const EpDispatchParams &params, const TileXR::CommArgs &commArgs,
    EpWindowConfig *window);
int TileXREpPrepareLaunchContext(const EpDispatchParams &params, EpHostLaunchContext *context);

} // namespace TileXREp

#endif // TILEXR_EP_HOST_EP_DISPATCH_HOST_H
```

- [ ] **Step 5: Implement host validation without TileXR communicator calls**

Create `src/ep/host/ep_dispatch_host.cpp` with:

```cpp
#include "ep_dispatch_host.h"

namespace TileXREp {

int TileXREpValidateBasicDispatchParams(const EpDispatchParams &params)
{
    if (params.x == nullptr || params.expertIds == nullptr || params.comm == nullptr ||
        params.expandXOut == nullptr || params.expertTokenNumsOut == nullptr ||
        params.epRecvCountsOut == nullptr || params.assistInfoForCombineOut == nullptr ||
        params.stream == nullptr || params.bs <= 0 || params.h <= 0 ||
        params.topK <= 0 || params.moeExpertNum <= 0 ||
        !TileXREpIsSupportedDataType(params.dtype)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpValidateDispatchConfig(const EpDispatchParams &params, const TileXR::CommArgs &commArgs,
    EpWindowConfig *window)
{
    if (window == nullptr || commArgs.rankSize <= 0 ||
        commArgs.rankSize > TileXR::TILEXR_MAX_RANK_SIZE ||
        commArgs.rank < 0 || commArgs.rank >= commArgs.rankSize) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    for (int rank = 0; rank < commArgs.rankSize; ++rank) {
        if (commArgs.peerMems[rank] == nullptr) {
            return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
        }
    }
    return TileXREpBuildWindowConfig(commArgs.rankSize, params.bs, params.h, params.topK,
        params.moeExpertNum, params.dtype, window);
}

} // namespace TileXREp
```

- [ ] **Step 6: Implement TileXR communicator context lookup**

Create `src/ep/host/ep_launch_context.cpp` with:

```cpp
#include "ep_dispatch_host.h"

namespace TileXREp {

int TileXREpPrepareLaunchContext(const EpDispatchParams &params, EpHostLaunchContext *context)
{
    if (context == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *context = EpHostLaunchContext {};

    int ret = TileXRGetCommArgsHost(params.comm, context->hostArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (context->hostArgs == nullptr) {
        *context = EpHostLaunchContext {};
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    ret = TileXRGetCommArgsDev(params.comm, context->devArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }
    if (context->devArgs == nullptr) {
        *context = EpHostLaunchContext {};
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    ret = TileXREpValidateDispatchConfig(params, *context->hostArgs, &context->window);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXREp
```

- [ ] **Step 7: Add temporary kernel launch stub and public entry**

Create `src/ep/host/ep_kernel_launch.h` with:

```cpp
#ifndef TILEXR_EP_HOST_EP_KERNEL_LAUNCH_H
#define TILEXR_EP_HOST_EP_KERNEL_LAUNCH_H

#include "ep_dispatch_host.h"

namespace TileXREp {

int TileXREpLaunchDispatchKernel(const EpDispatchParams &params, const EpHostLaunchContext &context);

} // namespace TileXREp

#endif // TILEXR_EP_HOST_EP_KERNEL_LAUNCH_H
```

Create `src/ep/host/ep_kernel_launch.cpp` with this temporary implementation:

```cpp
#include "ep_kernel_launch.h"

namespace TileXREp {

int TileXREpLaunchDispatchKernel(const EpDispatchParams &params, const EpHostLaunchContext &context)
{
    (void)params;
    (void)context;
    return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
}

} // namespace TileXREp
```

Create `src/ep/host/tilexr_ep_dispatch.cpp` with:

```cpp
#include "tilexr_ep.h"

#include "ep_dispatch_host.h"
#include "ep_kernel_launch.h"

int TileXRMoeEpDispatch(void *x, int32_t *expertIds, TileXRCommPtr comm,
    int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
    void *expandXOut, int64_t *expertTokenNumsOut, int32_t *epRecvCountsOut,
    int32_t *assistInfoForCombineOut, TileXR::TileXRDataType dtype, aclrtStream stream)
{
    TileXREp::EpDispatchParams params {};
    params.x = x;
    params.expertIds = expertIds;
    params.comm = comm;
    params.bs = bs;
    params.h = h;
    params.topK = topK;
    params.moeExpertNum = moeExpertNum;
    params.expandXOut = expandXOut;
    params.expertTokenNumsOut = expertTokenNumsOut;
    params.epRecvCountsOut = epRecvCountsOut;
    params.assistInfoForCombineOut = assistInfoForCombineOut;
    params.dtype = dtype;
    params.stream = stream;

    int ret = TileXREp::TileXREpValidateBasicDispatchParams(params);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    TileXREp::EpHostLaunchContext context {};
    ret = TileXREp::TileXREpPrepareLaunchContext(params, &context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    return TileXREp::TileXREpLaunchDispatchKernel(params, context);
}
```

- [ ] **Step 8: Wire host files into `tilexr-ep`**

Modify `src/ep/CMakeLists.txt`:

```cmake
add_library(tilexr-ep SHARED
    host/ep_layout.cpp
    host/ep_dispatch_host.cpp
    host/ep_launch_context.cpp
    host/ep_kernel_launch.cpp
    host/tilexr_ep_dispatch.cpp
)
```

Also add host/common private include directories if they are not already present:

```cmake
target_include_directories(tilexr-ep
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/host
)
```

- [ ] **Step 9: Run tests and commit**

Run:

```bash
cmake -S tests/ep -B tests/ep/build -DBUILD_TILEXR_EP_DEMO=OFF
cmake --build tests/ep/build --target test_tilexr_ep_host_validation -j"$(nproc)"
tests/ep/build/test_tilexr_ep_host_validation
```

Expected: host validation test exits `0`.

Commit:

```bash
git add src/ep/host/ep_dispatch_host.h src/ep/host/ep_dispatch_host.cpp src/ep/host/ep_launch_context.cpp src/ep/host/ep_kernel_launch.h src/ep/host/ep_kernel_launch.cpp src/ep/host/tilexr_ep_dispatch.cpp src/ep/CMakeLists.txt tests/ep/CMakeLists.txt tests/ep/unit/test_tilexr_ep_host_validation.cpp
git commit -m "feat: add ep host dispatch validation"
```

## Task 4: Kernel Source Skeleton, Real Launcher, and Kernel Source Tests

**Files:**
- Create: `src/ep/kernels/tilexr_ep_dispatch_kernel.cpp`
- Create: `tests/ep/unit/test_tilexr_ep_kernel_sources.cpp`
- Modify: `src/ep/CMakeLists.txt`
- Modify: `src/ep/host/ep_kernel_launch.cpp`
- Modify: `tests/ep/CMakeLists.txt`

- [ ] **Step 1: Write the failing kernel source test**

Create `tests/ep/unit/test_tilexr_ep_kernel_sources.cpp` with:

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
    const std::string fullPath = RepoPath(path);
    std::ifstream input(fullPath.c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << fullPath << std::endl;
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckContains(const std::string& path, const std::string& text, const std::string& needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected " << path << " to contain: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckDoesNotContain(const std::string& path, const std::string& text, const std::string& needle)
{
    const std::string::size_type pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "expected " << path << " not to contain: " << needle << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void TestKernelUsesTileXRPeerMemory()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    const std::string text = ReadFile(path);
    CheckContains(path, text, "extern \"C\" __global__ __aicore__ void tilexr_ep_dispatch_kernel");
    CheckContains(path, text, "launch_tilexr_ep_dispatch_kernel");
    CheckContains(path, text, "CommArgs");
    CheckContains(path, text, "peerMems");
    CheckContains(path, text, "IPC_DATA_OFFSET");
    CheckContains(path, text, "SyncCollectives");
    CheckContains(path, text, "DataCopyPad");
    CheckContains(path, text, "kEpStepWindowCleared");
    CheckContains(path, text, "kEpStepDispatchReady");
}

void TestNoForbiddenDependencies()
{
    const std::vector<std::string> paths = {
        "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp",
        "src/ep/host/ep_kernel_launch.cpp",
        "src/ep/CMakeLists.txt"
    };
    const std::vector<std::string> forbidden = {
        "src/mc2",
        "3rdparty/ops-transformer",
        "GetHcclContext",
        "TileXRUDMARegister",
        "UDMAPut",
        "shmem"
    };
    for (const std::string& path : paths) {
        const std::string text = ReadFile(path);
        for (const std::string& needle : forbidden) {
            CheckDoesNotContain(path, text, needle);
        }
    }
}

} // namespace

int main()
{
    TestKernelUsesTileXRPeerMemory();
    TestNoForbiddenDependencies();
    return g_failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Register the kernel source test**

Append to `tests/ep/CMakeLists.txt`:

```cmake
add_executable(test_tilexr_ep_kernel_sources
    unit/test_tilexr_ep_kernel_sources.cpp
)
target_compile_definitions(test_tilexr_ep_kernel_sources PRIVATE TILEXR_SOURCE_ROOT="${TILEXR_ROOT}")
add_test(NAME test_tilexr_ep_kernel_sources COMMAND test_tilexr_ep_kernel_sources)
install(TARGETS test_tilexr_ep_kernel_sources RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
```

- [ ] **Step 3: Run the kernel source test to verify it fails**

Run:

```bash
cmake -S tests/ep -B tests/ep/build -DBUILD_TILEXR_EP_DEMO=OFF
cmake --build tests/ep/build --target test_tilexr_ep_kernel_sources -j"$(nproc)"
tests/ep/build/test_tilexr_ep_kernel_sources
```

Expected: executable runs and reports missing `src/ep/kernels/tilexr_ep_dispatch_kernel.cpp`.

- [ ] **Step 4: Add the kernel file with the complete MVP algorithm**

Create `src/ep/kernels/tilexr_ep_dispatch_kernel.cpp`. Keep all row payload copies on `DataCopyPad`; direct scalar GM stores are used only for fixed-size control words and counters.

```cpp
#include "comm_args.h"
#include "ep_window.h"
#include "kernel_operator.h"
#include "tilexr_sync.h"

namespace {

constexpr uint32_t kEpUbBytes = 64 * 1024;
constexpr uint32_t kEpSyncUbBytes = 4 * 1024;
constexpr uint32_t kEpCopyTileBytes = kEpUbBytes - kEpSyncUbBytes;

__aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

__aicore__ inline int64_t SlotOffset(int64_t srcRank, int64_t slotBytes)
{
    return TileXREp::kEpWindowHeaderBytes + srcRank * slotBytes;
}

__aicore__ inline int64_t PayloadOffset(int64_t srcRank, int64_t slotBytes)
{
    return SlotOffset(srcRank, slotBytes) + TileXREp::kEpSrcSlotHeaderBytes;
}

__aicore__ inline int64_t AssistOffset(int64_t srcRank, int64_t slotBytes, int64_t payloadBytesPerSlot)
{
    return PayloadOffset(srcRank, slotBytes) + payloadBytesPerSlot;
}

__aicore__ inline void CopyBytesGmToGm(__gm__ uint8_t *dst, __gm__ uint8_t *src,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf, int64_t bytes)
{
    if (bytes <= 0) {
        return;
    }

    AscendC::LocalTensor<uint8_t> local =
        tBuf.GetWithOffset<uint8_t>(kEpCopyTileBytes, kEpSyncUbBytes);
    for (int64_t copied = 0; copied < bytes; copied += kEpCopyTileBytes) {
        int64_t tile = bytes - copied;
        if (tile > kEpCopyTileBytes) {
            tile = kEpCopyTileBytes;
        }

        AscendC::GlobalTensor<uint8_t> srcTensor;
        AscendC::GlobalTensor<uint8_t> dstTensor;
        srcTensor.SetGlobalBuffer(src + copied, static_cast<uint32_t>(tile));
        dstTensor.SetGlobalBuffer(dst + copied, static_cast<uint32_t>(tile));

        AscendC::DataCopyExtParams inParams {1, static_cast<uint32_t>(tile), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> padParams {false, 0, 0, 0};
        AscendC::DataCopyPad(local, srcTensor, inParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

        AscendC::DataCopyExtParams outParams {1, static_cast<uint32_t>(tile), 0, 0, 0};
        AscendC::DataCopyPad(dstTensor, local, outParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void ClearLocalWindow(__gm__ uint8_t *windowBase, int32_t rankSize,
    int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t slotBytes, int64_t payloadBytesPerSlot,
    int64_t assistBytesPerSlot, int64_t totalBytes)
{
    auto header = reinterpret_cast<__gm__ TileXREp::EpWindowHeader *>(windowBase);
    header->magic = TileXREp::kEpWindowMagic;
    header->rankSize = rankSize;
    header->maxRoutesPerSrc = maxRoutesPerSrc;
    header->rowBytes = rowBytes;
    header->slotBytes = slotBytes;
    header->totalBytes = totalBytes;
    header->reserved0 = 0;
    header->reserved1 = 0;
    header->reserved2 = 0;

    for (int32_t src = 0; src < rankSize; ++src) {
        auto slot = reinterpret_cast<__gm__ TileXREp::EpSrcSlotHeader *>(
            windowBase + SlotOffset(src, slotBytes));
        slot->count = 0;
        slot->srcRank = src;
        slot->payloadBytes = payloadBytesPerSlot;
        slot->assistBytes = assistBytesPerSlot;
        slot->reserved0 = 0;
        slot->reserved1 = 0;
        slot->reserved2 = 0;
        slot->reserved3 = 0;
        slot->reserved4 = 0;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void WriteAssist(__gm__ uint8_t *assistBase, int64_t record,
    int32_t srcRank, int32_t tokenId, int32_t topKId, int32_t expertId)
{
    auto tuple = reinterpret_cast<__gm__ TileXREp::EpAssistTuple *>(
        assistBase + record * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)));
    tuple->srcRank = srcRank;
    tuple->tokenId = tokenId;
    tuple->topKId = topKId;
    tuple->expertId = expertId;
}

} // namespace

extern "C" __global__ __aicore__ void tilexr_ep_dispatch_kernel(
    GM_ADDR commArgsGM, GM_ADDR xGM, GM_ADDR expertIdsGM, GM_ADDR expandXOutGM,
    GM_ADDR expertTokenNumsOutGM, GM_ADDR epRecvCountsOutGM, GM_ADDR assistInfoForCombineOutGM,
    int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum, int64_t dtypeBytes,
    int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t payloadBytesPerSlot,
    int64_t assistBytesPerSlot, int64_t slotBytes, int64_t totalBytes, int64_t magic)
{
    if constexpr (g_coreType == AscendC::AIV) {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }

        auto args = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
        const int32_t rank = args->rank;
        const int32_t rankSize = args->rankSize;
        if (rank < 0 || rank >= rankSize || rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE ||
            bs <= 0 || h <= 0 || topK <= 0 || moeExpertNum <= 0 || dtypeBytes <= 0) {
            return;
        }

        GM_ADDR shareAddrs[TileXR::TILEXR_MAX_RANK_SIZE];
        for (int32_t peer = 0; peer < rankSize; ++peer) {
            shareAddrs[peer] = args->peerMems[peer];
            if (shareAddrs[peer] == nullptr) {
                return;
            }
        }

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
        pipe.InitBuffer(tBuf, kEpUbBytes);

        SyncCollectives sync;
        sync.Init(rank, rankSize, shareAddrs, tBuf);

        __gm__ uint8_t *localWindow = reinterpret_cast<__gm__ uint8_t *>(
            shareAddrs[rank] + TileXR::IPC_DATA_OFFSET);
        ClearLocalWindow(localWindow, rankSize, maxRoutesPerSrc, rowBytes, slotBytes,
            payloadBytesPerSlot, assistBytesPerSlot, totalBytes);
        sync.SetInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepWindowCleared);
        for (int32_t src = 0; src < rankSize; ++src) {
            sync.WaitRankInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepWindowCleared, src);
        }

        int64_t dstCounts[TileXR::TILEXR_MAX_RANK_SIZE];
        for (int32_t dst = 0; dst < rankSize; ++dst) {
            dstCounts[dst] = 0;
        }

        auto expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdsGM);
        auto x = reinterpret_cast<__gm__ uint8_t *>(xGM);
        const int64_t localExpertNum = moeExpertNum / rankSize;
        for (int64_t token = 0; token < bs; ++token) {
            for (int64_t topKIdx = 0; topKIdx < topK; ++topKIdx) {
                const int64_t route = token * topK + topKIdx;
                const int32_t expert = expertIds[route];
                if (expert < 0 || expert >= moeExpertNum) {
                    continue;
                }
                const int32_t dstRank = expert / localExpertNum;
                if (dstRank < 0 || dstRank >= rankSize) {
                    continue;
                }
                const int64_t record = dstCounts[dstRank];
                if (record >= maxRoutesPerSrc) {
                    continue;
                }
                dstCounts[dstRank] = record + 1;

                __gm__ uint8_t *dstWindow = reinterpret_cast<__gm__ uint8_t *>(
                    shareAddrs[dstRank] + TileXR::IPC_DATA_OFFSET);
                __gm__ uint8_t *payload = dstWindow + PayloadOffset(rank, slotBytes);
                __gm__ uint8_t *assist = dstWindow + AssistOffset(rank, slotBytes, payloadBytesPerSlot);
                CopyBytesGmToGm(payload + record * rowBytes, x + token * rowBytes, tBuf, rowBytes);
                WriteAssist(assist, record, rank, static_cast<int32_t>(token),
                    static_cast<int32_t>(topKIdx), expert);
            }
        }

        for (int32_t dst = 0; dst < rankSize; ++dst) {
            __gm__ uint8_t *dstWindow = reinterpret_cast<__gm__ uint8_t *>(
                shareAddrs[dst] + TileXR::IPC_DATA_OFFSET);
            auto slot = reinterpret_cast<__gm__ TileXREp::EpSrcSlotHeader *>(
                dstWindow + SlotOffset(rank, slotBytes));
            slot->count = static_cast<int32_t>(dstCounts[dst]);
            slot->srcRank = rank;
            slot->payloadBytes = payloadBytesPerSlot;
            slot->assistBytes = assistBytesPerSlot;
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        sync.SetInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepDispatchReady);
        for (int32_t src = 0; src < rankSize; ++src) {
            sync.WaitRankInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepDispatchReady, src);
        }

        auto expandXOut = reinterpret_cast<__gm__ uint8_t *>(expandXOutGM);
        auto expertTokenNumsOut = reinterpret_cast<__gm__ int64_t *>(expertTokenNumsOutGM);
        auto epRecvCountsOut = reinterpret_cast<__gm__ int32_t *>(epRecvCountsOutGM);
        auto assistInfoForCombineOut = reinterpret_cast<__gm__ TileXREp::EpAssistTuple *>(assistInfoForCombineOutGM);
        for (int64_t localExpert = 0; localExpert < localExpertNum; ++localExpert) {
            expertTokenNumsOut[localExpert] = 0;
        }

        int64_t outRecord = 0;
        for (int32_t src = 0; src < rankSize; ++src) {
            __gm__ uint8_t *slotBase = localWindow + SlotOffset(src, slotBytes);
            auto slot = reinterpret_cast<__gm__ TileXREp::EpSrcSlotHeader *>(slotBase);
            const int32_t count = slot->count;
            epRecvCountsOut[src] = count;
            __gm__ uint8_t *payload = localWindow + PayloadOffset(src, slotBytes);
            auto assist = reinterpret_cast<__gm__ TileXREp::EpAssistTuple *>(
                localWindow + AssistOffset(src, slotBytes, payloadBytesPerSlot));
            for (int32_t record = 0; record < count; ++record) {
                CopyBytesGmToGm(expandXOut + outRecord * rowBytes, payload + record * rowBytes, tBuf, rowBytes);
                const TileXREp::EpAssistTuple tuple = assist[record];
                assistInfoForCombineOut[outRecord] = tuple;
                const int64_t localExpert = tuple.expertId % localExpertNum;
                expertTokenNumsOut[localExpert] = expertTokenNumsOut[localExpert] + 1;
                ++outRecord;
            }
        }
    }
}

void launch_tilexr_ep_dispatch_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs,
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR expandXOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR epRecvCountsOut, GM_ADDR assistInfoForCombineOut, int64_t bs, int64_t h,
    int64_t topK, int64_t moeExpertNum, int64_t dtypeBytes, int64_t maxRoutesPerSrc,
    int64_t rowBytes, int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot,
    int64_t slotBytes, int64_t totalBytes, int64_t magic)
{
    tilexr_ep_dispatch_kernel<<<blockDim, nullptr, stream>>>(commArgs, x, expertIds,
        expandXOut, expertTokenNumsOut, epRecvCountsOut, assistInfoForCombineOut,
        bs, h, topK, moeExpertNum, dtypeBytes, maxRoutesPerSrc, rowBytes,
        payloadBytesPerSlot, assistBytesPerSlot, slotBytes, totalBytes, magic);
}
```

- [ ] **Step 5: Replace the temporary host launcher with the real launcher call**

Replace `src/ep/host/ep_kernel_launch.cpp` with:

```cpp
#include "ep_kernel_launch.h"

extern void launch_tilexr_ep_dispatch_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs,
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR expandXOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR epRecvCountsOut, GM_ADDR assistInfoForCombineOut, int64_t bs, int64_t h,
    int64_t topK, int64_t moeExpertNum, int64_t dtypeBytes, int64_t maxRoutesPerSrc,
    int64_t rowBytes, int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot,
    int64_t slotBytes, int64_t totalBytes, int64_t magic);

namespace TileXREp {

int TileXREpLaunchDispatchKernel(const EpDispatchParams &params, const EpHostLaunchContext &context)
{
    int64_t magic = 0;
    const int magicRet = TileXRCommNextMagic(params.comm, &magic);
    if (magicRet != TileXR::TILEXR_SUCCESS) {
        return magicRet;
    }

    constexpr uint32_t kMvpBlockDim = 1;
    launch_tilexr_ep_dispatch_kernel(kMvpBlockDim, params.stream, context.devArgs,
        reinterpret_cast<GM_ADDR>(params.x), reinterpret_cast<GM_ADDR>(params.expertIds),
        reinterpret_cast<GM_ADDR>(params.expandXOut), reinterpret_cast<GM_ADDR>(params.expertTokenNumsOut),
        reinterpret_cast<GM_ADDR>(params.epRecvCountsOut), reinterpret_cast<GM_ADDR>(params.assistInfoForCombineOut),
        params.bs, params.h, params.topK, params.moeExpertNum, context.window.dtypeBytes,
        context.window.maxRoutesPerSrc, context.window.rowBytes, context.window.payloadBytesPerSlot,
        context.window.assistBytesPerSlot, context.window.slotBytes, context.window.totalBytes, magic);
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXREp
```

- [ ] **Step 6: Add the EP kernel build to `src/ep/CMakeLists.txt`**

Replace `src/ep/CMakeLists.txt` with this full file:

```cmake
include(GNUInstallDirs)

find_program(BISHENG_EXECUTABLE bisheng)
if(NOT BISHENG_EXECUTABLE)
    message(FATAL_ERROR "bisheng not found; source scripts/common_env.sh before building with -DTILEXR_BUILD_EP=ON")
endif()

if(TILEXR_EP_SOC_TYPE STREQUAL "")
    set(TILEXR_EP_SOC_TYPE "Ascend950")
endif()
set(TILEXR_EP_SOC_TYPE "${TILEXR_EP_SOC_TYPE}" CACHE STRING "SOC type used for the TileXR EP kernel")

if(TILEXR_EP_SOC_TYPE STREQUAL "Ascend950")
    set(TILEXR_EP_NPU_ARCH "dav-3510")
    set(TILEXR_EP_AICORE_ARCH "--cce-aicore-arch=dav-c310-vec")
    set(TILEXR_EP_CATLASS_ARCH "3510")
else()
    set(TILEXR_EP_NPU_ARCH "dav-2201")
    set(TILEXR_EP_AICORE_ARCH "--cce-aicore-arch=dav-c220-vec")
    set(TILEXR_EP_CATLASS_ARCH "2201")
endif()

execute_process(
    COMMAND ${BISHENG_EXECUTABLE} -v
    OUTPUT_VARIABLE TILEXR_EP_BISHENG_VERSION_OUTPUT
    ERROR_VARIABLE TILEXR_EP_BISHENG_VERSION_OUTPUT
    RESULT_VARIABLE TILEXR_EP_BISHENG_VERSION_RESULT
)
if(TILEXR_EP_BISHENG_VERSION_RESULT EQUAL 0 AND TILEXR_EP_BISHENG_VERSION_OUTPUT MATCHES "([0-9]{8})")
    set(TILEXR_EP_BISHENG_DATE "${CMAKE_MATCH_1}")
else()
    set(TILEXR_EP_BISHENG_DATE "0")
endif()

if(TILEXR_EP_BISHENG_DATE GREATER_EQUAL 20250428)
    set(TILEXR_EP_KERNEL_COMPILE_OPTIONS
        -xasc
        -Xhost-start -ftrapv -Xhost-end
        --npu-arch=${TILEXR_EP_NPU_ARCH}
        --cce-auto-infer-kernel-type=false
    )
    set(TILEXR_EP_KERNEL_LINK_OPTIONS --cce-fatobj-link)
else()
    set(TILEXR_EP_KERNEL_COMPILE_OPTIONS
        -xcce
        -Xhost-start -ftrapv -Xhost-end
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=true
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
        ${TILEXR_EP_AICORE_ARCH}
    )
    set(TILEXR_EP_KERNEL_LINK_OPTIONS --cce-fatobj-link ${TILEXR_EP_AICORE_ARCH})
endif()

set(TILEXR_EP_KERNEL_SO "${CMAKE_CURRENT_BINARY_DIR}/libtilexr_ep_dispatch_kernel.so")
set(TILEXR_EP_KERNEL_INCLUDES
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
    -I${CMAKE_SOURCE_DIR}/3rdparty
    -I${CMAKE_SOURCE_DIR}/src/include
    -I${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_custom_command(
    OUTPUT "${TILEXR_EP_KERNEL_SO}"
    COMMAND ${BISHENG_EXECUTABLE}
        ${TILEXR_EP_KERNEL_COMPILE_OPTIONS}
        -std=gnu++17
        -fPIC
        -shared
        ${TILEXR_EP_KERNEL_LINK_OPTIONS}
        -DCATLASS_ARCH=${TILEXR_EP_CATLASS_ARCH}
        ${TILEXR_EP_KERNEL_INCLUDES}
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_ep_dispatch_kernel.cpp"
        -L${ASCEND_DRIVER_PATH}/lib64/driver
        -L${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
        -L${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
        -lruntime
        -lascendcl
        -lstdc++
        -lm
        -ltiling_api
        -lplatform
        -lc_sec
        -ldl
        -lnnopbase
        -lpthread
        -o "${TILEXR_EP_KERNEL_SO}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_ep_dispatch_kernel.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/ep_window.h"
        "${CMAKE_SOURCE_DIR}/src/include/comm_args.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_sync.h"
    VERBATIM
    COMMENT "Building TileXR EP dispatch kernel with bisheng"
)
add_custom_target(tilexr_ep_dispatch_kernel ALL DEPENDS "${TILEXR_EP_KERNEL_SO}")

add_library(tilexr-ep SHARED
    host/ep_layout.cpp
    host/ep_dispatch_host.cpp
    host/ep_kernel_launch.cpp
    host/tilexr_ep_dispatch.cpp
)
add_dependencies(tilexr-ep tilexr_ep_dispatch_kernel)

target_include_directories(tilexr-ep
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/host
)

target_link_directories(tilexr-ep
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ASCEND_DRIVER_PATH}/lib64/driver
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
    ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)

target_link_libraries(tilexr-ep
    PRIVATE
    "${TILEXR_EP_KERNEL_SO}"
    tile-comm
    ascendcl
    runtime
    ascend_hal
)

install(TARGETS tilexr-ep LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES "${TILEXR_EP_KERNEL_SO}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_ep.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

- [ ] **Step 7: Run source tests and commit**

Run:

```bash
cmake -S tests/ep -B tests/ep/build -DBUILD_TILEXR_EP_DEMO=OFF
cmake --build tests/ep/build --target test_tilexr_ep_kernel_sources -j"$(nproc)"
tests/ep/build/test_tilexr_ep_kernel_sources
```

Expected: source test exits `0`.

Commit:

```bash
git add src/ep/kernels/tilexr_ep_dispatch_kernel.cpp src/ep/host/ep_kernel_launch.cpp src/ep/CMakeLists.txt tests/ep/CMakeLists.txt tests/ep/unit/test_tilexr_ep_kernel_sources.cpp
git commit -m "feat: add ep dispatch kernel path"
```

## Task 5: EP Test Build Script and README

**Files:**
- Create: `tests/ep/build.sh`
- Create: `tests/ep/README.md`

- [ ] **Step 1: Add `tests/ep/build.sh`**

Create `tests/ep/build.sh` with executable mode:

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TILEXR_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODE=${1:-source-only}

source "${TILEXR_ROOT}/scripts/common_env.sh"
export ARCH="${TILEXR_OS_ARCH}"

if [ "${MODE}" = "full" ]; then
    ROOT_BUILD_DIR="${TILEXR_ROOT}/build_ep"
    cmake -S "${TILEXR_ROOT}" -B "${ROOT_BUILD_DIR}" \
        -DCMAKE_INSTALL_PREFIX="${TILEXR_ROOT}/install" \
        -DTILEXR_BUILD_EP=ON
    cmake --build "${ROOT_BUILD_DIR}" --target install -j"$(nproc)"
    DEMO_OPTION="-DBUILD_TILEXR_EP_DEMO=ON"
elif [ "${MODE}" = "source-only" ]; then
    DEMO_OPTION="-DBUILD_TILEXR_EP_DEMO=OFF"
else
    echo "usage: bash tests/ep/build.sh [source-only|full]"
    exit 2
fi

BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    ${DEMO_OPTION}
cmake --build "${BUILD_DIR}" --target install -j"$(nproc)"

echo "TileXR EP test build complete: ${MODE}"
echo "Installed test binaries: ${INSTALL_DIR}/bin"
```

Run:

```bash
chmod +x tests/ep/build.sh
bash tests/ep/build.sh source-only
tests/ep/install/bin/test_tilexr_ep_layout
tests/ep/install/bin/test_tilexr_ep_api_sources
tests/ep/install/bin/test_tilexr_ep_host_validation
tests/ep/install/bin/test_tilexr_ep_kernel_sources
```

Expected: source-only build succeeds and all four installed tests exit `0`.

- [ ] **Step 2: Add README**

Create `tests/ep/README.md` with:

```markdown
# TileXR EP Dispatch Tests

This tree tests the standalone TileXR EP module under `src/ep`. The EP path is independent from `src/mc2` and does not use HCCL window helpers, shmem, or UDMA in the MVP route.

## Source-Only Tests

    source ../../scripts/common_env.sh
    bash build.sh source-only
    ./install/bin/test_tilexr_ep_layout
    ./install/bin/test_tilexr_ep_api_sources
    ./install/bin/test_tilexr_ep_host_validation
    ./install/bin/test_tilexr_ep_kernel_sources

## Full Hardware Demo

    source ../../scripts/common_env.sh
    bash build.sh full
    bash demo/run_tilexr_ep_dispatch_demo.sh 2

The full mode builds and installs `tile-comm`, `tilexr-ep`, and `libtilexr_ep_dispatch_kernel.so` under the repository `install` directory, then builds the EP demo.

## Remote Blue Verification

    bash demo/deploy_and_run_blue.sh

The remote script syncs the complete repository into `/home/d00520898/tilexr_ep_dispatch_verify/TileXR` on `blue`, initializes submodules, sources `scripts/common_env.sh`, builds full EP artifacts, and runs the two-rank dispatch demo.
```

- [ ] **Step 3: Commit**

Run:

```bash
git add tests/ep/build.sh tests/ep/README.md
git commit -m "test: add ep build helper"
```

## Task 6: Deterministic EP Dispatch Demo

**Files:**
- Create: `tests/ep/demo/tilexr_ep_dispatch_demo.cpp`
- Create: `tests/ep/demo/run_tilexr_ep_dispatch_demo.sh`
- Modify: `tests/ep/CMakeLists.txt`

- [ ] **Step 1: Add demo CMake wiring**

Append to `tests/ep/CMakeLists.txt`:

```cmake
if(BUILD_TILEXR_EP_DEMO)
    if(IS_ABSOLUTE "${TILEXR_INSTALL_LIBDIR}")
        set(TILEXR_INSTALL_LIB_SEARCH_DIR "${TILEXR_INSTALL_LIBDIR}")
    else()
        set(TILEXR_INSTALL_LIB_SEARCH_DIR "${TILEXR_INSTALL_PREFIX}/${TILEXR_INSTALL_LIBDIR}")
    endif()
    if(IS_ABSOLUTE "${TILEXR_INSTALL_INCLUDEDIR}")
        set(TILEXR_INSTALL_INCLUDE_SEARCH_DIR "${TILEXR_INSTALL_INCLUDEDIR}")
    else()
        set(TILEXR_INSTALL_INCLUDE_SEARCH_DIR "${TILEXR_INSTALL_PREFIX}/${TILEXR_INSTALL_INCLUDEDIR}")
    endif()

    find_library(TILEXR_EP_LIB tilexr-ep HINTS "${TILEXR_INSTALL_LIB_SEARCH_DIR}" REQUIRED)
    find_library(TILEXR_COMM_LIB tile-comm HINTS "${TILEXR_INSTALL_LIB_SEARCH_DIR}" REQUIRED)

    set(_tilexr_default_ascend_home_path "$ENV{ASCEND_HOME_PATH}")
    if(NOT _tilexr_default_ascend_home_path)
        set(_tilexr_default_ascend_home_path "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
    set(ASCEND_HOME_PATH "${_tilexr_default_ascend_home_path}" CACHE PATH "Ascend toolkit path")

    set(_tilexr_default_arch "$ENV{ARCH}")
    if(NOT _tilexr_default_arch)
        set(_tilexr_default_arch "$ENV{TILEXR_OS_ARCH}")
    endif()
    if(NOT _tilexr_default_arch)
        execute_process(COMMAND uname -m OUTPUT_VARIABLE _tilexr_default_arch OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_tilexr_default_arch STREQUAL "arm64")
            set(_tilexr_default_arch "aarch64")
        endif()
    endif()
    set(ARCH "${_tilexr_default_arch}" CACHE STRING "Target OS architecture")

    set(_tilexr_default_ascend_driver_path "$ENV{ASCEND_DRIVER_PATH}")
    if(NOT _tilexr_default_ascend_driver_path)
        set(_tilexr_default_ascend_driver_path "/usr/local/Ascend/driver")
    endif()
    set(ASCEND_DRIVER_PATH "${_tilexr_default_ascend_driver_path}" CACHE PATH "Ascend driver path")

    add_executable(tilexr_ep_dispatch_demo
        demo/tilexr_ep_dispatch_demo.cpp
    )
    target_include_directories(tilexr_ep_dispatch_demo PRIVATE
        ${TILEXR_INSTALL_INCLUDE_SEARCH_DIR}
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime
    )
    target_link_directories(tilexr_ep_dispatch_demo PRIVATE
        ${TILEXR_INSTALL_LIB_SEARCH_DIR}
        ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
        ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
        ${ASCEND_DRIVER_PATH}/lib64/driver
    )
    target_link_libraries(tilexr_ep_dispatch_demo
        ${TILEXR_EP_LIB}
        ${TILEXR_COMM_LIB}
        ascendcl
        runtime
        ascend_hal
    )
    install(TARGETS tilexr_ep_dispatch_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
```

- [ ] **Step 2: Add the demo source**

Create `tests/ep/demo/tilexr_ep_dispatch_demo.cpp`. Use FP16 bit patterns in `uint16_t` vectors and compare bitwise; no numeric half conversion is needed.

```cpp
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "tilexr_api.h"
#include "tilexr_ep.h"
#include "tilexr_types.h"

namespace {

constexpr int64_t kBs = 4;
constexpr int64_t kH = 8;
constexpr int64_t kTopK = 2;
constexpr int64_t kMoeExpertNum = 4;

int GetEnvInt(const char *name, int defaultValue)
{
    const char *value = std::getenv(name);
    return value == nullptr ? defaultValue : std::atoi(value);
}

int GetDeviceIdFromEnv(int rank, int npuCount, int firstNpu)
{
    const char *devices = std::getenv("TILEXR_DEMO_DEVICES");
    if (devices != nullptr && devices[0] != '\0') {
        std::string list(devices);
        size_t start = 0;
        int index = 0;
        while (start <= list.size()) {
            const size_t comma = list.find(',', start);
            const size_t end = comma == std::string::npos ? list.size() : comma;
            if (index == rank && end > start) {
                return std::atoi(list.substr(start, end - start).c_str());
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
            ++index;
        }
    }
    return rank % npuCount + firstNpu;
}

bool CheckAcl(int rank, const std::string &step, int ret)
{
    if (ret == ACL_SUCCESS) {
        std::cout << "[rank " << rank << "] " << step << " success" << std::endl;
        return true;
    }
    std::cerr << "[rank " << rank << "] ERROR: " << step << " failed ret=" << ret << std::endl;
    return false;
}

bool CheckTileXR(int rank, const std::string &step, int ret)
{
    if (ret == TileXR::TILEXR_SUCCESS) {
        std::cout << "[rank " << rank << "] " << step << " success" << std::endl;
        return true;
    }
    std::cerr << "[rank " << rank << "] ERROR: " << step << " failed ret=" << ret << std::endl;
    return false;
}

uint16_t XValue(int srcRank, int token, int col)
{
    return static_cast<uint16_t>(srcRank * 10000 + token * 100 + col);
}

std::vector<int32_t> ExpertIds()
{
    return std::vector<int32_t> {
        0, 2,
        1, 3,
        2, 0,
        3, 1
    };
}

bool RouteBelongsToRank(int expertId, int rank, int rankSize)
{
    const int localExpertNum = static_cast<int>(kMoeExpertNum / rankSize);
    return expertId / localExpertNum == rank;
}

bool ValidateOutputs(int rank, int rankSize, const std::vector<uint16_t> &expandX,
    const std::vector<int64_t> &expertTokenNums, const std::vector<int32_t> &recvCounts,
    const std::vector<int32_t> &assist)
{
    const std::vector<int32_t> expertIds = ExpertIds();
    const int localExpertNum = static_cast<int>(kMoeExpertNum / rankSize);
    int64_t outRecord = 0;
    std::vector<int64_t> expectedExpertCounts(localExpertNum, 0);

    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        int32_t expectedRecv = 0;
        for (int64_t token = 0; token < kBs; ++token) {
            for (int64_t topKIdx = 0; topKIdx < kTopK; ++topKIdx) {
                const int route = static_cast<int>(token * kTopK + topKIdx);
                const int expert = expertIds[route];
                if (!RouteBelongsToRank(expert, rank, rankSize)) {
                    continue;
                }
                ++expectedRecv;
                for (int64_t col = 0; col < kH; ++col) {
                    const uint16_t expected = XValue(srcRank, static_cast<int>(token), static_cast<int>(col));
                    const uint16_t actual = expandX[static_cast<size_t>(outRecord * kH + col)];
                    if (actual != expected) {
                        std::cerr << "[rank " << rank << "] DATA MISMATCH record=" << outRecord
                                  << " col=" << col << " actual=" << actual
                                  << " expected=" << expected << std::endl;
                        return false;
                    }
                }
                const size_t assistBase = static_cast<size_t>(outRecord * 4);
                if (assist[assistBase + 0] != srcRank ||
                    assist[assistBase + 1] != token ||
                    assist[assistBase + 2] != topKIdx ||
                    assist[assistBase + 3] != expert) {
                    std::cerr << "[rank " << rank << "] ASSIST MISMATCH record=" << outRecord << std::endl;
                    return false;
                }
                expectedExpertCounts[expert % localExpertNum] += 1;
                ++outRecord;
            }
        }
        if (recvCounts[srcRank] != expectedRecv) {
            std::cerr << "[rank " << rank << "] RECV COUNT MISMATCH src=" << srcRank
                      << " actual=" << recvCounts[srcRank] << " expected=" << expectedRecv << std::endl;
            return false;
        }
    }

    for (int expert = 0; expert < localExpertNum; ++expert) {
        if (expertTokenNums[expert] != expectedExpertCounts[expert]) {
            std::cerr << "[rank " << rank << "] EXPERT COUNT MISMATCH localExpert=" << expert
                      << " actual=" << expertTokenNums[expert]
                      << " expected=" << expectedExpertCounts[expert] << std::endl;
            return false;
        }
    }
    std::cout << "[rank " << rank << "] validation success records=" << outRecord << std::endl;
    return true;
}

void Cleanup(TileXRCommPtr comm, aclrtStream stream, void *xDev, void *expertIdsDev, void *expandXDev,
    void *expertTokenNumsDev, void *recvCountsDev, void *assistDev, int rank, int deviceId)
{
    if (assistDev != nullptr) {
        aclrtFree(assistDev);
    }
    if (recvCountsDev != nullptr) {
        aclrtFree(recvCountsDev);
    }
    if (expertTokenNumsDev != nullptr) {
        aclrtFree(expertTokenNumsDev);
    }
    if (expandXDev != nullptr) {
        aclrtFree(expandXDev);
    }
    if (expertIdsDev != nullptr) {
        aclrtFree(expertIdsDev);
    }
    if (xDev != nullptr) {
        aclrtFree(xDev);
    }
    if (comm != nullptr) {
        CheckTileXR(rank, "TileXRCommDestroy", TileXRCommDestroy(comm));
    }
    if (stream != nullptr) {
        CheckAcl(rank, "aclrtDestroyStream", aclrtDestroyStream(stream));
    }
    CheckAcl(rank, "aclrtResetDevice", aclrtResetDevice(deviceId));
    CheckAcl(rank, "aclFinalize", aclFinalize());
}

} // namespace

int main(int argc, char **argv)
{
    int argIndex = 1;
    const int rankSize = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("RANK_SIZE", 2);
    const int rank = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("RANK", 0);
    const int npuCount = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_DEMO_NPUS", rankSize);
    const int firstNpu = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_DEMO_FIRST_NPU", 0);
    const int deviceId = GetDeviceIdFromEnv(rank, npuCount, firstNpu);
    if (rankSize != 2 || rank < 0 || rank >= rankSize) {
        std::cerr << "ERROR: this MVP demo expects rankSize=2" << std::endl;
        return 1;
    }

    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
    void *xDev = nullptr;
    void *expertIdsDev = nullptr;
    void *expandXDev = nullptr;
    void *expertTokenNumsDev = nullptr;
    void *recvCountsDev = nullptr;
    void *assistDev = nullptr;

    if (!CheckAcl(rank, "aclInit", aclInit(nullptr)) ||
        !CheckAcl(rank, "aclrtSetDevice", aclrtSetDevice(deviceId)) ||
        !CheckAcl(rank, "aclrtCreateStream", aclrtCreateStream(&stream)) ||
        !CheckTileXR(rank, "TileXRCommInitRankLocal", TileXRCommInitRankLocal(rankSize, rank, &comm))) {
        Cleanup(comm, stream, xDev, expertIdsDev, expandXDev, expertTokenNumsDev, recvCountsDev, assistDev,
            rank, deviceId);
        return 1;
    }

    const size_t xCount = static_cast<size_t>(kBs * kH);
    const size_t routeCount = static_cast<size_t>(kBs * kTopK);
    const size_t maxOutRows = static_cast<size_t>(rankSize) * routeCount;
    const size_t localExpertNum = static_cast<size_t>(kMoeExpertNum / rankSize);
    std::vector<uint16_t> xHost(xCount);
    for (int64_t token = 0; token < kBs; ++token) {
        for (int64_t col = 0; col < kH; ++col) {
            xHost[static_cast<size_t>(token * kH + col)] = XValue(rank, token, col);
        }
    }
    std::vector<int32_t> expertIdsHost = ExpertIds();
    std::vector<uint16_t> expandXHost(maxOutRows * static_cast<size_t>(kH), 0xFFFF);
    std::vector<int64_t> expertTokenNumsHost(localExpertNum, -1);
    std::vector<int32_t> recvCountsHost(static_cast<size_t>(rankSize), -1);
    std::vector<int32_t> assistHost(maxOutRows * 4, -1);

    if (!CheckAcl(rank, "aclrtMalloc x", aclrtMalloc(&xDev, xHost.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc expertIds", aclrtMalloc(&expertIdsDev, expertIdsHost.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc expandX", aclrtMalloc(&expandXDev, expandXHost.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc expertTokenNums", aclrtMalloc(&expertTokenNumsDev, expertTokenNumsHost.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc recvCounts", aclrtMalloc(&recvCountsDev, recvCountsHost.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc assist", aclrtMalloc(&assistDev, assistHost.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST))) {
        Cleanup(comm, stream, xDev, expertIdsDev, expandXDev, expertTokenNumsDev, recvCountsDev, assistDev,
            rank, deviceId);
        return 1;
    }

    if (!CheckAcl(rank, "copy x", aclrtMemcpy(xDev, xHost.size() * sizeof(uint16_t), xHost.data(), xHost.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE)) ||
        !CheckAcl(rank, "copy expertIds", aclrtMemcpy(expertIdsDev, expertIdsHost.size() * sizeof(int32_t), expertIdsHost.data(), expertIdsHost.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE)) ||
        !CheckAcl(rank, "copy expandX", aclrtMemcpy(expandXDev, expandXHost.size() * sizeof(uint16_t), expandXHost.data(), expandXHost.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE)) ||
        !CheckAcl(rank, "copy expertTokenNums", aclrtMemcpy(expertTokenNumsDev, expertTokenNumsHost.size() * sizeof(int64_t), expertTokenNumsHost.data(), expertTokenNumsHost.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE)) ||
        !CheckAcl(rank, "copy recvCounts", aclrtMemcpy(recvCountsDev, recvCountsHost.size() * sizeof(int32_t), recvCountsHost.data(), recvCountsHost.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE)) ||
        !CheckAcl(rank, "copy assist", aclrtMemcpy(assistDev, assistHost.size() * sizeof(int32_t), assistHost.data(), assistHost.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE))) {
        Cleanup(comm, stream, xDev, expertIdsDev, expandXDev, expertTokenNumsDev, recvCountsDev, assistDev,
            rank, deviceId);
        return 1;
    }

    const int epRet = TileXRMoeEpDispatch(xDev, reinterpret_cast<int32_t *>(expertIdsDev), comm,
        kBs, kH, kTopK, kMoeExpertNum, expandXDev, reinterpret_cast<int64_t *>(expertTokenNumsDev),
        reinterpret_cast<int32_t *>(recvCountsDev), reinterpret_cast<int32_t *>(assistDev),
        TileXR::TILEXR_DATA_TYPE_FP16, stream);
    if (!CheckTileXR(rank, "TileXRMoeEpDispatch", epRet) ||
        !CheckAcl(rank, "aclrtSynchronizeStream", aclrtSynchronizeStream(stream))) {
        Cleanup(comm, stream, xDev, expertIdsDev, expandXDev, expertTokenNumsDev, recvCountsDev, assistDev,
            rank, deviceId);
        return 1;
    }

    if (!CheckAcl(rank, "copy expandX back", aclrtMemcpy(expandXHost.data(), expandXHost.size() * sizeof(uint16_t), expandXDev, expandXHost.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST)) ||
        !CheckAcl(rank, "copy expertTokenNums back", aclrtMemcpy(expertTokenNumsHost.data(), expertTokenNumsHost.size() * sizeof(int64_t), expertTokenNumsDev, expertTokenNumsHost.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST)) ||
        !CheckAcl(rank, "copy recvCounts back", aclrtMemcpy(recvCountsHost.data(), recvCountsHost.size() * sizeof(int32_t), recvCountsDev, recvCountsHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST)) ||
        !CheckAcl(rank, "copy assist back", aclrtMemcpy(assistHost.data(), assistHost.size() * sizeof(int32_t), assistDev, assistHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST))) {
        Cleanup(comm, stream, xDev, expertIdsDev, expandXDev, expertTokenNumsDev, recvCountsDev, assistDev,
            rank, deviceId);
        return 1;
    }

    const bool ok = ValidateOutputs(rank, rankSize, expandXHost, expertTokenNumsHost, recvCountsHost, assistHost);
    Cleanup(comm, stream, xDev, expertIdsDev, expandXDev, expertTokenNumsDev, recvCountsDev, assistDev,
        rank, deviceId);
    return ok ? 0 : 1;
}
```

- [ ] **Step 3: Add demo runner**

Create `tests/ep/demo/run_tilexr_ep_dispatch_demo.sh` with executable mode:

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EP_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
TILEXR_ROOT=$(cd "${EP_DIR}/../.." && pwd)
INSTALL_DIR="${EP_DIR}/install"

rank_size=${1:-2}
npu_count=${2:-${rank_size}}
first_npu=${3:-0}

source "${TILEXR_ROOT}/scripts/common_env.sh"

export TILEXR_COMM_ID=${TILEXR_COMM_ID:-127.0.0.1:10077}
export TILEXR_DEMO_NPUS=${npu_count}
export TILEXR_DEMO_FIRST_NPU=${first_npu}
export LD_LIBRARY_PATH="${TILEXR_ROOT}/install/lib:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

bin="${INSTALL_DIR}/bin/tilexr_ep_dispatch_demo"
if [ ! -x "${bin}" ]; then
    echo "ERROR: ${bin} not found. Run: cd ${EP_DIR} && bash build.sh full"
    exit 1
fi

log_dir="${EP_DIR}/logs/tilexr_ep_dispatch_demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${log_dir}"

echo "TileXR EP dispatch demo"
echo "Binary: ${bin}"
echo "Rank size: ${rank_size}"
echo "NPU count: ${npu_count}"
echo "First NPU: ${first_npu}"
echo "TILEXR_COMM_ID: ${TILEXR_COMM_ID}"
echo "Log dir: ${log_dir}"

pids=()
for rank in $(seq 0 $((rank_size - 1))); do
    log_file="${log_dir}/rank_${rank}.log"
    echo "Starting rank ${rank}, log=${log_file}"
    RANK=${rank} RANK_SIZE=${rank_size} "${bin}" "${rank_size}" "${rank}" "${npu_count}" "${first_npu}" \
        >"${log_file}" 2>&1 &
    pids+=("$!")
done

ret=0
for idx in "${!pids[@]}"; do
    pid=${pids[$idx]}
    rank=${idx}
    if wait "${pid}"; then
        echo "rank ${rank} finished successfully"
    else
        r=$?
        echo "rank ${rank} failed with exit code ${r}"
        ret=${r}
    fi
done

for rank in $(seq 0 $((rank_size - 1))); do
    log_file="${log_dir}/rank_${rank}.log"
    echo "----- rank ${rank}: ${log_file} -----"
    tail -n 120 "${log_file}" || true
done

exit "${ret}"
```

Run:

```bash
chmod +x tests/ep/demo/run_tilexr_ep_dispatch_demo.sh
```

- [ ] **Step 4: Build source-only tests locally**

Run:

```bash
bash tests/ep/build.sh source-only
tests/ep/install/bin/test_tilexr_ep_api_sources
tests/ep/install/bin/test_tilexr_ep_kernel_sources
```

Expected: both source checks exit `0`.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/ep/CMakeLists.txt tests/ep/demo/tilexr_ep_dispatch_demo.cpp tests/ep/demo/run_tilexr_ep_dispatch_demo.sh
git commit -m "test: add ep dispatch demo"
```

## Task 7: Remote Blue Complete Deployment

**Files:**
- Create: `tests/ep/demo/deploy_and_run_blue.sh`
- Modify: `tests/ep/README.md`

- [ ] **Step 1: Add the remote deployment script**

Create `tests/ep/demo/deploy_and_run_blue.sh` with executable mode:

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TILEXR_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

REMOTE=${TILEXR_EP_REMOTE:-blue}
REMOTE_BASE=${TILEXR_EP_REMOTE_BASE:-/home/d00520898/tilexr_ep_dispatch_verify}
REMOTE_REPO="${REMOTE_BASE}/TileXR"
REMOTE_LOG="${REMOTE_BASE}/deploy_$(date +%Y%m%d_%H%M%S).log"

branch=$(git -C "${TILEXR_ROOT}" rev-parse --abbrev-ref HEAD)
commit=$(git -C "${TILEXR_ROOT}" rev-parse HEAD)

echo "Deploying TileXR EP dispatch verification"
echo "Remote: ${REMOTE}"
echo "Remote repo: ${REMOTE_REPO}"
echo "Branch: ${branch}"
echo "Commit: ${commit}"

ssh "${REMOTE}" "mkdir -p '${REMOTE_BASE}' '${REMOTE_REPO}'"

rsync -a --delete \
    --exclude '.worktrees' \
    --exclude 'build' \
    --exclude 'build_*' \
    --exclude 'install' \
    --exclude 'tests/ep/build' \
    --exclude 'tests/ep/install' \
    --exclude 'tests/ep/logs' \
    "${TILEXR_ROOT}/" "${REMOTE}:${REMOTE_REPO}/"

ssh "${REMOTE}" "bash -lc '
set -euo pipefail
cd \"${REMOTE_REPO}\"
echo \"remote branch source: ${branch}\"
echo \"remote commit source: ${commit}\"
git submodule update --init --recursive
source scripts/common_env.sh
bash tests/ep/build.sh full
bash tests/ep/demo/run_tilexr_ep_dispatch_demo.sh 2
' 2>&1 | tee '${REMOTE_LOG}'"

echo "Remote verification log: ${REMOTE}:${REMOTE_LOG}"
```

Run:

```bash
chmod +x tests/ep/demo/deploy_and_run_blue.sh
```

- [ ] **Step 2: Ensure README names the remote path**

Confirm `tests/ep/README.md` contains:

```markdown
The remote script syncs the complete repository into `/home/d00520898/tilexr_ep_dispatch_verify/TileXR` on `blue`, initializes submodules, sources `scripts/common_env.sh`, builds full EP artifacts, and runs the two-rank dispatch demo.
```

- [ ] **Step 3: Commit**

Run:

```bash
git add tests/ep/demo/deploy_and_run_blue.sh tests/ep/README.md
git commit -m "test: add ep blue deployment script"
```

## Task 8: Local and Remote Verification

**Files:**
- No new source files.

- [ ] **Step 1: Run source-only verification locally**

Run:

```bash
bash tests/ep/build.sh source-only
tests/ep/install/bin/test_tilexr_ep_layout
tests/ep/install/bin/test_tilexr_ep_api_sources
tests/ep/install/bin/test_tilexr_ep_host_validation
tests/ep/install/bin/test_tilexr_ep_kernel_sources
```

Expected: all commands exit `0`.

- [ ] **Step 2: Run root EP build where CANN/bisheng are available**

Run:

```bash
source scripts/common_env.sh
cmake -S . -B build_ep -DCMAKE_INSTALL_PREFIX="${PWD}/install" -DTILEXR_BUILD_EP=ON
cmake --build build_ep --target install -j"$(nproc)"
```

Expected: `install/lib/libtilexr-ep.so`, `install/lib/libtilexr_ep_dispatch_kernel.so`, and `install/include/tilexr_ep.h` exist.

- [ ] **Step 3: Run local hardware demo where at least two NPUs are available**

Run:

```bash
bash tests/ep/build.sh full
bash tests/ep/demo/run_tilexr_ep_dispatch_demo.sh 2
```

Expected: both rank logs end with `validation success records=8`, and the script exits `0`.

- [ ] **Step 4: Run complete deployment on blue**

Run:

```bash
bash tests/ep/demo/deploy_and_run_blue.sh
```

Expected: remote script creates `/home/d00520898/tilexr_ep_dispatch_verify/TileXR`, syncs the complete repository, initializes submodules, builds EP in full mode, runs the two-rank demo, and prints the remote log path.

- [ ] **Step 5: Capture final status**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: only known pre-existing submodule modifications remain outside the committed EP work, and the recent commits show the EP layout, API, validation, kernel, tests, demo, and deployment script.

## Future UDMA Backend Note

The second route is recorded for a later plan: add a UDMA backend using TileXR UDMA registration and signal-put APIs, while keeping this peer-memory backend as fallback. Do not add UDMA headers, registration calls, or put/get calls in the MVP files created by this plan.
