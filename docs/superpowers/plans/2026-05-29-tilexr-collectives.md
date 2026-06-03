# TileXR Collectives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Important update:** Use the split layout in `docs/superpowers/plans/2026-05-29-tilexr-collectives-split-structure.md` before implementing this plan. `TileXR-infra` must remain `libtile-comm.so`; `TileXR-collectives` must be built as a separate optional `libtilexr-collectives.so`.

**Goal:** Add public TileXR AllGather and equal-size AllToAll APIs plus correctness and performance tests modeled after nccl-tests.

**Architecture:** `src/collectives` provides an optional host dispatch layer over the existing TileXR communicator and shared `CommArgs`. The device implementation is adapted from lcal CCL kernels, renamed into the TileXR namespace/symbol scheme, and embedded into `libtilexr-collectives.so`; `libtile-comm.so` stays infra-only and contains no collective API, collective kernel, or collective registration.

**Tech Stack:** C++14 host code, AscendC/CCE device kernels, CMake 3.16, CANN 9.1 runtime/ACL, TileXR `CommArgs`, TileXR public C API.

---

## File Structure

- Keep `src/include/tilexr_api.h`: infra-only public C API. Do not declare `TileXRAllGather` or `TileXRAllToAll` here.
- Create `src/include/tilexr_collectives.h`: optional collectives public C API.
- Create `src/collectives/host/`: public C wrappers, validation, blockDim calculation, kernel registration, and launch.
- Keep `src/comm/CMakeLists.txt`: infra-only `tile-comm` target. Do not add collectives sources or CCE blobs here.
- Create `src/collectives/CMakeLists.txt`: optional `tilexr-collectives` target linked against `tile-comm`.
- Create `src/collectives/kernels/`: TileXR-adapted AllGather/AllToAll AscendC kernel sources and minimal shared headers copied/adapted from lcal.
- Create `src/collectives/kernels/CMakeLists.txt`: build CCE object blobs for collectives.
- Create `tests/collectives/CMakeLists.txt`: build unit, integration, and perf tools.
- Create `tests/collectives/unit/test_tilexr_collectives_api.cpp`: compile/source checks for API export and dependency boundaries.
- Create `tests/collectives/integration/test_tilexr_collectives_correctness.cpp`: multi-rank correctness runner.
- Create `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`: nccl-tests style performance/correctness tool.
- Create `tests/collectives/README.md`: build/run documentation.

## Task 1: Public API and Source-Level Tests

**Files:**
- Modify: `src/include/tilexr_api.h`
- Create: `tests/collectives/CMakeLists.txt`
- Create: `tests/collectives/unit/test_tilexr_collectives_api.cpp`

- [ ] **Step 1: Write the failing API/source test**

Create `tests/collectives/unit/test_tilexr_collectives_api.cpp`:

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
        std::cerr << "failed to open " << path << std::endl;
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void ExpectContains(const std::string& path, const std::string& text, const std::string& needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << path << " missing: " << needle << std::endl;
        ++g_failures;
    }
}

void ExpectNotContains(const std::string& path, const std::string& text, const std::string& needle)
{
    if (text.find(needle) != std::string::npos) {
        std::cerr << path << " must not contain: " << needle << std::endl;
        ++g_failures;
    }
}

void TestPublicApiDeclarations()
{
    const std::string path = "src/include/tilexr_api.h";
    const std::string text = ReadFile(path);
    ExpectContains(path, text, "int TileXRAllGather(void *sendBuf, void *recvBuf, int64_t sendCount,");
    ExpectContains(path, text, "int TileXRAllToAll(void *sendBuf, void *recvBuf, int64_t sendCount,");
    ExpectContains(path, text, "TileXR::TileXRDataType dataType, TileXRCommPtr comm,");
    ExpectNotContains(path, text, "HcclDataType");
}

void TestCollectivesDoNotUseForbiddenDeps()
{
    const std::vector<std::string> paths = {
        "src/collectives/tilexr_collectives.h",
        "src/collectives/tilexr_collectives.cpp",
    };
    const std::vector<std::string> forbidden = {
        "hccl.h",
        "HcclDataType",
        "MKI_LOG",
        "mki/",
        "shmem",
        "aclshmem",
    };
    for (const auto& path : paths) {
        const std::string text = ReadFile(path);
        if (text.empty()) {
            continue;
        }
        for (const auto& needle : forbidden) {
            ExpectNotContains(path, text, needle);
        }
    }
}
} // namespace

int main()
{
    TestPublicApiDeclarations();
    TestCollectivesDoNotUseForbiddenDeps();
    if (g_failures != 0) {
        std::cerr << g_failures << " collectives API checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR collectives API/source checks passed" << std::endl;
    return 0;
}
```

- [ ] **Step 2: Add the collectives test CMake**

Create `tests/collectives/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(TileXR_Collectives_Tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_executable(test_tilexr_collectives_api
    unit/test_tilexr_collectives_api.cpp
)
target_compile_definitions(test_tilexr_collectives_api PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

install(TARGETS test_tilexr_collectives_api
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
```

- [ ] **Step 3: Run the source test and verify it fails**

Run:

```bash
source scripts/common_env.sh
cmake -S tests/collectives -B /tmp/tilexr-collectives-tests -DCMAKE_INSTALL_PREFIX=/tmp/tilexr-collectives-tests-install
cmake --build /tmp/tilexr-collectives-tests --target test_tilexr_collectives_api -j"$(nproc)"
/tmp/tilexr-collectives-tests/test_tilexr_collectives_api
```

Expected: FAIL because `TileXRAllGather`, `TileXRAllToAll`, and `src/collectives` do not exist yet.

- [ ] **Step 4: Add public API declarations**

Modify `src/include/tilexr_api.h` inside the `extern "C"` block after `TileXRGetUDMARegistryDev`:

```cpp
int TileXRAllGather(void *sendBuf, void *recvBuf, int64_t sendCount,
                    TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                    aclrtStream stream);

int TileXRAllToAll(void *sendBuf, void *recvBuf, int64_t sendCount,
                   TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                   aclrtStream stream);
```

Also add the include required for `aclrtStream`:

```cpp
#include <acl/acl_base.h>
```

- [ ] **Step 5: Create stub collectives files for dependency checks**

Create `src/collectives/tilexr_collectives.h`:

```cpp
#ifndef TILEXR_COLLECTIVES_H
#define TILEXR_COLLECTIVES_H

#include <cstdint>
#include <acl/acl_base.h>
#include "../include/tilexr_types.h"
#include "../include/tilexr_api.h"

namespace TileXR {
int AllGather(void *sendBuf, void *recvBuf, int64_t sendCount, TileXRDataType dataType,
              TileXRCommPtr comm, aclrtStream stream);
int AllToAll(void *sendBuf, void *recvBuf, int64_t sendCount, TileXRDataType dataType,
             TileXRCommPtr comm, aclrtStream stream);
} // namespace TileXR

#endif // TILEXR_COLLECTIVES_H
```

Create `src/collectives/tilexr_collectives.cpp`:

```cpp
#include "tilexr_collectives.h"

namespace TileXR {
int AllGather(void *, void *, int64_t, TileXRDataType, TileXRCommPtr, aclrtStream)
{
    return TILEXR_ERROR_INTERNAL;
}

int AllToAll(void *, void *, int64_t, TileXRDataType, TileXRCommPtr, aclrtStream)
{
    return TILEXR_ERROR_INTERNAL;
}
} // namespace TileXR

extern "C" int TileXRAllGather(void *sendBuf, void *recvBuf, int64_t sendCount,
    TileXR::TileXRDataType dataType, TileXRCommPtr comm, aclrtStream stream)
{
    return TileXR::AllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

extern "C" int TileXRAllToAll(void *sendBuf, void *recvBuf, int64_t sendCount,
    TileXR::TileXRDataType dataType, TileXRCommPtr comm, aclrtStream stream)
{
    return TileXR::AllToAll(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}
```

- [ ] **Step 6: Run the source test and verify it passes**

Run:

```bash
cmake --build /tmp/tilexr-collectives-tests --target test_tilexr_collectives_api -j"$(nproc)"
/tmp/tilexr-collectives-tests/test_tilexr_collectives_api
```

Expected: PASS with `TileXR collectives API/source checks passed`.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/include/tilexr_api.h src/collectives/tilexr_collectives.h src/collectives/tilexr_collectives.cpp tests/collectives/CMakeLists.txt tests/collectives/unit/test_tilexr_collectives_api.cpp
git commit -m "feat: declare TileXR collectives API"
```

## Task 2: Host Validation and Loopback Implementation

**Files:**
- Modify: `src/collectives/tilexr_collectives.h`
- Modify: `src/collectives/tilexr_collectives.cpp`
- Modify: `src/comm/CMakeLists.txt`
- Create: `tests/collectives/unit/test_tilexr_collectives_validation.cpp`
- Modify: `tests/collectives/CMakeLists.txt`

- [ ] **Step 1: Write the failing validation test**

Create `tests/collectives/unit/test_tilexr_collectives_validation.cpp`:

```cpp
#include <iostream>
#include "tilexr_collectives.h"

namespace {
int g_failures = 0;

void ExpectEq(const char* name, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << name << " expected " << expected << " got " << actual << std::endl;
        ++g_failures;
    }
}

void TestNullAndInvalidArgs()
{
    int send = 1;
    int recv = 0;
    ExpectEq("allgather null comm",
        TileXR::AllGather(&send, &recv, 1, TileXR::TILEXR_DATA_TYPE_INT32, nullptr, nullptr),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    ExpectEq("alltoall null comm",
        TileXR::AllToAll(&send, &recv, 1, TileXR::TILEXR_DATA_TYPE_INT32, nullptr, nullptr),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    ExpectEq("allgather null send",
        TileXR::AllGather(nullptr, &recv, 1, TileXR::TILEXR_DATA_TYPE_INT32, reinterpret_cast<TileXRCommPtr>(0x1), nullptr),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    ExpectEq("allgather null recv",
        TileXR::AllGather(&send, nullptr, 1, TileXR::TILEXR_DATA_TYPE_INT32, reinterpret_cast<TileXRCommPtr>(0x1), nullptr),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    ExpectEq("allgather zero count",
        TileXR::AllGather(&send, &recv, 0, TileXR::TILEXR_DATA_TYPE_INT32, reinterpret_cast<TileXRCommPtr>(0x1), nullptr),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    ExpectEq("allgather invalid type",
        TileXR::AllGather(&send, &recv, 1, TileXR::TILEXR_DATA_TYPE_RESERVED, reinterpret_cast<TileXRCommPtr>(0x1), nullptr),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}
} // namespace

int main()
{
    TestNullAndInvalidArgs();
    if (g_failures != 0) {
        std::cerr << g_failures << " validation checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR collectives validation checks passed" << std::endl;
    return 0;
}
```

- [ ] **Step 2: Add validation test target**

Append to `tests/collectives/CMakeLists.txt`:

```cmake
add_executable(test_tilexr_collectives_validation
    unit/test_tilexr_collectives_validation.cpp
)
target_include_directories(test_tilexr_collectives_validation PRIVATE
    ${TILEXR_ROOT}/src/collectives
    ${TILEXR_ROOT}/src/include
)
find_library(TILEXR_LIB tile-comm HINTS "${TILEXR_ROOT}/install/lib" "${TILEXR_ROOT}/build/src/comm" REQUIRED)
target_link_libraries(test_tilexr_collectives_validation ${TILEXR_LIB} ascendcl runtime ascend_hal)

install(TARGETS test_tilexr_collectives_validation
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
```

- [ ] **Step 3: Run the validation test and verify it fails**

Run:

```bash
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build-collectives -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build-collectives --target tile-comm -j"$(nproc)"
cmake --install /tmp/tilexr-build-collectives
cmake -S tests/collectives -B /tmp/tilexr-collectives-tests -DCMAKE_INSTALL_PREFIX=/tmp/tilexr-collectives-tests-install
cmake --build /tmp/tilexr-collectives-tests --target test_tilexr_collectives_validation -j"$(nproc)"
/tmp/tilexr-collectives-tests/test_tilexr_collectives_validation
```

Expected: FAIL because stubs return `TILEXR_ERROR_INTERNAL`.

- [ ] **Step 4: Implement validation helpers**

Replace `src/collectives/tilexr_collectives.cpp` with:

```cpp
#include "tilexr_collectives.h"

#include <acl/acl.h>
#include "../comm/tilexr_comm.h"
#include "../comm/tilexr_log.h"
#include "../comm/tilexr_internal.h"

namespace TileXR {
namespace {

bool IsSupportedDataType(TileXRDataType dataType)
{
    return dataType == TILEXR_DATA_TYPE_INT8 ||
           dataType == TILEXR_DATA_TYPE_INT16 ||
           dataType == TILEXR_DATA_TYPE_INT32 ||
           dataType == TILEXR_DATA_TYPE_INT64 ||
           dataType == TILEXR_DATA_TYPE_FP16 ||
           dataType == TILEXR_DATA_TYPE_FP32 ||
           dataType == TILEXR_DATA_TYPE_BFP16;
}

int ValidateArgs(const void *sendBuf, const void *recvBuf, int64_t sendCount,
                 TileXRDataType dataType, TileXRCommPtr comm)
{
    if (comm == nullptr || sendBuf == nullptr || recvBuf == nullptr) {
        TILEXR_LOG(ERROR) << "TileXR collective argument has null pointer";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (sendCount <= 0) {
        TILEXR_LOG(ERROR) << "TileXR collective sendCount must be positive";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (!IsSupportedDataType(dataType)) {
        TILEXR_LOG(ERROR) << "TileXR collective unsupported dataType: " << static_cast<int>(dataType);
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TILEXR_SUCCESS;
}

int LoopBack(const void *sendBuf, void *recvBuf, int64_t count, TileXRDataType dataType, aclrtStream stream)
{
    if (sendBuf == recvBuf) {
        return TILEXR_SUCCESS;
    }
    const int64_t bytes = Count2Size(count, dataType);
    if (bytes <= 0) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    aclError ret = aclrtMemcpyAsync(recvBuf, static_cast<size_t>(bytes), sendBuf, static_cast<size_t>(bytes),
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXR collective loopback memcpy failed: " << ret;
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

} // namespace

int AllGather(void *sendBuf, void *recvBuf, int64_t sendCount, TileXRDataType dataType,
              TileXRCommPtr comm, aclrtStream stream)
{
    int ret = ValidateArgs(sendBuf, recvBuf, sendCount, dataType, comm);
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    TileXRComm *tileComm = static_cast<TileXRComm *>(comm);
    if (tileComm->GetRankSize() <= 1) {
        return LoopBack(sendBuf, recvBuf, sendCount, dataType, stream);
    }
    return TILEXR_ERROR_INTERNAL;
}

int AllToAll(void *sendBuf, void *recvBuf, int64_t sendCount, TileXRDataType dataType,
             TileXRCommPtr comm, aclrtStream stream)
{
    int ret = ValidateArgs(sendBuf, recvBuf, sendCount, dataType, comm);
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    TileXRComm *tileComm = static_cast<TileXRComm *>(comm);
    if (tileComm->GetRankSize() <= 1) {
        return LoopBack(sendBuf, recvBuf, sendCount, dataType, stream);
    }
    if (tileComm->GetRankSize() % RANK_SIZE_TWO != 0) {
        TILEXR_LOG(ERROR) << "TileXRAllToAll requires even rankSize for multi-rank";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TILEXR_ERROR_INTERNAL;
}
} // namespace TileXR

extern "C" int TileXRAllGather(void *sendBuf, void *recvBuf, int64_t sendCount,
    TileXR::TileXRDataType dataType, TileXRCommPtr comm, aclrtStream stream)
{
    return TileXR::AllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

extern "C" int TileXRAllToAll(void *sendBuf, void *recvBuf, int64_t sendCount,
    TileXR::TileXRDataType dataType, TileXRCommPtr comm, aclrtStream stream)
{
    return TileXR::AllToAll(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}
```

- [ ] **Step 5: Add collectives sources to tile-comm**

Modify `src/comm/CMakeLists.txt` by adding the collectives files to `TILEXR_SOURCE_FILE`:

```cmake
        ../collectives/tilexr_collectives.h
        ../collectives/tilexr_collectives.cpp
```

Add `${CMAKE_CURRENT_SOURCE_DIR}/../collectives` to `target_include_directories(tile-comm PRIVATE ...)`.

- [ ] **Step 6: Run validation and tile-comm build**

Run:

```bash
cmake --build /tmp/tilexr-collectives-tests --target test_tilexr_collectives_validation -j"$(nproc)"
/tmp/tilexr-collectives-tests/test_tilexr_collectives_validation
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build-collectives -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build-collectives --target tile-comm -j"$(nproc)"
```

Expected: validation PASS; `tile-comm` builds.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/collectives src/comm/CMakeLists.txt tests/collectives
git commit -m "feat: add TileXR collectives host validation"
```

## Task 3: Host Launch Path and BlockDim Rules

**Files:**
- Modify: `src/comm/tilexr_internal.h`
- Modify: `src/comm/ccl_kernel_args.h`
- Modify: `src/comm/tilexr_internal.cpp`
- Modify: `src/comm/tilexr_comm.h`
- Modify: `src/collectives/tilexr_collectives.cpp`

- [ ] **Step 1: Consolidate kernel args type**

Modify `src/comm/tilexr_internal.h` to include `ccl_kernel_args.h` and remove the global `struct AscendCCLKernelArgs` definition:

```cpp
#include "ccl_kernel_args.h"
```

Keep this declaration using the namespaced type:

```cpp
int LoadMTE(TileXRType cclType, AscendCCLKernelArgs &args, uint32_t blockDim, TileXRDataType dataType, aclrtStream stream);
```

Ensure `src/comm/ccl_kernel_args.h` remains the only definition:

```cpp
namespace TileXR {
struct AscendCCLKernelArgs {
    const void *input = nullptr;
    const void *output = nullptr;
    const void *commArgsPtr = nullptr;
    int64_t count = 0;
    int64_t magic = 0;
    int op = 0;
    int root = 0;
    const void *scale = nullptr;
    int64_t scaleCount = 0;
    const void *offset = nullptr;
};
}
```

- [ ] **Step 2: Add magic and blockDim helpers, then launch implementation**

First add a public magic allocator to `src/comm/tilexr_comm.h`:

```cpp
int64_t NextMagic()
{
    return magic_++;
}
```

Then in `src/collectives/tilexr_collectives.cpp`, add these helpers inside the anonymous namespace:

```cpp
bool GetParallel()
{
    static int parallel = -1;
    if (parallel == -1) {
        const char *env = std::getenv("LCCL_PARALLEL");
        parallel = (env != nullptr && (std::string(env) == "1" || std::string(env) == "true")) ? 1 : 0;
    }
    return parallel == 1;
}

uint32_t GetAllGatherBlockNum(uint32_t rankSize, int64_t dataSize, uint32_t extraFlag)
{
    constexpr uint32_t axRankSize = 16;
    constexpr uint32_t twoBlockNum = 2;
    constexpr uint32_t quickOneshotRankSize = 2;
    constexpr uint32_t allGatherHDBRingBlockNum = 32;
    constexpr uint32_t cceSmallDataSize = 2 * 1024 * 1024;
    constexpr int64_t smallDataSize910a3 = 32 * 1024 * 1024;
    constexpr uint32_t smallRankSize = 8;

    if ((extraFlag & ExtraFlag::TOPO_910B2C) != 0 && rankSize == axRankSize) {
        constexpr uint32_t axBlockNum = 10;
        return axBlockNum;
    }
    if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) {
        return rankSize * twoBlockNum;
    }
    if (GetParallel()) {
        return rankSize;
    }
    if ((extraFlag & ExtraFlag::TOPO_910_93) != 0 &&
        (dataSize > smallDataSize910a3 || rankSize > smallRankSize) &&
        rankSize > quickOneshotRankSize && rankSize % quickOneshotRankSize == 0) {
        return allGatherHDBRingBlockNum;
    }
    return (rankSize == quickOneshotRankSize || dataSize >= cceSmallDataSize) ? rankSize * twoBlockNum : rankSize;
}

uint32_t GetAllToAllBlockNum(uint32_t rankSize, int64_t dataSize, uint32_t extraFlag)
{
    constexpr uint32_t twoStepBlockNum = 16;
    constexpr uint32_t twoBlockNum = 2;
    constexpr int64_t smallDataSize = 1 * 1024 * 1024;
    constexpr uint32_t smallRankSize = 8;

    if ((extraFlag & ExtraFlag::TOPO_910_93) != 0) {
        if (rankSize <= smallRankSize && dataSize > smallDataSize &&
            dataSize % static_cast<int64_t>(smallRankSize * smallRankSize * rankSize) == 0) {
            return twoStepBlockNum * twoBlockNum;
        }
        return rankSize <= twoStepBlockNum ? rankSize * twoBlockNum : twoStepBlockNum * twoBlockNum;
    }
    return rankSize * twoBlockNum;
}

int LaunchCollective(TileXRType type, TileXRComm *comm, void *sendBuf, void *recvBuf,
                     int64_t kernelCount, TileXRDataType dataType, aclrtStream stream)
{
    const int64_t dataSize = Count2Size(kernelCount, dataType);
    if (dataSize <= 0) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    const uint32_t rankSize = static_cast<uint32_t>(comm->GetRankSize());
    const uint32_t extraFlag = comm->GetCommArgs()->extraFlag;
    uint32_t blockDim = 0;
    if (type == TileXRType::ALL_GATHER) {
        blockDim = GetAllGatherBlockNum(rankSize, dataSize, extraFlag);
    } else {
        blockDim = GetAllToAllBlockNum(rankSize, dataSize, extraFlag);
    }
    AscendCCLKernelArgs args = {sendBuf, recvBuf, comm->GetCommArgsPtr(), kernelCount, 0, 0, 0};
    args.magic = comm->NextMagic();
    return LoadMTE(type, args, blockDim, dataType, stream);
}
```

- [ ] **Step 3: Replace multi-rank stubs with LoadMTE launch**

In `AllGather`, replace the final `return TILEXR_ERROR_INTERNAL;` with:

```cpp
return LaunchCollective(TileXRType::ALL_GATHER, tileComm, sendBuf, recvBuf, sendCount, dataType, stream);
```

In `AllToAll`, replace the final `return TILEXR_ERROR_INTERNAL;` with:

```cpp
const int64_t kernelCount = sendCount * static_cast<int64_t>(tileComm->GetRankSize());
if (kernelCount / tileComm->GetRankSize() != sendCount) {
    return TILEXR_ERROR_PARA_CHECK_FAIL;
}
return LaunchCollective(TileXRType::ALL2ALL, tileComm, sendBuf, recvBuf, kernelCount, dataType, stream);
```

- [ ] **Step 4: Build host code**

Run:

```bash
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build-collectives -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build-collectives --target tile-comm -j"$(nproc)"
```

Expected: compile succeeds. Runtime multi-rank calls may still fail until kernel binary embedding is complete.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/comm src/collectives
git commit -m "feat: launch TileXR collectives kernels from host API"
```

## Task 4: Kernel Binary Build and Registration

**Files:**
- Create: `src/collectives/kernels/CMakeLists.txt`
- Create: `src/collectives/kernels/tilexr_lccl_op.cpp`
- Create/Adapt: `src/collectives/kernels/*.h`
- Modify: `src/comm/CMakeLists.txt`
- Modify: `src/comm/tilexr_internal.cpp`
- Modify: `src/comm/tilexr_comm.cpp`

- [ ] **Step 1: Copy and adapt minimal lcal kernel sources**

Copy these lcal files into `src/collectives/kernels/` and rename include guards/namespaces from `Lcal/LCAL/Lcal...` to `TileXR/TILEXR/TileXR...`:

```text
3rdparty/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/sync_collectives.h
3rdparty/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/collectives.h
3rdparty/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/datacopy_gm2gm.h
3rdparty/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/datacopy_gm2gm_delay.h
3rdparty/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/ipc_queue.h
3rdparty/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/allgather.h
3rdparty/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/91093/allgather_hierarchy_double_ring.h
3rdparty/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/91093/all2all_hierarchy.h
3rdparty/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/91093/all2all_hierarchy_small.h
3rdparty/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/lccl_op.h
```

In adapted headers:

```cpp
#include "../../include/comm_args.h"
#include "../../include/tilexr_types.h"
```

Replace exported function prefixes:

```text
LcalAllGather_ -> TileXRAllGather_
LcalAll2All_   -> TileXRAll2All_
LCAL_MAX_RANK_SIZE -> TILEXR_MAX_RANK_SIZE
LCAL_BLOCK_NUM_MULTI -> 1
```

- [ ] **Step 2: Add TileXR kernel translation unit**

Create `src/collectives/kernels/tilexr_lccl_op.cpp`:

```cpp
#include "lccl_op.h"

LCCL_TYPE_AIV_FUNC(LCCL_ALLGATHER_FUNC_AUTO_DEF);
LCCL_TYPE_AIV_FUNC(LCCL_ALL2ALL_FUNC_AUTO_DEF);
```

Keep only macros needed by AllGather and All2All in the adapted `lccl_op.h`.

- [ ] **Step 3: Add kernel CMake**

Create `src/collectives/kernels/CMakeLists.txt`:

```cmake
enable_language(CCE)

set(CCE_COMPILE_OPTION
    -O2 -std=gnu++17
    --cce-aicore-only
    -Wno-deprecated-declarations
    "SHELL:-mllvm -cce-aicore-long-call"
    "SHELL:-mllvm -cce-aicore-function-stack-size=16000"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm --cce-aicore-jump-expand=true"
)

set(AIV_ARCH dav-c220-vec)
set_source_files_properties(tilexr_lccl_op.cpp PROPERTIES LANGUAGE CCE)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/include
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp
    ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/interface
)

add_library(tilexr_collectives_op_tmp OBJECT tilexr_lccl_op.cpp)
target_compile_options(tilexr_collectives_op_tmp PRIVATE
    ${CCE_COMPILE_OPTION}
    --cce-aicore-arch=${AIV_ARCH}
)

set(TILEXR_COLLECTIVES_OP "${CMAKE_CURRENT_BINARY_DIR}/tilexr_collectives_op.o")
add_custom_command(
    OUTPUT "${TILEXR_COLLECTIVES_OP}"
    COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
        "CMakeFiles/tilexr_collectives_op_tmp.dir/tilexr_lccl_op.cpp.o"
        --static -o "${TILEXR_COLLECTIVES_OP}" --allow-multiple-definition
    COMMAND truncate -c -s ${TILEXR_1OP_BIN_SIZE} "${TILEXR_COLLECTIVES_OP}"
    DEPENDS tilexr_collectives_op_tmp
    VERBATIM
)
add_custom_target(tilexr_collectives_op DEPENDS "${TILEXR_COLLECTIVES_OP}")
```

- [ ] **Step 4: Embed generated CCE object**

Modify `src/comm/CMakeLists.txt`:

```cmake
set(TILEXR_1OP_BIN_SIZE 3000000)
add_compile_definitions(TILEXR_1OP_BIN_SIZE=${TILEXR_1OP_BIN_SIZE})
add_subdirectory(../collectives/kernels ${CMAKE_CURRENT_BINARY_DIR}/collectives_kernels)
add_dependencies(tile-comm tilexr_collectives_op)
set_source_files_properties(tilexr_internal.cpp PROPERTIES
    OBJECT_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/collectives_kernels/tilexr_collectives_op.o"
)
```

Modify `src/comm/tilexr_internal.cpp` to replace the placeholder:

```cpp
extern const int TILEXR_CCE_BIN_STR[];
asm(R"(.section .rodata, "a", @progbits
.global TILEXR_CCE_BIN_STR
TILEXR_CCE_BIN_STR:
.incbin "collectives_kernels/tilexr_collectives_op.o"
.byte 0
.previous)");
```

If the relative `.incbin` path is not accepted, generate a header with `xxd -i` in CMake and include that header instead.

- [ ] **Step 5: Register kernels during comm init**

In `src/comm/tilexr_comm.cpp`, uncomment the registration in `InitCommon`:

```cpp
if (RegistKernel(isEnableMsprofOp_) != TILEXR_SUCCESS) {
    TILEXR_LOG(ERROR) << "RegistKernel failed";
    return TILEXR_ERROR_INTERNAL;
}
```

- [ ] **Step 6: Build tile-comm with kernels**

Run:

```bash
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build-collectives -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build-collectives --target tile-comm -j"$(nproc)"
cmake --install /tmp/tilexr-build-collectives
```

Expected: CCE object builds, `tile-comm` links, and install succeeds.

- [ ] **Step 7: Dependency verification**

Run:

```bash
ldd install/lib/libtile-comm.so | tee /tmp/tilexr-collectives-ldd.txt
grep -Ei "hcomm|hccl|shmem|aclshmem|ops-transformer|mki" /tmp/tilexr-collectives-ldd.txt
```

Expected: first command lists CANN runtime/driver libraries; second command produces no output and exits nonzero.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/collectives/kernels src/comm/CMakeLists.txt src/comm/tilexr_internal.cpp src/comm/tilexr_comm.cpp
git commit -m "feat: embed TileXR collectives kernels"
```

## Task 5: Multi-Rank Correctness Test Runner

**Files:**
- Create: `tests/collectives/integration/test_tilexr_collectives_correctness.cpp`
- Modify: `tests/collectives/CMakeLists.txt`
- Create: `tests/collectives/run_collectives_correctness.sh`

- [ ] **Step 1: Write correctness runner**

Create `tests/collectives/integration/test_tilexr_collectives_correctness.cpp`:

```cpp
#include <acl/acl.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "tilexr_api.h"

namespace {
int Pattern(int rank, int peer, int index)
{
    return rank * 1000000 + peer * 1000 + index;
}

int GetArg(int argc, char **argv, int index, int fallback)
{
    return argc > index ? std::atoi(argv[index]) : fallback;
}

bool CheckAcl(const std::string& name, aclError ret)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << name << " failed: " << ret << std::endl;
        return false;
    }
    return true;
}

bool CheckTileXR(const std::string& name, int ret)
{
    if (ret != TileXR::TILEXR_SUCCESS) {
        std::cerr << name << " failed: " << ret << std::endl;
        return false;
    }
    return true;
}

bool RunAllGather(int rank, int rankSize, int count, TileXRCommPtr comm, aclrtStream stream)
{
    std::vector<int> hostSend(count);
    std::vector<int> hostRecv(rankSize * count, -1);
    for (int i = 0; i < count; ++i) {
        hostSend[i] = Pattern(rank, 0, i);
    }
    int *devSend = nullptr;
    int *devRecv = nullptr;
    const size_t sendBytes = hostSend.size() * sizeof(int);
    const size_t recvBytes = hostRecv.size() * sizeof(int);
    if (!CheckAcl("aclrtMalloc allgather send", aclrtMalloc(reinterpret_cast<void **>(&devSend), sendBytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl("aclrtMalloc allgather recv", aclrtMalloc(reinterpret_cast<void **>(&devRecv), recvBytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
        return false;
    }
    bool ok = CheckAcl("copy allgather send", aclrtMemcpy(devSend, sendBytes, hostSend.data(), sendBytes, ACL_MEMCPY_HOST_TO_DEVICE)) &&
              CheckAcl("copy allgather recv", aclrtMemcpy(devRecv, recvBytes, hostRecv.data(), recvBytes, ACL_MEMCPY_HOST_TO_DEVICE)) &&
              CheckTileXR("TileXRAllGather", TileXRAllGather(devSend, devRecv, count, TileXR::TILEXR_DATA_TYPE_INT32, comm, stream)) &&
              CheckAcl("sync allgather", aclrtSynchronizeStream(stream)) &&
              CheckAcl("copy allgather result", aclrtMemcpy(hostRecv.data(), recvBytes, devRecv, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int src = 0; ok && src < rankSize; ++src) {
        for (int i = 0; i < count; ++i) {
            int expected = Pattern(src, 0, i);
            int actual = hostRecv[src * count + i];
            if (actual != expected) {
                std::cerr << "AllGather mismatch rank=" << rank << " src=" << src
                          << " i=" << i << " expected=" << expected << " actual=" << actual << std::endl;
                ok = false;
                break;
            }
        }
    }
    aclrtFree(devSend);
    aclrtFree(devRecv);
    return ok;
}

bool RunAllToAll(int rank, int rankSize, int count, TileXRCommPtr comm, aclrtStream stream)
{
    std::vector<int> hostSend(rankSize * count);
    std::vector<int> hostRecv(rankSize * count, -1);
    for (int dst = 0; dst < rankSize; ++dst) {
        for (int i = 0; i < count; ++i) {
            hostSend[dst * count + i] = Pattern(rank, dst, i);
        }
    }
    int *devSend = nullptr;
    int *devRecv = nullptr;
    const size_t bytes = hostSend.size() * sizeof(int);
    if (!CheckAcl("aclrtMalloc alltoall send", aclrtMalloc(reinterpret_cast<void **>(&devSend), bytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl("aclrtMalloc alltoall recv", aclrtMalloc(reinterpret_cast<void **>(&devRecv), bytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
        return false;
    }
    bool ok = CheckAcl("copy alltoall send", aclrtMemcpy(devSend, bytes, hostSend.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE)) &&
              CheckAcl("copy alltoall recv", aclrtMemcpy(devRecv, bytes, hostRecv.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE)) &&
              CheckTileXR("TileXRAllToAll", TileXRAllToAll(devSend, devRecv, count, TileXR::TILEXR_DATA_TYPE_INT32, comm, stream)) &&
              CheckAcl("sync alltoall", aclrtSynchronizeStream(stream)) &&
              CheckAcl("copy alltoall result", aclrtMemcpy(hostRecv.data(), bytes, devRecv, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int src = 0; ok && src < rankSize; ++src) {
        for (int i = 0; i < count; ++i) {
            int expected = Pattern(src, rank, i);
            int actual = hostRecv[src * count + i];
            if (actual != expected) {
                std::cerr << "AllToAll mismatch rank=" << rank << " src=" << src
                          << " i=" << i << " expected=" << expected << " actual=" << actual << std::endl;
                ok = false;
                break;
            }
        }
    }
    aclrtFree(devSend);
    aclrtFree(devRecv);
    return ok;
}
} // namespace

int main(int argc, char **argv)
{
    int rankSize = GetArg(argc, argv, 1, 2);
    int rank = GetArg(argc, argv, 2, 0);
    int count = GetArg(argc, argv, 3, 16);
    int firstNpu = GetArg(argc, argv, 4, 0);
    int deviceId = firstNpu + rank;

    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
    bool ok = CheckAcl("aclInit", aclInit(nullptr)) &&
              CheckAcl("aclrtSetDevice", aclrtSetDevice(deviceId)) &&
              CheckAcl("aclrtCreateStream", aclrtCreateStream(&stream)) &&
              CheckTileXR("TileXRCommInitRankLocal", TileXRCommInitRankLocal(rankSize, rank, &comm)) &&
              RunAllGather(rank, rankSize, count, comm, stream) &&
              RunAllToAll(rank, rankSize, count, comm, stream);
    if (comm != nullptr) {
        TileXRCommDestroy(comm);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ok ? 0 : 1;
}
```

- [ ] **Step 2: Add integration target**

Append to `tests/collectives/CMakeLists.txt`:

```cmake
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

find_library(TILEXR_LIB tile-comm HINTS "${TILEXR_ROOT}/install/lib" "${TILEXR_ROOT}/build/src/comm" REQUIRED)

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

add_executable(test_tilexr_collectives_correctness
    integration/test_tilexr_collectives_correctness.cpp
)
target_link_libraries(test_tilexr_collectives_correctness
    ${TILEXR_LIB}
    ascendcl
    runtime
    ascend_hal
)
install(TARGETS test_tilexr_collectives_correctness
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
```

- [ ] **Step 3: Add rank launcher script**

Create `tests/collectives/run_collectives_correctness.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

rank_size="${1:-2}"
count="${2:-16}"
first_npu="${3:-0}"
bin_dir="${4:-./install/bin}"

pids=()
for ((rank = 0; rank < rank_size; rank++)); do
  "${bin_dir}/test_tilexr_collectives_correctness" "${rank_size}" "${rank}" "${count}" "${first_npu}" \
    > "collectives_correctness_rank${rank}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  tail -n 80 collectives_correctness_rank*.log
fi
exit "$status"
```

Run:

```bash
chmod +x tests/collectives/run_collectives_correctness.sh
```

- [ ] **Step 4: Build and run correctness**

Run:

```bash
source scripts/common_env.sh
cmake -S tests/collectives -B /tmp/tilexr-collectives-tests -DCMAKE_INSTALL_PREFIX="$PWD/tests/collectives/install"
cmake --build /tmp/tilexr-collectives-tests --target test_tilexr_collectives_correctness -j"$(nproc)"
cmake --install /tmp/tilexr-collectives-tests
cd tests/collectives
./run_collectives_correctness.sh 2 16 0 ./install/bin
```

Expected: command exits `0`; per-rank logs contain no mismatch.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/collectives
git commit -m "test: add TileXR collectives correctness runner"
```

## Task 6: nccl-tests Style Performance Tool

**Files:**
- Create: `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`
- Modify: `tests/collectives/CMakeLists.txt`

- [ ] **Step 1: Write perf tool**

Create `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp` with a single binary supporting `--op allgather|alltoall`, `--min-bytes`, `--max-bytes`, `--step-factor`, `--iters`, `--warmup-iters`, `--datatype`, `--rank-size`, `--rank`, `--first-npu`, `--check`, and `--csv`.

Use these timing primitives:

```cpp
aclrtEvent start = nullptr;
aclrtEvent stop = nullptr;
aclrtCreateEvent(&start);
aclrtCreateEvent(&stop);
aclrtRecordEvent(start, stream);
for (int i = 0; i < iters; ++i) {
    TileXRAllGather(sendBuf, recvBuf, count, dataType, comm, stream);
}
aclrtRecordEvent(stop, stream);
aclrtSynchronizeEvent(stop);
float elapsedMs = 0.0f;
aclrtEventElapsedTime(&elapsedMs, start, stop);
double avgUs = static_cast<double>(elapsedMs) * 1000.0 / static_cast<double>(iters);
```

Use these bandwidth formulas:

```cpp
double algBwGbps = static_cast<double>(bytesPerRankOutput) / avgUs / 1000.0;
double busBwGbps = algBwGbps;
if (op == "allgather") {
    busBwGbps = algBwGbps * static_cast<double>(rankSize - 1) / static_cast<double>(rankSize);
} else if (op == "alltoall") {
    busBwGbps = algBwGbps * static_cast<double>(rankSize - 1) / static_cast<double>(rankSize);
}
```

Print this header once per rank 0:

```text
op dtype ranks bytes count iters algbw(GB/s) busbw(GB/s) avg(us) min(us) max(us) errors
```

- [ ] **Step 2: Add perf target**

Append to `tests/collectives/CMakeLists.txt`:

```cmake
add_executable(tilexr_collective_perf
    tilexr-tests/tilexr_collective_perf.cpp
)
target_link_libraries(tilexr_collective_perf
    ${TILEXR_LIB}
    ascendcl
    runtime
    ascend_hal
)
install(TARGETS tilexr_collective_perf
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
```

- [ ] **Step 3: Build perf tool**

Run:

```bash
source scripts/common_env.sh
cmake -S tests/collectives -B /tmp/tilexr-collectives-tests -DCMAKE_INSTALL_PREFIX="$PWD/tests/collectives/install"
cmake --build /tmp/tilexr-collectives-tests --target tilexr_collective_perf -j"$(nproc)"
cmake --install /tmp/tilexr-collectives-tests
```

Expected: `tests/collectives/install/bin/tilexr_collective_perf` exists.

- [ ] **Step 4: Run smoke perf**

Run rank 0 and rank 1 in separate processes:

```bash
cd tests/collectives
./install/bin/tilexr_collective_perf --op allgather --rank-size 2 --rank 0 --first-npu 0 --min-bytes 64 --max-bytes 4096 --iters 10 --warmup-iters 2 --datatype int32 --check 1 > perf_rank0.log 2>&1 &
./install/bin/tilexr_collective_perf --op allgather --rank-size 2 --rank 1 --first-npu 0 --min-bytes 64 --max-bytes 4096 --iters 10 --warmup-iters 2 --datatype int32 --check 1 > perf_rank1.log 2>&1 &
wait
```

Expected: rank 0 output includes the table header and `errors` column is `0`.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/collectives/tilexr-tests/tilexr_collective_perf.cpp tests/collectives/CMakeLists.txt
git commit -m "test: add TileXR collectives perf tool"
```

## Task 7: Documentation and Final Verification

**Files:**
- Create: `tests/collectives/README.md`
- Modify: `README.md`

- [ ] **Step 1: Document collectives tests**

Create `tests/collectives/README.md`:

```markdown
# TileXR Collectives Tests

This directory validates TileXR public collectives:

- `TileXRAllGather`
- `TileXRAllToAll`

## Build

```bash
cd /home/TileXR
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build-collectives -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build-collectives --target tile-comm -j"$(nproc)"
cmake --install /tmp/tilexr-build-collectives

cmake -S tests/collectives -B /tmp/tilexr-collectives-tests -DCMAKE_INSTALL_PREFIX="$PWD/tests/collectives/install"
cmake --build /tmp/tilexr-collectives-tests -j"$(nproc)"
cmake --install /tmp/tilexr-collectives-tests
```

## Correctness

```bash
cd /home/TileXR/tests/collectives
./run_collectives_correctness.sh 2 16 0 ./install/bin
```

Arguments are `rank_size`, `count`, and `first_npu`.

## Performance

```bash
./install/bin/tilexr_collective_perf --op allgather --rank-size 2 --rank 0 --first-npu 0 --min-bytes 64 --max-bytes 1M --iters 100 --warmup-iters 20 --datatype fp16 --check 1
```

Run one process per rank. The output columns are:

- `algbw(GB/s)`: bytes visible to the collective user divided by average kernel time.
- `busbw(GB/s)`: algorithm bandwidth scaled by `(rankSize - 1) / rankSize`.
- `errors`: correctness mismatches when `--check 1`.
```

- [ ] **Step 2: Update top-level README API list**

In `README.md`, add these entries under “Important host-side entry points”:

```markdown
- `TileXRAllGather`
- `TileXRAllToAll`
```

- [ ] **Step 3: Final build and dependency verification**

Run:

```bash
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build-collectives -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build-collectives --target tile-comm -j"$(nproc)"
cmake --install /tmp/tilexr-build-collectives
cmake -S tests/collectives -B /tmp/tilexr-collectives-tests -DCMAKE_INSTALL_PREFIX="$PWD/tests/collectives/install"
cmake --build /tmp/tilexr-collectives-tests -j"$(nproc)"
cmake --install /tmp/tilexr-collectives-tests
ldd install/lib/libtile-comm.so | tee /tmp/tilexr-final-ldd.txt
! grep -Ei "hcomm|hccl|shmem|aclshmem|ops-transformer|mki" /tmp/tilexr-final-ldd.txt
```

Expected: all builds pass; forbidden dependency grep prints nothing.

- [ ] **Step 4: Run available tests**

Run:

```bash
/tmp/tilexr-collectives-tests/test_tilexr_collectives_api
/tmp/tilexr-collectives-tests/test_tilexr_collectives_validation
```

On a machine with at least 2 available NPUs, also run:

```bash
cd tests/collectives
./run_collectives_correctness.sh 2 16 0 ./install/bin
```

Expected: all commands exit `0`.

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md tests/collectives/README.md
git commit -m "docs: document TileXR collectives tests"
```

## Self-Review

- Spec coverage: public C API, TileXRDataType, equal AllToAll, no forbidden dependencies, lcal-derived kernel path, correctness tests, and nccl-tests style perf tool are each covered by tasks.
- Placeholder scan: no task uses TBD/TODO language; every implementation step names concrete files, commands, and expected outcomes.
- Type consistency: public APIs use `TileXR::TileXRDataType`; host helpers use `TileXRDataType`; kernel launch uses existing `TileXRType::ALL_GATHER` and `TileXRType::ALL2ALL`.
