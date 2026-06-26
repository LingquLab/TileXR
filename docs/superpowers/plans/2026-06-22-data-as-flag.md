# DataAsFlag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable TileXR device-side DataAsFlag helper with init, GM-to-GM send, Sum-based check, and blocking check-and-recv APIs.

**Architecture:** Add one installed header-only AscendC utility, `src/include/tilexr_data_as_flag.h`, following the public-header style used by `tilexr_sdma.h` and `tilexr_udma.h`. The helper is scratch-driven: callers pass `LocalTensor<uint8_t>` buffers, and the header internally splits those buffers into send blocks, flag receive rows, Sum result/workspace, and receive-copy UB.

**Tech Stack:** C++14 host tests, AscendC AICore device code, CANN 9.1.0 target headers, CMake/CTest, remote CCE validation on `root@141.62.19.152`.

## Global Constraints

- User-facing reasoning and responses are Chinese.
- Read `CLAUDE.md` before execution; this plan assumes the repo guidance is active.
- CANN version target is 9.1.0.
- The new interface is header-only and installed with `tile-comm`.
- Do not add a host API or new comm lifecycle resource.
- DataAsFlag block layout is fixed: 512B total, 480B payload, 32B flag area.
- The flag value is the first `float` in each 32B flag area, fixed to `1.0f`; the send scratch init fills the whole send scratch with `1.0f`, then send overwrites only payload bytes.
- Data length is described by normal byte count, not token count and not caller-provided 512B block count.
- The implementation computes block count as `ceil(dataBytes / 480)`.
- `DataAsFlagCheck` is a non-blocking single-shot bool check.
- `DataAsFlagCheckAndRecv` is blocking and processes slices: poll one slice, copy that passed slice to `recvGM`, then continue.
- `DataAsFlagCheckAndRecv` splits `recvScratch` as `[flag receive area][sum area][copy-to-recvGM UB area]`.
- Sum area includes 32B-aligned result output and explicit Sum workspace, not only result storage.
- Single-round receive slice size is driven by the payload bytes that fit in the copy-to-`recvGM` UB area after reserving flag and Sum areas.
- Transfer files to `root@141.62.19.152:/home/h00580772` without filename suffixes first; after transfer completes, restore suffixes on the server.

---

## File Structure

- Create `src/include/tilexr_data_as_flag.h`: public device-side DataAsFlag constants and inline functions.
- Modify `src/comm/CMakeLists.txt`: install `tilexr_data_as_flag.h` with other TileXR public headers.
- Modify `CMakeLists.txt`: add `tests/data_as_flag` when testing is enabled.
- Create `tests/data_as_flag/CMakeLists.txt`: standalone/in-tree DataAsFlag test wiring that does not depend on collectives.
- Create `tests/data_as_flag/unit/test_tilexr_data_as_flag_header_compile.cpp`: host-side header compile and constant check.
- Create `tests/data_as_flag/unit/test_tilexr_data_as_flag_source_guard.cpp`: source guard for public header contents, installation wiring, explicit Sum workspace usage, and forbidden GM scalar APIs.
- Use remote scratch directory `/home/h00580772/tilexr_data_as_flag_verify_20260622` for CANN/AscendC validation.

### Task 1: Add Failing Tests And CMake Wiring

**Files:**
- Modify: `CMakeLists.txt`
- Create: `tests/data_as_flag/CMakeLists.txt`
- Create: `tests/data_as_flag/unit/test_tilexr_data_as_flag_header_compile.cpp`
- Create: `tests/data_as_flag/unit/test_tilexr_data_as_flag_source_guard.cpp`

**Interfaces:**
- Consumes: planned header `src/include/tilexr_data_as_flag.h`.
- Produces: CTest targets `test_tilexr_data_as_flag_header_compile` and `test_tilexr_data_as_flag_source_guard`.

- [ ] **Step 1: Register the DataAsFlag tests from the repo root**

In root `CMakeLists.txt`, add this block immediately after the existing `if(BUILD_TESTING OR TILEXR_BUILD_TESTS) enable_testing() endif()` block:

```cmake
if(BUILD_TESTING OR TILEXR_BUILD_TESTS)
    add_subdirectory(tests/data_as_flag)
endif()
```

- [ ] **Step 2: Create standalone test CMake**

Create `tests/data_as_flag/CMakeLists.txt` with:

```cmake
#
# Copyright (c) 2024-2026 TileXR Project
# CMakeLists.txt for TileXR DataAsFlag tests
#

cmake_minimum_required(VERSION 3.16)
project(TileXR_DataAsFlag_Tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(GNUInstallDirs)
enable_testing()

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_executable(test_tilexr_data_as_flag_header_compile
    unit/test_tilexr_data_as_flag_header_compile.cpp
)

target_include_directories(test_tilexr_data_as_flag_header_compile PRIVATE
    ${TILEXR_ROOT}/src/include
)

add_executable(test_tilexr_data_as_flag_source_guard
    unit/test_tilexr_data_as_flag_source_guard.cpp
)

target_compile_definitions(test_tilexr_data_as_flag_source_guard PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

add_test(NAME test_tilexr_data_as_flag_header_compile COMMAND test_tilexr_data_as_flag_header_compile)
add_test(NAME test_tilexr_data_as_flag_source_guard COMMAND test_tilexr_data_as_flag_source_guard)

install(TARGETS
    test_tilexr_data_as_flag_header_compile
    test_tilexr_data_as_flag_source_guard
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
```

- [ ] **Step 3: Create the header compile test**

Create `tests/data_as_flag/unit/test_tilexr_data_as_flag_header_compile.cpp` with:

```cpp
#include <cstdint>
#include <iostream>

#include "tilexr_data_as_flag.h"

int main()
{
    static_assert(TileXR::DATA_AS_FLAG_BLOCK_BYTES == 512U, "DataAsFlag block size must be 512B");
    static_assert(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES == 480U, "DataAsFlag payload size must be 480B");
    static_assert(TileXR::DATA_AS_FLAG_FLAG_BYTES == 32U, "DataAsFlag flag area must be 32B");
    static_assert(TileXR::DATA_AS_FLAG_FLAG_OFFSET_BYTES == 480U, "DataAsFlag flag offset must be 480B");
    static_assert(TileXR::DATA_AS_FLAG_FLAG_FLOATS == 8U, "DataAsFlag flag area must hold 8 floats");
    static_assert(TileXR::DATA_AS_FLAG_READY_VALUE == 1.0f, "DataAsFlag ready value must be float 1.0");

    if (TileXR::DataAsFlagBlockCountForPayloadBytes(0) != 0U) {
        std::cerr << "expected 0 payload bytes to require 0 DataAsFlag blocks" << std::endl;
        return 1;
    }
    if (TileXR::DataAsFlagBlockCountForPayloadBytes(480) != 1U) {
        std::cerr << "expected 480 payload bytes to require 1 DataAsFlag block" << std::endl;
        return 1;
    }
    if (TileXR::DataAsFlagBlockCountForPayloadBytes(481) != 2U) {
        std::cerr << "expected 481 payload bytes to require 2 DataAsFlag blocks" << std::endl;
        return 1;
    }

    std::cout << "TileXR DataAsFlag header compile check passed" << std::endl;
    return 0;
}
```

- [ ] **Step 4: Create the source guard test**

Create `tests/data_as_flag/unit/test_tilexr_data_as_flag_source_guard.cpp` with:

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
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "expected " << path << " not to contain: " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void TestHeaderShape()
{
    const std::string path = "src/include/tilexr_data_as_flag.h";
    const auto text = ReadFile(path);
    CheckContains(path, text, "DATA_AS_FLAG_BLOCK_BYTES = 512");
    CheckContains(path, text, "DATA_AS_FLAG_PAYLOAD_BYTES = 480");
    CheckContains(path, text, "DATA_AS_FLAG_FLAG_BYTES = 32");
    CheckContains(path, text, "DATA_AS_FLAG_FLAG_OFFSET_BYTES = 480");
    CheckContains(path, text, "DATA_AS_FLAG_FLAG_FLOATS");
    CheckContains(path, text, "DATA_AS_FLAG_READY_VALUE = 1.0f");
    CheckContains(path, text, "DataAsFlagBlockCountForPayloadBytes");
    CheckContains(path, text, "DataAsFlagInit");
    CheckContains(path, text, "DataAsFlagSend");
    CheckContains(path, text, "DataAsFlagCheck");
    CheckContains(path, text, "DataAsFlagCheckAndRecv");
}

void TestHeaderUsesExpectedAscendCApis()
{
    const std::string path = "src/include/tilexr_data_as_flag.h";
    const auto text = ReadFile(path);
    CheckContains(path, text, "#include \"adv_api/reduce/sum.h\"");
    CheckContains(path, text, "AscendC::DataCopyPad");
    CheckContains(path, text, "AscendC::Sum<");
    CheckContains(path, text, "AscendC::SumParams");
    CheckContains(path, text, "sharedTmpBuffer");
    CheckContains(path, text, "DataAsFlagSumWorkspaceBytes");
    CheckContains(path, text, "ReinterpretCast<float>");
    CheckContains(path, text, "GetSize()");
    CheckContains(path, text, "DATA_AS_FLAG_SUM_RESULT_BYTES");
    CheckContains(path, text, "DataAsFlagMaxRecvBlocks");
    CheckContains(path, text, "while (!DataAsFlagCheckBatch");
    CheckDoesNotContain(path, text, "GlobalTensor::GetValue");
    CheckDoesNotContain(path, text, "GlobalTensor::SetValue");
    CheckDoesNotContain(path, text, "checkScratchBlockCapacity");
}

void TestInstallWiring()
{
    const std::string path = "src/comm/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_data_as_flag.h");
    CheckContains(path, text, "DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}");
}

void TestRootCMakeWiring()
{
    const std::string path = "CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "add_subdirectory(tests/data_as_flag)");
}

void TestDataAsFlagCMakeWiring()
{
    const std::string path = "tests/data_as_flag/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "add_executable(test_tilexr_data_as_flag_header_compile");
    CheckContains(path, text, "unit/test_tilexr_data_as_flag_header_compile.cpp");
    CheckContains(path, text, "add_executable(test_tilexr_data_as_flag_source_guard");
    CheckContains(path, text, "unit/test_tilexr_data_as_flag_source_guard.cpp");
    CheckContains(path, text, "target_include_directories(test_tilexr_data_as_flag_header_compile PRIVATE");
    CheckContains(path, text, "target_compile_definitions(test_tilexr_data_as_flag_source_guard PRIVATE");
    CheckContains(path, text, "add_test(NAME test_tilexr_data_as_flag_header_compile");
    CheckContains(path, text, "add_test(NAME test_tilexr_data_as_flag_source_guard");
    CheckContains(path, text, "install(TARGETS");
    CheckContains(path, text, "test_tilexr_data_as_flag_header_compile");
    CheckContains(path, text, "test_tilexr_data_as_flag_source_guard");
}

} // namespace

int main()
{
    TestHeaderShape();
    TestHeaderUsesExpectedAscendCApis();
    TestInstallWiring();
    TestRootCMakeWiring();
    TestDataAsFlagCMakeWiring();
    if (g_failures != 0) {
        std::cerr << g_failures << " DataAsFlag source guard checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR DataAsFlag source guard checks passed" << std::endl;
    return 0;
}
```

- [ ] **Step 5: Run the source guard to verify the red state**

Run:

```powershell
cmake -S tests/data_as_flag -B build-dataasflag-tests
cmake --build build-dataasflag-tests --target test_tilexr_data_as_flag_source_guard
.\build-dataasflag-tests\test_tilexr_data_as_flag_source_guard.exe
```

Expected: the final command exits `1` and reports missing `src/include/tilexr_data_as_flag.h` plus missing install wiring.

- [ ] **Step 6: Commit failing tests**

Run:

```powershell
git add CMakeLists.txt tests/data_as_flag/CMakeLists.txt tests/data_as_flag/unit/test_tilexr_data_as_flag_header_compile.cpp tests/data_as_flag/unit/test_tilexr_data_as_flag_source_guard.cpp
git commit -m "test: add data as flag interface checks"
```

### Task 2: Validate Sum Workspace APIs On The Remote CANN Environment

**Files:**
- Remote-only create: `/home/h00580772/tilexr_data_as_flag_verify_20260622/probe_sum_api.cpp`
- Local temporary suffixless file: `D:\opencode_workspace\Codex\TileXR\probe_sum_api`
- No repository files are committed by this task.

**Interfaces:**
- Consumes: `adv_api/reduce/sum.h`, `kernel_operator.h`, and the remote CANN 9.1.0 compiler environment.
- Produces: compile confirmation for `Sum<float>(dst, src, sharedTmpBuffer, SumParams)` and `GetSumMaxMinTmpSize`.

- [ ] **Step 1: Create the remote directory**

Run from local PowerShell:

```powershell
ssh root@141.62.19.152 "mkdir -p /home/h00580772/tilexr_data_as_flag_verify_20260622"
```

Expected: command exits with status `0`.

If SSH prints `Host key verification failed`, run:

```powershell
ssh-keygen -R 141.62.19.152
ssh -o StrictHostKeyChecking=accept-new root@141.62.19.152 "mkdir -p /home/h00580772/tilexr_data_as_flag_verify_20260622"
```

Expected: the second command exits with status `0`.

- [ ] **Step 2: Create the local suffixless Sum probe**

Create local file `D:\opencode_workspace\Codex\TileXR\probe_sum_api` with:

```cpp
#include <cstdint>

#include "kernel_operator.h"
#include "adv_api/reduce/sum.h"

extern "C" __global__ __aicore__ void probe_sum_api_kernel(GM_ADDR debugGM)
{
    if constexpr (g_coreType == AscendC::AIV) {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, 4096);
        AscendC::LocalTensor<float> src = buf.Get<float>();
        AscendC::LocalTensor<float> dst = buf.GetWithOffset<float>(8, 256);
        AscendC::LocalTensor<uint8_t> tmp = buf.GetWithOffset<uint8_t>(2048, 512);
        AscendC::Duplicate<float>(src, 1.0f, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SumParams params {1U, 8U, 8U};
        AscendC::Sum<float>(dst, src, tmp, params);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        if (debugGM != nullptr) {
            reinterpret_cast<__gm__ float*>(debugGM)[0] = dst.GetValue(0);
        }
    }
}

extern "C" void probe_sum_tmp_size()
{
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSumMaxMinTmpSize(8U, sizeof(float), false, maxValue, minValue);
}
```

- [ ] **Step 3: Transfer without suffix, then restore suffix remotely**

Run from local PowerShell:

```powershell
scp D:\opencode_workspace\Codex\TileXR\probe_sum_api root@141.62.19.152:/home/h00580772/tilexr_data_as_flag_verify_20260622/probe_sum_api
ssh root@141.62.19.152 "mv /home/h00580772/tilexr_data_as_flag_verify_20260622/probe_sum_api /home/h00580772/tilexr_data_as_flag_verify_20260622/probe_sum_api.cpp"
```

Expected: both commands exit with status `0`.

- [ ] **Step 4: Compile the Sum probe remotely**

Run:

```powershell
ssh root@141.62.19.152 'bash -lc "cd /home/h00580772/tilexr_data_as_flag_verify_20260622 && source /home/h00580772/TileXR/scripts/common_env.sh && bisheng -xcce -std=gnu++17 -fPIC -shared --cce-aicore-only -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime -I${ASCEND_HOME_PATH}/${ARCH}-linux/include -I${ASCEND_HOME_PATH}/${ARCH}-linux/include/ascendc -I${ASCEND_HOME_PATH}/${ARCH}-linux/ascendc/include probe_sum_api.cpp -o libprobe_sum_api.so && test -f libprobe_sum_api.so"'
```

Expected: command exits with status `0`.

- [ ] **Step 5: Remove the local suffixless probe**

Run:

```powershell
Remove-Item -LiteralPath D:\opencode_workspace\Codex\TileXR\probe_sum_api
```

Expected: file is removed.

### Task 3: Implement The Public Header And Install It

**Files:**
- Create: `src/include/tilexr_data_as_flag.h`
- Modify: `src/comm/CMakeLists.txt`

**Interfaces:**
- Consumes: constants and signatures expected by Task 1, plus the Sum API confirmed by Task 2.
- Produces:
  - `TileXR::DataAsFlagBlockCountForPayloadBytes(uint64_t) -> uint32_t`
  - `TileXR::DataAsFlagInit(AscendC::LocalTensor<uint8_t>&) -> uint32_t`
  - `TileXR::DataAsFlagSend(__gm__ uint8_t*, const __gm__ uint8_t*, uint64_t, AscendC::LocalTensor<uint8_t>&) -> uint32_t`
  - `TileXR::DataAsFlagCheck(const __gm__ uint8_t*, uint64_t, AscendC::LocalTensor<uint8_t>&) -> bool`
  - `TileXR::DataAsFlagCheckAndRecv(const __gm__ uint8_t*, uint64_t, __gm__ uint8_t*, AscendC::LocalTensor<uint8_t>&) -> bool`

- [ ] **Step 1: Create `tilexr_data_as_flag.h`**

Create `src/include/tilexr_data_as_flag.h` with:

```cpp
/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_DATA_AS_FLAG_H
#define TILEXR_DATA_AS_FLAG_H

#include <cstdint>

#include "comm_args.h"

#if TILEXR_ASCENDC_AICORE_COMPILE
#include "adv_api/reduce/sum.h"
#endif

namespace TileXR {

constexpr uint32_t DATA_AS_FLAG_BLOCK_BYTES = 512;
constexpr uint32_t DATA_AS_FLAG_PAYLOAD_BYTES = 480;
constexpr uint32_t DATA_AS_FLAG_FLAG_BYTES = DATA_AS_FLAG_BLOCK_BYTES - DATA_AS_FLAG_PAYLOAD_BYTES;
constexpr uint32_t DATA_AS_FLAG_FLAG_OFFSET_BYTES = DATA_AS_FLAG_PAYLOAD_BYTES;
constexpr uint32_t DATA_AS_FLAG_ALIGN_BYTES = 32;
constexpr uint32_t DATA_AS_FLAG_FLOAT_BYTES = sizeof(float);
constexpr uint32_t DATA_AS_FLAG_FLAG_FLOATS = DATA_AS_FLAG_FLAG_BYTES / DATA_AS_FLAG_FLOAT_BYTES;
constexpr uint32_t DATA_AS_FLAG_SUM_RESULT_BYTES = DATA_AS_FLAG_ALIGN_BYTES;
constexpr float DATA_AS_FLAG_READY_VALUE = 1.0f;

inline uint32_t DataAsFlagBlockCountForPayloadBytes(uint64_t dataBytes)
{
    if (dataBytes == 0) {
        return 0;
    }
    return static_cast<uint32_t>(
        (dataBytes + DATA_AS_FLAG_PAYLOAD_BYTES - 1) / DATA_AS_FLAG_PAYLOAD_BYTES);
}

inline uint64_t DataAsFlagAlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

#if TILEXR_ASCENDC_AICORE_COMPILE

__aicore__ inline uint32_t DataAsFlagScratchBytes(const AscendC::LocalTensor<uint8_t>& scratch)
{
    return static_cast<uint32_t>(scratch.GetSize());
}

__aicore__ inline uint32_t DataAsFlagSumWorkspaceBytes(uint32_t valueCount)
{
    if (valueCount == 0) {
        return 0;
    }
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSumMaxMinTmpSize(valueCount, sizeof(float), false, maxValue, minValue);
    return static_cast<uint32_t>(DataAsFlagAlignUp(minValue, DATA_AS_FLAG_ALIGN_BYTES));
}

__aicore__ inline uint32_t DataAsFlagFlagBytes(uint32_t blockCount)
{
    return blockCount * DATA_AS_FLAG_FLAG_BYTES;
}

__aicore__ inline uint32_t DataAsFlagSumInputValueCount(uint32_t blockCount)
{
    return blockCount * DATA_AS_FLAG_FLAG_FLOATS;
}

__aicore__ inline uint32_t DataAsFlagSumBytes(uint32_t blockCount)
{
    return DATA_AS_FLAG_SUM_RESULT_BYTES +
        DataAsFlagSumWorkspaceBytes(DataAsFlagSumInputValueCount(blockCount));
}

__aicore__ inline bool DataAsFlagSplitCheckScratch(
    AscendC::LocalTensor<uint8_t>& scratch,
    uint32_t blockCount,
    AscendC::LocalTensor<float>& flagLocal,
    AscendC::LocalTensor<float>& sumOut,
    AscendC::LocalTensor<uint8_t>& sharedTmpBuffer)
{
    const uint32_t scratchBytes = DataAsFlagScratchBytes(scratch);
    const uint32_t flagBytes = DataAsFlagFlagBytes(blockCount);
    const uint32_t sumBytes = DataAsFlagSumBytes(blockCount);
    if (blockCount == 0 || scratchBytes < flagBytes + sumBytes) {
        return false;
    }

    flagLocal = scratch.template ReinterpretCast<float>();
    sumOut = scratch[flagBytes].template ReinterpretCast<float>();
    sharedTmpBuffer = scratch[flagBytes + DATA_AS_FLAG_SUM_RESULT_BYTES];
    return true;
}

__aicore__ inline uint32_t DataAsFlagMaxCheckBlocks(uint32_t scratchBytes)
{
    uint32_t blocks = scratchBytes / DATA_AS_FLAG_FLAG_BYTES;
    while (blocks > 0) {
        const uint32_t requiredBytes = DataAsFlagFlagBytes(blocks) + DataAsFlagSumBytes(blocks);
        if (scratchBytes >= requiredBytes) {
            return blocks;
        }
        --blocks;
    }
    return 0;
}

__aicore__ inline uint32_t DataAsFlagMaxRecvBlocks(uint32_t scratchBytes)
{
    uint32_t blocks = scratchBytes / DATA_AS_FLAG_PAYLOAD_BYTES;
    while (blocks > 0) {
        const uint64_t requiredBytes =
            static_cast<uint64_t>(DataAsFlagFlagBytes(blocks)) +
            DataAsFlagSumBytes(blocks) +
            static_cast<uint64_t>(blocks) * DATA_AS_FLAG_PAYLOAD_BYTES;
        if (static_cast<uint64_t>(scratchBytes) >= requiredBytes) {
            return blocks;
        }
        --blocks;
    }
    return 0;
}

__aicore__ inline uint32_t DataAsFlagInit(AscendC::LocalTensor<uint8_t>& sendScratch)
{
    const uint32_t sendBlocks = DataAsFlagScratchBytes(sendScratch) / DATA_AS_FLAG_BLOCK_BYTES;
    if (sendBlocks == 0) {
        return 0;
    }

    AscendC::LocalTensor<float> sendFloat = sendScratch.template ReinterpretCast<float>();
    AscendC::Duplicate<float>(
        sendFloat,
        DATA_AS_FLAG_READY_VALUE,
        sendBlocks * DATA_AS_FLAG_BLOCK_BYTES / sizeof(float));
    AscendC::PipeBarrier<PIPE_ALL>();
    return sendBlocks;
}

__aicore__ inline void DataAsFlagCopyPayloadToScratch(
    AscendC::LocalTensor<uint8_t>& sendScratch,
    const __gm__ uint8_t* srcGM,
    uint64_t srcOffset,
    uint32_t fullBlocks,
    uint32_t tailBytes)
{
    AscendC::GlobalTensor<uint8_t> srcGlobal;
    srcGlobal.SetGlobalBuffer(const_cast<__gm__ uint8_t*>(srcGM + srcOffset));
    AscendC::DataCopyPadExtParams<uint8_t> padParams {false, 0U, 0U, 0U};

    if (fullBlocks > 0) {
        AscendC::DataCopyExtParams fullParams {
            static_cast<uint16_t>(fullBlocks),
            DATA_AS_FLAG_PAYLOAD_BYTES,
            0U,
            DATA_AS_FLAG_FLAG_BYTES / DATA_AS_FLAG_ALIGN_BYTES,
            0U};
        AscendC::DataCopyPad(sendScratch, srcGlobal, fullParams, padParams);
    }
    if (tailBytes > 0) {
        AscendC::DataCopyExtParams tailParams {1U, tailBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(
            sendScratch[fullBlocks * DATA_AS_FLAG_BLOCK_BYTES],
            srcGlobal[static_cast<uint64_t>(fullBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES],
            tailParams,
            padParams);
    }
}

__aicore__ inline void DataAsFlagCopyScratchToDataAsFlagGM(
    __gm__ uint8_t* dstDataAsFlagGM,
    uint32_t dstBlockOffset,
    AscendC::LocalTensor<uint8_t>& sendScratch,
    uint32_t batchBlocks)
{
    AscendC::GlobalTensor<uint8_t> dstGlobal;
    dstGlobal.SetGlobalBuffer(dstDataAsFlagGM + static_cast<uint64_t>(dstBlockOffset) * DATA_AS_FLAG_BLOCK_BYTES);
    AscendC::DataCopyExtParams outParams {
        1U,
        batchBlocks * DATA_AS_FLAG_BLOCK_BYTES,
        0U,
        0U,
        0U};
    AscendC::DataCopyPad(dstGlobal, sendScratch, outParams);
}

__aicore__ inline uint32_t DataAsFlagSend(
    __gm__ uint8_t* dstDataAsFlagGM,
    const __gm__ uint8_t* srcGM,
    uint64_t dataBytes,
    AscendC::LocalTensor<uint8_t>& sendScratch)
{
    if (dstDataAsFlagGM == nullptr || srcGM == nullptr || dataBytes == 0) {
        return 0;
    }

    const uint32_t totalBlocks = DataAsFlagBlockCountForPayloadBytes(dataBytes);
    const uint32_t sendBlockCapacity = DataAsFlagScratchBytes(sendScratch) / DATA_AS_FLAG_BLOCK_BYTES;
    if (sendBlockCapacity == 0) {
        return 0;
    }

    uint32_t sentBlocks = 0;
    uint64_t sentBytes = 0;
    while (sentBlocks < totalBlocks) {
        const uint32_t remainingBlocks = totalBlocks - sentBlocks;
        const uint32_t batchBlocks =
            remainingBlocks < sendBlockCapacity ? remainingBlocks : sendBlockCapacity;
        const uint64_t maxBatchBytes = static_cast<uint64_t>(batchBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint64_t remainingBytes = dataBytes - sentBytes;
        const uint32_t batchPayloadBytes = static_cast<uint32_t>(
            remainingBytes < maxBatchBytes ? remainingBytes : maxBatchBytes);
        const uint32_t fullBlocks = batchPayloadBytes / DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint32_t tailBytes = batchPayloadBytes % DATA_AS_FLAG_PAYLOAD_BYTES;

        DataAsFlagCopyPayloadToScratch(sendScratch, srcGM, sentBytes, fullBlocks, tailBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataAsFlagCopyScratchToDataAsFlagGM(dstDataAsFlagGM, sentBlocks, sendScratch, batchBlocks);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

        sentBlocks += batchBlocks;
        sentBytes += batchPayloadBytes;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return totalBlocks;
}

__aicore__ inline bool DataAsFlagCheckBatch(
    const __gm__ uint8_t* dataAsFlagGM,
    uint32_t blockOffset,
    uint32_t batchBlocks,
    AscendC::LocalTensor<uint8_t>& recvScratch)
{
    AscendC::LocalTensor<float> flagLocal;
    AscendC::LocalTensor<float> sumOut;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;
    if (!DataAsFlagSplitCheckScratch(recvScratch, batchBlocks, flagLocal, sumOut, sharedTmpBuffer)) {
        return false;
    }

    AscendC::GlobalTensor<float> flagGlobal;
    flagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
        const_cast<__gm__ uint8_t*>(
            dataAsFlagGM + static_cast<uint64_t>(blockOffset) * DATA_AS_FLAG_BLOCK_BYTES +
            DATA_AS_FLAG_FLAG_OFFSET_BYTES)));
    AscendC::DataCopyExtParams flagParams {
        static_cast<uint16_t>(batchBlocks),
        sizeof(float),
        DATA_AS_FLAG_BLOCK_BYTES - sizeof(float),
        0U,
        0U};
    AscendC::DataCopyPadExtParams<float> flagPadParams {
        true,
        0U,
        static_cast<uint8_t>(DATA_AS_FLAG_FLAG_FLOATS - 1U),
        0.0f};
    AscendC::DataCopyPad(flagLocal, flagGlobal, flagParams, flagPadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    const uint32_t valueCount = DataAsFlagSumInputValueCount(batchBlocks);
    AscendC::SumParams sumParams {1U, valueCount, valueCount};
    AscendC::Sum<float>(sumOut, flagLocal, sharedTmpBuffer, sumParams);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    return sumOut.GetValue(0) == static_cast<float>(batchBlocks);
}

__aicore__ inline bool DataAsFlagCheck(
    const __gm__ uint8_t* dataAsFlagGM,
    uint64_t dataBytes,
    AscendC::LocalTensor<uint8_t>& recvScratch)
{
    if (dataAsFlagGM == nullptr || dataBytes == 0) {
        return false;
    }
    const uint32_t totalBlocks = DataAsFlagBlockCountForPayloadBytes(dataBytes);
    const uint32_t batchCapacity = DataAsFlagMaxCheckBlocks(DataAsFlagScratchBytes(recvScratch));
    if (batchCapacity == 0) {
        return false;
    }

    uint32_t checkedBlocks = 0;
    while (checkedBlocks < totalBlocks) {
        const uint32_t remainingBlocks = totalBlocks - checkedBlocks;
        const uint32_t batchBlocks = remainingBlocks < batchCapacity ? remainingBlocks : batchCapacity;
        if (!DataAsFlagCheckBatch(dataAsFlagGM, checkedBlocks, batchBlocks, recvScratch)) {
            return false;
        }
        checkedBlocks += batchBlocks;
    }
    return true;
}

__aicore__ inline void DataAsFlagCopyBatchToRecvGM(
    const __gm__ uint8_t* dataAsFlagGM,
    uint32_t blockOffset,
    uint64_t recvOffset,
    uint32_t batchBytes,
    __gm__ uint8_t* recvGM,
    AscendC::LocalTensor<uint8_t>& copyLocal)
{
    const uint32_t fullBlocks = batchBytes / DATA_AS_FLAG_PAYLOAD_BYTES;
    const uint32_t tailBytes = batchBytes % DATA_AS_FLAG_PAYLOAD_BYTES;
    AscendC::DataCopyPadExtParams<uint8_t> padParams {false, 0U, 0U, 0U};

    AscendC::GlobalTensor<uint8_t> dataAsFlagGlobal;
    dataAsFlagGlobal.SetGlobalBuffer(const_cast<__gm__ uint8_t*>(
        dataAsFlagGM + static_cast<uint64_t>(blockOffset) * DATA_AS_FLAG_BLOCK_BYTES));
    if (fullBlocks > 0) {
        AscendC::DataCopyExtParams payloadInParams {
            static_cast<uint16_t>(fullBlocks),
            DATA_AS_FLAG_PAYLOAD_BYTES,
            DATA_AS_FLAG_FLAG_BYTES,
            0U,
            0U};
        AscendC::DataCopyPad(copyLocal, dataAsFlagGlobal, payloadInParams, padParams);
    }
    if (tailBytes > 0) {
        AscendC::DataCopyExtParams tailInParams {1U, tailBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(
            copyLocal[fullBlocks * DATA_AS_FLAG_PAYLOAD_BYTES],
            dataAsFlagGlobal[static_cast<uint64_t>(fullBlocks) * DATA_AS_FLAG_BLOCK_BYTES],
            tailInParams,
            padParams);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<uint8_t> recvGlobal;
    recvGlobal.SetGlobalBuffer(recvGM + recvOffset);
    AscendC::DataCopyExtParams payloadOutParams {1U, batchBytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(recvGlobal, copyLocal, payloadOutParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline bool DataAsFlagCheckAndRecv(
    const __gm__ uint8_t* dataAsFlagGM,
    uint64_t dataBytes,
    __gm__ uint8_t* recvGM,
    AscendC::LocalTensor<uint8_t>& recvScratch)
{
    if (dataAsFlagGM == nullptr || recvGM == nullptr || dataBytes == 0) {
        return false;
    }
    const uint32_t totalBlocks = DataAsFlagBlockCountForPayloadBytes(dataBytes);
    const uint32_t batchCapacity = DataAsFlagMaxRecvBlocks(DataAsFlagScratchBytes(recvScratch));
    if (batchCapacity == 0) {
        return false;
    }

    uint32_t processedBlocks = 0;
    uint64_t processedBytes = 0;
    while (processedBlocks < totalBlocks) {
        const uint32_t remainingBlocks = totalBlocks - processedBlocks;
        const uint32_t batchBlocks = remainingBlocks < batchCapacity ? remainingBlocks : batchCapacity;
        while (!DataAsFlagCheckBatch(dataAsFlagGM, processedBlocks, batchBlocks, recvScratch)) {
        }

        const uint64_t remainingBytes = dataBytes - processedBytes;
        const uint64_t maxBatchBytes = static_cast<uint64_t>(batchBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint32_t batchBytes = static_cast<uint32_t>(
            remainingBytes < maxBatchBytes ? remainingBytes : maxBatchBytes);
        const uint32_t flagBytes = DataAsFlagFlagBytes(batchBlocks);
        const uint32_t sumBytes = DataAsFlagSumBytes(batchBlocks);
        AscendC::LocalTensor<uint8_t> copyLocal = recvScratch[flagBytes + sumBytes];
        DataAsFlagCopyBatchToRecvGM(
            dataAsFlagGM, processedBlocks, processedBytes, batchBytes, recvGM, copyLocal);
        processedBlocks += batchBlocks;
        processedBytes += batchBytes;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return true;
}

#endif // TILEXR_ASCENDC_AICORE_COMPILE

} // namespace TileXR

#endif // TILEXR_DATA_AS_FLAG_H
```

- [ ] **Step 2: Install the new header**

In `src/comm/CMakeLists.txt`, add this file to the `install(FILES ... )` list after `tilexr_sync.h`:

```cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_data_as_flag.h
```

- [ ] **Step 3: Run the DataAsFlag tests**

Run:

```powershell
cmake -S tests/data_as_flag -B build-dataasflag-tests
cmake --build build-dataasflag-tests --target test_tilexr_data_as_flag_source_guard test_tilexr_data_as_flag_header_compile
ctest --test-dir build-dataasflag-tests --output-on-failure
```

Expected: both tests pass and print:

```text
TileXR DataAsFlag source guard checks passed
TileXR DataAsFlag header compile check passed
```

- [ ] **Step 4: Commit public header implementation**

Run:

```powershell
git add src/include/tilexr_data_as_flag.h src/comm/CMakeLists.txt
git commit -m "feat: add data as flag device interface"
```

### Task 4: Remote Validate The New Header In A Kernel Compile

**Files:**
- Remote-only create: `/home/h00580772/tilexr_data_as_flag_verify_20260622/probe_data_as_flag.cpp`
- Local temporary suffixless file: `D:\opencode_workspace\Codex\TileXR\probe_data_as_flag`
- No repository files are committed by this task.

**Interfaces:**
- Consumes: implemented `src/include/tilexr_data_as_flag.h`.
- Produces: target CANN compile validation that all four device interfaces can be referenced from an AICore kernel.

- [ ] **Step 1: Create the local suffixless DataAsFlag probe**

Create `D:\opencode_workspace\Codex\TileXR\probe_data_as_flag` with:

```cpp
#include <cstdint>

#include "kernel_operator.h"
#include "tilexr_data_as_flag.h"

extern "C" __global__ __aicore__ void probe_data_as_flag_kernel(
    GM_ADDR dstDataAsFlagGM,
    GM_ADDR srcGM,
    GM_ADDR recvGM,
    uint64_t dataBytes)
{
    if constexpr (g_coreType == AscendC::AIV) {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> sendBuf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> recvBuf;
        pipe.InitBuffer(sendBuf, 4096);
        pipe.InitBuffer(recvBuf, 8192);
        AscendC::LocalTensor<uint8_t> sendScratch = sendBuf.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> recvScratch = recvBuf.Get<uint8_t>();
        const uint32_t initBlocks = TileXR::DataAsFlagInit(sendScratch);
        if (initBlocks == 0) {
            return;
        }
        (void)TileXR::DataAsFlagSend(
            reinterpret_cast<__gm__ uint8_t*>(dstDataAsFlagGM),
            reinterpret_cast<const __gm__ uint8_t*>(srcGM),
            dataBytes,
            sendScratch);
        (void)TileXR::DataAsFlagCheck(
            reinterpret_cast<const __gm__ uint8_t*>(dstDataAsFlagGM),
            dataBytes,
            recvScratch);
        (void)TileXR::DataAsFlagCheckAndRecv(
            reinterpret_cast<const __gm__ uint8_t*>(dstDataAsFlagGM),
            dataBytes,
            reinterpret_cast<__gm__ uint8_t*>(recvGM),
            recvScratch);
    }
}
```

- [ ] **Step 2: Transfer header and probe without suffixes, then restore suffixes remotely**

Run:

```powershell
scp D:\opencode_workspace\Codex\TileXR\src\include\tilexr_data_as_flag.h root@141.62.19.152:/home/h00580772/tilexr_data_as_flag_verify_20260622/tilexr_data_as_flag
scp D:\opencode_workspace\Codex\TileXR\probe_data_as_flag root@141.62.19.152:/home/h00580772/tilexr_data_as_flag_verify_20260622/probe_data_as_flag
ssh root@141.62.19.152 "cd /home/h00580772/tilexr_data_as_flag_verify_20260622 && mv tilexr_data_as_flag tilexr_data_as_flag.h && mv probe_data_as_flag probe_data_as_flag.cpp"
```

Expected: all commands exit with status `0`.

- [ ] **Step 3: Compile the DataAsFlag probe remotely**

Run:

```powershell
ssh root@141.62.19.152 'bash -lc "cd /home/h00580772/tilexr_data_as_flag_verify_20260622 && source /home/h00580772/TileXR/scripts/common_env.sh && bisheng -xcce -std=gnu++17 -fPIC -shared --cce-aicore-only -I. -I/home/h00580772/TileXR/src/include -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime -I${ASCEND_HOME_PATH}/${ARCH}-linux/include -I${ASCEND_HOME_PATH}/${ARCH}-linux/include/ascendc -I${ASCEND_HOME_PATH}/${ARCH}-linux/ascendc/include probe_data_as_flag.cpp -o libprobe_data_as_flag.so && test -f libprobe_data_as_flag.so"'
```

Expected: command exits with status `0`.

- [ ] **Step 4: Remove the local suffixless probe**

Run:

```powershell
Remove-Item -LiteralPath D:\opencode_workspace\Codex\TileXR\probe_data_as_flag
```

Expected: file is removed.

### Task 5: Run Final Local And Remote Verification

**Files:**
- No source files created.
- Uses build directories `build-dataasflag-tests` and `build-dataasflag-root`.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified repository state and final commits.

- [ ] **Step 1: Verify git status before final checks**

Run:

```powershell
git status --short
```

Expected: output is empty after Task 1 and Task 3 commits.

- [ ] **Step 2: Run local DataAsFlag tests**

Run:

```powershell
cmake -S tests/data_as_flag -B build-dataasflag-tests
cmake --build build-dataasflag-tests --target test_tilexr_data_as_flag_source_guard test_tilexr_data_as_flag_header_compile
ctest --test-dir build-dataasflag-tests --output-on-failure
```

Expected: both DataAsFlag tests pass.

- [ ] **Step 3: Build and install TileXR core headers**

Run:

```powershell
cmake -S . -B build-dataasflag-root -DTILEXR_BUILD_TESTS=ON
cmake --build build-dataasflag-root --target tile-comm
cmake --install build-dataasflag-root --prefix install
```

Expected: `install/include/tilexr_data_as_flag.h` exists.

- [ ] **Step 4: Check remote verification artifacts**

Run:

```powershell
ssh root@141.62.19.152 "ls -l /home/h00580772/tilexr_data_as_flag_verify_20260622/libprobe_sum_api.so /home/h00580772/tilexr_data_as_flag_verify_20260622/libprobe_data_as_flag.so"
```

Expected: both `.so` files are listed.

- [ ] **Step 5: Commit verification-driven source changes**

Run:

```powershell
git status --short
```

Expected: output is empty. If output lists source files changed by remote compile fixes, run:

```powershell
git add src/include/tilexr_data_as_flag.h src/comm/CMakeLists.txt CMakeLists.txt tests/data_as_flag/CMakeLists.txt tests/data_as_flag/unit/test_tilexr_data_as_flag_header_compile.cpp tests/data_as_flag/unit/test_tilexr_data_as_flag_source_guard.cpp
git commit -m "fix: validate data as flag ascendc integration"
```

Expected: commit succeeds when source files are listed by `git status --short`; when status is empty, no commit is created.

## Self-Review

- Spec coverage: Task 3 implements the installed header, fixed 512B/480B/32B layout, byte-count block calculation, four public interfaces, GM-to-GM send, Sum-based check, blocking `CheckAndRecv`, and scratch splitting. Task 1 covers source and header tests. Tasks 2 and 4 cover the remote CANN validation and suffixless transfer rule. Task 5 covers final local and remote verification.
- Placeholder scan: the plan contains concrete file paths, code, commands, and expected results. It avoids `TBD`, `TODO`, `fill in`, and vague test instructions.
- Type consistency: public scratch parameters are `AscendC::LocalTensor<uint8_t>&`; internal flag/result tensors use `AscendC::LocalTensor<float>`; Sum workspace uses `AscendC::LocalTensor<uint8_t>`; function names match the requested `Init`, `Send`, `Check`, and `CheckAndRecv` API set.
