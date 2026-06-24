# P2P Transport Trim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trim the UDMA P2P performance demo to expose only `direct_urma`, `memory`, and `data_as_flag`, with `direct_urma` using the current parallel multi-jetty implementation internally.

**Architecture:** Keep one UDMA benchmark path and two peer-memory comparison paths. The public transport enum/parser/CSV/docs will only know the three user-facing names, while the `direct_urma` host launcher calls the existing parallel multi-jetty kernel. Remove obsolete single-WQE, multi-WQE, serial multi-jetty, and fixed-WQE benchmark paths.

**Tech Stack:** C++14, Ascend C kernels, Bash demo runners, host-only C++ unit tests, GitHub PR docs.

## Global Constraints

- User-facing transport names must be exactly `direct_urma`, `memory`, and `data_as_flag`.
- `direct_urma` must use the current parallel multi-jetty implementation internally.
- `direct_urma` with `block_dim=1` and one QP must behave like the previous single-QP direct URMA path.
- `direct_urma` with `block_dim=N` and `TILEXR_UDMA_QP_NUM=N` must use up to `N` QPs/jettys in parallel.
- Keep QP-indexed device helpers and UDMA layout support.
- Remove fixed-WQE-only window sizing and mismatch-check logic.
- Do not touch unrelated untracked files such as `.kilo/`.
- Source repository guidance from `CLAUDE.md` remains in effect.

---

## File Structure

- `tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp`: host-only behavior tests for supported transports, rejected aliases, CSV formatting, window sizing, and aggregation.
- `tests/udma/demo/tilexr_udma_p2p_perf_config.h`: transport enum, parsing, names, validation, window sizing, effective byte accounting, mismatch checking, CSV row formatting, and row aggregation.
- `tests/udma/demo/tilexr_udma_demo.cpp`: host launch declarations and transport dispatch. This should dispatch `P2PTransport::DirectUrma` to the parallel multi-jetty launcher.
- `tests/udma/demo/tilexr_udma_demo_kernel.cpp`: Ascend C benchmark kernels and launch wrappers. Keep the parallel multi-jetty kernel, rename its launcher to the public `launch_tilexr_udma_p2p_perf`, and remove obsolete benchmark kernels/wrappers.
- `tests/udma/demo/run_tilexr_udma_p2p_perf.sh`: runner argument handling and `TILEXR_UDMA_QP_NUM` setup.
- `tests/udma/demo/run_tilexr_udma_p2p_concurrency_sweep.sh`: sweep defaults, already expected to remain `direct_urma,memory,data_as_flag`.
- `docs/TileXR_UDMA_P2P_PERF_GUIDE.md` and `tests/udma/demo/README.md`: user-facing documentation.

---

### Task 1: Update Config Tests First

**Files:**
- Modify: `tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp`
- Test: `tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp`

**Interfaces:**
- Consumes: current `TileXR::Demo::P2PTransport`, `ParseP2PTransport`, `P2PTransportName`, `P2PTransportWindowBytes`, `P2PEffectiveTransferBytes`, `CountP2PTransportMismatches`, `FormatP2PPerfCsvRow`, `ValidateP2PPerfOptions`
- Produces: expected behavior that later implementation must satisfy:
  - only `DirectUrma`, `Memory`, `DataAsFlag`, and `Invalid` transport enum values are referenced by tests
  - obsolete transport strings parse to `Invalid`
  - `direct_urma` CSV rows keep the name `direct_urma`
  - direct URMA effective bytes are the payload bytes, not `payload * blockDim`

- [ ] **Step 1: Replace transport name assertions**

Replace the block after `DirectionName` so it only checks the three supported names:

```cpp
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::DirectUrma) == "direct_urma",
        "direct_urma transport name mismatch");
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::Memory) == "memory",
        "memory transport name mismatch");
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::DataAsFlag) == "data_as_flag",
        "data_as_flag transport name mismatch");
```

- [ ] **Step 2: Replace parse assertions**

Keep the positive parse checks for `direct_urma`, `memory`, and `data_as_flag`, and add explicit rejection checks:

```cpp
    Require(TileXR::Demo::ParseP2PTransport("direct_urma") == TileXR::Demo::P2PTransport::DirectUrma,
        "direct_urma transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("udma") == TileXR::Demo::P2PTransport::DirectUrma,
        "udma alias parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("memory") == TileXR::Demo::P2PTransport::Memory,
        "memory transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("data_as_flag") == TileXR::Demo::P2PTransport::DataAsFlag,
        "data_as_flag transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("direct_urma_multi_wqe") == TileXR::Demo::P2PTransport::Invalid,
        "direct_urma_multi_wqe must be rejected");
    Require(TileXR::Demo::ParseP2PTransport("direct_urma_multi_jetty") == TileXR::Demo::P2PTransport::Invalid,
        "direct_urma_multi_jetty must be rejected");
    Require(TileXR::Demo::ParseP2PTransport("direct_urma_multi_jetty_parallel") == TileXR::Demo::P2PTransport::Invalid,
        "direct_urma_multi_jetty_parallel must be rejected");
    Require(TileXR::Demo::ParseP2PTransport("direct_urma_multi_jetty_parallel_fixed_wqe") ==
            TileXR::Demo::P2PTransport::Invalid,
        "direct_urma_multi_jetty_parallel_fixed_wqe must be rejected");
```

- [ ] **Step 3: Remove fixed-WQE helper assertions**

Delete assertions that call:

```cpp
TileXR::Demo::P2PFixedWqeStrideBytes(...)
TileXR::Demo::P2PFixedWqeWindowBytes(...)
TileXR::Demo::P2PTransportWindowBytes(TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallelFixedWqe, ...)
```

Add this replacement assertion:

```cpp
    Require(TileXR::Demo::P2PTransportWindowBytes(TileXR::Demo::P2PTransport::DirectUrma, 4096, 8) == 4096,
        "direct_urma window must equal payload bytes");
```

- [ ] **Step 4: Replace validation assertions for removed transports**

Remove validation checks for `DirectUrmaMultiWqe`, `DirectUrmaMultiJetty`, `DirectUrmaMultiJettyParallel`, and `DirectUrmaMultiJettyParallelFixedWqe`. Add this direct URMA validation check:

```cpp
    options.transport = TileXR::Demo::P2PTransport::DirectUrma;
    options.traffic = TileXR::Demo::P2PTraffic::BiDir;
    options.blockDim = 8;
    Require(TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error),
        "valid direct_urma multi-jetty options rejected");
```

- [ ] **Step 5: Replace mismatch-check assertions**

Delete the `fixedWqeBytes` setup and the two fixed-WQE mismatch-check assertions. Add this simple direct URMA assertion:

```cpp
    Require(TileXR::Demo::CountP2PTransportMismatches(
                bytes, pattern, 4096, TileXR::Demo::P2PTransport::DirectUrma, 8) == 1,
        "direct_urma mismatch checker must validate payload bytes");
```

This assertion comes after `bytes[17] ^= 0xff;`.

- [ ] **Step 6: Replace CSV assertions for removed transports**

Delete the CSV assertions for `DirectUrmaMultiWqe`, `DirectUrmaMultiJetty`, `DirectUrmaMultiJettyParallel`, and `DirectUrmaMultiJettyParallelFixedWqe`. Add this direct URMA multi-block CSV assertion:

```cpp
    row.transport = TileXR::Demo::P2PTransport::DirectUrma;
    row.traffic = TileXR::Demo::P2PTraffic::UniDir;
    row.blockDim = 8;
    const std::string directUrmaParallelCsv = TileXR::Demo::FormatP2PPerfCsvRow(row);
    Require(directUrmaParallelCsv ==
            "direct_urma,unidir,8,1to0,1,0,2,4096,20,8.000,0.000,0.000,0.512,0.512,0,0,logs/run\n",
        "direct_urma parallel csv row mismatch");
```

- [ ] **Step 7: Run focused unit test and verify RED**

Run:

```powershell
cmake --build tests/udma/build --target test_tilexr_udma_p2p_perf_config
```

If the build directory does not exist, run:

```powershell
cmake -S tests/udma -B tests/udma/build
cmake --build tests/udma/build --target test_tilexr_udma_p2p_perf_config
```

Expected before production changes: build fails because enum members such as `DirectUrmaMultiWqe` still exist in production but tests now expect old strings to parse as `Invalid`, or the executable fails with one of the new rejection messages.

- [ ] **Step 8: Commit failing tests only**

Do not commit a failing test state. This step is intentionally a review gate: inspect `git diff tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp`, then proceed to Task 2 without committing.

---

### Task 2: Trim Config Helpers

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_p2p_perf_config.h`
- Test: `tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp`

**Interfaces:**
- Consumes: failing tests from Task 1
- Produces:
  - `enum class P2PTransport { DirectUrma, Memory, DataAsFlag, Invalid }`
  - `ParseP2PTransport("direct_urma"|"udma") -> DirectUrma`
  - removed transport strings return `Invalid`
  - `P2PTransportWindowBytes(DirectUrma, bytes, blockDim) -> bytes`
  - `P2PEffectiveTransferBytes(DirectUrma, bytes, blockDim) -> bytes`

- [ ] **Step 1: Trim `P2PTransport` enum**

Change the enum to:

```cpp
enum class P2PTransport {
    DirectUrma,
    Memory,
    DataAsFlag,
    Invalid,
};
```

- [ ] **Step 2: Trim `P2PTransportName`**

Keep only:

```cpp
inline const char* P2PTransportName(P2PTransport transport)
{
    switch (transport) {
        case P2PTransport::DirectUrma:
            return "direct_urma";
        case P2PTransport::Memory:
            return "memory";
        case P2PTransport::DataAsFlag:
            return "data_as_flag";
        default:
            return "invalid";
    }
}
```

- [ ] **Step 3: Trim `ParseP2PTransport`**

Keep only supported names and aliases:

```cpp
inline P2PTransport ParseP2PTransport(const std::string& name)
{
    if (name == "direct_urma" || name == "udma") {
        return P2PTransport::DirectUrma;
    }
    if (name == "memory" || name == "ipc" || name == "datacopy") {
        return P2PTransport::Memory;
    }
    if (name == "data_as_flag" || name == "data-as-flag" || name == "daf") {
        return P2PTransport::DataAsFlag;
    }
    return P2PTransport::Invalid;
}
```

- [ ] **Step 4: Remove fixed-WQE helpers**

Delete:

```cpp
inline uint64_t P2PAlignUp(uint64_t value, uint64_t alignment)
inline uint64_t P2PFixedWqeStrideBytes(uint64_t payloadBytes)
inline uint64_t P2PFixedWqeWindowBytes(uint64_t payloadBytes, uint32_t blockDim)
```

Then simplify:

```cpp
inline uint64_t P2PTransportWindowBytes(P2PTransport transport, uint64_t payloadBytes, uint32_t blockDim)
{
    (void)blockDim;
    return P2PTransportWindowBytes(transport, payloadBytes);
}
```

- [ ] **Step 5: Simplify effective bytes**

Replace `P2PEffectiveTransferBytes` with:

```cpp
inline uint64_t P2PEffectiveTransferBytes(P2PTransport transport, uint64_t payloadBytes, uint32_t blockDim)
{
    (void)transport;
    (void)blockDim;
    return payloadBytes;
}
```

- [ ] **Step 6: Update validation message**

Change the invalid transport message to:

```cpp
return fail("transport must be direct_urma, memory, or data_as_flag");
```

- [ ] **Step 7: Simplify mismatch checker**

Replace `CountP2PTransportMismatches` with:

```cpp
inline uint64_t CountP2PTransportMismatches(
    const std::vector<uint8_t>& data, uint32_t pattern, uint64_t payloadBytes,
    P2PTransport transport, uint32_t blockDim)
{
    (void)transport;
    (void)blockDim;
    return CountP2PMismatches(data, pattern, payloadBytes);
}
```

- [ ] **Step 8: Run focused unit test and verify GREEN**

Run:

```powershell
cmake --build tests/udma/build --target test_tilexr_udma_p2p_perf_config
ctest --test-dir tests/udma/build -R test_tilexr_udma_p2p_perf_config --output-on-failure
```

Expected: target builds and the config test passes.

- [ ] **Step 9: Commit config test and helper changes**

Run:

```powershell
git add tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp tests/udma/demo/tilexr_udma_p2p_perf_config.h
git commit -m "test: trim P2P transport config modes"
```

---

### Task 3: Route `direct_urma` To Parallel Multi-Jetty And Remove Obsolete Kernels

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Modify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`
- Test: UDMA demo build target

**Interfaces:**
- Consumes:
  - `P2PTransport::DirectUrma`
  - `launch_tilexr_udma_p2p_perf(uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR debug, int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern, int32_t traffic)`
- Produces:
  - `launch_tilexr_udma_p2p_perf` launches the parallel multi-jetty kernel body
  - no host references to removed enum values
  - no benchmark wrappers named `launch_tilexr_udma_p2p_perf_multi_wqe`, `launch_tilexr_udma_p2p_perf_multi_jetty`, `launch_tilexr_udma_p2p_perf_multi_jetty_parallel`, or `launch_tilexr_udma_p2p_perf_multi_jetty_parallel_fixed_wqe`

- [ ] **Step 1: Remove obsolete extern declarations from host**

In `tests/udma/demo/tilexr_udma_demo.cpp`, delete declarations for:

```cpp
launch_tilexr_udma_p2p_perf_multi_wqe
launch_tilexr_udma_p2p_perf_multi_jetty
launch_tilexr_udma_p2p_perf_multi_jetty_parallel
launch_tilexr_udma_p2p_perf_multi_jetty_parallel_fixed_wqe
```

Keep the existing `launch_tilexr_udma_p2p_perf` declaration.

- [ ] **Step 2: Simplify host dispatch**

In `LaunchP2PKernel`, remove branches for removed transports. The final structure should be:

```cpp
    if (options.transport == TileXR::Demo::P2PTransport::Memory) {
        launch_tilexr_memory_p2p_perf(options.blockDim, stream, commArgsDev, srcDev, debugDev,
            options.srcRank, options.dstRank, dstOffset, transferBytes, pattern, traffic);
        return;
    }
    if (options.transport == TileXR::Demo::P2PTransport::DataAsFlag) {
        launch_tilexr_data_as_flag_p2p_perf(options.blockDim, stream, commArgsDev, srcDev, dstDev, debugDev,
            options.srcRank, options.dstRank, dstOffset, transferBytes, pattern, traffic);
        return;
    }
    launch_tilexr_udma_p2p_perf(options.blockDim, stream, commArgsDev, srcDev, debugDev,
        options.srcRank, options.dstRank, dstOffset, transferBytes, pattern, traffic);
```

- [ ] **Step 3: Remove fixed-WQE host window special cases**

In `RunP2PPerfMode`, delete `useFixedWqeSharedWindow` and `useSeparateDebugBuffer`. Replace the offset calculations with:

```cpp
    const uint64_t dstOffset = useIpcTransport ? TileXR::IPC_DATA_OFFSET : dstWindowBytes;
    const uint64_t localDstOffset = dstWindowBytes;
    const uint64_t debugOffset = useIpcTransport ? localDstOffset + dstWindowBytes : dstOffset + dstWindowBytes;
    const uint64_t payloadBytes = debugOffset + kP2PDebugWords * sizeof(uint32_t);
    const uint64_t registeredPayloadBytes = payloadBytes;
```

Remove `debugMemory` allocation and cleanup if it is only used by the fixed-WQE path after this change.

- [ ] **Step 4: Simplify host source/destination initialization**

Replace:

```cpp
const bool initSrc = !useFixedWqeSharedWindow || IsP2PSourceRank(rank, options);
const bool initDst = !useFixedWqeSharedWindow || IsP2PReceiveRank(rank, options);
```

with:

```cpp
const bool initSrc = true;
const bool initDst = true;
```

or inline the copy conditions so both local source and destination buffers are initialized on every rank.

- [ ] **Step 5: Simplify UDMA QP support warning**

Near the CLI option handling in `tilexr_udma_demo.cpp`, replace conditions that check removed transports:

```cpp
p2pOptions.transport == TileXR::Demo::P2PTransport::DirectUrmaMultiWqe ||
p2pOptions.transport == TileXR::Demo::P2PTransport::DirectUrmaMultiJetty ||
p2pOptions.transport == TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallel ||
p2pOptions.transport == TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallelFixedWqe
```

with:

```cpp
p2pOptions.transport == TileXR::Demo::P2PTransport::DirectUrma
```

- [ ] **Step 6: Rename the parallel multi-jetty kernel body**

In `tests/udma/demo/tilexr_udma_demo_kernel.cpp`, delete the old `tilexr_udma_p2p_perf_kernel` body and rename:

```cpp
tilexr_udma_p2p_perf_multi_jetty_parallel_kernel
```

to:

```cpp
tilexr_udma_p2p_perf_kernel
```

The resulting `tilexr_udma_p2p_perf_kernel` must retain the parallel logic:

```cpp
uint32_t qpNum = enabled ? TileXR::GetUDMAInfo(args)->qpNum : 0;
uint32_t jettyCount = blockNum < qpNum ? blockNum : qpNum;
if (blockIdx >= jettyCount) {
    TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, 0);
    return;
}
TileXRUdmaDemoWqeSlice(bytes, jettyCount, blockIdx, offset, sliceBytes);
TileXR::UDMAPutNbiQp<uint8_t>(args, peer, blockIdx, src + offset, dstByteOffset + offset, sliceBytes);
uint32_t status = TileXR::UDMAQuietStatusQp(args, peer, blockIdx);
```

- [ ] **Step 7: Delete obsolete benchmark kernels**

Delete these kernel functions:

```cpp
tilexr_udma_p2p_perf_multi_wqe_kernel
tilexr_udma_p2p_perf_multi_jetty_kernel
tilexr_udma_p2p_perf_multi_jetty_parallel_fixed_wqe_kernel
```

Keep helper `TileXRUdmaDemoWqeSlice` because the renamed direct URMA kernel uses it. Delete `TileXRUdmaDemoFixedWqeSlice` if it becomes unused.

- [ ] **Step 8: Delete obsolete launch wrappers**

Delete wrappers:

```cpp
launch_tilexr_udma_p2p_perf_multi_wqe
launch_tilexr_udma_p2p_perf_multi_jetty
launch_tilexr_udma_p2p_perf_multi_jetty_parallel
launch_tilexr_udma_p2p_perf_multi_jetty_parallel_fixed_wqe
```

Make existing `launch_tilexr_udma_p2p_perf` launch the renamed parallel kernel:

```cpp
void launch_tilexr_udma_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern, int32_t traffic)
{
    tilexr_udma_p2p_perf_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, src, debug, srcRank, dstRank, dstByteOffset, bytes, pattern, traffic);
}
```

- [ ] **Step 9: Search for removed enum/function references**

Run:

```powershell
rg -n "DirectUrmaMulti|direct_urma_multi|multi_wqe|multi_jetty_parallel_fixed_wqe|fixed_wqe|launch_tilexr_udma_p2p_perf_multi" tests src docs
```

Expected after Task 3 code changes but before docs cleanup: only docs or tests planned for Task 4 may still mention removed user-facing names.

- [ ] **Step 10: Build demo/unit targets**

Run:

```powershell
cmake --build tests/udma/build
ctest --test-dir tests/udma/build -R "test_tilexr_udma_p2p_perf_config|test_tilexr_udma_transport_layout" --output-on-failure
```

Expected: build succeeds; focused host-only tests pass. If CANN/Ascend build dependencies are unavailable on this host, capture the exact failure and continue to docs cleanup.

- [ ] **Step 11: Commit host/kernel trim**

Run:

```powershell
git add tests/udma/demo/tilexr_udma_demo.cpp tests/udma/demo/tilexr_udma_demo_kernel.cpp
git commit -m "feat: fold P2P direct URMA into parallel jetty path"
```

---

### Task 4: Update Scripts, Documentation, And PR Description

**Files:**
- Modify: `tests/udma/demo/run_tilexr_udma_p2p_perf.sh`
- Modify: `tests/udma/demo/run_tilexr_udma_p2p_concurrency_sweep.sh`
- Modify: `docs/TileXR_UDMA_P2P_PERF_GUIDE.md`
- Modify: `tests/udma/demo/README.md`
- External: GitHub PR #40 description

**Interfaces:**
- Consumes: supported transport names from Task 2
- Produces: scripts and docs that mention only `direct_urma`, `memory`, and `data_as_flag`

- [ ] **Step 1: Update runner QP setup**

In `tests/udma/demo/run_tilexr_udma_p2p_perf.sh`, replace the transport conditional with:

```bash
if [ "${transport}" = "direct_urma" ] || [ "${transport}" = "udma" ]; then
    export TILEXR_UDMA_QP_NUM="${block_dim}"
else
    export TILEXR_UDMA_QP_NUM="${TILEXR_UDMA_QP_NUM:-1}"
fi
```

- [ ] **Step 2: Verify sweep defaults**

Confirm `tests/udma/demo/run_tilexr_udma_p2p_concurrency_sweep.sh` default remains:

```bash
transports_csv=${8:-direct_urma,memory,data_as_flag}
```

If it differs, change it to that exact line.

- [ ] **Step 3: Update docs transport model**

In `docs/TileXR_UDMA_P2P_PERF_GUIDE.md`, update the transport list to:

```markdown
The P2P perf mode supports three user-facing transport modes:

- `direct_urma`: registered-memory UDMA transfer. Internally this path uses
  the parallel multi-jetty kernel; `block_dim=1` with one QP is the single-jetty
  baseline, while `block_dim=N` with `TILEXR_UDMA_QP_NUM=N` uses up to `N`
  QPs/jettys in parallel.
- `memory`: peer-memory IPC comparison using Ascend C `DataCopyPad`.
- `data_as_flag`: peer-memory IPC comparison where each 512B block carries
  480B payload plus a 32B ready flag.
```

Also update argument descriptions so `transport` says:

```markdown
- `transport`: `direct_urma`, `memory`, or `data_as_flag`; default is `direct_urma`.
```

- [ ] **Step 4: Remove obsolete docs wording**

Run:

```powershell
rg -n "direct_urma_multi|multi-WQE|multi WQE|fixed-WQE|fixed_wqe|direct_urma_multi_jetty_parallel" docs tests/udma/demo
```

Edit `docs/TileXR_UDMA_P2P_PERF_GUIDE.md` and `tests/udma/demo/README.md` until no user-facing obsolete transport names remain. It is acceptable for the design spec and plan files under `docs/superpowers/` to mention removed names as historical scope.

- [ ] **Step 5: Update PR #40 body with `gh`**

Use the installed GitHub CLI full path if `gh` is not active in `PATH`:

```powershell
& 'C:\Users\h30059441\AppData\Local\GitHubCLI\bin\gh.exe' pr edit 40 --repo LingquLab/TileXR --body @"
## Summary

Extend the UDMA P2P performance demo with a trimmed three-mode transport matrix.

This PR adds:

- `test_type=4` UDMA P2P performance mode and usage documentation
- `direct_urma` registered-memory UDMA transfer with built-in parallel multi-jetty support
- `memory` peer-memory IPC comparison
- `data_as_flag` peer-memory IPC comparison with embedded ready flags
- unidirectional and bidirectional P2P traffic modes
- CSV output, plotting helper, P2P runner script, and concurrency sweep script
- host-side unit coverage for P2P config parsing/window sizing and UDMA transport layout metadata

## Test plan

- [x] `git diff --cached --check`
- [ ] `bash tests/udma/run_tests.sh`
- [ ] Run `tests/udma/demo/run_tilexr_udma_p2p_perf.sh` on supported Ascend hardware
- [ ] Run `tests/udma/demo/run_tilexr_udma_p2p_concurrency_sweep.sh` on supported Ascend hardware
"@
```

- [ ] **Step 6: Run final searches**

Run:

```powershell
rg -n "direct_urma_multi|multi-WQE|multi WQE|fixed-WQE|fixed_wqe|DirectUrmaMulti|launch_tilexr_udma_p2p_perf_multi" tests src docs --glob "!docs/superpowers/**"
```

Expected: no matches.

- [ ] **Step 7: Run verification**

Run:

```powershell
git diff --check
ctest --test-dir tests/udma/build -R "test_tilexr_udma_p2p_perf_config|test_tilexr_udma_transport_layout" --output-on-failure
```

If `tests/udma/build` is unavailable, run:

```powershell
cmake -S tests/udma -B tests/udma/build
cmake --build tests/udma/build
ctest --test-dir tests/udma/build -R "test_tilexr_udma_p2p_perf_config|test_tilexr_udma_transport_layout" --output-on-failure
```

Expected: whitespace check passes; focused tests pass. If environment dependencies prevent building, record the exact command and failure.

- [ ] **Step 8: Commit docs/script cleanup**

Run:

```powershell
git add tests/udma/demo/run_tilexr_udma_p2p_perf.sh tests/udma/demo/run_tilexr_udma_p2p_concurrency_sweep.sh docs/TileXR_UDMA_P2P_PERF_GUIDE.md tests/udma/demo/README.md
git commit -m "docs: describe trimmed P2P transport modes"
```

---

## Self-Review Checklist

- Spec coverage:
  - Three transport names covered in Tasks 1, 2, and 4.
  - `direct_urma` folded into parallel multi-jetty path covered in Task 3.
  - Removed modes covered in Tasks 1, 2, 3, and 4.
  - Script QP behavior covered in Task 4.
  - TDD red/green covered in Tasks 1 and 2.
- Placeholder scan: no unfinished placeholder wording is intentional in executable steps.
- Type consistency:
  - `P2PTransport::DirectUrma`, `Memory`, `DataAsFlag`, and `Invalid` are the only public enum values used after Task 2.
  - `launch_tilexr_udma_p2p_perf` remains the host-visible UDMA launcher name.
