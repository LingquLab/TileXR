# DataAsFlag Epoch-Ordered P2P Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a side-by-side `data_as_flag_epoch_ordered` P2P transport that uses per-launch epochs and ordered payload/flag MTE3 writes before later replacing legacy `data_as_flag`.

**Architecture:** The implementation extends the existing UDMA P2P demo transport table, keeps the legacy data-as-flag helpers intact, and adds epoch-ordered helper APIs in `src/include/tilexr_data_as_flag.h`. The new P2P kernel reuses the existing data-as-flag role and slice logic, but sender writes payload and epoch flag in separate MTE3 stages while receiver waits on one batch commit flag.

**Tech Stack:** C++14 host code, AscendC AICore device code, CMake UDMA test targets, remote Ascend950 validation through `docs/ASCEND_REMOTE_BUILD_RUNBOOK.md`.

## Global Constraints

- Keep the existing block layout: `512B = 480B payload + 32B flag`.
- Keep ready information embedded in the data-as-flag window; do not move it to `SyncCollectives` or an independent flag region.
- Use per-launch `magic` and `step` to build a 64-bit epoch.
- Do not depend on payload and flag visibility within the same MTE3 operation.
- Phase 1 must keep legacy `data_as_flag` selectable for A/B comparison.
- `data_as_flag_epoch_ordered` must use the IPC peer data window and the same window byte calculation as legacy `data_as_flag`.
- Remote validation targets the existing Ascend environment in `docs/ASCEND_REMOTE_BUILD_RUNBOOK.md`; prefer NPU6/7 if 4/5 or 2/3 are suspect.
- Do not store the remote password in repo files or scripts.

---

## File Structure

- Modify `tests/udma/demo/tilexr_udma_p2p_perf_config.h`
  - Add the `DataAsFlagEpochOrdered` transport enum value and helpers.
  - Keep legacy `DataAsFlag` behavior unchanged.
- Modify `tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp`
  - Add failing then passing config tests for the new transport.
- Modify `src/include/tilexr_data_as_flag.h`
  - Add epoch construction and epoch-ordered send/receive helpers.
  - Keep existing `DataAsFlagSend` and `DataAsFlagCheckAndRecv` untouched for legacy mode.
- Modify `tests/udma/demo/tilexr_udma_demo_kernel.cpp`
  - Add the new epoch-ordered P2P kernel and launcher wrapper.
  - Reuse existing data-as-flag role resolution and slicing helpers.
- Modify `tests/udma/demo/tilexr_udma_demo.cpp`
  - Add the launcher declaration and host dispatch.
  - Pass `magic` and `step` to the new mode.
  - Ensure legacy data-as-flag per-iteration clear barriers do not apply to the new mode.
- Modify `tests/udma/unit/test_tilexr_udma_p2p_source_guard.cpp`
  - Add source structure checks for the new helper, kernel, launcher, and host wiring.
- Optionally modify `tests/udma/demo/run_tilexr_udma_p2p_concurrency_sweep.sh`
  - Include `data_as_flag_epoch_ordered` in default sweeps only after smoke validation passes.
- Add CSV outputs under `run/csv_full/` only during validation, not as part of core implementation commits unless the user explicitly asks to keep results.

---

### Task 1: Add Transport Config and Unit Tests

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_p2p_perf_config.h`
- Modify: `tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp`

**Interfaces:**
- Consumes: Existing `P2PTransport`, `P2PTransportName`, `ParseP2PTransport`, `P2PTransportWindowBytes`, `P2PTransportUsesIpc`, `P2PTransportBothRanksActive`, `ValidateP2PPerfOptions`, `FormatP2PPerfCsvRow`.
- Produces:
  - `P2PTransport::DataAsFlagEpochOrdered`
  - Transport name: `"data_as_flag_epoch_ordered"`
  - Aliases: `"data-as-flag-epoch-ordered"`, `"daf_epoch_ordered"`

- [ ] **Step 1: Write failing enum/name/parse tests**

Add these checks near the existing `DataAsFlag` transport checks in `tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp`:

```cpp
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered) ==
            "data_as_flag_epoch_ordered",
        "data_as_flag_epoch_ordered transport name mismatch");
    Require(TileXR::Demo::ParseP2PTransport("data_as_flag_epoch_ordered") ==
            TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered,
        "data_as_flag_epoch_ordered transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("data-as-flag-epoch-ordered") ==
            TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered,
        "data-as-flag-epoch-ordered alias parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("daf_epoch_ordered") ==
            TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered,
        "daf_epoch_ordered alias parse mismatch");
```

- [ ] **Step 2: Write failing layout/helper tests**

Add these checks after the existing data-as-flag window checks:

```cpp
    Require(TileXR::Demo::P2PTransportWindowBytes(
                TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered, 0) == 0,
        "data_as_flag_epoch_ordered zero layout mismatch");
    Require(TileXR::Demo::P2PTransportWindowBytes(
                TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered, 480) == 512,
        "data_as_flag_epoch_ordered 480B layout mismatch");
    Require(TileXR::Demo::P2PTransportWindowBytes(
                TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered, 481) == 1024,
        "data_as_flag_epoch_ordered 481B layout mismatch");
    Require(TileXR::Demo::P2PTransportUsesIpc(TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered),
        "data_as_flag_epoch_ordered must use IPC peer window");
    Require(TileXR::Demo::P2PTransportBothRanksActive(
                TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered, TileXR::Demo::P2PTraffic::UniDir),
        "data_as_flag_epoch_ordered unidir must keep receiver active");
```

- [ ] **Step 3: Write failing validation and CSV tests**

Add these checks near existing `DataAsFlag` validation and CSV assertions:

```cpp
    options.transport = TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered;
    options.traffic = TileXR::Demo::P2PTraffic::UniDir;
    options.blockDim = 4;
    options.maxBytes = 16384;
    Require(TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error),
        "valid data_as_flag_epoch_ordered options rejected");

    row.transport = TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered;
    row.traffic = TileXR::Demo::P2PTraffic::UniDir;
    row.blockDim = 4;
    const std::string epochOrderedCsv = TileXR::Demo::FormatP2PPerfCsvRow(row);
    Require(epochOrderedCsv ==
            "data_as_flag_epoch_ordered,unidir,4,1to0,1,0,2,4096,20,8.000,0.000,0.000,0.512,0.512,0,0,logs/run\n",
        "data_as_flag_epoch_ordered csv row mismatch");
```

- [ ] **Step 4: Run the config test and confirm it fails**

Run on a Linux/Ascend build environment:

```bash
cd /home/h30059441/TileXR
source scripts/common_env.sh
cd tests/udma
bash build.sh
./install/bin/test_tilexr_udma_p2p_perf_config
```

Expected before implementation: compile failure mentioning `DataAsFlagEpochOrdered` is not a member of `P2PTransport`.

- [ ] **Step 5: Implement transport config**

Modify `tests/udma/demo/tilexr_udma_p2p_perf_config.h`:

```cpp
enum class P2PTransport {
    DirectUrma,
    DirectUrmaPostOnly,
    Memory,
    MemoryConsume,
    DataAsFlag,
    DataAsFlagEpochOrdered,
    Invalid,
};
```

Add name mapping:

```cpp
        case P2PTransport::DataAsFlagEpochOrdered:
            return "data_as_flag_epoch_ordered";
```

Add parser mapping:

```cpp
    if (name == "data_as_flag_epoch_ordered" ||
        name == "data-as-flag-epoch-ordered" ||
        name == "daf_epoch_ordered") {
        return P2PTransport::DataAsFlagEpochOrdered;
    }
```

Update window bytes:

```cpp
    return (transport == P2PTransport::DataAsFlag ||
            transport == P2PTransport::DataAsFlagEpochOrdered) ?
        DataAsFlagWindowBytes(payloadBytes) : payloadBytes;
```

Update IPC and active-rank helpers:

```cpp
inline bool P2PTransportUsesIpc(P2PTransport transport)
{
    return transport == P2PTransport::Memory ||
        transport == P2PTransport::MemoryConsume ||
        transport == P2PTransport::DataAsFlag ||
        transport == P2PTransport::DataAsFlagEpochOrdered;
}

inline bool P2PTransportBothRanksActive(P2PTransport transport, P2PTraffic traffic)
{
    return traffic == P2PTraffic::BiDir ||
        transport == P2PTransport::MemoryConsume ||
        transport == P2PTransport::DataAsFlag ||
        transport == P2PTransport::DataAsFlagEpochOrdered;
}
```

Update the validation error string:

```cpp
return fail("transport must be direct_urma, direct_urma_post_only, memory, memory_consume, data_as_flag, or data_as_flag_epoch_ordered");
```

- [ ] **Step 6: Run the config test and confirm it passes**

Run:

```bash
cd /home/h30059441/TileXR/tests/udma
bash build.sh
./install/bin/test_tilexr_udma_p2p_perf_config
```

Expected: exit code `0`.

- [ ] **Step 7: Commit Task 1**

```bash
git add tests/udma/demo/tilexr_udma_p2p_perf_config.h \
        tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp
git commit -m "feat: add epoch ordered data-as-flag transport config"
```

---

### Task 2: Add Epoch-Ordered Device Helpers

**Files:**
- Modify: `src/include/tilexr_data_as_flag.h`

**Interfaces:**
- Consumes: Existing constants `DATA_AS_FLAG_BLOCK_BYTES`, `DATA_AS_FLAG_PAYLOAD_BYTES`, `DATA_AS_FLAG_FLAG_BYTES`, `DATA_AS_FLAG_FLAG_OFFSET_BYTES`, `DataAsFlagBlockCountForPayloadBytes`, `DataAsFlagScratchBytes`, `DataAsFlagMaxRecvBlocks`, `DataAsFlagCopyPayloadToScratch`, `DataAsFlagCopyBatchToRecvGM`.
- Produces:
  - `uint64_t DataAsFlagEpoch(int32_t magic, int32_t step)`
  - `uint32_t DataAsFlagMaxEpochOrderedSendBlocks(uint32_t scratchBytes)`
  - `uint32_t DataAsFlagMaxEpochOrderedRecvBlocks(uint32_t scratchBytes)`
  - `uint32_t DataAsFlagSendEpochOrdered(...)`
  - `bool DataAsFlagCheckAndRecvEpochOrdered(...)`

- [ ] **Step 1: Add source guard tests before implementation**

This task's implementation will be guarded in Task 5, but first verify absence manually:

```bash
rg -n "DataAsFlagEpoch|DataAsFlagSendEpochOrdered|DataAsFlagCheckAndRecvEpochOrdered" src/include/tilexr_data_as_flag.h
```

Expected before implementation: no matches.

- [ ] **Step 2: Add epoch helper**

Add this helper in `src/include/tilexr_data_as_flag.h` outside the AscendC-only block so unit/source checks can find it:

```cpp
TILEXR_DATA_AS_FLAG_INLINE uint64_t DataAsFlagEpoch(int32_t magic, int32_t step)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(magic)) << 32) |
        static_cast<uint32_t>(step);
}
```

- [ ] **Step 3: Add epoch ordered batch capacity helpers**

Inside `#if TILEXR_ASCENDC_AICORE_COMPILE`, add:

```cpp
__aicore__ inline uint32_t DataAsFlagMaxEpochOrderedSendBlocks(uint32_t scratchBytes)
{
    uint32_t blocks = scratchBytes / DATA_AS_FLAG_BLOCK_BYTES;
    while (blocks > 0U) {
        const uint64_t requiredBytes =
            static_cast<uint64_t>(blocks) * DATA_AS_FLAG_BLOCK_BYTES +
            static_cast<uint64_t>(blocks) * DATA_AS_FLAG_FLAG_BYTES;
        if (static_cast<uint64_t>(scratchBytes) >= requiredBytes) {
            return blocks;
        }
        --blocks;
    }
    return 0U;
}

__aicore__ inline uint32_t DataAsFlagMaxEpochOrderedRecvBlocks(uint32_t scratchBytes)
{
    return DataAsFlagMaxRecvBlocks(scratchBytes);
}
```

This conservative v1 uses one scratch region for packed payload blocks and a second scratch region for epoch flags.

- [ ] **Step 4: Add flag fill and flag copy helpers**

Add:

```cpp
__aicore__ inline void DataAsFlagFillEpochFlags(
    AscendC::LocalTensor<uint8_t>& flagScratch,
    uint32_t blockCount,
    uint64_t epoch)
{
    AscendC::LocalTensor<uint64_t> flagWords = flagScratch.template ReinterpretCast<uint64_t>();
    const uint32_t words = blockCount * DATA_AS_FLAG_FLAG_BYTES / sizeof(uint64_t);
    for (uint32_t i = 0; i < words; ++i) {
        flagWords.SetValue(i, epoch);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
}

__aicore__ inline void DataAsFlagCopyEpochFlagsToGM(
    __gm__ uint8_t* dstDataAsFlagGM,
    uint32_t dstBlockOffset,
    AscendC::LocalTensor<uint8_t>& flagScratch,
    uint32_t batchBlocks)
{
    AscendC::GlobalTensor<uint8_t> flagGlobal;
    flagGlobal.SetGlobalBuffer(
        dstDataAsFlagGM + static_cast<uint64_t>(dstBlockOffset) * DATA_AS_FLAG_BLOCK_BYTES +
        DATA_AS_FLAG_FLAG_OFFSET_BYTES);
    AscendC::DataCopyExtParams flagParams {
        static_cast<uint16_t>(batchBlocks),
        DATA_AS_FLAG_FLAG_BYTES,
        0U,
        DATA_AS_FLAG_PAYLOAD_BYTES / DATA_AS_FLAG_ALIGN_BYTES,
        0U};
    AscendC::DataCopyPadExtParams<uint8_t> padParams {false, 0U, 0U, 0U};
    AscendC::DataCopyPad(flagGlobal, flagScratch, flagParams, padParams);
}
```

- [ ] **Step 5: Add `DataAsFlagSendEpochOrdered`**

Add:

```cpp
__aicore__ inline uint32_t DataAsFlagSendEpochOrdered(
    __gm__ uint8_t* dstDataAsFlagGM,
    const __gm__ uint8_t* srcGM,
    uint64_t dataBytes,
    uint64_t epoch,
    AscendC::LocalTensor<uint8_t>& scratch)
{
    if (dstDataAsFlagGM == nullptr || srcGM == nullptr || dataBytes == 0U) {
        return 0U;
    }

    const uint32_t totalBlocks = DataAsFlagBlockCountForPayloadBytes(dataBytes);
    const uint32_t batchCapacity = DataAsFlagMaxEpochOrderedSendBlocks(DataAsFlagScratchBytes(scratch));
    if (batchCapacity == 0U) {
        return 0U;
    }

    uint32_t sentBlocks = 0U;
    uint64_t sentBytes = 0U;
    while (sentBlocks < totalBlocks) {
        const uint32_t remainingBlocks = totalBlocks - sentBlocks;
        const uint32_t batchBlocks = remainingBlocks < batchCapacity ? remainingBlocks : batchCapacity;
        const uint64_t maxBatchBytes = static_cast<uint64_t>(batchBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint64_t remainingBytes = dataBytes - sentBytes;
        const uint32_t batchPayloadBytes = static_cast<uint32_t>(
            remainingBytes < maxBatchBytes ? remainingBytes : maxBatchBytes);
        const uint32_t fullBlocks = batchPayloadBytes / DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint32_t tailBytes = batchPayloadBytes % DATA_AS_FLAG_PAYLOAD_BYTES;

        AscendC::LocalTensor<uint8_t> payloadScratch = scratch;
        AscendC::LocalTensor<uint8_t> flagScratch =
            scratch[static_cast<uint64_t>(batchBlocks) * DATA_AS_FLAG_BLOCK_BYTES];

        AscendC::Duplicate<uint8_t>(payloadScratch, 0U, batchBlocks * DATA_AS_FLAG_BLOCK_BYTES);
        AscendC::PipeBarrier<PIPE_ALL>();
        DataAsFlagCopyPayloadToScratch(payloadScratch, srcGM, sentBytes, fullBlocks, tailBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataAsFlagCopyScratchToDataAsFlagGM(dstDataAsFlagGM, sentBlocks, payloadScratch, batchBlocks);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

        DataAsFlagFillEpochFlags(flagScratch, batchBlocks, epoch);
        DataAsFlagCopyEpochFlagsToGM(dstDataAsFlagGM, sentBlocks, flagScratch, batchBlocks);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

        sentBlocks += batchBlocks;
        sentBytes += batchPayloadBytes;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return totalBlocks;
}
```

Note: the first MTE3 writes packed 512B blocks with zero flags. The second MTE3 writes current epoch flags. This satisfies the ordered protocol while avoiding a risky first version of strided payload-only GM writes.

- [ ] **Step 6: Add commit flag polling helper**

Add:

```cpp
__aicore__ inline uint64_t DataAsFlagLoadEpochFlag(
    const __gm__ uint8_t* dataAsFlagGM,
    uint32_t blockIndex,
    AscendC::LocalTensor<uint8_t>& scratch)
{
    AscendC::GlobalTensor<uint64_t> flagGlobal;
    flagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(
        const_cast<__gm__ uint8_t*>(
            dataAsFlagGM + static_cast<uint64_t>(blockIndex) * DATA_AS_FLAG_BLOCK_BYTES +
            DATA_AS_FLAG_FLAG_OFFSET_BYTES)));
    AscendC::LocalTensor<uint64_t> flagLocal = scratch.template ReinterpretCast<uint64_t>();
    AscendC::DataCopyExtParams params {1U, sizeof(uint64_t), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint64_t> padParams {false, 0U, 0U, 0U};
    AscendC::DataCopyPad(flagLocal, flagGlobal, params, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return flagLocal.GetValue(0);
}
```

- [ ] **Step 7: Add strict batch checker**

Add:

```cpp
__aicore__ inline bool DataAsFlagCheckBatchEpochStrict(
    const __gm__ uint8_t* dataAsFlagGM,
    uint32_t blockOffset,
    uint32_t batchBlocks,
    uint64_t epoch,
    AscendC::LocalTensor<uint8_t>& scratch)
{
    if (batchBlocks == 0U) {
        return false;
    }
    for (uint32_t i = 0; i < batchBlocks; ++i) {
        if (DataAsFlagLoadEpochFlag(dataAsFlagGM, blockOffset + i, scratch) != epoch) {
            return false;
        }
    }
    return true;
}
```

- [ ] **Step 8: Add `DataAsFlagCheckAndRecvEpochOrdered`**

Add:

```cpp
__aicore__ inline bool DataAsFlagCheckAndRecvEpochOrdered(
    const __gm__ uint8_t* dataAsFlagGM,
    uint64_t dataBytes,
    __gm__ uint8_t* recvGM,
    uint64_t epoch,
    AscendC::LocalTensor<uint8_t>& recvScratch,
    bool strict)
{
    if (dataAsFlagGM == nullptr || recvGM == nullptr) {
        return false;
    }
    if (dataBytes == 0U) {
        return true;
    }

    const uint32_t totalBlocks = DataAsFlagBlockCountForPayloadBytes(dataBytes);
    const uint32_t batchCapacity = DataAsFlagMaxEpochOrderedRecvBlocks(DataAsFlagScratchBytes(recvScratch));
    if (batchCapacity == 0U) {
        return false;
    }

    uint32_t processedBlocks = 0U;
    uint64_t processedBytes = 0U;
    while (processedBlocks < totalBlocks) {
        const uint32_t remainingBlocks = totalBlocks - processedBlocks;
        const uint32_t batchBlocks = remainingBlocks < batchCapacity ? remainingBlocks : batchCapacity;
        const uint32_t lastBlock = processedBlocks + batchBlocks - 1U;
        while (DataAsFlagLoadEpochFlag(dataAsFlagGM, lastBlock, recvScratch) != epoch) {
        }
        if (strict && !DataAsFlagCheckBatchEpochStrict(dataAsFlagGM, processedBlocks, batchBlocks, epoch, recvScratch)) {
            return false;
        }

        const uint64_t remainingBytes = dataBytes - processedBytes;
        const uint64_t maxBatchBytes = static_cast<uint64_t>(batchBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint32_t batchBytes = static_cast<uint32_t>(
            remainingBytes < maxBatchBytes ? remainingBytes : maxBatchBytes);
        DataAsFlagCopyBatchToRecvGM(
            dataAsFlagGM, processedBlocks, processedBytes, batchBytes, recvGM, recvScratch);
        processedBlocks += batchBlocks;
        processedBytes += batchBytes;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return true;
}
```

- [ ] **Step 9: Build and fix compiler issues**

Run:

```bash
cd /home/h30059441/TileXR
source scripts/common_env.sh
cd tests/udma
bash build.sh
```

Expected: build succeeds. If AscendC rejects `Duplicate<uint8_t>` or `DataCopyPad` template parameters, adjust only the helper internals while preserving the same signatures and two-stage MTE3 protocol.

- [ ] **Step 10: Commit Task 2**

```bash
git add src/include/tilexr_data_as_flag.h
git commit -m "feat: add epoch ordered data-as-flag helpers"
```

---

### Task 3: Add P2P Kernel and Launcher

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`

**Interfaces:**
- Consumes:
  - `TileXRUdmaDemoResolveDataAsFlagRole`
  - `TileXRUdmaDemoDataAsFlagSlice`
  - `TileXR::DataAsFlagEpoch`
  - `TileXR::DataAsFlagSendEpochOrdered`
  - `TileXR::DataAsFlagCheckAndRecvEpochOrdered`
- Produces:
  - `tilexr_data_as_flag_epoch_ordered_p2p_perf_kernel`
  - `launch_tilexr_data_as_flag_epoch_ordered_p2p_perf`

- [ ] **Step 1: Add failing source guard expectation**

Temporarily add these checks to `tests/udma/unit/test_tilexr_udma_p2p_source_guard.cpp` in a new helper `TestDataAsFlagEpochOrderedSource()`:

```cpp
void TestDataAsFlagEpochOrderedSource()
{
    const std::string kernelPath = "tests/udma/demo/tilexr_udma_demo_kernel.cpp";
    const std::string kernelText = ReadFile(kernelPath);
    CheckContains(kernelPath, kernelText, "tilexr_data_as_flag_epoch_ordered_p2p_perf_kernel");
    CheckContains(kernelPath, kernelText, "launch_tilexr_data_as_flag_epoch_ordered_p2p_perf");
    CheckContains(kernelPath, kernelText, "DataAsFlagEpoch(magic, step)");
    CheckContains(kernelPath, kernelText, "DataAsFlagSendEpochOrdered");
    CheckContains(kernelPath, kernelText, "DataAsFlagCheckAndRecvEpochOrdered");
    CheckContains(kernelPath, kernelText, "int32_t magic, int32_t step");
}
```

Call it from `main()`:

```cpp
    TestDataAsFlagEpochOrderedSource();
```

Run:

```bash
cd /home/h30059441/TileXR/tests/udma
bash build.sh
./install/bin/test_tilexr_udma_p2p_source_guard
```

Expected before kernel implementation: source guard failure listing missing strings.

- [ ] **Step 2: Add the epoch-ordered kernel**

Add this kernel beside `tilexr_data_as_flag_p2p_perf_kernel` in `tests/udma/demo/tilexr_udma_demo_kernel.cpp`:

```cpp
extern "C" __global__ __aicore__ void tilexr_data_as_flag_epoch_ordered_p2p_perf_kernel(
    GM_ADDR commArgsGM, GM_ADDR srcGM, GM_ADDR dstGM, GM_ADDR debugGM,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset,
    uint32_t bytes, uint32_t pattern, int32_t traffic, int32_t magic, int32_t step)
{
    if constexpr (g_coreType == AscendC::AIV) {
        auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
        auto debug = reinterpret_cast<__gm__ uint32_t*>(debugGM);
        auto src = reinterpret_cast<__gm__ uint8_t*>(srcGM);
        auto dst = reinterpret_cast<__gm__ uint8_t*>(dstGM);

        int32_t rank = args->rank;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        const uint64_t epoch = TileXR::DataAsFlagEpoch(magic, step);
        if (debug != nullptr && blockIdx == 0) {
            debug[0] = TILEXR_UDMA_DEMO_MAGIC;
            debug[1] = rank;
            debug[2] = 1;
            debug[3] = bytes;
            debug[4] = pattern;
            debug[5] = 0xffffffffu;
            debug[6] = blockNum;
            debug[7] = static_cast<uint32_t>(epoch & 0xffffffffu);
        }

        bool isSender = false;
        bool isReceiver = false;
        int32_t peer = -1;
        if (blockNum == 0 ||
            !TileXRUdmaDemoResolveDataAsFlagRole(rank, srcRank, dstRank, traffic, isSender, isReceiver, peer) ||
            peer < 0 || peer >= args->rankSize) {
            return;
        }

        AscendC::GlobalTensor<GM_ADDR> peerMems;
        peerMems.SetGlobalBuffer(&(args->peerMems[0]), TileXR::TILEXR_MAX_RANK_SIZE);
        GM_ADDR peerBase = peerMems.GetValue(peer);
        GM_ADDR localBase = peerMems.GetValue(rank);
        if (peerBase == nullptr || localBase == nullptr || dst == nullptr || (isSender && src == nullptr)) {
            uint32_t status = TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, 2);
            if (debug != nullptr && blockIdx == 0) {
                debug[5] = status;
            }
            return;
        }

        uint32_t payloadOffset = 0;
        uint32_t sliceBytes = 0;
        uint32_t dataAsFlagOffset = 0;
        TileXRUdmaDemoDataAsFlagSlice(bytes, blockNum, blockIdx, payloadOffset, sliceBytes, dataAsFlagOffset);
        if (debug != nullptr && blockIdx < 8) {
            debug[16 + blockIdx] = payloadOffset;
            debug[24 + blockIdx] = sliceBytes;
        }
        if (sliceBytes == 0) {
            TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, 0);
            return;
        }

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
        pipe.InitBuffer(tBuf, TILEXR_UDMA_DEMO_P2P_UB_BYTES);
        AscendC::LocalTensor<uint8_t> scratch = tBuf.Get<uint8_t>();

        uint32_t status = 0;
        if (isSender) {
            uint32_t sentBlocks = TileXR::DataAsFlagSendEpochOrdered(
                reinterpret_cast<__gm__ uint8_t*>(peerBase + dstByteOffset + dataAsFlagOffset),
                src + payloadOffset, sliceBytes, epoch, scratch);
            if (sentBlocks == 0U) {
                status = 3;
            }
        }
        if (isReceiver && status == 0) {
            bool strict = false;
            bool received = TileXR::DataAsFlagCheckAndRecvEpochOrdered(
                reinterpret_cast<__gm__ uint8_t*>(localBase + dstByteOffset + dataAsFlagOffset),
                sliceBytes, dst + payloadOffset, epoch, scratch, strict);
            if (!received) {
                status = 4;
            }
        }
        TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, status);
        if (debug != nullptr && blockIdx == 0) {
            debug[5] = status;
        }
    }
}
```

- [ ] **Step 3: Add launcher wrapper**

Add near the existing `launch_tilexr_data_as_flag_p2p_perf`:

```cpp
void launch_tilexr_data_as_flag_epoch_ordered_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR dst, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset,
    uint32_t bytes, uint32_t pattern, int32_t traffic, int32_t magic, int32_t step)
{
    tilexr_data_as_flag_epoch_ordered_p2p_perf_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, src, dst, debug, srcRank, dstRank, dstByteOffset, bytes, pattern, traffic, magic, step);
}
```

- [ ] **Step 4: Build and run source guard**

Run:

```bash
cd /home/h30059441/TileXR
source scripts/common_env.sh
cd tests/udma
bash build.sh
./install/bin/test_tilexr_udma_p2p_source_guard
```

Expected: build succeeds and source guard exits `0`.

- [ ] **Step 5: Commit Task 3**

```bash
git add tests/udma/demo/tilexr_udma_demo_kernel.cpp \
        tests/udma/unit/test_tilexr_udma_p2p_source_guard.cpp
git commit -m "feat: add epoch ordered data-as-flag p2p kernel"
```

---

### Task 4: Wire Host Launch and Clear-Window Behavior

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_p2p_source_guard.cpp`

**Interfaces:**
- Consumes:
  - `P2PTransport::DataAsFlagEpochOrdered`
  - `launch_tilexr_data_as_flag_epoch_ordered_p2p_perf(...)`
- Produces:
  - Host dispatch from CLI transport `data_as_flag_epoch_ordered`.
  - No per-iteration clear-window barrier for the new transport.

- [ ] **Step 1: Extend source guard for host wiring**

Add checks to `TestDataAsFlagEpochOrderedSource()`:

```cpp
    const std::string hostPath = "tests/udma/demo/tilexr_udma_demo.cpp";
    const std::string hostText = ReadFile(hostPath);
    CheckContains(hostPath, hostText, "launch_tilexr_data_as_flag_epoch_ordered_p2p_perf");
    CheckContains(hostPath, hostText, "P2PTransport::DataAsFlagEpochOrdered");
    CheckContains(hostPath, hostText, "magic, step");
    CheckContains(hostPath, hostText, "useLegacyDataAsFlagTransport");
```

Run:

```bash
cd /home/h30059441/TileXR/tests/udma
bash build.sh
./install/bin/test_tilexr_udma_p2p_source_guard
```

Expected before host implementation: source guard failure.

- [ ] **Step 2: Add launcher declaration**

In `tests/udma/demo/tilexr_udma_demo.cpp`, add:

```cpp
extern void launch_tilexr_data_as_flag_epoch_ordered_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR dst, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset,
    uint32_t bytes, uint32_t pattern, int32_t traffic, int32_t magic, int32_t step);
```

- [ ] **Step 3: Dispatch the new transport**

In `LaunchP2PKernel`, add the new branch before the legacy `DataAsFlag` branch:

```cpp
    if (options.transport == TileXR::Demo::P2PTransport::DataAsFlagEpochOrdered) {
        launch_tilexr_data_as_flag_epoch_ordered_p2p_perf(
            options.blockDim, stream, commArgsDev, srcDev, dstDev, debugDev,
            options.srcRank, options.dstRank, dstOffset, transferBytes, pattern, traffic, magic, step);
        return;
    }
```

- [ ] **Step 4: Split legacy clear behavior from new behavior**

Replace:

```cpp
    const bool useDataAsFlagTransport = options.transport == TileXR::Demo::P2PTransport::DataAsFlag;
```

with:

```cpp
    const bool useLegacyDataAsFlagTransport = options.transport == TileXR::Demo::P2PTransport::DataAsFlag;
```

Then replace warmup/measured per-iteration data-as-flag clear branches to use `useLegacyDataAsFlagTransport` only:

```cpp
            if (useLegacyDataAsFlagTransport && IsP2PReceiveRank(rank, options) &&
                !ClearLocalPeerWindow(rank, commArgsHost, dstOffset,
                    TileXR::Demo::P2PTransportWindowBytes(options.transport, bytes), "p2p data_as_flag warmup window")) {
                ok = false;
                break;
            }
            if (useLegacyDataAsFlagTransport &&
                !DemoBarrierAll(rank, rankSize,
                    "p2p data_as_flag warmup clear bytes=" + std::to_string(bytes) +
                    " iter=" + std::to_string(i))) {
                ok = false;
                break;
            }
```

And measured:

```cpp
        if (useLegacyDataAsFlagTransport && IsP2PReceiveRank(rank, options) &&
            !ClearLocalPeerWindow(rank, commArgsHost, dstOffset,
                TileXR::Demo::P2PTransportWindowBytes(options.transport, bytes), "p2p data_as_flag measured window")) {
            ok = false;
            break;
        }
        if (useLegacyDataAsFlagTransport &&
            !DemoBarrierAll(rank, rankSize,
                "p2p data_as_flag measured clear bytes=" + std::to_string(bytes))) {
            ok = false;
            break;
        }
```

Keep the existing once-per-size IPC destination clear:

```cpp
        if (useIpcTransport && IsP2PReceiveRank(rank, options)) {
            const uint64_t clearBytes = TileXR::Demo::P2PTransportWindowBytes(options.transport, bytes);
            ...
        }
```

This one-time clear remains valid for epoch-ordered startup hygiene.

- [ ] **Step 5: Build and run unit/source tests**

Run:

```bash
cd /home/h30059441/TileXR
source scripts/common_env.sh
cd tests/udma
bash build.sh
./install/bin/test_tilexr_udma_p2p_perf_config
./install/bin/test_tilexr_udma_p2p_source_guard
```

Expected: both tests exit `0`.

- [ ] **Step 6: Commit Task 4**

```bash
git add tests/udma/demo/tilexr_udma_demo.cpp \
        tests/udma/unit/test_tilexr_udma_p2p_source_guard.cpp
git commit -m "feat: wire epoch ordered data-as-flag p2p host path"
```

---

### Task 5: Add Strict Mode Host Control

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`
- Modify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_p2p_source_guard.cpp`

**Interfaces:**
- Consumes: `DataAsFlagCheckAndRecvEpochOrdered(..., bool strict)`.
- Produces:
  - Environment variable `TILEXR_DATA_AS_FLAG_STRICT`.
  - Launcher and kernel parameter `int32_t strict`.

- [ ] **Step 1: Add source guard checks**

Add checks:

```cpp
    CheckContains(hostPath, hostText, "TILEXR_DATA_AS_FLAG_STRICT");
    CheckContains(kernelPath, kernelText, "int32_t strict");
    CheckContains(kernelPath, kernelText, "strict != 0");
```

Run source guard and expect failure before implementation.

- [ ] **Step 2: Extend launcher signatures**

In `tests/udma/demo/tilexr_udma_demo.cpp`, change the extern declaration:

```cpp
extern void launch_tilexr_data_as_flag_epoch_ordered_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR dst, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset,
    uint32_t bytes, uint32_t pattern, int32_t traffic, int32_t magic, int32_t step, int32_t strict);
```

In `tests/udma/demo/tilexr_udma_demo_kernel.cpp`, extend the launcher wrapper and kernel parameters with `int32_t strict`.

- [ ] **Step 3: Pass strict from host**

In `LaunchP2PKernel`, compute:

```cpp
    const int32_t dataAsFlagStrict = GetEnvFlag("TILEXR_DATA_AS_FLAG_STRICT", false) ? 1 : 0;
```

Pass it only to the epoch-ordered launcher:

```cpp
        launch_tilexr_data_as_flag_epoch_ordered_p2p_perf(
            options.blockDim, stream, commArgsDev, srcDev, dstDev, debugDev,
            options.srcRank, options.dstRank, dstOffset, transferBytes, pattern, traffic,
            magic, step, dataAsFlagStrict);
```

- [ ] **Step 4: Use strict in kernel**

Replace:

```cpp
            bool strict = false;
```

with:

```cpp
            bool strictMode = strict != 0;
```

Pass `strictMode` to `DataAsFlagCheckAndRecvEpochOrdered`.

- [ ] **Step 5: Build and run source guard**

Run:

```bash
cd /home/h30059441/TileXR
source scripts/common_env.sh
cd tests/udma
bash build.sh
./install/bin/test_tilexr_udma_p2p_source_guard
```

Expected: source guard exits `0`.

- [ ] **Step 6: Commit Task 5**

```bash
git add tests/udma/demo/tilexr_udma_demo.cpp \
        tests/udma/demo/tilexr_udma_demo_kernel.cpp \
        tests/udma/unit/test_tilexr_udma_p2p_source_guard.cpp
git commit -m "feat: add strict flag checks for epoch ordered data-as-flag"
```

---

### Task 6: Remote Smoke and Sweep Validation

**Files:**
- Read: `docs/ASCEND_REMOTE_BUILD_RUNBOOK.md`
- May create local validation outputs under: `run/csv_full/`

**Interfaces:**
- Consumes: Built demo binary `tests/udma/install/bin/tilexr_udma_demo`.
- Produces:
  - Remote logs for focused smoke and sweep runs.
  - Local CSV copies for old/new transport comparison.

- [ ] **Step 1: Sync changed files to remote**

Use Paramiko from local PowerShell. Keep the password in the environment only:

```powershell
@'
import os
import pathlib
import paramiko

host = "141.62.19.144"
user = "root"
password = os.environ["TILEXR_REMOTE_PASS"]
local_root = pathlib.Path(r"D:\workspace\TileXR")
remote_root = "/home/h30059441/TileXR"

files = [
    "src/include/tilexr_data_as_flag.h",
    "tests/udma/demo/tilexr_udma_demo.cpp",
    "tests/udma/demo/tilexr_udma_demo_kernel.cpp",
    "tests/udma/demo/tilexr_udma_p2p_perf_config.h",
    "tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp",
    "tests/udma/unit/test_tilexr_udma_p2p_source_guard.cpp",
]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username=user, password=password, timeout=15,
            banner_timeout=15, auth_timeout=15)
sftp = ssh.open_sftp()
for path in files:
    sftp.put(str(local_root / path), remote_root + "/" + path)
    print("uploaded", path)
sftp.close()
ssh.close()
'@ | python -
```

- [ ] **Step 2: Build and run tests remotely**

Run:

```powershell
@'
import os
import paramiko
host = "141.62.19.144"
user = "root"
password = os.environ["TILEXR_REMOTE_PASS"]
cmd = r'''
cd /home/h30059441/TileXR
pids=$(ps -eo pid=,comm= | awk '$2=="tilexr_udma_demo" {print $1}')
[ -n "$pids" ] && kill -TERM $pids || true
sleep 2
pids=$(ps -eo pid=,comm= | awk '$2=="tilexr_udma_demo" {print $1}')
[ -n "$pids" ] && kill -KILL $pids || true
source scripts/common_env.sh
cd tests/udma
bash build.sh
bash run_tests.sh
'''
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username=user, password=password, timeout=15,
            banner_timeout=15, auth_timeout=15)
stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True, timeout=900)
for line in iter(stdout.readline, ""):
    print(line, end="")
err = stderr.read().decode("utf-8", errors="replace")
if err:
    print(err, end="")
rc = stdout.channel.recv_exit_status()
ssh.close()
raise SystemExit(rc)
'@ | python -
```

Expected: build succeeds, `run_tests.sh` summary reports PASS for layout, registry, P2P perf config, source guard, and single-process integration.

- [ ] **Step 3: Run focused fast-path smoke on NPU6/7**

Run a 4KB to 1MB smoke:

```bash
cd /home/h30059441/TileXR
source scripts/common_env.sh
cd tests/udma
export TILEXR_COMM_ID=127.0.0.1:10267
export TILEXR_DEMO_NPUS=2
export TILEXR_DEMO_FIRST_NPU=6
export LD_LIBRARY_PATH="$PWD/install/lib:$PWD/install/lib64:/home/h30059441/TileXR/install/lib:/home/h30059441/TileXR/install/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"
log_dir="$PWD/logs/tilexr_daf_epoch_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
csv="$log_dir/p2p_perf.csv"
bin="$PWD/install/bin/tilexr_udma_demo"
for rank in 0 1; do
  RANK=$rank RANK_SIZE=2 TILEXR_P2P_LOG_DIR="$log_dir" TILEXR_P2P_CSV="$csv" "$bin" \
    2 "$rank" 4 0 2 6 \
    0 1 4096 1048576 2 \
    20 5 1 "$csv" "$log_dir" data_as_flag_epoch_ordered \
    1 unidir >"$log_dir/rank_${rank}.log" 2>&1 &
  pids[$rank]=$!
done
ret=0
for pid in ${pids[0]} ${pids[1]}; do
  wait "$pid" || ret=$?
done
echo "log_dir=$log_dir"
cat "$csv"
exit $ret
```

Expected: all CSV rows have `status=0` and `errors=0`.

- [ ] **Step 4: Run strict smoke**

Repeat Step 3 with:

```bash
export TILEXR_DATA_AS_FLAG_STRICT=1
```

Use a smaller range:

```text
8 65536 2
```

Expected: all CSV rows have `status=0` and `errors=0`.

- [ ] **Step 5: Run full unidir sweep 8B to 64MB**

Run with:

```text
min_bytes=8
max_bytes=67108864
step_factor=2
iters=20
warmup_iters=5
transport=data_as_flag_epoch_ordered
block_dim=1
traffic=unidir
```

Expected: 24 data rows, all `status=0`, all `errors=0`.

- [ ] **Step 6: Download CSVs**

Download the new sweep:

```powershell
@'
import os
import pathlib
import paramiko
host = "141.62.19.144"
user = "root"
password = os.environ["TILEXR_REMOTE_PASS"]
local_csv = pathlib.Path(r"D:\workspace\TileXR\run\csv_full\data_as_flag_epoch_ordered_unidir_bd1_8B_64MB.csv")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username=user, password=password, timeout=15,
            banner_timeout=15, auth_timeout=15)
stdin, stdout, stderr = ssh.exec_command(
    "ls -td /home/h30059441/TileXR/tests/udma/logs/tilexr_daf_epoch_* 2>/dev/null | head -1",
    timeout=30)
log_dir = stdout.read().decode("utf-8", errors="replace").strip()
err = stderr.read().decode("utf-8", errors="replace").strip()
if err:
    raise RuntimeError(err)
if not log_dir:
    raise RuntimeError("no tilexr_daf_epoch log directory found")
remote_csv = log_dir + "/p2p_perf.csv"
sftp = ssh.open_sftp()
sftp.get(remote_csv, str(local_csv))
sftp.close()
ssh.close()
print("downloaded", local_csv)
'@ | python -
```

- [ ] **Step 7: Confirm no stale demo processes**

Run:

```bash
ps -eo pid=,comm=,args= | awk '$2=="tilexr_udma_demo" {print}'
```

Expected: no output.

- [ ] **Step 8: Commit Task 6 if validation artifacts should be tracked**

Only commit CSVs or plot changes if the user asks to keep them in the branch:

```bash
git add run/csv_full/data_as_flag_epoch_ordered_unidir_bd1_8B_64MB.csv
git commit -m "test: add epoch ordered data-as-flag p2p sweep results"
```

If the user does not ask to track artifacts, do not commit this task.

---

### Task 7: Decide Migration to Public `data_as_flag`

**Files:**
- Modify later only after user approval:
  - `tests/udma/demo/tilexr_udma_p2p_perf_config.h`
  - `tests/udma/demo/tilexr_udma_demo.cpp`
  - `tests/udma/demo/tilexr_udma_demo_kernel.cpp`
  - `tests/udma/unit/test_tilexr_udma_p2p_perf_config.cpp`
  - `tests/udma/unit/test_tilexr_udma_p2p_source_guard.cpp`
  - plotting scripts under `run/csv_full/` if needed

**Interfaces:**
- Consumes: successful Task 6 validation evidence.
- Produces: a follow-up plan or patch that maps `data_as_flag` to the epoch-ordered implementation.

- [ ] **Step 1: Summarize validation**

Prepare a short table:

```text
transport                       range        block_dim  strict  status/errors
data_as_flag                    8B-64MB      1          no      ...
data_as_flag_epoch_ordered      8B-64MB      1          no      ...
data_as_flag_epoch_ordered      8B-64KB      1          yes     ...
```

- [ ] **Step 2: Ask for migration approval**

Ask the user:

```text
Validation is clean. Should I now replace legacy data_as_flag with the epoch-ordered implementation and keep data_as_flag_epoch_ordered as a temporary alias?
```

- [ ] **Step 3: Stop if approval is not explicit**

Do not remove legacy `data_as_flag` in this plan without explicit user approval after validation.

---

## Self-Review Checklist

- Spec coverage:
  - New transport config: Task 1.
  - Per-launch epoch: Tasks 2, 3, 4.
  - Embedded 512B block layout unchanged: Tasks 2 and 3.
  - Separate ordered MTE3 payload/flag stages: Task 2.
  - Fast receiver checks last flag: Task 2.
  - Strict/debug path: Task 5 and Task 6 strict smoke.
  - Side-by-side validation with old `data_as_flag`: Task 6 and Task 7.
  - Later migration to public `data_as_flag`: Task 7.
- Placeholder scan: no red-flag placeholder words or unnamed tests remain.
- Type consistency:
  - Enum name is consistently `DataAsFlagEpochOrdered`.
  - Public transport string is consistently `data_as_flag_epoch_ordered`.
  - Device epoch type is consistently `uint64_t`.
  - Launcher carries `int32_t magic`, `int32_t step`, and later `int32_t strict`.
