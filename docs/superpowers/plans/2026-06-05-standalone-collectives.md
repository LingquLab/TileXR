# Standalone Collectives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `TileXRAllReduce`, `TileXRReduceScatter`, and `TileXRBroadcast` to the optional TileXR collectives library, with SUM-only reduction support and hardware verification on `ssh blue`.

**Architecture:** Keep collectives in `libtilexr-collectives.so`; `libtile-comm.so` remains infrastructure-only. Extend the existing collectives host dispatch and CCE registration path, import only the missing lcal standalone kernel families into `src/collectives/kernels`, and validate behavior through unit tests, integration correctness tests, perf smoke tests, and final Ascend hardware runs.

**Tech Stack:** C++14, CMake, CANN 9.1.0, Ascend C/CCE, ACL runtime, TileXR communicator APIs.

---

## File Structure

Modify these public headers:

- `src/include/tilexr_types.h`: add `TileXRReduceOp`.
- `src/include/tilexr_collectives.h`: add the three public API declarations.

Modify these collectives host files:

- `src/collectives/host/tilexr_collectives.cpp`: add wrappers and validation for AllReduce, ReduceScatter, Broadcast.
- `src/collectives/host/collective_utils.h`: declare reduce-op support and new blockDim helpers.
- `src/collectives/host/collective_utils.cpp`: implement SUM-only reduce-op support and lcal-derived blockDim helpers.
- `src/collectives/host/collective_kernel.h`: add launch attributes for reduce op/root.
- `src/collectives/host/collective_kernel.cpp`: register and launch all five standalone collectives.

Modify these collectives kernel files:

- `src/collectives/kernels/lccl_op.h`: include imported kernel families and define new wrapper macros.
- `src/collectives/kernels/tilexr_lccl_op.cpp`: instantiate AllReduce, ReduceScatter, and Broadcast kernels.
- `src/collectives/kernels/CMakeLists.txt`: increase padded binary size only if the CCE link output exceeds the current 5 MiB pad.

Create these imported kernel files by copying from `reference/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels`:

- `src/collectives/kernels/allreduce_one_shot.h`
- `src/collectives/kernels/allreduce_two_shot.h`
- `src/collectives/kernels/allreduce_big_data.h`
- `src/collectives/kernels/91093/allreduce_big_data_sio.h`
- `src/collectives/kernels/91093/allreduce_hierarchy_double_ring.h`
- `src/collectives/kernels/reduce_scatter.h`
- `src/collectives/kernels/91093/reduce_scatter_big_data_91093_4step.h`
- `src/collectives/kernels/91093/reduce_scatter_hierarchy_double_ring.h`
- `src/collectives/kernels/kernels/lcal_allreduce_2npu_read.cce`
- `src/collectives/kernels/kernels/lcal_allreduce_2npu_write.cce`
- `src/collectives/kernels/kernels/lcal_allreduce_2npu_big_write.cce`
- `src/collectives/kernels/kernels/lcal_allreduce_two_shot.cce`
- `src/collectives/kernels/kernels/lcal_allreduce_big_data.cce`
- `src/collectives/kernels/kernels/lcal_allreduce_two_shot_910B2C.cce`
- `src/collectives/kernels/kernels/lcal_allreduce_big_data_910B2C.cce`
- `src/collectives/kernels/kernels/lcal_allreduce_deterministic.cce`
- `src/collectives/kernels/kernels/lcal_allreduce_deterministic_big_data.cce`
- `src/collectives/kernels/kernels/lcal_reduce_scatter.cce`
- `src/collectives/kernels/kernels/lcal_reduce_scatter_big_data.cce`
- `src/collectives/kernels/kernels/lcal_reduce_scatter_write.cce`
- `src/collectives/kernels/kernels/lcal_reduce_scatter_big_data_write.cce`
- `src/collectives/kernels/kernels/lcal_broadcast_write.cce`
- `src/collectives/kernels/kernels/lcal_broadcast_big_data.cce`

Modify these tests and tools:

- `tests/collectives/unit/test_tilexr_collectives_api.cpp`: public API ownership checks.
- `tests/collectives/unit/test_tilexr_collectives_header_compile.cpp`: compile-check function pointers and enum values.
- `tests/collectives/unit/test_tilexr_collectives_stub_behavior.cpp`: validation behavior for initialized single-rank comm without launching kernels.
- `tests/collectives/unit/test_tilexr_collectives_uninitialized_comm.cpp`: uninitialized comm behavior.
- `tests/collectives/unit/test_collective_host_utils.cpp`: blockDim and reduce-op helper tests.
- `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`: imported kernel ownership and registration checks.
- `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`: CLI/docs source checks for new ops.
- `tests/collectives/integration/test_tilexr_collectives_correctness.cpp`: INT32 correctness for new ops.
- `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`: perf/check support for new ops.
- `tests/collectives/README.md`: document new ops and message-size semantics.

---

### Task 1: Public API Tests

**Files:**
- Modify: `tests/collectives/unit/test_tilexr_collectives_api.cpp`
- Modify: `tests/collectives/unit/test_tilexr_collectives_header_compile.cpp`
- Test: `test_tilexr_collectives_api`
- Test: `test_tilexr_collectives_header_compile`

- [ ] **Step 1: Extend public API ownership test**

In `tests/collectives/unit/test_tilexr_collectives_api.cpp`, update `TestCollectivesHeaderDeclaresPublicApis()` to include these checks after the existing AllToAll check:

```cpp
    CheckContains(path, text, "int TileXRAllReduce(void *sendBuf, void *recvBuf, int64_t count,");
    CheckContains(path, text, "TileXR::TileXRReduceOp op,");
    CheckContains(path, text, "int TileXRReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount,");
    CheckContains(path, text, "int TileXRBroadcast(void *buf, int64_t count,");
```

In `TestCoreApiHeaderDoesNotDeclareCollectives()`, add:

```cpp
    CheckDoesNotContain(path, text, "TileXRAllReduce");
    CheckDoesNotContain(path, text, "TileXRReduceScatter");
    CheckDoesNotContain(path, text, "TileXRBroadcast");
```

- [ ] **Step 2: Extend header compile test**

Replace the function pointer section in `tests/collectives/unit/test_tilexr_collectives_header_compile.cpp` with:

```cpp
using AllGatherFn = int (*)(void *, void *, int64_t, TileXR::TileXRDataType, TileXRCommPtr, aclrtStream);
using AllToAllFn = int (*)(void *, void *, int64_t, TileXR::TileXRDataType, TileXRCommPtr, aclrtStream);
using AllReduceFn = int (*)(void *, void *, int64_t, TileXR::TileXRDataType, TileXR::TileXRReduceOp,
                            TileXRCommPtr, aclrtStream);
using ReduceScatterFn = int (*)(void *, void *, int64_t, TileXR::TileXRDataType, TileXR::TileXRReduceOp,
                                TileXRCommPtr, aclrtStream);
using BroadcastFn = int (*)(void *, int64_t, TileXR::TileXRDataType, int, TileXRCommPtr, aclrtStream);
```

In `main()`, after the existing function pointers, add:

```cpp
    AllReduceFn allReduce = &TileXRAllReduce;
    ReduceScatterFn reduceScatter = &TileXRReduceScatter;
    BroadcastFn broadcast = &TileXRBroadcast;
    const bool reduceOpsPresent =
        TileXR::TILEXR_REDUCE_SUM != TileXR::TILEXR_REDUCE_RESERVED &&
        TileXR::TILEXR_REDUCE_MAX != TileXR::TILEXR_REDUCE_RESERVED &&
        TileXR::TILEXR_REDUCE_MIN != TileXR::TILEXR_REDUCE_RESERVED &&
        TileXR::TILEXR_REDUCE_PROD != TileXR::TILEXR_REDUCE_RESERVED;
```

Change the return expression to:

```cpp
    return (allGather != nullptr && allToAll != nullptr && allReduce != nullptr &&
            reduceScatter != nullptr && broadcast != nullptr && reduceOpsPresent &&
            header.magic == TileXR::TILEXR_PERF_TRACE_MAGIC && config.enabled == 1) ? 0 : 1;
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
source scripts/common_env.sh
cmake -S . -B build -DTILEXR_BUILD_COLLECTIVES=ON -DCMAKE_INSTALL_PREFIX=install
cmake --build build --target test_tilexr_collectives_api test_tilexr_collectives_header_compile -j$(nproc)
ctest --test-dir build -R 'test_tilexr_collectives_api|test_tilexr_collectives_header_compile' --output-on-failure
```

Expected: FAIL. The API ownership test reports missing strings, and the header compile test reports missing `TileXRReduceOp` and missing new API symbols.

- [ ] **Step 4: Commit failing tests**

```bash
git add tests/collectives/unit/test_tilexr_collectives_api.cpp tests/collectives/unit/test_tilexr_collectives_header_compile.cpp
git commit -m "test: specify standalone collective public APIs"
```

---

### Task 2: Public API Headers

**Files:**
- Modify: `src/include/tilexr_types.h`
- Modify: `src/include/tilexr_collectives.h`
- Test: `test_tilexr_collectives_api`
- Test: `test_tilexr_collectives_header_compile`

- [ ] **Step 1: Add reduce-op enum**

In `src/include/tilexr_types.h`, add this enum immediately after `enum TileXRDataType`.
The ordinal values are public ABI and must match the HCCL/repo convention:
SUM=0, PROD=1, MAX=2, MIN=3.

```cpp
enum TileXRReduceOp {
    TILEXR_REDUCE_SUM = 0,
    TILEXR_REDUCE_PROD = 1,
    TILEXR_REDUCE_MAX = 2,
    TILEXR_REDUCE_MIN = 3,
    TILEXR_REDUCE_RESERVED = 255
};
```

- [ ] **Step 2: Add public API declarations**

In `src/include/tilexr_collectives.h`, add these declarations inside the existing `extern "C"` block after `TileXRAllToAll`:

```cpp
int TileXRAllReduce(void *sendBuf, void *recvBuf, int64_t count,
                    TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                    TileXRCommPtr comm, aclrtStream stream);
int TileXRReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount,
                        TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                        TileXRCommPtr comm, aclrtStream stream);
int TileXRBroadcast(void *buf, int64_t count,
                    TileXR::TileXRDataType dataType, int root,
                    TileXRCommPtr comm, aclrtStream stream);
```

- [ ] **Step 3: Run tests to verify pass**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collectives_api test_tilexr_collectives_header_compile -j$(nproc)
ctest --test-dir build -R 'test_tilexr_collectives_api|test_tilexr_collectives_header_compile' --output-on-failure
```

Expected: PASS for both tests.

- [ ] **Step 4: Commit API headers**

```bash
git add src/include/tilexr_types.h src/include/tilexr_collectives.h
git commit -m "feat: declare standalone collective APIs"
```

---

### Task 3: Host Utility Tests

**Files:**
- Modify: `tests/collectives/unit/test_collective_host_utils.cpp`
- Test: `test_collective_host_utils`

- [ ] **Step 1: Add reduce-op helper test**

In `tests/collectives/unit/test_collective_host_utils.cpp`, add this function after `TestDataTypeSupport()`:

```cpp
void TestReduceOpSupport()
{
    using TileXRCollectives::Host::IsSupportedReduceOp;

    CheckBool("SUM reduce op supported", IsSupportedReduceOp(TileXR::TILEXR_REDUCE_SUM), true);
    CheckBool("MAX reduce op unsupported", IsSupportedReduceOp(TileXR::TILEXR_REDUCE_MAX), false);
    CheckBool("MIN reduce op unsupported", IsSupportedReduceOp(TileXR::TILEXR_REDUCE_MIN), false);
    CheckBool("PROD reduce op unsupported", IsSupportedReduceOp(TileXR::TILEXR_REDUCE_PROD), false);
    CheckBool("reserved reduce op unsupported", IsSupportedReduceOp(TileXR::TILEXR_REDUCE_RESERVED), false);
    CheckBool("unknown reduce op unsupported",
              IsSupportedReduceOp(static_cast<TileXR::TileXRReduceOp>(999)), false);
}
```

Call it in `main()` after `TestDataTypeSupport();`:

```cpp
    TestReduceOpSupport();
```

- [ ] **Step 2: Add AllReduce blockDim tests**

Add this function after `TestAllToAllBlockNum()`:

```cpp
void TestAllReduceBlockNum()
{
    using TileXRCollectives::Host::GetAllReduceBlockNum;

    CheckUint32("rank2 small allreduce uses two blocks per rank",
                GetAllReduceBlockNum(Args(2, 0), 1024),
                4);
    CheckUint32("rank4 small allreduce uses one block per rank",
                GetAllReduceBlockNum(Args(4, 0), 1024),
                4);
    CheckUint32("rank4 large allreduce uses two blocks per rank",
                GetAllReduceBlockNum(Args(4, 0), 2 * 1024 * 1024),
                8);
    CheckUint32("pcie allreduce uses two blocks per rank",
                GetAllReduceBlockNum(Args(4, TileXR::ExtraFlag::TOPO_PCIE), 1024),
                8);
    CheckUint32("910B2C rank16 small allreduce follows lcal split",
                GetAllReduceBlockNum(Args(16, TileXR::ExtraFlag::TOPO_910B2C), 1024),
                16);
    CheckUint32("910B2C rank16 large allreduce follows lcal split",
                GetAllReduceBlockNum(Args(16, TileXR::ExtraFlag::TOPO_910B2C), 2 * 1024 * 1024),
                26);
    CheckUint32("910_93 rank8 large allreduce uses double ring",
                GetAllReduceBlockNum(Args(8, TileXR::ExtraFlag::TOPO_910_93), 33LL * 1024 * 1024),
                34);
}
```

Call it in `main()` after `TestAllToAllBlockNum();`:

```cpp
    TestAllReduceBlockNum();
```

- [ ] **Step 3: Add ReduceScatter and Broadcast blockDim tests**

Add this function after `TestAllReduceBlockNum()`:

```cpp
void TestReduceScatterBlockNum()
{
    using TileXRCollectives::Host::GetReduceScatterBlockNum;

    CheckUint32("rank2 small reducescatter uses two blocks per rank",
                GetReduceScatterBlockNum(Args(2, 0), 1024),
                4);
    CheckUint32("rank4 small reducescatter uses one block per rank",
                GetReduceScatterBlockNum(Args(4, 0), 1024),
                4);
    CheckUint32("rank4 large reducescatter uses two blocks per rank",
                GetReduceScatterBlockNum(Args(4, 0), 2 * 1024 * 1024),
                8);
    CheckUint32("pcie reducescatter uses two blocks per rank",
                GetReduceScatterBlockNum(Args(4, TileXR::ExtraFlag::TOPO_PCIE), 1024),
                8);
    CheckUint32("910_93 rank4 double ring reducescatter",
                GetReduceScatterBlockNum(Args(4, TileXR::ExtraFlag::TOPO_910_93), 512 * 1024),
                36);
    CheckUint32("910_93 rank16 large reducescatter uses four-step block count",
                GetReduceScatterBlockNum(Args(16, TileXR::ExtraFlag::TOPO_910_93), 2LL * 1024 * 1024),
                34);
}

void TestBroadcastBlockNum()
{
    using TileXRCollectives::Host::GetBroadcastBlockNum;

    CheckUint32("rank1 broadcast has one block",
                GetBroadcastBlockNum(Args(1, 0), 1024),
                1);
    CheckUint32("rank8 broadcast uses rankSize blocks",
                GetBroadcastBlockNum(Args(8, 0), 1024),
                8);
    CheckUint32("rank9 broadcast unsupported",
                GetBroadcastBlockNum(Args(9, 0), 1024),
                0);
}
```

Call them in `main()` after `TestAllReduceBlockNum();`:

```cpp
    TestReduceScatterBlockNum();
    TestBroadcastBlockNum();
```

- [ ] **Step 4: Run test to verify failure**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_collective_host_utils -j$(nproc)
ctest --test-dir build -R test_collective_host_utils --output-on-failure
```

Expected: FAIL at compile time because `IsSupportedReduceOp`, `GetAllReduceBlockNum`, `GetReduceScatterBlockNum`, and `GetBroadcastBlockNum` do not exist.

- [ ] **Step 5: Commit failing host utility tests**

```bash
git add tests/collectives/unit/test_collective_host_utils.cpp
git commit -m "test: specify standalone collective host helpers"
```

---

### Task 4: Host Utility Implementation

**Files:**
- Modify: `src/collectives/host/collective_utils.h`
- Modify: `src/collectives/host/collective_utils.cpp`
- Test: `test_collective_host_utils`

- [ ] **Step 1: Declare helper APIs**

In `src/collectives/host/collective_utils.h`, add after `IsSupportedDataType`:

```cpp
bool IsSupportedReduceOp(TileXR::TileXRReduceOp reduceOp);
```

Add after `GetAllToAllBlockNum`:

```cpp
uint32_t GetAllReduceBlockNum(const TileXR::CommArgs &commArgs, int64_t dataSize);

uint32_t GetReduceScatterBlockNum(const TileXR::CommArgs &commArgs, int64_t dataSize);

uint32_t GetBroadcastBlockNum(const TileXR::CommArgs &commArgs, int64_t dataSize);
```

- [ ] **Step 2: Add constants**

In `src/collectives/host/collective_utils.cpp`, add these constants in the anonymous namespace after the existing constants:

```cpp
constexpr uint32_t THREE_STEP_BLOCK_NUM = 3;
constexpr uint32_t ALL_REDUCE_DB_RING_BLOCK_NUM = 34;
constexpr uint32_t REDUCE_SCATTER_FOUR_STEP_BLOCK_NUM = 34;
constexpr uint32_t REDUCE_SCATTER_DB_RING_BLOCK_NUM = 36;
constexpr uint32_t BROADCAST_MAX_RANK_SIZE = 8;
constexpr int64_t ONE_MIB = 1LL * 1024 * 1024;
constexpr int64_t TWO_MIB = 2LL * 1024 * 1024;
constexpr int64_t THIRTY_TWO_MIB = 32LL * 1024 * 1024;
```

- [ ] **Step 3: Implement SUM-only reduce-op validation**

In `src/collectives/host/collective_utils.cpp`, add after `IsSupportedDataType`:

```cpp
bool IsSupportedReduceOp(TileXR::TileXRReduceOp reduceOp)
{
    return reduceOp == TileXR::TILEXR_REDUCE_SUM;
}
```

- [ ] **Step 4: Implement AllReduce blockDim helper**

In `src/collectives/host/collective_utils.cpp`, add after `GetAllToAllBlockNum`:

```cpp
uint32_t GetAllReduceBlockNum(const TileXR::CommArgs &commArgs, int64_t dataSize)
{
    const uint32_t rankSize = RankSizeOrZero(commArgs);
    if (rankSize == 0 || dataSize < 0) {
        return 0;
    }

    const uint32_t extraFlag = commArgs.extraFlag;
    if ((extraFlag & TileXR::ExtraFlag::TOPO_PCIE) != 0) {
        return rankSize * TWO_BLOCK_NUM;
    }
    if ((extraFlag & TileXR::ExtraFlag::TOPO_910B2C) != 0 && rankSize > SMALL_RANK_SIZE) {
        return dataSize < TWO_MIB ? rankSize :
            (rankSize / TWO_BLOCK_NUM * THREE_STEP_BLOCK_NUM + TWO_BLOCK_NUM);
    }
    if (GetParallel()) {
        return rankSize;
    }
    if ((extraFlag & TileXR::ExtraFlag::TOPO_910_93) != 0 &&
        dataSize > THIRTY_TWO_MIB && rankSize != TileXR::RANK_SIZE_TWO) {
        return rankSize % TileXR::RANK_SIZE_TWO == 0 ?
            ALL_REDUCE_DB_RING_BLOCK_NUM : rankSize * THREE_STEP_BLOCK_NUM;
    }
    return (rankSize == TileXR::RANK_SIZE_TWO || dataSize >= TWO_MIB) ?
        rankSize * TWO_BLOCK_NUM : rankSize;
}
```

- [ ] **Step 5: Implement ReduceScatter and Broadcast blockDim helpers**

In `src/collectives/host/collective_utils.cpp`, add after `GetAllReduceBlockNum`:

```cpp
uint32_t GetReduceScatterBlockNum(const TileXR::CommArgs &commArgs, int64_t dataSize)
{
    const uint32_t rankSize = RankSizeOrZero(commArgs);
    if (rankSize == 0 || dataSize < 0) {
        return 0;
    }

    const uint32_t extraFlag = commArgs.extraFlag;
    if ((extraFlag & TileXR::ExtraFlag::TOPO_PCIE) != 0) {
        return rankSize * TWO_BLOCK_NUM;
    }

    const bool isDbRing = (rankSize == 4 || rankSize == SMALL_RANK_SIZE) &&
        (dataSize * SMALL_RANK_SIZE > TWO_MIB && dataSize * SMALL_RANK_SIZE <= THIRTY_TWO_MIB);
    if ((extraFlag & TileXR::ExtraFlag::TOPO_910_93) != 0 &&
        ((rankSize > SMALL_RANK_SIZE && rankSize % TWO_BLOCK_NUM == 0) || isDbRing)) {
        if (isDbRing) {
            return REDUCE_SCATTER_DB_RING_BLOCK_NUM;
        }
        return dataSize <= ONE_MIB ? rankSize : REDUCE_SCATTER_FOUR_STEP_BLOCK_NUM;
    }
    return (rankSize == TileXR::RANK_SIZE_TWO || dataSize >= TWO_MIB) ?
        rankSize * TWO_BLOCK_NUM : rankSize;
}

uint32_t GetBroadcastBlockNum(const TileXR::CommArgs &commArgs, int64_t dataSize)
{
    const uint32_t rankSize = RankSizeOrZero(commArgs);
    if (rankSize == 0 || dataSize < 0 || rankSize > BROADCAST_MAX_RANK_SIZE) {
        return 0;
    }
    return rankSize;
}
```

- [ ] **Step 6: Run test to verify pass**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_collective_host_utils -j$(nproc)
ctest --test-dir build -R test_collective_host_utils --output-on-failure
```

Expected: PASS.

- [ ] **Step 7: Commit host utilities**

```bash
git add src/collectives/host/collective_utils.h src/collectives/host/collective_utils.cpp
git commit -m "feat: add standalone collective host helpers"
```

---

### Task 5: Host Wrapper Validation Tests

**Files:**
- Modify: `tests/collectives/unit/test_tilexr_collectives_stub_behavior.cpp`
- Modify: `tests/collectives/unit/test_tilexr_collectives_uninitialized_comm.cpp`
- Test: `test_tilexr_collectives_stub_behavior`
- Test: `test_tilexr_collectives_uninitialized_comm`

- [ ] **Step 1: Extend stub validation cases**

In `tests/collectives/unit/test_tilexr_collectives_stub_behavior.cpp`, add this struct after `ValidationCase`:

```cpp
struct BroadcastValidationCase {
    const char *name;
    void *buf;
    int64_t count;
    TileXR::TileXRDataType dataType;
    int root;
    TileXRCommPtr comm;
};
```

After the existing loop that checks AllGather and AllToAll, add:

```cpp
    for (const auto& testCase : cases) {
        CheckStatus("TileXRAllReduce", testCase,
                    TileXRAllReduce(testCase.sendBuf, testCase.recvBuf, testCase.sendCount,
                                    testCase.dataType, TileXR::TILEXR_REDUCE_SUM,
                                    testCase.comm, nullptr));
        CheckStatus("TileXRReduceScatter", testCase,
                    TileXRReduceScatter(testCase.sendBuf, testCase.recvBuf, testCase.sendCount,
                                        testCase.dataType, TileXR::TILEXR_REDUCE_SUM,
                                        testCase.comm, nullptr));
    }

    const ValidationCase unsupportedReduceOps[] = {
        { "max reduce op", sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, comm },
        { "min reduce op", sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, comm },
        { "prod reduce op", sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, comm },
        { "reserved reduce op", sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, comm },
    };
    const TileXR::TileXRReduceOp reduceOps[] = {
        TileXR::TILEXR_REDUCE_MAX,
        TileXR::TILEXR_REDUCE_MIN,
        TileXR::TILEXR_REDUCE_PROD,
        TileXR::TILEXR_REDUCE_RESERVED,
    };
    for (size_t i = 0; i < sizeof(reduceOps) / sizeof(reduceOps[0]); ++i) {
        CheckStatus("TileXRAllReduce", unsupportedReduceOps[i],
                    TileXRAllReduce(unsupportedReduceOps[i].sendBuf, unsupportedReduceOps[i].recvBuf,
                                    unsupportedReduceOps[i].sendCount, unsupportedReduceOps[i].dataType,
                                    reduceOps[i], unsupportedReduceOps[i].comm, nullptr));
        CheckStatus("TileXRReduceScatter", unsupportedReduceOps[i],
                    TileXRReduceScatter(unsupportedReduceOps[i].sendBuf, unsupportedReduceOps[i].recvBuf,
                                        unsupportedReduceOps[i].sendCount, unsupportedReduceOps[i].dataType,
                                        reduceOps[i], unsupportedReduceOps[i].comm, nullptr));
    }

    const BroadcastValidationCase broadcastCases[] = {
        { "null comm", recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, 0, nullptr },
        { "buffer null", nullptr, 1, TileXR::TILEXR_DATA_TYPE_FP16, 0, comm },
        { "zero count", recvStorage, 0, TileXR::TILEXR_DATA_TYPE_INT32, 0, comm },
        { "negative count", recvStorage, -1, TileXR::TILEXR_DATA_TYPE_INT32, 0, comm },
        { "negative root", recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, -1, comm },
        { "root past rank size", recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, 1, comm },
        { "unsupported uint8 datatype", recvStorage, 1024, TileXR::TILEXR_DATA_TYPE_UINT8, 0, comm },
        { "unknown datatype", recvStorage, 1024, static_cast<TileXR::TileXRDataType>(999), 0, comm },
    };
    for (const auto& testCase : broadcastCases) {
        ValidationCase adapted { testCase.name, testCase.buf, testCase.buf, testCase.count,
            testCase.dataType, testCase.comm };
        CheckStatus("TileXRBroadcast", adapted,
                    TileXRBroadcast(testCase.buf, testCase.count, testCase.dataType,
                                    testCase.root, testCase.comm, nullptr));
    }
```

Add `#include <cstddef>` at the top because the new loop uses `size_t`.

- [ ] **Step 2: Extend uninitialized comm test**

In `tests/collectives/unit/test_tilexr_collectives_uninitialized_comm.cpp`, add after the existing AllToAll check:

```cpp
    CheckStatus("TileXRAllReduce(uninitialized comm)",
                TileXRAllReduce(sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32,
                                TileXR::TILEXR_REDUCE_SUM, comm, nullptr),
                TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    CheckStatus("TileXRReduceScatter(uninitialized comm)",
                TileXRReduceScatter(sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32,
                                    TileXR::TILEXR_REDUCE_SUM, comm, nullptr),
                TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    CheckStatus("TileXRBroadcast(uninitialized comm)",
                TileXRBroadcast(recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, 0, comm, nullptr),
                TileXR::TILEXR_ERROR_NOT_INITIALIZED);
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collectives_stub_behavior test_tilexr_collectives_uninitialized_comm -j$(nproc)
ctest --test-dir build -R 'test_tilexr_collectives_stub_behavior|test_tilexr_collectives_uninitialized_comm' --output-on-failure
```

Expected: FAIL at link time because the new APIs are declared but not defined.

- [ ] **Step 4: Commit failing validation tests**

```bash
git add tests/collectives/unit/test_tilexr_collectives_stub_behavior.cpp tests/collectives/unit/test_tilexr_collectives_uninitialized_comm.cpp
git commit -m "test: specify standalone collective validation"
```

---

### Task 6: Host Wrapper Implementation

**Files:**
- Modify: `src/collectives/host/collective_kernel.h`
- Modify: `src/collectives/host/collective_kernel.cpp`
- Modify: `src/collectives/host/tilexr_collectives.cpp`
- Test: `test_tilexr_collectives_stub_behavior`
- Test: `test_tilexr_collectives_uninitialized_comm`
- Test: `test_collective_host_utils`

- [ ] **Step 1: Add launch attributes type**

In `src/collectives/host/collective_kernel.h`, add before `LaunchCollectiveKernel`:

```cpp
struct CollectiveLaunchAttrs {
    int op = 0;
    int root = 0;
};
```

Change the `LaunchCollectiveKernel` declaration to:

```cpp
int LaunchCollectiveKernel(TileXRCommPtr comm, TileXR::TileXRType type, const HostLaunchContext &context,
                           void *sendBuf, void *recvBuf, int64_t kernelCount,
                           TileXR::TileXRDataType dataType, uint32_t blockDim,
                           aclrtStream stream,
                           CollectiveLaunchAttrs attrs = CollectiveLaunchAttrs {});
```

- [ ] **Step 2: Accept all standalone collective types in launch validation**

In `src/collectives/host/collective_kernel.cpp`, add this helper inside the anonymous namespace:

```cpp
bool IsStandaloneCollectiveType(TileXR::TileXRType type)
{
    return type == TileXR::TileXRType::ALL_GATHER ||
        type == TileXR::TileXRType::ALL2ALL ||
        type == TileXR::TileXRType::ALL_REDUCE ||
        type == TileXR::TileXRType::REDUCE_SCATTER ||
        type == TileXR::TileXRType::BROADCAST;
}
```

Replace the type validation at the start of `LaunchCollectiveKernel` with:

```cpp
    if (!IsStandaloneCollectiveType(type) ||
        comm == nullptr || context.hostArgs == nullptr || context.devArgs == nullptr || sendBuf == nullptr || recvBuf == nullptr ||
        kernelCount <= 0 || blockDim == 0 || !IsSupportedDataType(dataType)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
```

Change the function signature in the definition to match the header and set the kernel args fields:

```cpp
                           aclrtStream stream,
                           CollectiveLaunchAttrs attrs)
```

Replace:

```cpp
    args.op = 0;
```

with:

```cpp
    args.op = attrs.op;
    args.root = attrs.root;
```

- [ ] **Step 3: Add host validation helpers**

In `src/collectives/host/tilexr_collectives.cpp`, add after `ValidateCommon`:

```cpp
int ValidateReduce(void *sendBuf, void *recvBuf, int64_t count,
                   TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                   TileXRCommPtr comm)
{
    const int ret = ValidateCommon(sendBuf, recvBuf, count, dataType, comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    return TileXRCollectives::Host::IsSupportedReduceOp(op) ?
        TileXR::TILEXR_SUCCESS : TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
}

int ValidateBroadcast(void *buf, int64_t count, TileXR::TileXRDataType dataType,
                      int root, TileXRCommPtr comm)
{
    if (comm == nullptr || buf == nullptr || count <= 0 ||
        !TileXRCollectives::Host::IsSupportedDataType(dataType)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (TileXRCollectives::Host::CountToBytes(count, dataType) < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    TileXRCollectives::Host::HostLaunchContext context;
    const int prepareRet = TileXRCollectives::Host::PrepareHostLaunchContext(comm, context);
    if (prepareRet != TileXR::TILEXR_SUCCESS) {
        return prepareRet;
    }
    if (root < 0 || root >= context.hostArgs->rankSize) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXR::TILEXR_SUCCESS;
}
```

- [ ] **Step 4: Implement `TileXRAllReduce`**

In `src/collectives/host/tilexr_collectives.cpp`, add after `TileXRAllToAll`:

```cpp
int TileXRAllReduce(void *sendBuf, void *recvBuf, int64_t count,
                    TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                    TileXRCommPtr comm, aclrtStream stream)
{
    int ret = ValidateReduce(sendBuf, recvBuf, count, dataType, op, comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    TileXRCollectives::Host::HostLaunchContext context;
    ret = TileXRCollectives::Host::PrepareHostLaunchContext(comm, context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    const int64_t bytes = TileXRCollectives::Host::CountToBytes(count, dataType);
    if (context.hostArgs->rankSize <= 1) {
        return LoopbackCopy(sendBuf, recvBuf, bytes, stream);
    }

    const uint32_t blockDim = TileXRCollectives::Host::GetAllReduceBlockNum(*context.hostArgs, bytes);
    return TileXRCollectives::Host::LaunchCollectiveKernel(comm, TileXR::TileXRType::ALL_REDUCE, context,
        sendBuf, recvBuf, count, dataType, blockDim, stream,
        TileXRCollectives::Host::CollectiveLaunchAttrs { static_cast<int>(op), 0 });
}
```

- [ ] **Step 5: Implement `TileXRReduceScatter`**

In `src/collectives/host/tilexr_collectives.cpp`, add after `TileXRAllReduce`:

```cpp
int TileXRReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount,
                        TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                        TileXRCommPtr comm, aclrtStream stream)
{
    int ret = ValidateReduce(sendBuf, recvBuf, recvCount, dataType, op, comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    TileXRCollectives::Host::HostLaunchContext context;
    ret = TileXRCollectives::Host::PrepareHostLaunchContext(comm, context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    const int64_t bytes = TileXRCollectives::Host::CountToBytes(recvCount, dataType);
    const int rankSize = context.hostArgs->rankSize;
    if (rankSize <= 1) {
        return LoopbackCopy(sendBuf, recvBuf, bytes, stream);
    }
    if (recvCount > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(rankSize)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const int64_t inputCount = recvCount * static_cast<int64_t>(rankSize);
    const int64_t inputBytes = TileXRCollectives::Host::CountToBytes(inputCount, dataType);
    if (inputBytes < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const uint32_t blockDim = TileXRCollectives::Host::GetReduceScatterBlockNum(*context.hostArgs, bytes);
    return TileXRCollectives::Host::LaunchCollectiveKernel(comm, TileXR::TileXRType::REDUCE_SCATTER, context,
        sendBuf, recvBuf, recvCount, dataType, blockDim, stream,
        TileXRCollectives::Host::CollectiveLaunchAttrs { static_cast<int>(op), 0 });
}
```

- [ ] **Step 6: Implement `TileXRBroadcast`**

In `src/collectives/host/tilexr_collectives.cpp`, add after `TileXRReduceScatter`:

```cpp
int TileXRBroadcast(void *buf, int64_t count,
                    TileXR::TileXRDataType dataType, int root,
                    TileXRCommPtr comm, aclrtStream stream)
{
    int ret = ValidateBroadcast(buf, count, dataType, root, comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    TileXRCollectives::Host::HostLaunchContext context;
    ret = TileXRCollectives::Host::PrepareHostLaunchContext(comm, context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    const int64_t bytes = TileXRCollectives::Host::CountToBytes(count, dataType);
    if (context.hostArgs->rankSize <= 1) {
        return TileXR::TILEXR_SUCCESS;
    }

    const uint32_t blockDim = TileXRCollectives::Host::GetBroadcastBlockNum(*context.hostArgs, bytes);
    if (blockDim == 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXRCollectives::Host::LaunchCollectiveKernel(comm, TileXR::TileXRType::BROADCAST, context,
        buf, buf, count, dataType, blockDim, stream,
        TileXRCollectives::Host::CollectiveLaunchAttrs { 0, root });
}
```

- [ ] **Step 7: Run validation tests**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collectives_stub_behavior test_tilexr_collectives_uninitialized_comm test_collective_host_utils -j$(nproc)
ctest --test-dir build -R 'test_tilexr_collectives_stub_behavior|test_tilexr_collectives_uninitialized_comm|test_collective_host_utils' --output-on-failure
```

Expected: PASS.

- [ ] **Step 8: Commit host wrapper implementation**

```bash
git add src/collectives/host/collective_kernel.h src/collectives/host/collective_kernel.cpp src/collectives/host/tilexr_collectives.cpp
git commit -m "feat: add standalone collective host wrappers"
```

---

### Task 7: Kernel Import Ownership Tests

**Files:**
- Modify: `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`
- Test: `test_tilexr_collectives_kernel_ownership`

- [ ] **Step 1: Update scoped-kernel source test**

In `TestCollectivesKernelSourcesAreScoped()`, replace the existing checks that forbid AllReduce, ReduceScatter, and Broadcast in `tilexr_lccl_op.cpp` with:

```cpp
    CheckContains(kernelTuPath, kernelTu, "LCCL_TYPE_AIV_FUNC(LCCL_ALL_REDUCE_FUNC_AUTO_DEF)");
    CheckContains(kernelTuPath, kernelTu, "LCCL_TYPE_AIV_FUNC(LCCL_REDUCE_SCATTER_FUNC_AUTO_DEF)");
    CheckContains(kernelTuPath, kernelTu, "LCCL_BROADCAST_FUNC_AUTO_DEF()");
```

In the loop over `kernelFiles`, add booleans before the loop:

```cpp
    bool sawAllReduceCce = false;
    bool sawReduceScatterCce = false;
    bool sawBroadcastCce = false;
```

Inside the loop, add:

```cpp
        CheckDoesNotContain(path, text, "reference/ascend-transformer-boost");
        CheckDoesNotContain(path, text, "LCAL_MAX_RANK_SIZE");
        if (path.find("lcal_allreduce") != std::string::npos && path.find(".cce") != std::string::npos) {
            sawAllReduceCce = true;
        }
        if (path.find("lcal_reduce_scatter") != std::string::npos && path.find(".cce") != std::string::npos) {
            sawReduceScatterCce = true;
        }
        if (path.find("lcal_broadcast") != std::string::npos && path.find(".cce") != std::string::npos) {
            sawBroadcastCce = true;
        }
```

After the existing `sawAllToAllCce` check, add:

```cpp
    CheckTrue(sawAllReduceCce, "expected copied allreduce .cce sources under src/collectives/kernels");
    CheckTrue(sawReduceScatterCce, "expected copied reduce_scatter .cce sources under src/collectives/kernels");
    CheckTrue(sawBroadcastCce, "expected copied broadcast .cce sources under src/collectives/kernels");
```

- [ ] **Step 2: Update registration ownership test**

In `TestHostRegistrationLivesInCollectives()`, add after the existing AllToAll type checks:

```cpp
    CheckContains(kernelPath, kernel, "TileXR::TileXRType::ALL_REDUCE");
    CheckContains(kernelPath, kernel, "TileXR::TileXRType::REDUCE_SCATTER");
    CheckContains(kernelPath, kernel, "TileXR::TileXRType::BROADCAST");
```

- [ ] **Step 3: Update comm forbidden symbol list**

In `TestCommDoesNotOwnCollectiveRuntime()`, add to the `forbidden` array:

```cpp
        "TileXRAllReduce",
        "TileXRReduceScatter",
        "TileXRBroadcast",
```

- [ ] **Step 4: Run test to verify failure**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collectives_kernel_ownership -j$(nproc)
ctest --test-dir build -R test_tilexr_collectives_kernel_ownership --output-on-failure
```

Expected: FAIL because imported kernel files and registration entries do not exist.

- [ ] **Step 5: Commit failing kernel ownership tests**

```bash
git add tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp
git commit -m "test: specify standalone collective kernel ownership"
```

---

### Task 8: Import and Adapt Kernel Sources

**Files:**
- Create: all imported files listed in File Structure
- Modify: `src/collectives/kernels/lccl_op.h`
- Modify: `src/collectives/kernels/tilexr_lccl_op.cpp`
- Test: `test_tilexr_collectives_kernel_ownership`

- [ ] **Step 1: Copy missing lcal source files**

Run:

```bash
set -euo pipefail
src=reference/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels
dst=src/collectives/kernels
cp "$src/allreduce_one_shot.h" "$dst/allreduce_one_shot.h"
cp "$src/allreduce_two_shot.h" "$dst/allreduce_two_shot.h"
cp "$src/allreduce_big_data.h" "$dst/allreduce_big_data.h"
cp "$src/91093/allreduce_big_data_sio.h" "$dst/91093/allreduce_big_data_sio.h"
cp "$src/91093/allreduce_hierarchy_double_ring.h" "$dst/91093/allreduce_hierarchy_double_ring.h"
cp "$src/reduce_scatter.h" "$dst/reduce_scatter.h"
cp "$src/91093/reduce_scatter_big_data_91093_4step.h" "$dst/91093/reduce_scatter_big_data_91093_4step.h"
cp "$src/91093/reduce_scatter_hierarchy_double_ring.h" "$dst/91093/reduce_scatter_hierarchy_double_ring.h"
cp "$src/../kernels/lcal_allreduce_2npu_read.cce" "$dst/kernels/lcal_allreduce_2npu_read.cce"
cp "$src/../kernels/lcal_allreduce_2npu_write.cce" "$dst/kernels/lcal_allreduce_2npu_write.cce"
cp "$src/../kernels/lcal_allreduce_2npu_big_write.cce" "$dst/kernels/lcal_allreduce_2npu_big_write.cce"
cp "$src/../kernels/lcal_allreduce_two_shot.cce" "$dst/kernels/lcal_allreduce_two_shot.cce"
cp "$src/../kernels/lcal_allreduce_big_data.cce" "$dst/kernels/lcal_allreduce_big_data.cce"
cp "$src/../kernels/lcal_allreduce_two_shot_910B2C.cce" "$dst/kernels/lcal_allreduce_two_shot_910B2C.cce"
cp "$src/../kernels/lcal_allreduce_big_data_910B2C.cce" "$dst/kernels/lcal_allreduce_big_data_910B2C.cce"
cp "$src/../kernels/lcal_allreduce_deterministic.cce" "$dst/kernels/lcal_allreduce_deterministic.cce"
cp "$src/../kernels/lcal_allreduce_deterministic_big_data.cce" "$dst/kernels/lcal_allreduce_deterministic_big_data.cce"
cp "$src/../kernels/lcal_reduce_scatter.cce" "$dst/kernels/lcal_reduce_scatter.cce"
cp "$src/../kernels/lcal_reduce_scatter_big_data.cce" "$dst/kernels/lcal_reduce_scatter_big_data.cce"
cp "$src/../kernels/lcal_reduce_scatter_write.cce" "$dst/kernels/lcal_reduce_scatter_write.cce"
cp "$src/../kernels/lcal_reduce_scatter_big_data_write.cce" "$dst/kernels/lcal_reduce_scatter_big_data_write.cce"
cp "$src/../kernels/lcal_broadcast_write.cce" "$dst/kernels/lcal_broadcast_write.cce"
cp "$src/../kernels/lcal_broadcast_big_data.cce" "$dst/kernels/lcal_broadcast_big_data.cce"
```

Expected: command exits with status 0 and creates the listed files.

- [ ] **Step 2: Mechanically adapt copied source names**

Run:

```bash
set -euo pipefail
files="$(find src/collectives/kernels -type f \( -name 'allreduce*.h' -o -name 'reduce_scatter*.h' -o -name 'lcal_allreduce*.cce' -o -name 'lcal_reduce_scatter*.cce' -o -name 'lcal_broadcast*.cce' \))"
perl -0pi -e 's/\bLcal\b/TileXR/g; s/\bLCAL_/TILEXR_/g; s/\bLcalAllReduce/TileXRAllReduce/g; s/\bLcalReduceScatter/TileXRReduceScatter/g; s/\bLcalBroadcast/TileXRBroadcast/g; s/\bLcalAll2All/TileXRAll2All/g; s/\bLcalAllGather/TileXRAllGather/g' $files
perl -0pi -e 's/LCAL_MAX_RANK_SIZE/TILEXR_MAX_RANK_SIZE/g; s/Attr_Section_Lcal/Attr_Section_TileXR/g' $files
```

Expected: command exits with status 0.

- [ ] **Step 3: Add AllReduce includes and macro to `lccl_op.h`**

In `src/collectives/kernels/lccl_op.h`, add these includes after the existing AllGather includes:

```cpp
#include "allreduce_one_shot.h"
#include "allreduce_two_shot.h"
#include "allreduce_big_data.h"
#include "91093/allreduce_big_data_sio.h"
#include "91093/allreduce_hierarchy_double_ring.h"
#include "reduce_scatter.h"
#include "91093/reduce_scatter_big_data_91093_4step.h"
#include "91093/reduce_scatter_hierarchy_double_ring.h"
```

Add these kernel includes before `lcal_all2all_transpose.cce`:

```cpp
#include "kernels/lcal_allreduce_2npu_read.cce"
#include "kernels/lcal_allreduce_2npu_write.cce"
#include "kernels/lcal_allreduce_2npu_big_write.cce"
#include "kernels/lcal_allreduce_two_shot.cce"
#include "kernels/lcal_allreduce_big_data.cce"
#include "kernels/lcal_allreduce_two_shot_910B2C.cce"
#include "kernels/lcal_allreduce_big_data_910B2C.cce"
#include "kernels/lcal_allreduce_deterministic.cce"
#include "kernels/lcal_allreduce_deterministic_big_data.cce"
#include "kernels/lcal_reduce_scatter_big_data_write.cce"
#include "kernels/lcal_reduce_scatter_write.cce"
#include "kernels/lcal_reduce_scatter.cce"
#include "kernels/lcal_reduce_scatter_big_data.cce"
#include "kernels/lcal_broadcast_write.cce"
#include "kernels/lcal_broadcast_big_data.cce"
```

Copy the adapted `LCCL_ALL_REDUCE_FUNC_AUTO_DEF`, `LCCL_REDUCE_SCATTER_FUNC_AUTO_DEF`, and `LCCL_BROADCAST_FUNC_AUTO_DEF` macro bodies from `reference/ascend-transformer-boost/src/kernels/lcal/src/ascendc_kernels/lccl_op.h`, then apply the same name substitutions used by existing TileXR macros:

- wrapper names use `TileXRAllReduce_`, `TileXRReduceScatter_`, and `TileXRBroadcast`
- `shareAddrs` arrays use `TILEXR_MAX_RANK_SIZE`
- kernel calls use `TileXR...` function names

- [ ] **Step 4: Instantiate kernels**

In `src/collectives/kernels/tilexr_lccl_op.cpp`, add after the AllGather instantiation:

```cpp
LCCL_TYPE_AIV_FUNC(LCCL_ALL_REDUCE_FUNC_AUTO_DEF);
LCCL_TYPE_AIV_FUNC(LCCL_REDUCE_SCATTER_FUNC_AUTO_DEF);
LCCL_BROADCAST_FUNC_AUTO_DEF();
```

- [ ] **Step 5: Check for stale reference names**

Run:

```bash
rg -n 'reference/ascend-transformer-boost|LCAL_MAX_RANK_SIZE|Attr_Section_Lcal|LcalAllReduce|LcalReduceScatter|LcalBroadcast' src/collectives/kernels
```

Expected: no output.

- [ ] **Step 6: Run kernel ownership test**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collectives_kernel_ownership -j$(nproc)
ctest --test-dir build -R test_tilexr_collectives_kernel_ownership --output-on-failure
```

Expected: fails only on host registration checks until Task 9 is complete. File ownership checks pass.

- [ ] **Step 7: Commit imported kernels**

```bash
git add src/collectives/kernels tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp
git commit -m "feat: import standalone collective kernels"
```

---

### Task 9: Kernel Registration and CCE Build

**Files:**
- Modify: `src/collectives/host/collective_kernel.cpp`
- Modify: `src/collectives/kernels/CMakeLists.txt`
- Test: `test_tilexr_collectives_kernel_ownership`
- Build target: `tilexr-collectives`

- [ ] **Step 1: Register new collective types**

In `src/collectives/host/collective_kernel.cpp`, extend `kCollectiveTypes` to:

```cpp
const TileXR::TileXRType kCollectiveTypes[] = {
    TileXR::TileXRType::ALL_GATHER,
    TileXR::TileXRType::ALL2ALL,
    TileXR::TileXRType::ALL_REDUCE,
    TileXR::TileXRType::REDUCE_SCATTER,
    TileXR::TileXRType::BROADCAST,
};
```

- [ ] **Step 2: Handle Broadcast kernel name**

Replace `KernelName` with:

```cpp
std::string KernelName(TileXR::TileXRType type, const DataTypeRegistration &dataType)
{
    if (type == TileXR::TileXRType::BROADCAST) {
        return "TileXRBroadcast";
    }
    return TileXR::TILEXR_TYPE2NAME.at(type) + "_" + dataType.kernelTypeName;
}
```

If CCE compilation shows typed Broadcast symbols instead, revert this function to typed naming and update the Broadcast macro in `lccl_op.h` to produce typed names. The final state must make `rtFunctionRegister` names match symbols in `tilexr_collectives_op.o`.

- [ ] **Step 3: Build collectives CCE binary**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target tilexr-collectives -j$(nproc)
```

Expected: PASS. If the custom link or truncate step reports that the output exceeds `TILEXR_COLLECTIVES_1OP_BIN_SIZE`, increase the value in `src/collectives/kernels/CMakeLists.txt` from `5242880` to `10485760`, rerun the build, and commit that explicit size change.

- [ ] **Step 4: Verify symbols in CCE object**

Run:

```bash
nm -a build/src/collectives/kernels/tilexr_collectives_op.o | rg 'TileXR(AllReduce_|ReduceScatter_|Broadcast)'
```

Expected: output includes `TileXRAllReduce_`, `TileXRReduceScatter_`, and `TileXRBroadcast` symbols.

- [ ] **Step 5: Run ownership test**

Run:

```bash
ctest --test-dir build -R test_tilexr_collectives_kernel_ownership --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Commit registration and CCE build**

```bash
git add src/collectives/host/collective_kernel.cpp src/collectives/kernels/CMakeLists.txt
git commit -m "feat: register standalone collective kernels"
```

---

### Task 10: Integration Correctness Tests

**Files:**
- Modify: `tests/collectives/integration/test_tilexr_collectives_correctness.cpp`
- Modify: `tests/collectives/run_collectives_correctness.sh`
- Test binary: `test_tilexr_collectives_correctness`

- [ ] **Step 1: Extend op enum and parser**

In `tests/collectives/integration/test_tilexr_collectives_correctness.cpp`, replace `CollectiveOp` with:

```cpp
enum class CollectiveOp {
    ALLGATHER,
    ALLTOALL,
    ALLREDUCE,
    REDUCESCATTER,
    BROADCAST,
    BOTH,
};
```

Update `PrintUsage` op text to:

```cpp
              << " --rank-size N --rank R --count C --first-npu D [--op allgather|alltoall|allreduce|reducescatter|broadcast|both]\n"
```

Add parser branches in `ParseOp`:

```cpp
    if (value == "allreduce") {
        op = CollectiveOp::ALLREDUCE;
        return true;
    }
    if (value == "reducescatter") {
        op = CollectiveOp::REDUCESCATTER;
        return true;
    }
    if (value == "broadcast") {
        op = CollectiveOp::BROADCAST;
        return true;
    }
```

Update the error text to:

```cpp
                std::cerr << "ERROR: --op must be allgather, alltoall, allreduce, reducescatter, broadcast, or both" << std::endl;
```

- [ ] **Step 2: Add expected-value helpers**

Add after the existing `using` declarations:

```cpp
int32_t ExpectedAllReduceSum(int rankSize, int64_t index)
{
    int64_t sum = 0;
    for (int rank = 0; rank < rankSize; ++rank) {
        sum += ExpectedAllGatherValue(rankSize, rank, index);
    }
    return static_cast<int32_t>(sum);
}

int32_t ExpectedReduceScatterSum(int rankSize, int rank, int64_t recvCount, int64_t index)
{
    const int64_t globalIndex = static_cast<int64_t>(rank) * recvCount + index;
    int64_t sum = 0;
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        sum += ExpectedAllGatherValue(rankSize, srcRank, globalIndex);
    }
    return static_cast<int32_t>(sum);
}
```

- [ ] **Step 3: Add `RunAllReduce`**

Add after `RunAllToAll`:

```cpp
bool RunAllReduce(const Options &options, TileXRCommPtr comm, aclrtStream stream)
{
    const int64_t count = options.count;
    std::vector<int32_t> hostSend(static_cast<size_t>(count));
    std::vector<int32_t> hostRecv(static_cast<size_t>(count), -1);
    for (int64_t i = 0; i < count; ++i) {
        hostSend[static_cast<size_t>(i)] = ExpectedAllGatherValue(options.rankSize, options.rank, i);
    }

    int32_t *devSend = nullptr;
    int32_t *devRecv = nullptr;
    const size_t bytes = hostSend.size() * sizeof(int32_t);
    bool ok = AllocDeviceInt32(options.rank, "allreduce send", count, &devSend) &&
        AllocDeviceInt32(options.rank, "allreduce recv", count, &devRecv) &&
        CopyHostToDevice(options.rank, devSend, bytes, hostSend.data(), bytes, "allreduce send") &&
        CopyHostToDevice(options.rank, devRecv, bytes, hostRecv.data(), bytes, "allreduce recv") &&
        CheckTileXR(options.rank, "TileXRAllReduce",
            TileXRAllReduce(devSend, devRecv, count, TileXR::TILEXR_DATA_TYPE_INT32,
                            TileXR::TILEXR_REDUCE_SUM, comm, stream)) &&
        CheckAcl(options.rank, "aclrtSynchronizeStream allreduce", aclrtSynchronizeStream(stream)) &&
        CopyDeviceToHost(options.rank, hostRecv.data(), bytes, devRecv, bytes, "allreduce result");

    if (ok) {
        for (int64_t i = 0; i < count; ++i) {
            const int32_t expected = ExpectedAllReduceSum(options.rankSize, i);
            if (hostRecv[static_cast<size_t>(i)] != expected) {
                std::cerr << "[rank " << options.rank << "] allreduce mismatch index=" << i
                          << " got=" << hostRecv[static_cast<size_t>(i)]
                          << " expected=" << expected << std::endl;
                ok = false;
                break;
            }
        }
    }
    FreeDevice(devSend);
    FreeDevice(devRecv);
    return ok;
}
```

- [ ] **Step 4: Add `RunReduceScatter` and `RunBroadcast`**

Add after `RunAllReduce`:

```cpp
bool RunReduceScatter(const Options &options, TileXRCommPtr comm, aclrtStream stream)
{
    const int64_t recvCount = options.count;
    const int64_t sendCount = recvCount * options.rankSize;
    std::vector<int32_t> hostSend(static_cast<size_t>(sendCount));
    std::vector<int32_t> hostRecv(static_cast<size_t>(recvCount), -1);
    for (int64_t i = 0; i < sendCount; ++i) {
        hostSend[static_cast<size_t>(i)] = ExpectedAllGatherValue(options.rankSize, options.rank, i);
    }

    int32_t *devSend = nullptr;
    int32_t *devRecv = nullptr;
    const size_t sendBytes = hostSend.size() * sizeof(int32_t);
    const size_t recvBytes = hostRecv.size() * sizeof(int32_t);
    bool ok = AllocDeviceInt32(options.rank, "reducescatter send", sendCount, &devSend) &&
        AllocDeviceInt32(options.rank, "reducescatter recv", recvCount, &devRecv) &&
        CopyHostToDevice(options.rank, devSend, sendBytes, hostSend.data(), sendBytes, "reducescatter send") &&
        CopyHostToDevice(options.rank, devRecv, recvBytes, hostRecv.data(), recvBytes, "reducescatter recv") &&
        CheckTileXR(options.rank, "TileXRReduceScatter",
            TileXRReduceScatter(devSend, devRecv, recvCount, TileXR::TILEXR_DATA_TYPE_INT32,
                                TileXR::TILEXR_REDUCE_SUM, comm, stream)) &&
        CheckAcl(options.rank, "aclrtSynchronizeStream reducescatter", aclrtSynchronizeStream(stream)) &&
        CopyDeviceToHost(options.rank, hostRecv.data(), recvBytes, devRecv, recvBytes, "reducescatter result");

    if (ok) {
        for (int64_t i = 0; i < recvCount; ++i) {
            const int32_t expected = ExpectedReduceScatterSum(options.rankSize, options.rank, recvCount, i);
            if (hostRecv[static_cast<size_t>(i)] != expected) {
                std::cerr << "[rank " << options.rank << "] reducescatter mismatch index=" << i
                          << " got=" << hostRecv[static_cast<size_t>(i)]
                          << " expected=" << expected << std::endl;
                ok = false;
                break;
            }
        }
    }
    FreeDevice(devSend);
    FreeDevice(devRecv);
    return ok;
}

bool RunBroadcast(const Options &options, TileXRCommPtr comm, aclrtStream stream)
{
    const int64_t count = options.count;
    const int root = 0;
    std::vector<int32_t> hostData(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        hostData[static_cast<size_t>(i)] = options.rank == root ?
            ExpectedAllGatherValue(options.rankSize, root, i) : -1;
    }

    int32_t *devBuf = nullptr;
    const size_t bytes = hostData.size() * sizeof(int32_t);
    bool ok = AllocDeviceInt32(options.rank, "broadcast buffer", count, &devBuf) &&
        CopyHostToDevice(options.rank, devBuf, bytes, hostData.data(), bytes, "broadcast buffer") &&
        CheckTileXR(options.rank, "TileXRBroadcast",
            TileXRBroadcast(devBuf, count, TileXR::TILEXR_DATA_TYPE_INT32, root, comm, stream)) &&
        CheckAcl(options.rank, "aclrtSynchronizeStream broadcast", aclrtSynchronizeStream(stream)) &&
        CopyDeviceToHost(options.rank, hostData.data(), bytes, devBuf, bytes, "broadcast result");

    if (ok) {
        for (int64_t i = 0; i < count; ++i) {
            const int32_t expected = ExpectedAllGatherValue(options.rankSize, root, i);
            if (hostData[static_cast<size_t>(i)] != expected) {
                std::cerr << "[rank " << options.rank << "] broadcast mismatch index=" << i
                          << " got=" << hostData[static_cast<size_t>(i)]
                          << " expected=" << expected << std::endl;
                ok = false;
                break;
            }
        }
    }
    FreeDevice(devBuf);
    return ok;
}
```

- [ ] **Step 5: Dispatch new ops in `main`**

Find the existing boolean expression that runs AllGather and AllToAll. Replace it with:

```cpp
        ok = (options.op == CollectiveOp::ALLGATHER || options.op == CollectiveOp::BOTH ?
              RunAllGather(options, comm, stream) : true) &&
             (options.op == CollectiveOp::ALLTOALL || options.op == CollectiveOp::BOTH ?
              RunAllToAll(options, comm, stream) : true) &&
             (options.op == CollectiveOp::ALLREDUCE ?
              RunAllReduce(options, comm, stream) : true) &&
             (options.op == CollectiveOp::REDUCESCATTER ?
              RunReduceScatter(options, comm, stream) : true) &&
             (options.op == CollectiveOp::BROADCAST ?
              RunBroadcast(options, comm, stream) : true);
```

- [ ] **Step 6: Extend correctness script accepted ops**

In `tests/collectives/run_collectives_correctness.sh`, update the usage and case validation so accepted ops are:

```bash
allgather|alltoall|allreduce|reducescatter|broadcast|both
```

The command line should still pass `--op "${op}"` to the binary.

- [ ] **Step 7: Build correctness binary**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collectives_correctness -j$(nproc)
```

Expected: PASS.

- [ ] **Step 8: Commit integration test support**

```bash
git add tests/collectives/integration/test_tilexr_collectives_correctness.cpp tests/collectives/run_collectives_correctness.sh
git commit -m "test: add standalone collective correctness cases"
```

---

### Task 11: Perf Tool Support

**Files:**
- Modify: `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`
- Modify: `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`
- Modify: `tests/collectives/README.md`
- Test: `test_tilexr_collectives_tools_sources`
- Build target: `tilexr_collective_perf`

- [ ] **Step 1: Add source checks for new CLI ops**

In `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`, update checks that mention `--op allgather|alltoall` to require:

```cpp
CheckContains(path, text, "--op allgather|alltoall|allreduce|reducescatter|broadcast");
CheckContains(path, text, "TileXRAllReduce");
CheckContains(path, text, "TileXRReduceScatter");
CheckContains(path, text, "TileXRBroadcast");
CheckContains(path, text, "reducescatter: count * rank_size * dtype_size");
CheckContains(path, text, "broadcast: count * dtype_size");
```

- [ ] **Step 2: Run source test to verify failure**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target test_tilexr_collectives_tools_sources -j$(nproc)
ctest --test-dir build -R test_tilexr_collectives_tools_sources --output-on-failure
```

Expected: FAIL because perf tool and README do not mention the new ops.

- [ ] **Step 3: Extend perf op enum and usage**

In `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`, replace `CollectiveOp` with:

```cpp
enum class CollectiveOp {
    ALLGATHER,
    ALLTOALL,
    ALLREDUCE,
    REDUCESCATTER,
    BROADCAST,
};
```

Update `PrintUsage`:

```cpp
        << "  --op allgather|alltoall|allreduce|reducescatter|broadcast\n"
```

Update parse logic for `--op` to accept:

```cpp
            if (op == "allgather") {
                options.op = CollectiveOp::ALLGATHER;
            } else if (op == "alltoall") {
                options.op = CollectiveOp::ALLTOALL;
            } else if (op == "allreduce") {
                options.op = CollectiveOp::ALLREDUCE;
            } else if (op == "reducescatter") {
                options.op = CollectiveOp::REDUCESCATTER;
            } else if (op == "broadcast") {
                options.op = CollectiveOp::BROADCAST;
            } else {
                std::cerr << "ERROR: --op must be allgather, alltoall, allreduce, reducescatter, or broadcast" << std::endl;
                return false;
            }
```

- [ ] **Step 4: Add message-size and allocation semantics**

In `tilexr_collective_perf.cpp`, update the helper that converts bytes to counts so:

- AllGather, AllReduce, and Broadcast use `count = bytes / dtype.bytes`.
- AllToAll and ReduceScatter use `count = bytes / (rankSize * dtype.bytes)`.

Use this exact expression at the point where per-iteration `count` is derived:

```cpp
const int64_t divisor = (options.op == CollectiveOp::ALLTOALL ||
                         options.op == CollectiveOp::REDUCESCATTER) ?
    static_cast<int64_t>(options.rankSize) * static_cast<int64_t>(options.dtype.bytes) :
    static_cast<int64_t>(options.dtype.bytes);
```

- [ ] **Step 5: Launch new ops**

In the function that currently calls `TileXRAllGather` or `TileXRAllToAll`, replace the final dispatch with:

```cpp
    if (options.op == CollectiveOp::ALLGATHER) {
        return CheckTileXR(options.rank, "TileXRAllGather",
            TileXRAllGather(sendBuf, recvBuf, count, options.dtype.type, comm, stream));
    }
    if (options.op == CollectiveOp::ALLTOALL) {
        return CheckTileXR(options.rank, "TileXRAllToAll",
            TileXRAllToAll(sendBuf, recvBuf, count, options.dtype.type, comm, stream));
    }
    if (options.op == CollectiveOp::ALLREDUCE) {
        return CheckTileXR(options.rank, "TileXRAllReduce",
            TileXRAllReduce(sendBuf, recvBuf, count, options.dtype.type,
                            TileXR::TILEXR_REDUCE_SUM, comm, stream));
    }
    if (options.op == CollectiveOp::REDUCESCATTER) {
        return CheckTileXR(options.rank, "TileXRReduceScatter",
            TileXRReduceScatter(sendBuf, recvBuf, count, options.dtype.type,
                                TileXR::TILEXR_REDUCE_SUM, comm, stream));
    }
    return CheckTileXR(options.rank, "TileXRBroadcast",
        TileXRBroadcast(sendBuf, count, options.dtype.type, 0, comm, stream));
```

For Broadcast, pass the same pointer for input and output allocation in the surrounding code path by setting `recvBuf = sendBuf` before launch when `options.op == CollectiveOp::BROADCAST`.

- [ ] **Step 6: Add INT32 check logic for new ops**

Extend the existing check logic:

- AllReduce expects sum across ranks at each index.
- ReduceScatter expects sum across ranks for the segment owned by `rank`.
- Broadcast expects root 0 values on every rank.

Use the same formulas from Task 10:

```cpp
ExpectedAllReduceSum(options.rankSize, index)
ExpectedReduceScatterSum(options.rankSize, options.rank, count, index)
ExpectedAllGatherValue(options.rankSize, 0, index)
```

- [ ] **Step 7: Update op name output**

Where `opName` is computed, replace the binary expression with:

```cpp
        std::string opName;
        switch (options.op) {
            case CollectiveOp::ALLGATHER:
                opName = "allgather";
                break;
            case CollectiveOp::ALLTOALL:
                opName = "alltoall";
                break;
            case CollectiveOp::ALLREDUCE:
                opName = "allreduce";
                break;
            case CollectiveOp::REDUCESCATTER:
                opName = "reducescatter";
                break;
            case CollectiveOp::BROADCAST:
                opName = "broadcast";
                break;
        }
```

- [ ] **Step 8: Update README**

In `tests/collectives/README.md`, update the operation list and message-size paragraph to include:

```text
--op allgather|alltoall|allreduce|reducescatter|broadcast
```

and:

```text
Message-size semantics: allgather/allreduce/broadcast: count * dtype_size; alltoall/reducescatter: count * rank_size * dtype_size.
```

- [ ] **Step 9: Build and run source test**

Run:

```bash
source scripts/common_env.sh
cmake --build build --target tilexr_collective_perf test_tilexr_collectives_tools_sources -j$(nproc)
ctest --test-dir build -R test_tilexr_collectives_tools_sources --output-on-failure
```

Expected: PASS.

- [ ] **Step 10: Commit perf support**

```bash
git add tests/collectives/tilexr-tests/tilexr_collective_perf.cpp tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp tests/collectives/README.md
git commit -m "test: add standalone collective perf tooling"
```

---

### Task 12: Local Build and Test Verification

**Files:**
- No source edits expected.
- Verification only.

- [ ] **Step 1: Clean configure collectives build**

Run:

```bash
source scripts/common_env.sh
cmake -S . -B build -DTILEXR_BUILD_COLLECTIVES=ON -DCMAKE_INSTALL_PREFIX=install
```

Expected: configuration completes with no fatal errors.

- [ ] **Step 2: Build all local targets**

Run:

```bash
cmake --build build -j$(nproc)
```

Expected: build completes successfully, including `tilexr-collectives` and collectives test targets.

- [ ] **Step 3: Run local tests**

Run:

```bash
ctest --test-dir build --output-on-failure
```

Expected: PASS for all registered tests. Hardware-only integration binaries are built but not registered as CTest tests.

- [ ] **Step 4: Check exported host symbols**

Run:

```bash
nm -D build/src/collectives/libtilexr-collectives.so | rg 'TileXR(AllGather|AllToAll|AllReduce|ReduceScatter|Broadcast)$'
```

Expected: output includes all five API symbols:

```text
TileXRAllGather
TileXRAllToAll
TileXRAllReduce
TileXRReduceScatter
TileXRBroadcast
```

- [ ] **Step 5: Check kernel registration source**

Run:

```bash
rg -n 'ALL_REDUCE|REDUCE_SCATTER|BROADCAST' src/collectives/host/collective_kernel.cpp src/collectives/kernels/tilexr_lccl_op.cpp src/collectives/kernels/lccl_op.h
```

Expected: output shows registration and wrapper generation for all three new collective types.

- [ ] **Step 6: Commit verification note if a size-only fix was needed**

If Task 9 required changing `TILEXR_COLLECTIVES_1OP_BIN_SIZE`, commit that change now:

```bash
git add src/collectives/kernels/CMakeLists.txt
git commit -m "build: resize collectives kernel binary padding"
```

Expected: no commit is created if no size-only fix was needed.

---

### Task 13: Remote Hardware Verification on `ssh blue`

**Files:**
- No source edits expected.
- Verification only.

- [ ] **Step 1: Check remote access and environment**

Run:

```bash
ssh blue 'cd /home/TileXR && source scripts/common_env.sh && npu-smi info'
```

Expected: command exits with status 0 and prints visible Ascend NPU devices.

- [ ] **Step 2: Sync branch to `blue`**

If `/home/TileXR` on `blue` is the same shared checkout, run:

```bash
ssh blue 'cd /home/TileXR && git status --short && git rev-parse --short HEAD'
```

Expected: the commit matches the local branch. If it does not match, push/pull through the project remote or use the repository's established deployment method before continuing.

- [ ] **Step 3: Build on `blue`**

Run:

```bash
ssh blue 'cd /home/TileXR && source scripts/common_env.sh && cmake -S . -B build -DTILEXR_BUILD_COLLECTIVES=ON -DCMAKE_INSTALL_PREFIX=install && cmake --build build -j$(nproc)'
```

Expected: build completes successfully on the Ascend server.

- [ ] **Step 4: Run correctness for existing ops**

Run:

```bash
ssh blue 'cd /home/TileXR && source scripts/common_env.sh && bash tests/collectives/run_collectives_correctness.sh 2 0 build/tests/collectives --op allgather --count 16'
ssh blue 'cd /home/TileXR && source scripts/common_env.sh && bash tests/collectives/run_collectives_correctness.sh 2 0 build/tests/collectives --op alltoall --count 16'
```

Expected: both commands pass.

- [ ] **Step 5: Run correctness for new ops**

Run:

```bash
ssh blue 'cd /home/TileXR && source scripts/common_env.sh && bash tests/collectives/run_collectives_correctness.sh 2 0 build/tests/collectives --op allreduce --count 16'
ssh blue 'cd /home/TileXR && source scripts/common_env.sh && bash tests/collectives/run_collectives_correctness.sh 2 0 build/tests/collectives --op reducescatter --count 16'
ssh blue 'cd /home/TileXR && source scripts/common_env.sh && bash tests/collectives/run_collectives_correctness.sh 2 0 build/tests/collectives --op broadcast --count 16'
```

Expected: all three commands pass.

- [ ] **Step 6: Run perf smoke for new ops**

Run:

```bash
ssh blue 'cd /home/TileXR && source scripts/common_env.sh && bash tests/collectives/run_collective_perf.sh 2 0 build/tests/collectives --op allreduce --min-bytes 64 --max-bytes 64 --iters 3 --warmup-iters 1 --datatype int32 --check 1'
ssh blue 'cd /home/TileXR && source scripts/common_env.sh && bash tests/collectives/run_collective_perf.sh 2 0 build/tests/collectives --op reducescatter --min-bytes 64 --max-bytes 64 --iters 3 --warmup-iters 1 --datatype int32 --check 1'
ssh blue 'cd /home/TileXR && source scripts/common_env.sh && bash tests/collectives/run_collective_perf.sh 2 0 build/tests/collectives --op broadcast --min-bytes 64 --max-bytes 64 --iters 3 --warmup-iters 1 --datatype int32 --check 1'
```

Expected: all three commands pass and report latency/bandwidth rows.

- [ ] **Step 7: Record verification result in final implementation report**

The final implementation report must include:

- local build result
- local CTest result
- `ssh blue` environment check result
- correctness result for all five ops
- perf smoke result for the three new ops

If `ssh blue` is unreachable or NPU devices are unavailable, state that the implementation is not fully verified.

---

## Plan Self-Review

Spec coverage:

- Public APIs and SUM-only reduce-op support are covered by Tasks 1, 2, 5, and 6.
- Host validation and blockDim helpers are covered by Tasks 3, 4, 5, and 6.
- Kernel imports, wrapper generation, and registration are covered by Tasks 7, 8, and 9.
- Integration correctness and perf smoke coverage are covered by Tasks 10 and 11.
- Local and `ssh blue` verification are covered by Tasks 12 and 13.

Type consistency:

- Public signatures use `TileXR::TileXRDataType`, `TileXR::TileXRReduceOp`, `TileXRCommPtr`, and `aclrtStream`.
- Reduction validation supports only `TileXR::TILEXR_REDUCE_SUM`.
- `ReduceScatter` uses `recvCount` as output count and `recvCount * rankSize` as input count.
- Broadcast uses in-place `buf` and root rank 0 in tests/perf smoke.
