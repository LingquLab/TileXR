# UDMA Grouped Fullmesh AllToAll Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone 32-core grouped-fullmesh UDMA AllToAll kernel with two complete registered receive planes, explicit same-QP payload signals, and no device-wide or cross-rank barriers between invocations.

**Architecture:** Peers are scheduled in symmetric circular groups of at most 16. Cores 0-15 send one peer per lane with `UDMAPutSignalNbiOnQp` followed by immediate same-QP quiet; cores 16-31 independently wait for one source per lane and copy its source-rank receive slot into output. Two full receive planes alternate by invocation ID, so ordered stream execution needs neither ACK nor Host rank barriers.

**Tech Stack:** C++14 Host/layout code, Ascend C/Bisheng device code, TileXR UDMA APIs, Python 3 trace conversion, CMake, physical Ascend950 2x8 validation with CANN `/home/pkg/b101/cann`.

## Global Constraints

- Keep the existing 35-core big-data fullmesh kernel behavior unchanged.
- Add grouped fullmesh as dedicated demo `testType=8`; do not add mode branches to the old big-data worker loop.
- Support `8 <= rankSize <= 128` and `rankSize % 8 == 0` only, matching the
  current `TILEXR_MAX_RANK_SIZE` and `CommArgs` peer-array limit.
- Use exactly 32 AIV blocks: 16 send workers and 16 receive workers.
- Use a fixed group width of 16, with eight forward and eight backward circular distances.
- Allocate `payloadPlane[2][rankSize][elementsPerPeer]` and `signalPlane[2][rankSize][128 bytes]` in one registered region.
- Reject any complete registered layout above 1 GiB per rank.
- Use one source-rank slot per global rank; never address payload by local worker lane.
- Payload plus ready signal and immediate quiet must use the same explicit max-weight QP.
- Do not use `SyncAll()` in the grouped kernel and do not insert Host rank barriers between invocations.
- Submit all invocations in identical order on one stream per rank; concurrent grouped collectives on multiple streams are out of scope.
- Commit locally, create and verify a Git bundle, then upload the bundle before every remote build or physical validation.
- Build and run against `/home/pkg/b101/cann`, with runtime driver libraries ahead of any CANN `devlib` stubs.
- Set `TILEXR_IPC_PID_MODE=pid` for physical multi-process runs.

---

### Task 1: Group Schedule, Token, And Registered Layout

**Files:**
- Create: `tests/udma/demo/tilexr_udma_alltoall_group_layout.h`
- Create: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`
- Modify: `tests/udma/CMakeLists.txt`

**Interfaces:**
- Produces: `TileXR::Demo::AllToAllGroupPeer(int rank, int rankSize, uint32_t group, uint32_t lane) -> int32_t`.
- Produces: `TileXR::Demo::AllToAllGroupCount(int rankSize) -> uint32_t`.
- Produces: `TileXR::Demo::AllToAllGroupToken(uint32_t invocationId, uint32_t group, uint32_t pass) -> uint64_t`.
- Produces: `TileXR::Demo::PlanAllToAllGroup(int rankSize, int32_t elementsPerPeer, int32_t chunkElements) -> AllToAllGroupPlan`.
- Produces: layout offsets consumed by Host allocation and the device launch arguments in Tasks 2 and 3.

- [ ] **Step 1: Write the failing group-layout unit test**

Create a focused test executable with checks equivalent to:

```cpp
#include <cstdint>
#include <set>
#include <vector>
#include "demo/tilexr_udma_alltoall_group_layout.h"

static void CheckSchedule(int rankSize)
{
    for (int rank = 0; rank < rankSize; ++rank) {
        std::set<int> peers;
        for (uint32_t group = 0; group < TileXR::Demo::AllToAllGroupCount(rankSize); ++group) {
            for (uint32_t lane = 0; lane < 16; ++lane) {
                int peer = TileXR::Demo::AllToAllGroupPeer(rank, rankSize, group, lane);
                if (peer < 0) continue;
                CHECK_NE(peer, rank);
                CHECK_EQ(peers.insert(peer).second, true);
                bool symmetric = false;
                for (uint32_t remoteLane = 0; remoteLane < 16; ++remoteLane) {
                    symmetric |= TileXR::Demo::AllToAllGroupPeer(
                        peer, rankSize, group, remoteLane) == rank;
                }
                CHECK_EQ(symmetric, true);
            }
        }
        CHECK_EQ(peers.size(), static_cast<size_t>(rankSize - 1));
    }
}

int main()
{
    for (int rankSize : {8, 16, 24, 32, 40, 64}) CheckSchedule(rankSize);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(64), 4U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 0, 0), 1);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 0, 8), 63);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 3, 7), 32);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 3, 15), -1);
    return 0;
}
```

Add `test_tilexr_udma_alltoall_group_layout` to `tests/udma/CMakeLists.txt`, its include path, `INSTALL_TARGETS`, and the normal install set.

- [ ] **Step 2: Run the test to verify RED**

Commit the failing test only:

```bash
git add tests/udma/CMakeLists.txt tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp
git commit -m "test(udma): require grouped alltoall layout"
git bundle create tmp/grouped-alltoall-layout-red.bundle HEAD
git bundle verify tmp/grouped-alltoall-layout-red.bundle
```

Deploy to the build host and run:

```bash
source /home/pkg/b101/cann/set_env.sh
export ASCEND_HOME_PATH=/home/pkg/b101/cann
cmake -S tests/udma -B tests/udma/build_b101 \
  -DCMAKE_INSTALL_PREFIX=$PWD/tests/udma/install_b101 \
  -DTILEXR_UDMA_DEMO_SOC_TYPE=Ascend950
cmake --build tests/udma/build_b101 -j8
```

Expected: FAIL because `tilexr_udma_alltoall_group_layout.h` does not exist.

- [ ] **Step 3: Implement the host layout API**

Create the header with these constants and data structure:

```cpp
constexpr uint32_t kAllToAllGroupWidth = 16U;
constexpr uint32_t kAllToAllGroupHalfWidth = 8U;
constexpr uint32_t kAllToAllGroupPingPongSlots = 2U;
constexpr uint32_t kAllToAllGroupSignalSlotBytes = 128U;
constexpr uint32_t kAllToAllGroupBlockDim = 32U;
constexpr size_t kAllToAllGroupMaxRegisteredBytes = 1ULL << 30;

struct AllToAllGroupPlan {
    bool valid = false;
    uint32_t groupCount = 0;
    uint32_t passCount = 0;
    int32_t chunkElements = 0;
    size_t bytesPerPeer = 0;
    size_t payloadPlaneBytes = 0;
    size_t payloadOffset[2] = {0, 0};
    size_t signalPlaneBytes = 0;
    size_t signalOffset[2] = {0, 0};
    size_t controlOffset = 0;
    size_t controlBytes = 4096;
    size_t registeredBytes = 0;
};
```

Implement peer mapping without constructing vectors:

```cpp
inline int32_t AllToAllGroupPeer(
    int rank, int rankSize, uint32_t group, uint32_t lane)
{
    if (rankSize < 8 || rankSize > 128 || rankSize % 8 != 0 ||
        rank < 0 || rank >= rankSize || lane >= 16U) {
        return -1;
    }
    const uint32_t index = lane < 8U ? lane : lane - 8U;
    const int32_t distance = static_cast<int32_t>(group * 8U + index + 1U);
    const int32_t diameter = rankSize / 2;
    if (distance > diameter || (lane >= 8U && distance == diameter)) return -1;
    return lane < 8U ? (rank + distance) % rankSize :
        (rank - distance + rankSize) % rankSize;
}
```

Use checked multiplication/addition and 512-byte alignment in
`PlanAllToAllGroup`. Set `valid=true` only when all dimensions are positive,
the rank size is supported, token fields fit, every range is disjoint, and
`registeredBytes <= 1 GiB`.

Build tokens exactly as specified:

```cpp
inline uint64_t AllToAllGroupToken(
    uint32_t invocationId, uint32_t group, uint32_t pass)
{
    const uint64_t invocation = static_cast<uint64_t>(invocationId) + 1ULL;
    const uint64_t slot = static_cast<uint64_t>(invocationId & 1U);
    return (invocation << 32U) | (slot << 31U) |
        (static_cast<uint64_t>(group) << 16U) |
        (static_cast<uint64_t>(pass) + 1ULL);
}
```

- [ ] **Step 4: Extend tests for memory and token boundaries**

Add assertions for:

```cpp
auto plan = TileXR::Demo::PlanAllToAllGroup(16, 2 * 1024 * 1024, 2 * 1024 * 1024);
CHECK_EQ(plan.valid, true);
CHECK_EQ(plan.groupCount, 1U);
CHECK_EQ(plan.passCount, 1U);
CHECK_EQ(plan.payloadPlaneBytes, 128ULL * 1024ULL * 1024ULL);
CHECK_EQ(plan.payloadOffset[1] >= plan.payloadOffset[0] + plan.payloadPlaneBytes, true);
CHECK_EQ(plan.signalOffset[0] >= plan.payloadOffset[1] + plan.payloadPlaneBytes, true);
CHECK_EQ(plan.registeredBytes <= (1ULL << 30), true);
CHECK_NE(TileXR::Demo::AllToAllGroupToken(48, 0, 0),
         TileXR::Demo::AllToAllGroupToken(49, 0, 0));
CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(7, 1024, 1024).valid, false);
CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(18, 1024, 1024).valid, false);
```

Construct the smallest payload above the 1 GiB layout boundary and verify it
is rejected without integer wraparound.

- [ ] **Step 5: Run GREEN verification and commit**

Run:

```bash
cmake --build tests/udma/build_b101 -j8
./tests/udma/build_b101/test_tilexr_udma_alltoall_group_layout
./tests/udma/build_b101/test_tilexr_udma_alltoall_layout
```

Expected: both executables print their pass messages and exit 0.

Commit:

```bash
git add tests/udma/demo/tilexr_udma_alltoall_group_layout.h \
  tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp tests/udma/CMakeLists.txt
git commit -m "feat(udma): add grouped alltoall layout"
```

---

### Task 2: Standalone 32-Core Grouped Kernel

**Files:**
- Create: `tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp`
- Modify: `tests/udma/CMakeLists.txt`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`

**Interfaces:**
- Consumes: Task 1 plan offsets, dimensions, peer mapping semantics, and token format.
- Produces: `launch_tilexr_udma_all_to_all_group(uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output, GM_ADDR registeredMemory, GM_ADDR debug, uint32_t invocationId, int32_t elementsPerPeer, int32_t chunkElements, uint32_t passCount, uint32_t groupCount, uint64_t payloadOffset0, uint64_t payloadOffset1, uint64_t signalOffset0, uint64_t signalOffset1)`.

- [ ] **Step 1: Add failing kernel-structure checks**

Read the new kernel source from the unit test and require these strings:

```cpp
CHECK_CONTAINS(groupKernel, "tilexr_udma_all_to_all_group_kernel");
CHECK_CONTAINS(groupKernel, "TILEXR_ALLTOALL_GROUP_SEND_CORES");
CHECK_CONTAINS(groupKernel, "UDMAPutSignalNbiOnQp<int32_t>");
CHECK_CONTAINS(groupKernel, "UDMAQuietStatusOnQp");
CHECK_CONTAINS(groupKernel, "AllToAllGroupWaitTokenMte");
CHECK_CONTAINS(groupKernel, "observed >= expectedToken");
CHECK_NOT_CONTAINS(groupKernel, "UDMAPutSignalNbi<int32_t>");
CHECK_NOT_CONTAINS(groupKernel, "SyncAll");
```

Also inspect the function slice containing the put and quiet calls and verify
both receive the same local variable `qpIdx`.

- [ ] **Step 2: Run the source test to verify RED**

Run the layout test. Expected: FAIL because the standalone kernel file and
required symbols do not exist.

- [ ] **Step 3: Add the kernel to the Bisheng shared-library build**

Update the custom Bisheng command to compile and link both sources:

```cmake
"${CMAKE_CURRENT_SOURCE_DIR}/demo/tilexr_udma_demo_kernel.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/demo/tilexr_udma_alltoall_group_kernel.cpp"
```

Add the new source and group layout header to `DEPENDS`. Do not move or edit
the existing big-data kernel implementation.

- [ ] **Step 4: Implement device-only helpers**

Implement small helpers in an anonymous namespace:

```cpp
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SEND_CORES = 16U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_BLOCK_DIM = 32U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE = 128U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_RELAY_BYTES = 64U * 1024U;

__aicore__ inline int32_t AllToAllGroupDevicePeer(
    int32_t rank, int32_t rankSize, uint32_t group, uint32_t lane);

__aicore__ inline uint64_t AllToAllGroupDeviceToken(
    uint32_t invocationId, uint32_t group, uint32_t pass);

__aicore__ inline uint32_t AllToAllGroupSelectMaxWeightQp(
    const __gm__ TileXR::CommArgs* args, int32_t peer);

__aicore__ inline void AllToAllGroupCopyMte(
    __gm__ uint8_t* dst, const __gm__ uint8_t* src, uint32_t bytes,
    AscendC::LocalTensor<uint8_t> relayLocal);

__aicore__ inline bool AllToAllGroupWaitTokenMte(
    __gm__ uint64_t* signal, uint64_t expectedToken, uint64_t timeoutCycles,
    AscendC::LocalTensor<uint8_t> relayLocal, uint64_t& observed);
```

QP selection must scan `GetUDMAInfo(args)->qpNum` and choose the largest
`UDMAGetQpWeight(info, peer, qpIdx)`, preserving the lowest QP index on ties.

The wait helper repeatedly copies one 8-byte token GM-to-UB, performs the
required MTE event synchronization, and returns when `observed >= expectedToken`
or `GetSystemCycle() - begin >= timeoutCycles`.

- [ ] **Step 5: Implement send and receive workers**

The send path must use the source rank for the remote slot:

```cpp
const uint32_t slot = invocationId & 1U;
const uint64_t remotePayloadOffset = payloadOffsets[slot] +
    static_cast<uint64_t>(rank) * bytesPerPeer + chunkByteOffset;
const uint64_t remoteSignalOffset = signalOffsets[slot] +
    static_cast<uint64_t>(rank) * TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE;
const uint32_t qpIdx = AllToAllGroupSelectMaxWeightQp(args, peer);
TileXR::UDMAPutSignalNbiOnQp<int32_t>(
    args, peer, qpIdx, localSrc, remotePayloadOffset, chunkBytes,
    remoteSignalOffset, expectedToken);
const uint32_t quietStatus = TileXR::UDMAQuietStatusOnQp(args, peer, qpIdx);
```

The receive path uses the source peer for its local slot:

```cpp
auto signal = reinterpret_cast<__gm__ uint64_t*>(
    registeredMemory + signalOffsets[slot] +
    static_cast<uint64_t>(peer) * TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE);
uint64_t observed = 0;
if (!AllToAllGroupWaitTokenMte(signal, expectedToken, timeoutCycles, relayLocal, observed)) {
    AllToAllGroupRecordError(debug, blockIdx, stage, group, pass, peer, 0U,
        expectedToken, observed);
    return;
}
auto relaySrc = registeredMemory + payloadOffsets[slot] +
    static_cast<uint64_t>(peer) * bytesPerPeer + chunkByteOffset;
auto relayDst = reinterpret_cast<__gm__ uint8_t*>(
    output + static_cast<uint64_t>(peer) * elementsPerPeer + chunkElementOffset);
AllToAllGroupCopyMte(relayDst, relaySrc, chunkBytes, relayLocal);
```

Receive workers shard self-copy before entering peer groups. Each block owns
one `TPipe` and one 64 KiB relay buffer for its full lifetime. Inactive lanes
skip work. There is no `SyncAll()` and no shared mutable progress counter.

- [ ] **Step 6: Add first-error debug recording**

Reserve one fixed debug record per block with fields:

```cpp
struct AllToAllGroupDeviceError {
    uint32_t valid;
    uint32_t stage;
    uint32_t group;
    uint32_t pass;
    int32_t peer;
    uint32_t qpIdx;
    uint32_t quietStatus;
    uint32_t reserved;
    uint64_t expectedToken;
    uint64_t observedToken;
};
```

Only write when `valid == 0`. Quiet failure records immediately; receive wait
timeout records the expected and observed token and returns.

- [ ] **Step 7: Build, run source guards, and commit**

Run:

```bash
cmake --build tests/udma/build_b101 -j8
./tests/udma/build_b101/test_tilexr_udma_alltoall_group_layout
nm -D tests/udma/build_b101/libtilexr_udma_demo_kernel.so | \
  grep launch_tilexr_udma_all_to_all_group
```

Expected: build succeeds, source checks pass, and the launch wrapper is exported.

Commit:

```bash
git add tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp \
  tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp tests/udma/CMakeLists.txt
git commit -m "feat(udma): add grouped fullmesh alltoall kernel"
```

---

### Task 3: Dedicated Host Mode And Ping-Pong Invocation Loop

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`

**Interfaces:**
- Consumes: `PlanAllToAllGroup` and `launch_tilexr_udma_all_to_all_group`.
- Produces: demo `testType=8`, `TILEXR_DEMO_ALLTOALL_GROUP_CHUNK_ELEMENTS`, Host warmup/repeat invocation IDs, full output validation, and grouped debug reporting.

- [ ] **Step 1: Add failing Host source checks**

Require the Host source to contain:

```cpp
CHECK_CONTAINS(demo, "testType == 8");
CHECK_CONTAINS(demo, "PlanAllToAllGroup");
CHECK_CONTAINS(demo, "launch_tilexr_udma_all_to_all_group");
CHECK_CONTAINS(demo, "grouped alltoall registeredBytes=");
CHECK_CONTAINS(demo, "grouped alltoall warmup=");
CHECK_NOT_CONTAINS(groupModeSlice, "DemoBarrierAll");
```

The group-mode slice begins at its `if (testType == 8)` allocation branch and
ends before the existing `testType == 7` branch.

- [ ] **Step 2: Run RED verification**

Run the group-layout test. Expected: FAIL because `testType=8` plumbing is absent.

- [ ] **Step 3: Add plan validation and allocation**

Treat `testType=8` as strict AllToAll UDMA with output and no IPC fallback.
Build the plan from:

```cpp
const int32_t requestedChunkElements = std::max(
    1, GetEnvInt("TILEXR_DEMO_ALLTOALL_GROUP_CHUNK_ELEMENTS", elementsPerRank));
const auto groupPlan = TileXR::Demo::PlanAllToAllGroup(
    rankSize, elementsPerRank, requestedChunkElements);
```

Reject invalid plans before allocation. Allocate normal `groupInput` and
`groupOutput`, allocate `groupPlan.registeredBytes` with `aclrtMalloc`, zero
both signal planes and controls once, and register the complete region with
`TileXRUDMARegister`.

Print exact offsets, plane sizes, group count, pass count, and total bytes.

- [ ] **Step 4: Add warmup and measured invocation launches**

Use one monotonically increasing invocation ID across warmup and measured calls:

```cpp
uint32_t invocationId = 0;
for (int iter = 0; iter < allToAllWarmup; ++iter, ++invocationId) {
    launch_tilexr_udma_all_to_all_group(
        32U, stream, commArgsDev, reinterpret_cast<GM_ADDR>(groupInput),
        reinterpret_cast<GM_ADDR>(groupOutput),
        reinterpret_cast<GM_ADDR>(registeredMemory), reinterpret_cast<GM_ADDR>(debug),
        invocationId, elementsPerRank, groupPlan.chunkElements,
        groupPlan.passCount, groupPlan.groupCount,
        groupPlan.payloadOffset[0], groupPlan.payloadOffset[1],
        groupPlan.signalOffset[0], groupPlan.signalOffset[1]);
}
if (!CheckAcl(rank, "aclrtSynchronizeStream grouped warmup",
        aclrtSynchronizeStream(stream))) {
    return 1;
}

auto begin = std::chrono::steady_clock::now();
for (int iter = 0; iter < allToAllRepeat; ++iter, ++invocationId) {
    launch_tilexr_udma_all_to_all_group(
        32U, stream, commArgsDev, reinterpret_cast<GM_ADDR>(groupInput),
        reinterpret_cast<GM_ADDR>(groupOutput),
        reinterpret_cast<GM_ADDR>(registeredMemory), reinterpret_cast<GM_ADDR>(debug),
        invocationId, elementsPerRank, groupPlan.chunkElements,
        groupPlan.passCount, groupPlan.groupCount,
        groupPlan.payloadOffset[0], groupPlan.payloadOffset[1],
        groupPlan.signalOffset[0], groupPlan.signalOffset[1]);
}
if (!CheckAcl(rank, "aclrtSynchronizeStream grouped measured",
        aclrtSynchronizeStream(stream))) {
    return 1;
}
auto end = std::chrono::steady_clock::now();
```

Do not synchronize or call `DemoBarrierAll` between invocations. One stream
must serialize all launches.

- [ ] **Step 5: Add result and debug validation**

Copy the full output once after measured synchronization and call the existing
`ValidateAllToAllData`. Copy all 32 debug records and reject any record with
`valid != 0`, printing every recorded field.

Ensure cleanup unregisters before freeing the registered region and frees the
normal input/output exactly once on every error path.

- [ ] **Step 6: Build, run Host source tests, and commit**

Run:

```bash
cmake --build tests/udma/build_b101 -j8
./tests/udma/build_b101/test_tilexr_udma_alltoall_group_layout
./tests/udma/build_b101/test_tilexr_udma_alltoall_layout
```

Expected: both tests pass and `tilexr_udma_demo` links against the new wrapper.

Commit:

```bash
git add tests/udma/demo/tilexr_udma_demo.cpp \
  tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp
git commit -m "feat(udma): add grouped alltoall demo mode"
```

---

### Task 4: Compact Grouped-Kernel GM Trace

**Files:**
- Create: `tests/udma/demo/tilexr_udma_alltoall_group_trace.h`
- Create: `tests/udma/demo/tilexr_udma_alltoall_group_trace_to_chrome.py`
- Create: `tests/udma/unit/test_tilexr_udma_alltoall_group_trace_to_chrome.py`
- Modify: `tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp`
- Modify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Modify: `tests/udma/CMakeLists.txt`

**Interfaces:**
- Consumes: invocation, slot, group, pass, lane, peer, and QP from Tasks 2 and 3.
- Produces: optional 8 MiB per-rank raw trace and Chrome Trace JSON phases `kernel`, `self-copy`, `send-put-signal`, `send-quiet`, `receive-wait`, and `receive-copy`.
- Extends: `launch_tilexr_udma_all_to_all_group` with trailing arguments
  `GM_ADDR groupTrace, uint32_t traceIteration`; warmup passes `nullptr`, and
  measured invocation `iter` passes the trace allocation and `iter`.

- [ ] **Step 1: Write failing Python trace tests**

Construct a synthetic binary with one invocation and verify:

```python
events = module.build_chrome_trace([module.read_rank_trace(path)])
names = {event["name"] for event in events["traceEvents"] if event["ph"] == "X"}
self.assertEqual(names, {
    "kernel", "self-copy", "send-put-signal", "send-quiet",
    "receive-wait", "receive-copy",
})
self.assertEqual(events["otherData"]["displayTimeUnit"], "ns")
```

Also test rejection of bad magic, partial spans, out-of-capacity dimensions,
and non-8-MiB files.

- [ ] **Step 2: Run trace tests to verify RED**

Run:

```bash
python3 -m unittest tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
```

Expected: FAIL because the converter does not exist.

- [ ] **Step 3: Implement the trace binary layout and converter**

Use an 8 MiB fixed allocation with:

```text
header
kernelSpan[50][32]
taskSpan[50][32][groupCount][passCount][5]
```

The five task phases exclude `kernel`; receive cores record self-copy in the
group-0/pass-0 self-copy cell reserved for that core. The Host validates the computed required bytes
before enabling trace.

Write Chrome JSON incrementally with `json.dump`; use `displayTimeUnit: "ns"`
or omit the field, never `"us"`. Normalize cycles independently per rank and
invocation.

- [ ] **Step 4: Instrument the kernel without changing synchronization**

Add begin/end cycle recording around exactly these operations:

```text
self MTE copy
UDMAPutSignalNbiOnQp
UDMAQuietStatusOnQp
signal wait loop
receive MTE copy
whole kernel block
```

Trace writes use disjoint offsets per block/group/pass/phase. Do not add
barriers, flags, or shared counters for tracing.

- [ ] **Step 5: Add Host allocation and output**

Enable with:

```text
TILEXR_UDMA_GROUP_TRACE=1
TILEXR_UDMA_GROUP_TRACE_DIR=/tmp/tilexr_group_trace
```

Allocate trace GM separately from registered UDMA memory, initialize it once,
pass `traceIteration=iter` for measured launches, copy it back after stream
synchronization, and write:

```text
tilexr_group_trace_rank_{rank}.bin
```

- [ ] **Step 6: Run all trace tests and commit**

Run:

```bash
python3 -m unittest tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
cmake --build tests/udma/build_b101 -j8
./tests/udma/build_b101/test_tilexr_udma_alltoall_group_layout
```

Expected: all Python tests pass, Bisheng build succeeds, and layout/source
guards remain green.

Commit:

```bash
git add tests/udma/demo/tilexr_udma_alltoall_group_trace.h \
  tests/udma/demo/tilexr_udma_alltoall_group_trace_to_chrome.py \
  tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp \
  tests/udma/demo/tilexr_udma_demo.cpp \
  tests/udma/unit/test_tilexr_udma_alltoall_group_trace_to_chrome.py \
  tests/udma/CMakeLists.txt
git commit -m "feat(udma): trace grouped alltoall pipeline"
```

---

### Task 5: Full Verification, Bundle, And Two-Host Deployment

**Files:**
- Modify only if verification exposes a grouped-mode defect in files from Tasks 1-4.
- Create artifact: `tmp/udma-grouped-fullmesh-alltoall.bundle`

**Interfaces:**
- Consumes: committed grouped layout, kernel, Host mode, and trace converter.
- Produces: one verified complete Git bundle and identical source commits on `141.61.50.31` and `141.61.49.223`.

- [ ] **Step 1: Run the complete local/source verification set**

Run:

```bash
git diff --check
rg -n "SyncAll|UDMAPutSignalNbi<int32_t>|UDMAQuietStatus\(" \
  tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp
python3 -m unittest \
  tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
```

Expected: `git diff --check` is clean; `rg` finds no forbidden grouped-kernel
calls; Python tests pass.

- [ ] **Step 2: Build and test on the b101 build host**

On `141.61.49.223`:

```bash
source /home/pkg/b101/cann/set_env.sh
export ASCEND_HOME_PATH=/home/pkg/b101/cann
export PATH=/home/pkg/b101/cann/bin:$PATH
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$PWD/install
cmake --build build -j8 && cmake --install build
cmake -S tests/udma -B tests/udma/build_b101 \
  -DCMAKE_INSTALL_PREFIX=$PWD/tests/udma/install_b101 \
  -DTILEXR_UDMA_DEMO_SOC_TYPE=Ascend950
cmake --build tests/udma/build_b101 -j8 && cmake --install tests/udma/build_b101
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_group_layout
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_layout
./tests/udma/install_b101/bin/test_tilexr_udma_transport_layout
python3 -m unittest tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
```

Expected: all commands exit 0.

- [ ] **Step 3: Commit any verification fix separately**

If verification required a code correction, rerun Step 1 and Step 2 and commit
only that correction with the focused message
`fix(udma): correct grouped alltoall verification defect`. Do not bundle
uncommitted changes.

- [ ] **Step 4: Create and verify the deployment bundle**

Run locally:

```bash
git status --short
git bundle create tmp/udma-grouped-fullmesh-alltoall.bundle HEAD
git bundle verify tmp/udma-grouped-fullmesh-alltoall.bundle
```

Expected: only known unrelated untracked paths remain; bundle verification
reports a complete history containing current HEAD.

- [ ] **Step 5: Deploy identical sources to both hosts**

Upload the bundle to both hosts, fetch it into
`/home/h30059441/tilexr_grouped_alltoall_b101`, and detach at the exact local
HEAD. Verify `git rev-parse HEAD` matches on both.

`141.61.50.31` currently has no `cmake`. Build on `141.61.49.223`, then copy
these exact artifacts to the same install paths on `141.61.50.31`:

```text
install/lib64/libtile-comm.so
tests/udma/install_b101/lib/libtilexr_udma_demo_kernel.so
tests/udma/install_b101/bin/tilexr_udma_demo
```

Compare SHA-256 hashes on both hosts before running.

---

### Task 6: Physical 2x8 Correctness And Performance Validation

**Files:**
- Create artifacts under: `tmp/grouped_alltoall_b101_2x8/`
- Do not commit raw trace binaries or run logs.

**Interfaces:**
- Consumes: `testType=8`, 32-core kernel, b101 artifacts, Host warmup/repeat, and grouped raw trace.
- Produces: 16-rank correctness evidence, Host timing, iteration-49 Chrome trace, and per-phase summaries.

- [ ] **Step 1: Run a small correctness smoke test**

Use ranks 0-7 on `141.61.50.31` and ranks 8-15 on `141.61.49.223`. Export on
both hosts:

```bash
export ASCEND_HOME_PATH=/home/pkg/b101/cann
export PATH=/home/pkg/b101/cann/bin:$PATH
export LD_LIBRARY_PATH=$PWD/tests/udma/install_b101/lib:$PWD/install/lib64:\
/home/pkg/b101/cann/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/driver:\
/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64
export TILEXR_COMM_ID=141.61.50.31:64720
export TILEXR_DEMO_BARRIER_HOST=141.61.50.31
export TILEXR_IPC_PID_MODE=pid
export TILEXR_UDMA_ROUTE_POLICY=all
export TILEXR_UDMA_QP_NUM=4
export TILEXR_DEMO_ALLTOALL_WARMUP=1
export TILEXR_DEMO_ALLTOALL_REPEAT=2
```

Run each rank with:

```bash
timeout 180s ./tests/udma/install_b101/bin/tilexr_udma_demo \
  16 "$rank" 8 4096 8 0
```

Expected: 16 success lines, no timeout, no quiet error, and full output
validation succeeds on every rank.

- [ ] **Step 2: Run the target 128 MiB per-rank matrix**

Set:

```bash
export TILEXR_DEMO_ALLTOALL_WARMUP=5
export TILEXR_DEMO_ALLTOALL_REPEAT=50
export TILEXR_DEMO_ALLTOALL_GROUP_CHUNK_ELEMENTS=2097152
export TILEXR_UDMA_GROUP_TRACE=1
export TILEXR_UDMA_GROUP_TRACE_DIR=/home/h30059441/grouped_alltoall_b101_2x8
```

Run:

```bash
timeout 300s ./tests/udma/install_b101/bin/tilexr_udma_demo \
  16 "$rank" 8 2097152 8 0
```

This is 8 MiB per peer and 128 MiB input/output per rank. Expected: 16 success
lines and sixteen 8 MiB trace files.

- [ ] **Step 3: Verify ping-pong and absence of barriers**

Parse all traces and verify:

- With warmup 5, measured iteration 48 uses invocation ID 53 and slot 1;
  measured iteration 49 uses invocation ID 54 and slot 0.
- Every rank has 15 active peer send and receive sequences.
- No lane waits for self or the duplicate diameter peer.
- Every send event records an explicit valid QP.
- No device `SyncAll` phase exists.
- All waits complete without timeout.

- [ ] **Step 4: Generate loop-49 Chrome trace and phase summary**

Convert only measured invocation 49 into:

```text
tmp/grouped_alltoall_b101_2x8/grouped_alltoall_loop49_trace.json
```

Report per core and aggregate:

```text
self-copy
send-put-signal
send-quiet
receive-wait
receive-copy
kernel envelope
Host mean/min/max
```

Compute payload bandwidth from 128 MiB per rank and compare the kernel envelope
against the old fullmesh 2x8 baseline. Keep correctness and performance
conclusions separate.

- [ ] **Step 5: Final regression verification**

After physical validation, rerun on the build host:

```bash
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_group_layout
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_layout
./tests/udma/install_b101/bin/test_tilexr_udma_transport_layout
python3 -m unittest tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
git status --short
```

Expected: all tests pass; only ignored/untracked run artifacts remain; no
tracked source differs from the deployed commit.
