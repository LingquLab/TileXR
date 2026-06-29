# UDMA Multi-EID Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the PR 45 multi-EID direct UDMA P2P route capability into the current bigdata-isolated branch without merging unrelated PR changes.

**Architecture:** Keep the current single-route path as default. Add opt-in multi-route route discovery and queue expansion on the host, expose per-QP route weights in the device-visible UDMA layout, and use those weights for P2P kernel slicing.

**Tech Stack:** C++14, Ascend C device headers, ACL runtime memory copy/allocation, TileXR socket exchange, existing UDMA layout unit tests.

## Global Constraints

- Target commit range: `06cd19c42846eb79ab75b5a075b82d9b656c2852..07afc489467e7e9a369eac6679913de208d169b1`.
- Do not merge unrelated PR 45 checker, collectives, documentation, or demo restructuring changes.
- Existing single-route behavior remains the default.
- Multi-route mode is opt-in through `TILEXR_UDMA_ROUTE_POLICY=all`.
- Preserve graceful UDMA fallback on unsupported hardware.
- Do not change public host APIs.

---

## File Structure

- `src/include/tilexr_udma_types.h`: device-visible `UDMAInfo` layout. Add `qpWeightPtr`.
- `src/comm/udma/tilexr_udma_layout.h`: layout builder declarations and route helper declarations.
- `src/comm/udma/tilexr_udma_layout.cpp`: serialize weights, build QP-to-EID vectors, build QP weights, parse explicit EID lists.
- `src/include/tilexr_udma.h`: device helper `UDMAGetQpWeight()`.
- `src/comm/udma/tilexr_udma_transport.h`: transport state for multi-route local/remote EID vectors, expanded QP route vectors, route weights, and per-peer remote memory handles.
- `src/comm/udma/tilexr_udma_transport.cpp`: host route selection, context creation, queue creation, import/export, image refresh, and cleanup.
- `tests/udma/unit/test_tilexr_udma_transport_layout.cpp`: host-only unit coverage for layout and route helper behavior.
- `tests/udma/demo/tilexr_udma_demo_kernel.cpp`: weighted slice computation for P2P perf kernels.

---

### Task 1: Layout and Device Weight Helpers

**Files:**
- Modify: `src/include/tilexr_udma_types.h`
- Modify: `src/comm/udma/tilexr_udma_layout.h`
- Modify: `src/comm/udma/tilexr_udma_layout.cpp`
- Modify: `src/include/tilexr_udma.h`
- Test: `tests/udma/unit/test_tilexr_udma_transport_layout.cpp`

**Interfaces:**
- Consumes: existing `UDMAInfo`, `BuildUDMAInfoImage()`, `UDMAInfo::qpNum`.
- Produces:
  - `UDMAInfo::qpWeightPtr`
  - `int BuildUDMAInfoImage(uintptr_t deviceBase, uint32_t qpNum, const std::vector<UDMAWQCtx>& sq, const std::vector<UDMAWQCtx>& rq, const std::vector<UDMACQCtx>& scq, const std::vector<UDMACQCtx>& rcq, const std::vector<UDMAMemInfo>& mem, const std::vector<uint32_t>& qpWeights, UDMAInfo& info, std::vector<uint8_t>& bytes)`
  - `std::vector<uint32_t> BuildUDMAMultiRouteQpToEid(const std::vector<uint32_t>& routeEids, uint32_t qpsPerRoute)`
  - `std::vector<uint32_t> BuildUDMAMultiRouteQpWeights(const std::vector<uint32_t>& routeEids, const std::map<uint32_t, uint32_t>& routeWeights, uint32_t qpsPerRoute)`
  - `std::vector<uint32_t> SelectExplicitUDMARouteEids(const char* routeList, const std::vector<uint32_t>& candidateEids)`
  - `__aicore__ inline uint32_t UDMAGetQpWeight(__gm__ UDMAInfo* udmaInfo, uint32_t pe, uint32_t qpIdx)`

- [ ] **Step 1: Write failing layout tests**

Add tests to `tests/udma/unit/test_tilexr_udma_transport_layout.cpp`:

```cpp
#include <map>
```

In `TestHostLayoutUsesDeviceRelativePointers()`, add:

```cpp
std::vector<uint32_t> weights = {3, 1};
const int ret = TileXR::BuildUDMAInfoImage(deviceBase, 1, sq, rq, scq, rcq, mem, weights, info, bytes);
CHECK_TRUE(info.qpWeightPtr > info.memPtr);
CHECK_EQ(bytes.size(),
         sizeof(TileXR::UDMAInfo) + 2 * sizeof(TileXR::UDMAWQCtx) +
             2 * sizeof(TileXR::UDMAWQCtx) + 2 * sizeof(TileXR::UDMACQCtx) +
             2 * sizeof(TileXR::UDMACQCtx) + 2 * sizeof(TileXR::UDMAMemInfo) +
             2 * sizeof(uint32_t));
const auto* imageWeights = reinterpret_cast<const uint32_t*>(
    bytes.data() + (info.qpWeightPtr - deviceBase));
CHECK_EQ(imageWeights[0], 3U);
CHECK_EQ(imageWeights[1], 1U);
```

Add helper tests:

```cpp
void TestMultiRouteQpMappingRepeatsEachRoute()
{
    const std::vector<uint32_t> routeEids = {7, 8};
    const std::vector<uint32_t> qpToEid = TileXR::BuildUDMAMultiRouteQpToEid(routeEids, 2);
    CHECK_EQ(qpToEid.size(), static_cast<size_t>(4));
    CHECK_EQ(qpToEid[0], 7U);
    CHECK_EQ(qpToEid[1], 7U);
    CHECK_EQ(qpToEid[2], 8U);
    CHECK_EQ(qpToEid[3], 8U);
}

void TestMultiRouteQpMappingRejectsEmptyInputs()
{
    CHECK_TRUE(TileXR::BuildUDMAMultiRouteQpToEid({}, 1).empty());
    CHECK_TRUE(TileXR::BuildUDMAMultiRouteQpToEid({7}, 0).empty());
}

void TestMultiRouteQpWeightsUseRouteBandwidth()
{
    const std::vector<uint32_t> routeEids = {7, 8};
    const std::map<uint32_t, uint32_t> routeWeights = {{7, 6}, {8, 2}};
    const std::vector<uint32_t> qpWeights = TileXR::BuildUDMAMultiRouteQpWeights(routeEids, routeWeights, 1);
    CHECK_EQ(qpWeights.size(), static_cast<size_t>(2));
    CHECK_EQ(qpWeights[0], 6U);
    CHECK_EQ(qpWeights[1], 2U);
}

void TestExplicitRouteSelectionKeepsRequestedCandidateOrder()
{
    const std::vector<uint32_t> candidates = {7, 8};
    const std::vector<uint32_t> selected = TileXR::SelectExplicitUDMARouteEids("8,7,9,bad", candidates);
    CHECK_EQ(selected.size(), static_cast<size_t>(2));
    CHECK_EQ(selected[0], 8U);
    CHECK_EQ(selected[1], 7U);
}

void TestExplicitRouteSelectionRejectsMissingInputs()
{
    const std::vector<uint32_t> candidates = {7, 8};
    CHECK_TRUE(TileXR::SelectExplicitUDMARouteEids("", candidates).empty());
    CHECK_TRUE(TileXR::SelectExplicitUDMARouteEids("9", candidates).empty());
    CHECK_TRUE(TileXR::SelectExplicitUDMARouteEids("8", {}).empty());
}
```

Call the helper tests from `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_tilexr_udma_transport_layout
```

Expected: compile failure mentioning missing `UDMAInfo::qpWeightPtr`, missing overload, or missing helper declarations.

- [ ] **Step 3: Implement layout and helper declarations**

Add `uint64_t qpWeightPtr;` to `UDMAInfo` in `src/include/tilexr_udma_types.h`.

Add to `src/comm/udma/tilexr_udma_layout.h`:

```cpp
#include <map>

int BuildUDMAInfoImage(
    uintptr_t deviceBase,
    uint32_t qpNum,
    const std::vector<UDMAWQCtx>& sq,
    const std::vector<UDMAWQCtx>& rq,
    const std::vector<UDMACQCtx>& scq,
    const std::vector<UDMACQCtx>& rcq,
    const std::vector<UDMAMemInfo>& mem,
    const std::vector<uint32_t>& qpWeights,
    UDMAInfo& info,
    std::vector<uint8_t>& bytes);

std::vector<uint32_t> BuildUDMAMultiRouteQpToEid(
    const std::vector<uint32_t>& routeEids,
    uint32_t qpsPerRoute);

std::vector<uint32_t> BuildUDMAMultiRouteQpWeights(
    const std::vector<uint32_t>& routeEids,
    const std::map<uint32_t, uint32_t>& routeWeights,
    uint32_t qpsPerRoute);

std::vector<uint32_t> SelectExplicitUDMARouteEids(
    const char* routeList,
    const std::vector<uint32_t>& candidateEids);
```

- [ ] **Step 4: Implement layout serialization and helpers**

In `src/comm/udma/tilexr_udma_layout.cpp`, include:

```cpp
#include <algorithm>
#include <climits>
#include <cstdlib>
```

Change the existing `BuildUDMAInfoImage()` to forward to the new overload with `std::vector<uint32_t>(mem.size(), 1)`. In the new overload, validate `qpWeights.size() == sq.size()`, append the weight array after `mem`, assign `info.qpWeightPtr`, and copy the weights into `bytes`.

Add the three helper functions exactly as declared in Step 3.

- [ ] **Step 5: Implement device weight lookup**

Add to `src/include/tilexr_udma.h` after `UDMAGetRemoteMemInfo()`:

```cpp
__aicore__ inline uint32_t UDMAGetQpWeight(__gm__ UDMAInfo* udmaInfo, uint32_t pe, uint32_t qpIdx)
{
    uint32_t qpNum = udmaInfo->qpNum;
    if (udmaInfo->qpWeightPtr == 0) {
        return 1;
    }
    auto weights = reinterpret_cast<__gm__ uint32_t*>(udmaInfo->qpWeightPtr);
    uint32_t weight = weights[pe * qpNum + qpIdx];
    return weight == 0 ? 1 : weight;
}
```

- [ ] **Step 6: Run layout test**

Run:

```bash
cmake --build build --target test_tilexr_udma_transport_layout
ctest --test-dir build -R test_tilexr_udma_transport_layout --output-on-failure
```

Expected: layout test passes. If `build` is not configured, run the repository's existing CMake configure command first.

- [ ] **Step 7: Commit**

```bash
git add src/include/tilexr_udma_types.h src/comm/udma/tilexr_udma_layout.h src/comm/udma/tilexr_udma_layout.cpp src/include/tilexr_udma.h tests/udma/unit/test_tilexr_udma_transport_layout.cpp
git commit -m "feat: add udma route weight layout"
```

---

### Task 2: Host Multi-EID Route Expansion

**Files:**
- Modify: `src/comm/udma/tilexr_udma_transport.h`
- Modify: `src/comm/udma/tilexr_udma_transport.cpp`
- Test: `tests/udma/unit/test_tilexr_udma_transport_layout.cpp`

**Interfaces:**
- Consumes: Task 1 helpers `BuildUDMAMultiRouteQpToEid()`, `BuildUDMAMultiRouteQpWeights()`, `SelectExplicitUDMARouteEids()`.
- Produces:
  - `TileXRUDMATransport::qpsPerRoute_`
  - `peerLocalEids_`, `peerRemoteEids_`
  - `peerQpRouteEids_`, `peerQpRouteWeights_`
  - `remoteMemHandlesByPeer_`

- [ ] **Step 1: Add transport state fields**

In `src/comm/udma/tilexr_udma_transport.h`, add:

```cpp
uint32_t qpsPerRoute_ = 1;
std::map<int, std::vector<uint32_t>> peerLocalEids_;
std::map<int, std::vector<uint32_t>> peerRemoteEids_;
std::map<int, std::vector<uint32_t>> peerQpRouteEids_;
std::map<int, std::vector<uint32_t>> peerQpRouteWeights_;
std::map<int, std::vector<void*>> remoteMemHandlesByPeer_;
```

Replace `std::vector<void*> remoteMemHandles_;` with `remoteMemHandlesByPeer_`.

- [ ] **Step 2: Add topology route helpers**

In `src/comm/udma/tilexr_udma_transport.cpp`, extend `TileXRRootInfo` with:

```cpp
std::unordered_map<uint32_t, std::map<uint32_t, uint32_t>> portCountByEidByLocalId;
```

In `ParseRootInfo()`, store port counts per EID:

```cpp
const std::vector<std::string> ports = JsonStringArrayField(addrObj, "ports");
root.portCountByEidByLocalId[localId][eidIndex] = static_cast<uint32_t>(ports.size());
for (const std::string& port : ports) {
    root.portToEidByLocalId[localId][port] = eidIndex;
}
```

Add helpers:

```cpp
std::vector<uint32_t> ResolveLocalEidRoutes(
    const TileXRRootInfo& root, const std::vector<TileXRTopoEdge>& edges, uint32_t localId, uint32_t peerLocalId);

std::vector<uint32_t> ResolveLocalAggregateEidRoutes(const TileXRRootInfo& root, uint32_t localId);
```

`ResolveLocalEidRoutes()` walks all matching topology edges, maps local ports to EID indices, de-duplicates while preserving order, and only returns EIDs present in `root.eidByLocalId[localId]`.

`ResolveLocalAggregateEidRoutes()` returns local EIDs with port count greater than `1`, preserving map order.

- [ ] **Step 3: Expand `BuildRoutes()`**

At the start of `BuildRoutes()`:

```cpp
qpsPerRoute_ = qpNum_;
const bool useAllRoutes = std::getenv("TILEXR_UDMA_ROUTE_POLICY") != nullptr &&
    std::strcmp(std::getenv("TILEXR_UDMA_ROUTE_POLICY"), "all") == 0;
const uint32_t maxEidsPerPeer = GetEnvUint("TILEXR_UDMA_MAX_EIDS_PER_PEER", UINT32_MAX, 1, UINT32_MAX);
```

When assigning routes per peer:

- In default mode, keep one EID per peer.
- In `useAllRoutes`, resolve local EID vectors with `ResolveLocalEidRoutes()`.
- If peer-specific routes are empty, use `ResolveLocalAggregateEidRoutes()`.
- If `TILEXR_UDMA_ROUTE_EIDS` selects valid candidates, replace the vector with that explicit list.
- Resize to `routeSlots = useAllRoutes ? max(1, min(eidCount_, maxEidsPerPeer)) : 1`.
- Exchange the route slots with peers.
- Pair local and remote route vectors with the same count.
- Build expanded per-QP route vectors and weight vectors:

```cpp
peerQpRouteEids_[peer] = BuildUDMAMultiRouteQpToEid(localRoutes, qpsPerRoute_);
peerQpRouteWeights_[peer] = BuildUDMAMultiRouteQpWeights(localRoutes, weightByEid, qpsPerRoute_);
```

Finally set:

```cpp
size_t maxRouteCount = 1;
for (const auto& entry : peerQpRouteEids_) {
    maxRouteCount = std::max(maxRouteCount, entry.second.size());
}
qpNum_ = static_cast<uint32_t>(maxRouteCount);
```

- [ ] **Step 4: Create contexts for all selected local EIDs**

In `CreateContexts()`, build `contextEids` from every value in `peerLocalEids_`. Fall back to existing `peerLocalEid_` if the vector map is empty. Create one RA context/token per unique EID.

- [ ] **Step 5: Create queues per route**

In `CreateQueues()`, size per-EID queue vectors by `qpsPerRoute_`, not `qpNum_`:

```cpp
state.qpHandles.assign(qpsPerRoute_, nullptr);
state.remoteQpHandlesByQp.assign(qpsPerRoute_, std::vector<void*>(options_.rankSize, nullptr));
state.localWqs.resize(qpsPerRoute_);
state.localCqs.resize(qpsPerRoute_);
```

When creating peer queues, only create a peer queue for a state if that state EID appears in `peerLocalEids_[peer]`.

- [ ] **Step 6: Expand `RefreshUDMAInfo()` image entries**

Build `sq/rq/scq/rcq/mem/qpWeights` with `options_.rankSize * qpNum_` entries. For each `(rank, qpIdx)`:

- Resolve `localEid` from `peerQpRouteEids_[rank][qpIdx]` when available.
- Resolve `remoteEid` from the matching index in `peerRemoteEids_[rank]`.
- Use `routeQpIdx = qpIdx % qpsPerRoute_`.
- Read WQ/CQ metadata from `states_[localEid].localWqs[routeQpIdx]` and `.localCqs[routeQpIdx]`.
- Read remote TPN from `states_[localEid].tpnListByQp[routeQpIdx][rank]`.
- Read per-QP weight from `peerQpRouteWeights_[rank][qpIdx]`, defaulting to `1`.
- Call the weighted `BuildUDMAInfoImage()` overload.

- [ ] **Step 7: Import and cleanup remote memory per route**

In `ExchangeAndImportMemory()`, replace the single remote handle per peer with:

```cpp
remoteMemHandlesByPeer_.clear();
std::vector<void*> remoteHandles(localRoutes.size(), nullptr);
```

For each paired `localEid/remoteEid`, find the peer's exchanged MR for `remoteEid`, import it through `ctxHandleByEid_[localEid]`, and store it in `remoteHandles[routeIdx]`.

In `CleanupMemory()`, iterate `remoteMemHandlesByPeer_`, use the same route index to find the local EID, and call `RaCtxRmemUnimport()` for each non-null handle.

- [ ] **Step 8: Reset multi-route state during shutdown**

In `Shutdown()`, clear:

```cpp
peerLocalEids_.clear();
peerRemoteEids_.clear();
peerQpRouteEids_.clear();
peerQpRouteWeights_.clear();
qpsPerRoute_ = 1;
qpNum_ = 1;
remoteMemHandlesByPeer_.clear();
```

- [ ] **Step 9: Build UDMA transport objects**

Run:

```bash
cmake --build build --target test_tilexr_udma_transport_layout
```

Expected: compile succeeds and the layout test target still builds.

- [ ] **Step 10: Commit**

```bash
git add src/comm/udma/tilexr_udma_transport.h src/comm/udma/tilexr_udma_transport.cpp
git commit -m "feat: expand udma p2p across multi-eid routes"
```

---

### Task 3: Weighted P2P Demo Kernel Slicing

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`

**Interfaces:**
- Consumes: Task 1 `UDMAGetQpWeight()`.
- Produces: `TileXRUdmaDemoWeightedWqeSlice()`.

- [ ] **Step 1: Add weighted slice helper**

Add after `TileXRUdmaDemoWqeSlice()`:

```cpp
__aicore__ inline void TileXRUdmaDemoWeightedWqeSlice(
    __gm__ TileXR::UDMAInfo* udmaInfo, int32_t peer, uint32_t total,
    uint32_t wqeCount, uint32_t wqeIdx, uint32_t& offset, uint32_t& bytes)
{
    uint32_t weightSum = 0;
    uint32_t prefixWeight = 0;
    for (uint32_t i = 0; i < wqeCount; ++i) {
        uint32_t weight = TileXR::UDMAGetQpWeight(udmaInfo, peer, i);
        if (i < wqeIdx) {
            prefixWeight += weight;
        }
        weightSum += weight;
    }
    if (wqeCount == 0 || weightSum == 0 || wqeIdx >= wqeCount) {
        offset = total;
        bytes = 0;
        return;
    }

    uint64_t rawStart = static_cast<uint64_t>(total) * prefixWeight / weightSum;
    uint64_t rawEnd = static_cast<uint64_t>(total) *
        (prefixWeight + TileXR::UDMAGetQpWeight(udmaInfo, peer, wqeIdx)) / weightSum;
    uint32_t alignedStart = static_cast<uint32_t>(
        (rawStart / TileXR::BLOCK_UNIT_BYTE) * TileXR::BLOCK_UNIT_BYTE);
    uint32_t alignedEnd = wqeIdx + 1 == wqeCount ? total : static_cast<uint32_t>(
        ((rawEnd + TileXR::BLOCK_UNIT_BYTE - 1) / TileXR::BLOCK_UNIT_BYTE) * TileXR::BLOCK_UNIT_BYTE);
    if (alignedEnd > total) {
        alignedEnd = total;
    }
    if (alignedStart >= alignedEnd) {
        offset = total;
        bytes = 0;
        return;
    }
    offset = alignedStart;
    bytes = alignedEnd - alignedStart;
}
```

- [ ] **Step 2: Use weighted slicing in P2P perf kernels**

In both `tilexr_udma_p2p_perf_kernel()` and `tilexr_udma_p2p_post_only_perf_kernel()`, replace:

```cpp
TileXRUdmaDemoWqeSlice(bytes, jettyCount, blockIdx, offset, sliceBytes);
```

with:

```cpp
auto udmaInfo = TileXR::GetUDMAInfo(args);
TileXRUdmaDemoWeightedWqeSlice(udmaInfo, peer, bytes, jettyCount, blockIdx, offset, sliceBytes);
```

In `tilexr_udma_p2p_post_only_perf_kernel()`, remove the later duplicate `auto udmaInfo = TileXR::GetUDMAInfo(args);` declaration.

- [ ] **Step 3: Build demo kernel target**

Run the existing UDMA build entrypoint:

```bash
bash tests/udma/build.sh
```

Expected: compile succeeds. If local Windows host cannot run the Linux/CANN build script, record that hardware build verification was not runnable in this environment and still run Task 4 host tests.

- [ ] **Step 4: Commit**

```bash
git add tests/udma/demo/tilexr_udma_demo_kernel.cpp
git commit -m "perf: weight udma p2p demo slices"
```

---

### Task 4: Verification and Integration Check

**Files:**
- Verify: `src/include/tilexr_udma_types.h`
- Verify: `src/comm/udma/tilexr_udma_layout.h`
- Verify: `src/comm/udma/tilexr_udma_layout.cpp`
- Verify: `src/include/tilexr_udma.h`
- Verify: `src/comm/udma/tilexr_udma_transport.h`
- Verify: `src/comm/udma/tilexr_udma_transport.cpp`
- Verify: `tests/udma/unit/test_tilexr_udma_transport_layout.cpp`
- Verify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: verified local changes and a concise implementation summary.

- [ ] **Step 1: Check diff scope**

Run:

```bash
git diff --name-status HEAD
```

Expected: only the eight UDMA implementation/test files are modified if commits were not made per task. If task commits were made, run:

```bash
git diff --name-status ae07f69..HEAD
```

Expected: only the eight UDMA implementation/test files are changed by implementation commits.

- [ ] **Step 2: Run host layout unit test**

Run:

```bash
ctest --test-dir build -R test_tilexr_udma_transport_layout --output-on-failure
```

Expected: `TileXR UDMA transport layout checks passed`.

- [ ] **Step 3: Run focused source search**

Run:

```bash
rg -n "qpWeightPtr|BuildUDMAMultiRoute|TILEXR_UDMA_ROUTE_POLICY|TILEXR_UDMA_ROUTE_EIDS|remoteMemHandlesByPeer|TileXRUdmaDemoWeightedWqeSlice" src tests
```

Expected: matches appear only in UDMA layout, UDMA transport, UDMA device wrapper, and UDMA demo/test files.

- [ ] **Step 4: Inspect compatibility fallback**

Run:

```bash
git diff ae07f69..HEAD -- src/comm/udma/tilexr_udma_transport.cpp src/comm/udma/tilexr_udma_layout.cpp src/include/tilexr_udma.h
```

Expected:

- `BuildRoutes()` uses multi-route only when `TILEXR_UDMA_ROUTE_POLICY=all`.
- Existing `BuildUDMAInfoImage()` overload still exists.
- `UDMAGetQpWeight()` returns `1` when `qpWeightPtr == 0` or the entry is zero.
- `Shutdown()` clears all multi-route maps.

- [ ] **Step 5: Commit final verification adjustments if needed**

If verification required fixes, commit them:

```bash
git add src/include/tilexr_udma_types.h src/comm/udma/tilexr_udma_layout.h src/comm/udma/tilexr_udma_layout.cpp src/include/tilexr_udma.h src/comm/udma/tilexr_udma_transport.h src/comm/udma/tilexr_udma_transport.cpp tests/udma/unit/test_tilexr_udma_transport_layout.cpp tests/udma/demo/tilexr_udma_demo_kernel.cpp
git commit -m "fix: stabilize udma multi-eid route port"
```

Expected: commit succeeds only if there were verification fixes.

