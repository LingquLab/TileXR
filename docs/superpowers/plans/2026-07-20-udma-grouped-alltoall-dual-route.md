# UDMA Grouped AllToAll Dual-Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute grouped AllToAll cross-node peers over the six-port and two-port aggregate UDMA routes in a balanced 6:2 ratio without changing the payload/ready protocol.

**Architecture:** Add a host-testable route-policy header for node and 6:2 peer assignment. The grouped device kernel uses an equivalent device-local predicate because Bisheng does not reliably link ordinary host inline functions into AICore objects, discovers the highest and second-highest distinct QP weights, and continues issuing payload plus signal and quiet on one selected QP per peer.

**Tech Stack:** C++14 host tests, Ascend C/Bisheng device code, TileXR UDMA QP-weight metadata, CMake, Chrome Trace JSON, and physical Ascend950 2x8 validation with CANN `/home/pkg/b101/cann`.

## Global Constraints

- Keep the existing 35-core big-data fullmesh kernel unchanged.
- Keep one complete peer payload on one route; do not split payload within a peer.
- Keep `UDMAPutSignalNbiOnQp` and `UDMAQuietStatusOnQp` on the same selected QP.
- Do not add signals, ACKs, barriers, registered regions, or `SyncAll()`.
- Apply dual-route selection only when `rank / 8 != peer / 8`.
- Continue supporting `8 <= rankSize <= 128` and `rankSize % 8 == 0`.
- Treat each contiguous group of eight ranks as one node.
- Primary QP is the lowest index with maximum weight.
- Secondary QP is the lowest index with the largest weight strictly below the primary weight.
- Fall back to primary when no distinct lower-weight QP exists.
- Commit locally, create and verify a complete Git bundle, then upload the bundle before every remote build or run.
- Build and run against `/home/pkg/b101/cann`; set `TILEXR_IPC_PID_MODE=pid` for physical runs.

---

### Task 1: Host-Testable Balanced Peer Policy

**Files:**
- Create: `tests/udma/demo/tilexr_udma_alltoall_group_route.h`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`
- Modify: `tests/udma/CMakeLists.txt`

**Interfaces:**
- Produces: `AllToAllGroupIsCrossNode(int rank, int peer) -> bool`.
- Produces: `AllToAllGroupUseSecondaryRoute(int rank, int peer) -> bool`.
- Produces: `kAllToAllGroupRanksPerNode=8U` and `kAllToAllGroupPrimaryPeersPerNode=6U`.

- [ ] **Step 1: Write the failing route-policy test**

Add the missing header include and this test to
`tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`:

```cpp
#include "demo/tilexr_udma_alltoall_group_route.h"

void TestDualRoutePeerPolicy()
{
    using TileXR::Demo::AllToAllGroupIsCrossNode;
    using TileXR::Demo::AllToAllGroupUseSecondaryRoute;
    CHECK_EQ(AllToAllGroupIsCrossNode(0, 7), false);
    CHECK_EQ(AllToAllGroupIsCrossNode(0, 8), true);
    CHECK_EQ(AllToAllGroupIsCrossNode(15, 8), false);
    CHECK_EQ(AllToAllGroupIsCrossNode(15, 0), true);

    for (int sourceNode = 0; sourceNode < 3; ++sourceNode) {
        for (int targetNode = 0; targetNode < 3; ++targetNode) {
            if (sourceNode == targetNode) continue;
            int rowSecondary[8] = {};
            int columnSecondary[8] = {};
            for (int sourceLocal = 0; sourceLocal < 8; ++sourceLocal) {
                for (int targetLocal = 0; targetLocal < 8; ++targetLocal) {
                    const int source = sourceNode * 8 + sourceLocal;
                    const int target = targetNode * 8 + targetLocal;
                    if (AllToAllGroupUseSecondaryRoute(source, target)) {
                        ++rowSecondary[sourceLocal];
                        ++columnSecondary[targetLocal];
                    }
                }
            }
            for (int local = 0; local < 8; ++local) {
                CHECK_EQ(rowSecondary[local], 2);
                CHECK_EQ(columnSecondary[local], 2);
            }
        }
    }
}
```

Call `TestDualRoutePeerPolicy()` from `main()`.

- [ ] **Step 2: Verify RED and commit the test**

```bash
git diff --check
git add tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp
git commit -m "test(udma): require balanced grouped dual-route policy"
git bundle create tmp/grouped-dual-route-red.bundle HEAD
git bundle verify tmp/grouped-dual-route-red.bundle
```

Deploy to `141.61.49.223` and run:

```bash
source /home/pkg/b101/cann/set_env.sh
export ASCEND_HOME_PATH=/home/pkg/b101/cann
cmake --build tests/udma/build_b101 --target test_tilexr_udma_alltoall_group_layout -j8
```

Expected: compilation fails because `tilexr_udma_alltoall_group_route.h` is missing.

- [ ] **Step 3: Implement the minimal peer policy**

Create `tests/udma/demo/tilexr_udma_alltoall_group_route.h`:

```cpp
#ifndef TILEXR_UDMA_ALLTOALL_GROUP_ROUTE_H
#define TILEXR_UDMA_ALLTOALL_GROUP_ROUTE_H

#include <cstdint>

namespace TileXR {
namespace Demo {
constexpr uint32_t kAllToAllGroupRanksPerNode = 8U;
constexpr uint32_t kAllToAllGroupPrimaryPeersPerNode = 6U;

inline bool AllToAllGroupIsCrossNode(int rank, int peer)
{
    return rank >= 0 && peer >= 0 &&
        rank / static_cast<int>(kAllToAllGroupRanksPerNode) !=
        peer / static_cast<int>(kAllToAllGroupRanksPerNode);
}

inline bool AllToAllGroupUseSecondaryRoute(int rank, int peer)
{
    if (!AllToAllGroupIsCrossNode(rank, peer)) return false;
    const uint32_t sourceLocal =
        static_cast<uint32_t>(rank) % kAllToAllGroupRanksPerNode;
    const uint32_t targetLocal =
        static_cast<uint32_t>(peer) % kAllToAllGroupRanksPerNode;
    return (sourceLocal + targetLocal) % kAllToAllGroupRanksPerNode >=
        kAllToAllGroupPrimaryPeersPerNode;
}
} // namespace Demo
} // namespace TileXR
#endif
```

Add this header to the grouped-kernel custom-command `DEPENDS` list in
`tests/udma/CMakeLists.txt`.

- [ ] **Step 4: Verify GREEN and commit**

Bundle and deploy the implementation, then run on `141.61.49.223`:

```bash
cmake --build tests/udma/build_b101 --target test_tilexr_udma_alltoall_group_layout -j8
./tests/udma/build_b101/test_tilexr_udma_alltoall_group_layout
```

Expected: `TileXR grouped all-to-all layout checks passed`.

```bash
git add tests/udma/demo/tilexr_udma_alltoall_group_route.h \
  tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp tests/udma/CMakeLists.txt
git commit -m "feat(udma): add grouped dual-route peer policy"
```

---

### Task 2: Device QP Discovery And Selection

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`

**Interfaces:**
- Consumes: route-policy constants and formula from Task 1.
- Produces: `AllToAllGroupSelectRouteQps(args, peer, primaryQp, secondaryQp)`.
- Produces: device helper `AllToAllGroupUseSecondaryRouteDevice(rank, peer)`.
- Produces: one selected QP reused by payload plus signal, quiet, debug, and trace.

- [ ] **Step 1: Add failing grouped-kernel source guards**

Extend `TestKernelStructure()`:

```cpp
CHECK_CONTAINS(kernel, "#include \"tilexr_udma_alltoall_group_route.h\"");
CHECK_CONTAINS(kernel, "AllToAllGroupSelectRouteQps");
CHECK_CONTAINS(kernel, "AllToAllGroupUseSecondaryRouteDevice(rank, peer)");
CHECK_CONTAINS(kernel, "secondaryQp");
CHECK_CONTAINS(kernel, "selectedQp");
CHECK_CONTAINS(kernel, "UDMAQuietStatusOnQp(args, peer, selectedQp)");
```

Run the grouped layout test. Expected: source-guard failures for these names.

- [ ] **Step 2: Implement distinct-weight QP discovery**

Include the route-policy header. Add the device-local peer predicate so the
kernel does not depend on linking a normal host inline function:

```cpp
__aicore__ inline bool AllToAllGroupUseSecondaryRouteDevice(
    int32_t rank, int32_t peer)
{
    if (rank < 0 || peer < 0 ||
        rank / static_cast<int32_t>(TileXR::Demo::kAllToAllGroupRanksPerNode) ==
        peer / static_cast<int32_t>(TileXR::Demo::kAllToAllGroupRanksPerNode)) {
        return false;
    }
    const uint32_t sourceLocal = static_cast<uint32_t>(rank) %
        TileXR::Demo::kAllToAllGroupRanksPerNode;
    const uint32_t targetLocal = static_cast<uint32_t>(peer) %
        TileXR::Demo::kAllToAllGroupRanksPerNode;
    return (sourceLocal + targetLocal) % TileXR::Demo::kAllToAllGroupRanksPerNode >=
        TileXR::Demo::kAllToAllGroupPrimaryPeersPerNode;
}
```

Then replace the max-only QP helper with:

```cpp
__aicore__ inline void AllToAllGroupSelectRouteQps(
    const __gm__ TileXR::CommArgs* args, int32_t peer,
    uint32_t& primaryQp, uint32_t& secondaryQp)
{
    auto info = TileXR::GetUDMAInfo(args);
    const uint32_t qpCount = info->qpNum == 0U ? 1U : info->qpNum;
    primaryQp = 0U;
    uint32_t primaryWeight = TileXR::UDMAGetQpWeight(info, peer, 0U);
    for (uint32_t qp = 1U; qp < qpCount; ++qp) {
        const uint32_t weight = TileXR::UDMAGetQpWeight(info, peer, qp);
        if (weight > primaryWeight) {
            primaryQp = qp;
            primaryWeight = weight;
        }
    }
    secondaryQp = primaryQp;
    uint32_t secondaryWeight = 0U;
    for (uint32_t qp = 0U; qp < qpCount; ++qp) {
        const uint32_t weight = TileXR::UDMAGetQpWeight(info, peer, qp);
        if (weight < primaryWeight && weight > secondaryWeight) {
            secondaryQp = qp;
            secondaryWeight = weight;
        }
    }
}
```

Strict `>` comparisons preserve the lowest QP index on equal weights.

- [ ] **Step 3: Select once per peer and reuse the QP**

Before the send pass loop:

```cpp
uint32_t primaryQp = 0U;
uint32_t secondaryQp = 0U;
AllToAllGroupSelectRouteQps(args, peer, primaryQp, secondaryQp);
const uint32_t selectedQp =
    AllToAllGroupUseSecondaryRouteDevice(rank, peer) ?
    secondaryQp : primaryQp;
```

Use `selectedQp` for `UDMAPutSignalNbiOnQp`, `UDMAQuietStatusOnQp`, both send
trace records, and quiet-error debug. Do not modify receive-side waits or copy.

- [ ] **Step 4: Build, regress, and commit**

Commit, bundle, deploy, then run on `141.61.49.223`:

```bash
source /home/pkg/b101/cann/set_env.sh
export ASCEND_HOME_PATH=/home/pkg/b101/cann
cmake --build tests/udma/build_b101 -j8
./tests/udma/build_b101/test_tilexr_udma_alltoall_group_layout
./tests/udma/build_b101/test_tilexr_udma_alltoall_layout
./tests/udma/build_b101/test_tilexr_udma_transport_layout
python3 -m unittest tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
```

Expected: Bisheng and Host builds succeed and every test exits zero.

```bash
git add tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp \
  tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp
git commit -m "feat(udma): balance grouped peers across UDMA routes"
```

---

### Task 3: Physical 2x8 Comparison

**Files:**
- Create artifact: `tmp/udma-grouped-alltoall-dual-route.bundle`
- Create run artifacts under: `tmp/grouped_alltoall_dual_route_b101_2x8/`
- Do not commit raw binaries, JSON traces, or run logs.

**Interfaces:**
- Consumes: grouped dual-route kernel from Task 2.
- Produces: correctness, QP distribution, Host timing, and loop49 kernel envelope.

- [ ] **Step 1: Create final bundle and deploy both hosts**

```bash
git diff --check
git status --short
git bundle create tmp/udma-grouped-alltoall-dual-route.bundle HEAD
git bundle verify tmp/udma-grouped-alltoall-dual-route.bundle
```

Fetch it into `/home/h30059441/tilexr_grouped_alltoall_b101` on
`141.61.50.31` and `141.61.49.223`. Build/install on `141.61.49.223`, copy
the grouped kernel library and demo executable to `141.61.50.31`, then verify
commit IDs and SHA-256 hashes match.

- [ ] **Step 2: Run 16-rank smoke correctness**

```bash
export TILEXR_COMM_ID=141.61.50.31:64720
export TILEXR_IPC_PID_MODE=pid
export TILEXR_UDMA_ROUTE_POLICY=all
export TILEXR_UDMA_QP_NUM=4
export TILEXR_DEMO_ALLTOALL_WARMUP=1
export TILEXR_DEMO_ALLTOALL_REPEAT=2
timeout 180s ./tests/udma/install_b101/bin/tilexr_udma_demo \
  16 "$rank" 8 4096 8 0
```

Expected: all 16 ranks succeed with no quiet error, timeout, or mismatch.

- [ ] **Step 3: Run 128 MiB/rank with trace**

```bash
export TILEXR_DEMO_ALLTOALL_WARMUP=5
export TILEXR_DEMO_ALLTOALL_REPEAT=50
export TILEXR_DEMO_ALLTOALL_GROUP_CHUNK_ELEMENTS=2097152
export TILEXR_UDMA_GROUP_TRACE=1
export TILEXR_UDMA_GROUP_TRACE_DIR=/home/h30059441/grouped_alltoall_dual_route_b101_2x8
timeout 300s ./tests/udma/install_b101/bin/tilexr_udma_demo \
  16 "$rank" 8 2097152 8 0
```

Expected: 16 successful ranks and sixteen 8 MiB traces.

- [ ] **Step 4: Validate loop49 and compare performance**

Convert and extract iteration49. Require:

```text
cross-node total = 128
cross-node QP0 = 96
cross-node QP4 = 32
same-node total = 112
invalid QP = 0
unique send peers/rank = 15
unique receive peers/rank = 15
complete loop49 spans = 1728
```

Report Host mean/min/max and loop49 kernel-envelope mean/min/max beside the
single-route baseline `824.632 us` Host mean and `822.583 us` kernel mean.

- [ ] **Step 5: Run final regressions**

```bash
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_group_layout
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_layout
./tests/udma/install_b101/bin/test_tilexr_udma_transport_layout
python3 -m unittest tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
git status --short
```

Expected: every test passes and no tracked source differs from deployed HEAD.
