# UDMA Grouped AllToAll Configurable Copyout Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make grouped AllToAll copyout workers selectable as 8 or 16 and compare both configurations on the physical 2x8, 128 MiB/rank workload.

**Architecture:** Put copyout validation, block-dimension calculation, and worker-to-lane mapping in the existing grouped layout header so they are host-testable. Pass the selected count from Host to the kernel; use an equivalent device-local lane helper because Bisheng does not reliably link ordinary host inline functions into AICore objects. Active receive cores iterate their owned logical lanes while trace tasks remain indexed by logical lane.

**Tech Stack:** C++14, Ascend C/Bisheng, TileXR UDMA, CMake, GM trace converter, CANN `/home/pkg/b101/cann`, physical Ascend950 2x8.

## Global Constraints

- Accept only 8 or 16 copyout workers; default to 16.
- Keep send cores 0-15 and all dual-route behavior unchanged.
- Launch 24 blocks for 8 workers and 32 blocks for 16 workers.
- Do not change payload, signal, token, ping-pong, or registered-memory layouts.
- Do not add barriers, ACKs, signals, `SyncAll`, or generic UDMA routes.
- Index receive wait/copy trace tasks by logical core `16 + lane`.
- Index kernel and self-copy trace spans by physical `blockIdx`.
- Commit and verify a complete bundle before each remote build or run.

---

### Task 1: Copyout Layout Policy And Host Configuration

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_alltoall_group_layout.h`
- Modify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`

**Interfaces:**
- Produces: `AllToAllGroupValidCopyoutWorkers(uint32_t) -> bool`.
- Produces: `AllToAllGroupBlockDim(uint32_t) -> uint32_t`.
- Produces: `AllToAllGroupCopyoutLane(worker, assignment, workers) -> int32_t`.
- Produces: environment variable `TILEXR_DEMO_ALLTOALL_GROUP_COPYOUT_WORKERS`.
- Extends: grouped launch with trailing `uint32_t copyoutWorkers`.

- [ ] **Step 1: Write failing layout and Host source tests**

Add checks equivalent to:

```cpp
CHECK_EQ(TileXR::Demo::AllToAllGroupValidCopyoutWorkers(8U), true);
CHECK_EQ(TileXR::Demo::AllToAllGroupValidCopyoutWorkers(16U), true);
CHECK_EQ(TileXR::Demo::AllToAllGroupValidCopyoutWorkers(4U), false);
CHECK_EQ(TileXR::Demo::AllToAllGroupBlockDim(8U), 24U);
CHECK_EQ(TileXR::Demo::AllToAllGroupBlockDim(16U), 32U);

std::set<int32_t> lanes;
for (uint32_t worker = 0; worker < 8U; ++worker) {
    for (uint32_t assignment = 0; assignment < 2U; ++assignment) {
        lanes.insert(TileXR::Demo::AllToAllGroupCopyoutLane(worker, assignment, 8U));
    }
}
CHECK_EQ(lanes.size(), 16U);
CHECK_EQ(*lanes.begin(), 0);
CHECK_EQ(*lanes.rbegin(), 15);
CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(7U, 1U, 16U), -1);
```

Require Host source strings for the environment variable, validation, dynamic
block dimension, and launch argument.

- [ ] **Step 2: Verify RED and commit the test**

Commit and bundle the test, deploy to `141.61.49.223`, then build
`test_tilexr_udma_alltoall_group_layout`. Expected: compile/source-guard
failures because the copyout policy and Host plumbing do not exist.

- [ ] **Step 3: Implement layout helpers**

Add to the grouped layout header:

```cpp
constexpr uint32_t kAllToAllGroupSendCoreCount = 16U;

inline bool AllToAllGroupValidCopyoutWorkers(uint32_t workers)
{
    return workers == 8U || workers == 16U;
}

inline uint32_t AllToAllGroupBlockDim(uint32_t workers)
{
    return AllToAllGroupValidCopyoutWorkers(workers) ?
        kAllToAllGroupSendCoreCount + workers : 0U;
}

inline int32_t AllToAllGroupCopyoutLane(
    uint32_t worker, uint32_t assignment, uint32_t workers)
{
    if (!AllToAllGroupValidCopyoutWorkers(workers) || worker >= workers) return -1;
    const uint32_t lane = worker + assignment * workers;
    return lane < kAllToAllGroupWidth ? static_cast<int32_t>(lane) : -1;
}
```

- [ ] **Step 4: Add Host configuration**

Read the environment in `RunGroupedAllToAll`, reject values other than 8/16,
print the selected count, launch `AllToAllGroupBlockDim(copyoutWorkers)`, and
pass `copyoutWorkers` to warmup and measured kernel launches.

- [ ] **Step 5: Verify GREEN and commit**

Bundle/deploy, build the full demo, and run the grouped layout test. Commit as:

```bash
git commit -m "feat(udma): configure grouped copyout workers"
```

---

### Task 2: Kernel Lane Scheduling And Trace Safety

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`

**Interfaces:**
- Consumes: `copyoutWorkers` and Task 1 lane helpers.
- Produces: physical receive workers that handle one or two logical lanes.
- Preserves: one trace wait/copy cell per logical lane.

- [ ] **Step 1: Add failing kernel source guards**

Require:

```cpp
CHECK_CONTAINS(kernel, "copyoutWorkers");
CHECK_CONTAINS(kernel, "AllToAllGroupCopyoutLaneDevice");
CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_SEND_CORES + copyoutWorkers");
CHECK_CONTAINS(kernel, "traceCore = TILEXR_ALLTOALL_GROUP_SEND_CORES + lane");
CHECK_NOT_CONTAINS(kernel, "elementsPerPeer) * lane /");
```

Expected RED: the existing receive branch assumes 16 workers and one lane.

- [ ] **Step 2: Implement active-core and self-copy mapping**

Validate `copyoutWorkers` as 8 or 16. Return inactive blocks at or above
`16 + copyoutWorkers`. Compute receive worker as `blockIdx - 16`, and divide
self-copy using `copyoutWorkers` rather than 16.

- [ ] **Step 3: Iterate logical lanes inside each group**

Add the device-local equivalent of the tested Host helper:

```cpp
__aicore__ inline int32_t AllToAllGroupCopyoutLaneDevice(
    uint32_t worker, uint32_t assignment, uint32_t workers)
{
    const uint32_t lane = worker + assignment * workers;
    return lane < TILEXR_ALLTOALL_GROUP_SEND_CORES ?
        static_cast<int32_t>(lane) : -1;
}
```

Use it in the receive loop:

```cpp
for (uint32_t group = 0U; group < groupCount; ++group) {
    for (uint32_t assignment = 0U; ; ++assignment) {
        const int32_t laneValue = AllToAllGroupCopyoutLaneDevice(
            worker, assignment, copyoutWorkers);
        if (laneValue < 0) break;
        const uint32_t lane = static_cast<uint32_t>(laneValue);
        const int32_t peer = AllToAllGroupDevicePeer(rank, rankSize, group, lane);
        if (peer < 0) continue;
        // existing pass wait and copy loop
    }
}
```

Record wait/copy tasks with `traceCore = 16U + lane`; retain physical blockIdx
for kernel, self-copy, and debug.

- [ ] **Step 4: Build, regress, and commit**

Bundle/deploy and run the Bisheng build, grouped/existing layout tests,
transport test, and Python trace tests. Commit as:

```bash
git commit -m "feat(udma): schedule grouped copyout on 8 or 16 cores"
```

---

### Task 3: Physical 16-Worker And 8-Worker Matrix

**Files:**
- Create artifact: `tmp/udma-grouped-alltoall-copyout-workers.bundle`
- Create artifacts under: `tmp/grouped_alltoall_copyout_workers_b101_2x8/`

- [ ] **Step 1: Deploy identical final artifacts**

Verify the complete bundle, deploy both hosts, build/install on
`141.61.49.223`, copy the grouped kernel and demo to `141.61.50.31`, and
compare commit IDs and SHA-256 hashes.

- [ ] **Step 2: Run 16-worker target**

Run 128 MiB/rank, warmup5/repeat50, dual route and trace enabled with:

```bash
export TILEXR_DEMO_ALLTOALL_GROUP_COPYOUT_WORKERS=16
```

Require 16 successful ranks and sixteen complete 8 MiB traces.

- [ ] **Step 3: Run 8-worker target**

Repeat the identical workload with:

```bash
export TILEXR_DEMO_ALLTOALL_GROUP_COPYOUT_WORKERS=8
```

Require 16 successful ranks and sixteen complete 8 MiB traces.

- [ ] **Step 4: Compare traces and timing**

For each mode report Host mean/min/max, loop49 kernel envelope, self-copy,
send-put-signal, send-quiet, receive-wait, receive-copy, complete event count,
and QP distribution. Verify every rank still has 15 send/receive peers and
cross-node QP0:QP4 remains 96:32.

- [ ] **Step 5: Final regression and bundle verification**

Run all grouped/existing layout, transport, and Python trace tests, confirm no
tracked deployment differences, and verify the final bundle contains HEAD.
