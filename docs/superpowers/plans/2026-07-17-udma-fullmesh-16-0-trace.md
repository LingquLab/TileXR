# Full-Mesh 16:0 Trace Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a 16:0 full-mesh remote-send variant for a physical 2x8 trace run.

**Architecture:** Reuse the current two-worker protocol and change only the shard split constants. Core17 retains synchronization and completion duties with a zero-byte segment.

**Tech Stack:** C++14, Ascend C/Bisheng, Git bundle, Python trace tooling.

## Global Constraints

- Preserve max/min weighted QP selection.
- Preserve ready and ACK routing.
- Preserve the 8 MiB GM trace layout and converter.
- Commit before bundle deployment.
- Use `TILEXR_IPC_PID_MODE=pid` for the physical run.

---

### Task 1: Change And Deploy The Split

**Files:**
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_layout.cpp`
- Modify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`

**Interfaces:**
- Consumes: `BigDataRemoteSendSegmentRange` and the existing segment-done protocol.
- Produces: core16 `[0, 16)` and core17 `[16, 16)` payload ranges.

- [ ] **Step 1: Write the failing source assertions**

Require exact constants `PRIMARY_SHARD_END = 16U` and
`SECONDARY_AGGREGATOR = 15U`.

- [ ] **Step 2: Verify RED**

Run the layout source test and confirm it fails while the kernel still uses
`12U` and `11U`.

- [ ] **Step 3: Implement the 16:0 constants**

Change only the two constants; retain the existing `segmentBytes > 0U` guard
and unconditional segment-done publication.

- [ ] **Step 4: Commit, bundle, and deploy**

Commit the test and implementation, create a complete verified bundle, upload
it to both hosts, and build the demo plus unit tests.

### Task 2: Run And Export The Trace

**Files:**
- Create: `tmp/test0701_fullmesh_trace_16_0_2x8/` artifacts.

**Interfaces:**
- Consumes: the committed 16:0 demo.
- Produces: sixteen raw traces, a complete Chrome trace, and an iteration49 trace.

- [ ] **Step 1: Run physical 2x8**

Use 128 MiB/rank, profile stage8, repeat50, full-mesh trace enabled, and PID IPC mode.

- [ ] **Step 2: Validate the split**

For iteration49, require eight core16 `data-put` events and zero core17
`data-put` events per rank.

- [ ] **Step 3: Export JSON**

Download all raw traces, convert the full trace, and extract iteration49.
