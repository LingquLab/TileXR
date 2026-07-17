# UDMA Full-Mesh Control Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the full-mesh 12:4 payload split while binding payload and ready completion to explicit QPs and removing generic-route UDMA ACK publication.

**Architecture:** The two remote-send workers select max/min weighted QPs for their existing primary/secondary segments. The primary worker publishes ready on its selected QP after both segments complete. Receive completion writes the token directly to the sender's registered ACK control slot.

**Tech Stack:** C++14, Ascend C device code, TileXR UDMA device helpers, source-structure unit tests.

## Global Constraints

- Preserve `TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END = 12U`.
- Do not import XY pipeline, trace, scheduling, or 16:0 changes.
- Commit before creating and uploading a Git bundle.
- Validate physical 2x8 with `TILEXR_IPC_PID_MODE=pid`.

---

### Task 1: Add Failing Full-Mesh Route Tests

**Files:**
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_layout.cpp`

**Interfaces:**
- Consumes: `SliceBetween`, `CHECK_CONTAINS`, and `CHECK_NOT_CONTAINS`.
- Produces: source-structure requirements for full-mesh payload, ready, and ACK traffic.

- [ ] **Step 1: Add isolated source checks**

```cpp
const std::string fullMeshReady = SliceBetween(
    kernel, "BigDataPublishReadySignal", "BigDataRemoteSendSegmentWorker");
const std::string fullMeshSend = SliceBetween(
    kernel, "BigDataRemoteSendSegmentWorker", "BigDataRemotePutOnlySendWorker");
const std::string fullMeshRecv = SliceBetween(
    kernel, "BigDataRecvPeerWorker", "BigDataWaitCopyDoneRange");
CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END = 12U");
CHECK_CONTAINS(fullMeshReady, "UDMAPutSignalNbiOnQp<uint64_t>");
CHECK_CONTAINS(fullMeshReady, "UDMAQuietStatusOnQp(args, peer, qpIdx)");
CHECK_CONTAINS(fullMeshSend, "BigDataSelectWeightedQp(");
CHECK_CONTAINS(fullMeshSend, "UDMAPutNbiOnQp<int32_t>");
CHECK_CONTAINS(fullMeshSend, "UDMAQuietStatusOnQp(args, peer, qpIdx)");
CHECK_NOT_CONTAINS(fullMeshSend, "TileXR::UDMAPutNbi<int32_t>(args, peer");
CHECK_CONTAINS(fullMeshRecv, "BigDataRemoteRegisteredControlSlot(");
CHECK_CONTAINS(fullMeshRecv, "BigDataStoreTokenMte(remoteAck, token, relayLocal)");
CHECK_NOT_CONTAINS(fullMeshRecv, "BigDataPublishAckSignalUdma(");
```

- [ ] **Step 2: Commit the failing test and create a RED bundle**

```bash
git add tests/udma/unit/test_tilexr_udma_alltoall_layout.cpp
git commit -m "test(udma): require explicit fullmesh control routes"
git bundle create tmp/test0701-fullmesh-control-route-red.bundle test0701-bundle
git bundle verify tmp/test0701-fullmesh-control-route-red.bundle
```

- [ ] **Step 3: Verify RED remotely**

Upload the committed bundle to both hosts, fetch it in `/home/h30059441/tilexr_stage0_y_stagger`, build target `test_tilexr_udma_alltoall_layout` in `tests/udma/build-test0701`, and run it. Expected: nonzero exit caused by the new checks for `UDMAPutSignalNbiOnQp`, explicit full-mesh payload QP, and direct registered ACK.

### Task 2: Route Full-Mesh Data And Control Explicitly

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`

**Interfaces:**
- Consumes: `BigDataSelectWeightedQp(args, peer, selectMax)` and `BigDataRemoteRegisteredControlSlot(...)`.
- Produces: `BigDataPublishReadySignal(..., uint32_t qpIdx)` and direct registered ACK publication.

- [ ] **Step 1: Bind payload and ready to the segment QP**

In `BigDataRemoteSendSegmentWorker`, preserve the current segment ranges and add:

```cpp
const uint32_t qpIdx = BigDataSelectWeightedQp(
    args, peer,
    segmentId == TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT);
TileXR::UDMAPutNbiOnQp<int32_t>(
    args, peer, qpIdx, localSrc, remoteDataOffset, segmentBytes);
status = TileXR::UDMAQuietStatusOnQp(args, peer, qpIdx);
```

Pass `qpIdx` to `BigDataPublishReadySignal`, whose put and quiet become:

```cpp
TileXR::UDMAPutSignalNbiOnQp<uint64_t>(
    args, peer, qpIdx, localSrc, localReadyPayloadOffset,
    sizeof(uint64_t), remoteReadyOffset, token);
(void)TileXR::UDMAQuietStatusOnQp(args, peer, qpIdx);
```

- [ ] **Step 2: Replace generic UDMA ACK publication**

For both multi-node ACK publication sites in `BigDataRecvPeerWorker`, resolve and write the remote registered slot:

```cpp
auto remoteAck = BigDataRemoteRegisteredControlSlot(
    args, peer, ackSignalOffset, slot, rankSize, shardCount, rank, 0U);
if (remoteAck == nullptr) {
    if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
        debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
            TILEXR_UDMA_DEMO_ACK_TIMEOUT_STATUS;
    }
    return;
}
BigDataStoreTokenMte(remoteAck, token, relayLocal);
```

Remove `BigDataPublishAckSignalUdma` after its callers are gone.

- [ ] **Step 3: Commit, bundle, and verify GREEN remotely**

```bash
git add tests/udma/demo/tilexr_udma_demo_kernel.cpp
git commit -m "fix(udma): pin fullmesh control traffic to valid routes"
git bundle create tmp/test0701-fullmesh-control-route.bundle test0701-bundle
git bundle verify tmp/test0701-fullmesh-control-route.bundle
```

Upload the bundle to both hosts, fetch it, rebuild `tilexr_udma_demo` and `test_tilexr_udma_alltoall_layout`, and run the unit test. Expected: exit 0. Also run `git diff --check` locally.

### Task 3: Verify Physical 2x8

**Files:**
- No tracked file changes.

**Interfaces:**
- Consumes: committed Task 2 HEAD.
- Produces: physical 2x8 correctness evidence and remote result logs.

- [ ] **Step 1: Run repeat1 full-mesh correctness**

```bash
export TILEXR_DEMO_BIGDATA_REMOTE_PUT_ONLY=0
export TILEXR_DEMO_BIGDATA_PROFILE_STAGE=8
export TILEXR_DEMO_ALLTOALL_REPEAT=1
export TILEXR_IPC_PID_MODE=pid
./tests/udma/install-test0701/bin/tilexr_udma_demo 16 RANK 7 2097152 8 0
```

Expected: all 16 ranks exit 0, print `TileXR UDMA demo success`, and report no `CQ incomplete`, `MISMATCH`, or `ERROR`.

- [ ] **Step 2: Run repeat50 only after repeat1 passes**

Change `TILEXR_DEMO_ALLTOALL_REPEAT` to `50`. Report max per-rank kernel time and retain the remote result directory. If repeat1 fails, stop and analyze that run instead.
