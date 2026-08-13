# MoonEP Combine V2 Self-Copy Pipeline Design

Status: Implemented; B150 compile and bounded hardware correctness complete

Baseline: `c0d40d7 perf: batch MoonEP combine v2 WQE construction`

Target environment: Ascend 3510, CANN `/home/pkg/910_B150/cann-9.1.0`

The environment exposes the requested B150 build as `910_B150`, not the
lowercase literal path `/home/pkg/b150`. On 2026-08-12 the implementation was
synced through Mutagen to 9号柜 CPU1 (`141.61.55.118`) and compiled with
`/home/pkg/910_B150/cann-9.1.0`.

Completed evidence:

- the pure AICore kernel, Combine V2 Host library, and hardware probe compile
  with B150;
- all six focused Combine V2 Host tests pass in the B150 build tree;
- the send-path UB layout is compile-time fixed at 216,096 bytes, leaving
  5,088 bytes of the 216 KiB budget;
- the obsolete Self mismatch allowance is removed from the probe and launcher;
- single-machine 8P correctness passes with the B150 build and runtime on
  9号柜 CPU1 for BS=128 and BS=8192, H=3584, K=16, BF16, ExpNum=64; and
- BS=8192 exercises selector batches above 128 routes and resume processing.

The B150 hardware checks used `warmup=0` and `iterations=1` as bounded
correctness runs. Their timing output is not a formal performance result. The
controller log is
`/home/h00580772/tilexr_selfcopy_b150/logs/selfcopy_8p_bs128_8192_correctness.log`.

This design restores `SendSelfStep()` after the SIMT route-selection and WQE
construction optimization. It reuses the active route-selection path and adds a
dedicated double-buffered GM-to-UB-to-GM copy pipeline for locally routed rows.

This document supersedes the following decisions in
`2026-08-12-moonep-combine-v2-simt-wqe-design.md`:

- self-copy no longer returns immediately;
- old compare/gather/descriptor UB buffers and their dead implementation are
  deleted rather than retained under `#if 0`; and
- the remote WQE issue allocation becomes one continuous 260-WQE region split
  into fixed 194- and 66-WQE lane regions.

The untracked `tilexr_moonep_combine_v2_kernel_back.h` file is an experimental
reference only. It is not an implementation source and must not be copied over
the active kernel. In particular, its reuse of old selection buffers is not
part of this design.

## Goals

1. Restore correct local-route movement from the registered workspace to the
   current scratch epoch.
2. Reuse the current SIMT selector and its compacted `RouteEntry` output.
3. Overlap Self MTE2 reads and MTE3 writes with two dedicated 64 KiB UB relay
   buffers.
4. Copy up to eight complete rows per relay batch when the runtime row size
   permits it, and reduce the row count automatically for larger hidden sizes.
5. Preserve the remote 3:1 QP distribution, control-WQE ordering, CQ behavior,
   and doorbell ordering.
6. Recover UB capacity by deleting obsolete compare/gather selection storage
   and reducing the WQE issue capacity to the actual maximum plus controls.

## Non-Goals

This change does not:

- change the Planner route encoding;
- replace route division and modulo with a packed bit-field representation;
- use UDMA WQEs, SQs, doorbells, grants, done tokens, or CQs for Self traffic;
- add a SIMT function for Self address construction;
- expose relay size or rows per copy batch as a Host API parameter;
- change the reduction algorithm; or
- reuse WQE, route, cursor, compare, or gather storage as the Self relay.

## Existing Route Contract

The inverse route table remains:

```text
dstLocal[expertRecvSlot] = srcRank * NvS + token * K + topk
```

For an entry selected for the current peer, the active SIMT selector produces:

```cpp
struct MoonEpCombineV2RouteEntry {
    uint32_t sourceSlotIndex;
    uint32_t targetSlot;
};
```

For Self traffic:

```cpp
source = workspace_ +
    static_cast<uint64_t>(route.sourceSlotIndex) * rowBytes_;
target = scratch_ +
    static_cast<uint64_t>(route.targetSlot) * rowBytes_;
```

`sourceSlotIndex` is already absolute. `SendSelfStep()` must not add the current
selection chunk offset a second time. `targetSlot` is already decoded by the
selector; Self must not reload `dstLocal` or repeat the modulo operation.

Workspace input and scratch output are different GM regions. The design does
not support or require an in-place overlapping copy.

## Reused Remote Selection Path

The following active Remote components are reused without a second selection
implementation:

- `dstSlotBuf_`, including its 64 KiB GM-to-UB chunk load;
- `LoadSelectionChunk()`;
- `MoonEpCombineV2SelectPeerRoutesVf()`;
- `SelectPeerRoutes()`;
- `routeEntryBuf_`;
- `threadMaxSlotIdxBuf_`;
- `selectStateBuf_`;
- the peer-lifetime `curWqeNum` initialization;
- `firstPass` and per-thread resume cursors;
- `pausedThreadCount` and repeated scans over a resident chunk; and
- the maximum of 256 selected `RouteEntry` objects per selector invocation.

Self invokes this path with `peer == rank_`. The selector continues to perform
the existing division and modulo and writes the same route representation used
by the Remote payload builder.

The following Remote stages are not used by Self:

- operator- or peer-level WQE prefill;
- remote-field resolution;
- payload WQE construction;
- 3:1 QP assignment;
- SQ publication and doorbells;
- grant and done control WQEs;
- SQ head, CQ target, completion-count, and outstanding-row updates; and
- remote admission or final-CQ handling.

`WaitInboundDone()` already treats the local source rank as ready. Self does
not publish a local done token.

## Deleted Legacy Selection State

The active kernel will delete the declarations, allocations, initialization,
and unreachable implementation associated with the old compare/gather path:

- `dstRankBuf_`;
- `slotIndexBuf_`;
- `selectedIndexBuf_`;
- `compareMaskBuf_`;
- the old 4 KiB `relayBuf_`;
- `descriptorBuf_`;
- `SelectPeer()`;
- `CreateVecIndex`, `Compares`, and `GatherMask` selection code;
- the selected-index-based Self loop; and
- obsolete descriptor append code.

These elements are deleted from the active source rather than moved to another
inactive block. The experimental back file remains outside this cleanup and is
not added to the build.

## Continuous WQE Issue Region

The maximum selector output remains 256 payload WQEs. The 3:1 distribution has
the following maxima:

```text
six-port payload maximum: 192
two-port payload maximum:  64
```

Each QP lane may append at most two control WQEs: grant and done. The issue
capacities therefore become:

```cpp
constexpr uint32_t kSixPortPayloadCapacity = 192U;
constexpr uint32_t kTwoPortPayloadCapacity = 64U;
constexpr uint32_t kControlWqesPerLane = 2U;
constexpr uint32_t kSixPortIssueCapacity = 194U;
constexpr uint32_t kTwoPortIssueCapacity = 66U;
constexpr uint32_t kTotalIssueCapacity = 260U;
```

One `TBuf` owns all 260 consecutive 64-byte WQEs:

```cpp
TBuf<QuePosition::VECCALC> wqeIssueBuf_;

pipe_->InitBuffer(wqeIssueBuf_, kTotalIssueCapacity * kWqeBytes);
```

The physical UB layout is fixed:

```text
entry 0                                               entry 259
+--------------------------------------------------------------+
| six-port region: 194 entries | two-port region: 66 entries   |
| payload capacity 192 + 2     | payload capacity 64 + 2       |
+--------------------------------------------------------------+
^ entry 0                       ^ entry 194
```

The two logical views are derived from the one allocation:

```cpp
LocalTensor<uint8_t> allIssue = wqeIssueBuf_.Get<uint8_t>();
LocalTensor<uint8_t> sixPortIssue = allIssue;
LocalTensor<uint8_t> twoPortIssue =
    allIssue[kSixPortIssueCapacity * kWqeBytes];
```

The four control capacities are reserved capacity, not fixed control indices.
For a short final batch, control WQEs are appended immediately after the actual
payload count in that lane:

```text
lane region: [active payload][grant, if any][done][unused capacity]
```

This preserves a compact active prefix for each lane. The fixed boundary at
entry 194 can leave unused space between the active six-port prefix and the
two-port region in a short batch. That space is not a data gap within either
QP submission because the two regions are copied to different SQs.

Operator and peer prefill cover the 192 and 64 payload capacities. Scalar
`AppendControlWqe()` clears and fully constructs each appended control entry.
When a short final batch overwrites an entry inside the payload-capacity part of
a region, the next peer-level prefill restores it as a payload template.

Each lane still normally needs one MTE3 operation for its compact active
prefix. `CopyIssueToSq()` uses two MTE3 operations for that lane only when the
target SQ range wraps at the ring boundary. The continuous 260-entry UB
allocation cannot be published as one MTE3 because the lane regions target two
different SQs.

## Send-Path UB Layout

The proposed active send-path allocation is:

| Buffer | Bytes | Purpose |
| --- | ---: | --- |
| `dstSlotBuf_` | 65,536 | 16,384 inverse-route entries |
| continuous `wqeIssueBuf_` | 16,640 | 260 complete WQEs, split 194/66 |
| `routeEntryBuf_` | 2,048 | Up to 256 compact Self or Remote routes |
| `threadMaxSlotIdxBuf_` | 512 | 128 selector resume cursors |
| `selectStateBuf_` | 32 | Selector atomic and batch state |
| `wqeContextBuf_` | 256 | Operator, peer, and build context |
| `selfCopyQueue_` | 131,072 | Two independent 64 KiB relay buffers |
| **Total** | **216,096** | |

The 216 KiB budget is 221,184 bytes, leaving 5,088 bytes. Compile-time
assertions must prove the total, the 194/66 split, and the maximum payload plus
two controls per lane.

The Self relay is a dedicated allocation. It does not alias the WQE region,
route entries, selector state, or cursor storage. This keeps the Self MTE
lifetime independent of both SIMT and Remote template restoration.

## Self Relay and Dynamic Batch Size

The Self copy uses two 64 KiB relay buffers:

```cpp
constexpr uint32_t kSelfRelayHalfBytes = 64U * 1024U;
constexpr uint32_t kSelfMaxBatchRows = 8U;

TQue<QuePosition::VECIN, 1> selfCopyQueue_;
pipe_->InitBuffer(selfCopyQueue_, 2U, kSelfRelayHalfBytes);
```

`rowBytes_`, not `h_`, is the authoritative transfer size. For BF16 without
additional row padding it is `H * sizeof(bfloat16_t)`, but the copy path must
continue to work when the launch contract supplies another valid row size.

For a row that fits in one relay half:

```cpp
const uint64_t localRowStride = AlignUp(rowBytes_, kUbAlignBytes);
const uint32_t rowsPerBatch = static_cast<uint32_t>(Min(
    static_cast<uint64_t>(kSelfMaxBatchRows),
    static_cast<uint64_t>(kSelfRelayHalfBytes) / localRowStride));
```

The division is evaluated only after checking `rowBytes_ != 0` and proving
that `AlignUp` cannot overflow. `rowsPerBatch` is a computed value and need not
be a power of two. It is capped at eight to avoid creating an unnecessarily
large group of non-contiguous DMA commands for small rows.

Examples for BF16 rows are:

| H | `rowBytes_` | Computed rows in 64 KiB | Active batch rows |
| ---: | ---: | ---: | ---: |
| 3,584 | 7,168 | 9 | 8 |
| 4,096 | 8,192 | 8 | 8 |
| 7,168 | 14,336 | 4 | 4 |
| 8,192 | 16,384 | 4 | 4 |
| 16,384 | 32,768 | 2 | 2 |
| 32,768 | 65,536 | 1 | 1 |

Each row occupies `localRowStride` bytes in UB, but MTE2 and MTE3 copy exactly
`rowBytes_` logical bytes. Padding between relay rows is not written back to GM.

## Oversized-Row Fallback

If `localRowStride > kSelfRelayHalfBytes`, a complete row cannot reside in one
relay half and `rowsPerBatch` would be zero. The implementation must not reject
the shape merely for that reason and must not return to the old serialized
4 KiB helper.

Instead, flatten each selected row into ordered copy tasks:

```text
(route index, row byte offset, tile bytes)
```

where:

```cpp
tileBytes = Min(rowBytes_ - rowOffset, kSelfRelayHalfBytes);
```

The source and destination for a tile retain the same byte offset:

```cpp
source = workspace_ + sourceSlotIndex * rowBytes_ + rowOffset;
target = scratch_ + targetSlot * rowBytes_ + rowOffset;
```

The same two-buffer pipeline processes these tiles. All tiles for a route must
be copied, but no ordering relationship is required between different target
rows beyond completion before `SendSelfStep()` returns.

## Copy Helper Responsibilities

The old per-row `CopyBytesGmToGm()` helper is not the top-level Self copy
interface. The new helper consumes the current compacted route batch:

```cpp
__aicore__ inline bool CopySelfRouteBatch(uint32_t selectedCount);
```

Its responsibilities are:

1. Read `selectedCount` entries from `routeEntryBuf_`.
2. Compute the dynamic row grouping or oversized-row tile sequence.
3. Issue GM-to-UB copies into one relay buffer.
4. Publish the filled relay buffer to the MTE3 consumer.
5. While MTE3 drains the pending buffer, issue the next MTE2 group into the
   other buffer.
6. Wait before reusing a buffer whose MTE3 is still active.
7. Drain the last pending MTE3 before returning.

No Self-specific SIMT address-builder is introduced initially. The MTE commands
are issued by scalar code, and each route needs only two address multiply-adds.
An extra SIMT launch and cross-model barrier would need measured evidence before
being added.

## MTE2/MTE3 Pipeline

For complete rows, one buffer group contains up to `rowsPerBatch` non-contiguous
rows. The logical schedule is:

```text
relay 0: [MTE2 group 0]                 [MTE3 group 0]
relay 1:                  [MTE2 group 1]                 [MTE3 group 1]
relay 0:                                   [MTE2 group 2] ...
```

Each group issues one `DataCopyPad` per active row because the route-derived GM
addresses are generally non-contiguous. Batching reduces synchronization and
buffer-turnover cost; it does not make the scatter rows one DMA range.

The required dependencies are:

1. MTE2 completion for a relay buffer before MTE3 reads that buffer.
2. MTE3 completion for a relay buffer before it is freed or reused by MTE2.
3. Completion of the final MTE3 before `CopySelfRouteBatch()` returns.
4. Completion of the route batch before `SelectPeerRoutes()` is invoked again
   and overwrites `routeEntryBuf_`.
5. Completion of all Self copies before reduction reads `scratch_`.

The implementation should use the target-supported `TQue` ownership and event
handoffs rather than a `PIPE_ALL` barrier per row. The existing dispatch helper
`CopyContiguousBytesGmToGmPipelined()` is a useful queue-lifetime reference, but
it cannot be called directly because Self source and destination rows are a
scatter operation.

Exact `TQue`, `DataCopyPad`, `DataCopyExtParams`, and HardEvent usage must be
confirmed by compiling against `/home/pkg/b150`. In particular, GM-to-UB and
UB-to-GM overloads have direction-specific signatures and tail semantics; a
working overload in another CANN version is not treated as proof for b150.

## SendSelfStep Flow

`SendSelfStep(peer)` follows the same chunk and resume structure as Remote:

```cpp
state->curWqeNum = 0U;

for (each dstLocal chunk) {
    LoadSelectionChunk(chunkStart, chunkElements);

    bool firstPass = true;
    uint32_t pausedThreadCount = 0U;
    do {
        const uint32_t selectedCount = SelectPeerRoutes(
            peer, chunkStart, chunkElements,
            firstPass, pausedThreadCount);

        if (selectedCount != 0U) {
            CopySelfRouteBatch(selectedCount);
        }
        firstPass = false;
    } while (pausedThreadCount != 0U);
}
```

Unlike Remote, Self does not submit an empty final batch. There are no control
WQEs to publish when no Self routes match.

The peer argument is expected to equal `rank_`. Safety-enabled builds should
reject or report a violation rather than silently copying routes for another
peer through the Self path.

## Remote Submit Behavior After WQE Resize

Remote payload construction continues to compact each lane independently. The
active prefix of a final lane submission is:

```text
[payload entries][grant when another step exists][done]
```

For example, 99 selected payload WQEs are distributed according to
`sequenceBase & 3`. Depending on the phase, the split is 75/24 or 74/25. Each
lane appends its one or two controls to its own compact prefix. With no SQ wrap,
the submit therefore issues two MTE3 calls: one to each lane's SQ. It does not
issue extra MTE3 calls because of unused capacity in the continuous UB region.

`CopyIssueToSq()` must accept the two logical tensor views and copy only the
active count. Its existing ring-wrap split remains unchanged. Doorbells are
rung only after all MTE3 writes for both lanes have completed.

## Profiling

The selection metrics retain their current meaning:

- `SELECTION_LOAD`: GM-to-UB load of the inverse-route chunk;
- `SELECTION_SELECT`: SIMT peer filtering, division/modulo decode, route
  compaction, and selector synchronization.

The old `SELF_ROUTE_DECODE` metric must not claim that a separate scalar decode
still exists. It should either remain zero/reserved for profile ABI stability or
be renamed only with an explicit profile-version change.

`SELF_COPY` measures the complete Self data-movement section, including scalar
address generation, MTE2, MTE3, queue/event handoffs, and final drain. If more
detail is temporarily needed during performance work, profiling may split it
into read, write, and wait components without putting timestamp collection into
the normal non-profiling build.

## Correctness and Boundary Conditions

The implementation must preserve these invariants:

- `selectedCount <= 256`;
- every selected route is copied exactly once;
- no unselected or `-1` route is copied;
- each copy reads exactly `rowBytes_` from its source row;
- each copy writes exactly `rowBytes_` to its target row;
- aligned UB stride padding never reaches GM;
- row and tile offsets use 64-bit arithmetic;
- the relay buffer is not reused before its MTE3 completes;
- `routeEntryBuf_` is not overwritten while a Self copy still consumes it;
- Self never changes `issuedRows_`, lane heads, CQ targets, or completion
  counts;
- each lane's payload plus controls fits 194 or 66 entries; and
- every WQE is fully constructed in UB and published through MTE3 before its
  SQ doorbell.

Planner is expected to provide the valid inverse-route mapping and unique
target ownership required by Combine V2. Safety checks continue to validate the
encoded destination range when enabled; this change does not add an expensive
hot-path duplicate-target detector.

## Validation Plan

Implementation acceptance requires:

1. Compile-time assertions for the 216 KiB UB total and the continuous 260-WQE
   region.
2. Unit checks for the fixed WQE offsets: six-port at entry 0 and two-port at
   entry 194.
3. Unit checks for maximum payload/control counts: 192+2 and 64+2.
4. Unit checks for dynamic batch rows at representative `rowBytes_`, including
   8, 4, 2, and 1 row cases.
5. Unit checks for a row larger than 64 KiB and its final partial tile.
6. Source guards proving the obsolete compare/gather/descriptor buffers and
   dead path are removed.
7. Source guards proving Self consumes `routeEntryBuf_` and does not add
   `chunkStart` to `sourceSlotIndex` again.
8. Source guards for MTE2-before-MTE3, MTE3-before-buffer-reuse, final drain,
   and SQ MTE3-before-doorbell ordering.
9. A target compile with `/home/pkg/b150` CANN.
10. Focused hardware correctness for no Self routes, one route, short batches,
    more than 128 Self routes, and multiple selector resume passes.
11. Hardware correctness for H values producing 8, 4, 2, and 1 rows per relay
    group, plus an oversized-row case if the public shape contract permits it.
12. Profiling that reports selection and Self copy separately.
13. Performance comparison with the same warmup, loop count, shape, rank count,
    and device placement used for the current SIMT WQE baseline.

Correctness is required again once `SendSelfStep()` is restored; the prior
`self_only_failed` allowance is no longer an acceptable final result.

## Review Decisions Recorded

- Reuse the current SIMT selector and `RouteEntry` representation for Self.
- Delete old compare/gather/descriptor UB allocations and implementation.
- Allocate a dedicated two-half Self relay; do not alias other send storage.
- Use two 64 KiB relay halves.
- Compute rows per batch from relay-half bytes divided by aligned runtime row
  bytes, capped at eight.
- Use the same ping-pong pipeline for an oversized row by splitting it into
  tiles.
- Keep Self address generation scalar for the first implementation.
- Replace the two oversized WQE buffers with one physically continuous
  260-entry allocation.
- Split that allocation into fixed 194- and 66-entry QP regions.
- Reserve capacity for at most two control WQEs per QP and append controls
  immediately after the actual payload prefix.
- Preserve separate MTE3 publication to the two SQs and the existing ring-wrap
  split.
- Validate the exact data-copy and queue synchronization APIs against CANN
  `/home/pkg/b150` before hardware testing.
