# MoonEP Combine V2 Vector Selection and Direct WQE Build Design

Status: Implemented and statically reviewed; build and runtime validation not run

Date: 2026-08-14

## Goal

Replace the contended SIMT route selector in MoonEP Combine V2 with vector
comparison and `GatherMask`, then let the existing 128-thread SIMT WQE builder
consume compacted source indices directly.

The new path must:

1. remove every per-route `asc_atomic_add` from route selection;
2. remove `MoonEpCombineV2RouteEntry` and its construction SIMT function;
3. preserve the existing `dstLocal` encoding for arbitrary valid `NvS`;
4. keep at most 128 payload WQEs in one submission batch;
5. ring each non-empty QP once per batch;
6. submit the final done/end and grant controls even when a Remote peer has no
   payload; and
7. migrate Remote and Self to the same compacted-index representation.

This is a kernel-internal optimization. It does not change the public MoonEP
API, Planner route ABI, workspace layout, schedule, QP ownership, or Host launch
arguments.

## Existing Route Contract

The inverse route table remains:

```text
dstLocal[expertRecvSlot] = srcRank * NvS + token * K + topk
```

For peer `p`, define:

```cpp
peerBase = static_cast<uint64_t>(p) * slots;
peerEnd = peerBase + slots;
```

An entry belongs to `p` exactly when:

```text
encoded >= 0 && peerBase <= encoded < peerEnd
```

For a selected entry:

```cpp
sourceSlotIndex = chunkStart + chunkIndex;
targetSlot = static_cast<uint64_t>(dstSlot[chunkIndex]) - peerBase;
```

`NvS` does not have to be a power of two. The new path uses comparisons and
subtraction, not shift/mask decoding.

## Key Representation Decision

`GatherMask` compacts `int16_t chunkIndex` values, not encoded `dstSlot`
values.

Compacting only `dstSlot` would lose the original array position. That position
is required to derive the local workspace source address and is also required
by Self copy. A compacted chunk-relative index preserves both addresses:

```cpp
const uint32_t relativeIndex =
    static_cast<uint16_t>(selectedIndex[densePosition]);
const uint32_t sourceSlotIndex = chunkStart + relativeIndex;
const int32_t encoded = dstSlot[relativeIndex];
const uint32_t targetSlot = static_cast<uint32_t>(
    static_cast<uint64_t>(encoded) - peerBase);
```

The selection chunk contains at most 8192 entries, so every valid relative
index is in `[0, 8191]` and is exactly representable by `int16_t`.

## Per-Core Index Initialization

Each AICore initializes the chunk-relative source index tensor once in
`InitBuffers()`:

```cpp
CreateVecIndex(slotIndexBuf_.Get<int16_t>(),
    static_cast<int16_t>(0), kSelectionChunkRows);
PipeBarrier<PIPE_V>();
```

The tensor is the reusable sequence `[0, 1, ..., 8191]`. It is not recreated
for a new peer, chunk, selector batch, or schedule step. Every chunk uses the
same relative index domain, including the final short chunk. `GatherMask`
receives `chunkElements`, so entries outside a short tail are not selected.

Initialization occurs once per core per kernel invocation. The implementation
must establish the Vector dependency before the first `GatherMask` reads the
tensor.

## UB Layout

Halve the selection chunk from 16,384 to 8192 entries:

```cpp
constexpr uint32_t kSelectionChunkRows = 8192U;
constexpr uint32_t kPayloadBatchRows = 128U;
```

Use these simultaneously live buffers:

| Buffer | Size | Purpose |
| --- | ---: | --- |
| original `dstSlot` | 32,768 B | 8192 `int32_t` encoded destinations |
| reusable `slotIndex` | 16,384 B | 8192 `int16_t` relative indices |
| compacted `selectedIndex` | 16,384 B | up to 8192 selected indices |
| lower-bound mask | 1,024 B | one packed bit per input entry |
| upper-bound mask | 1,024 B | one packed bit per input entry |
| WQE issue region | 8,448 B | 98 six-port plus 34 two-port WQEs |
| WQE context | 256 B | peer, chunk, batch, QP, and address fields |
| Self relay queue | 131,072 B | existing two 64 KiB Self relays |
| **Total** | **207,360 B** | |

The 216 KiB UB budget is 221,184 bytes, leaving 13,824 bytes. Compile-time
assertions must prove every component and the total.

The packed-mask size is:

```cpp
maskBytes = AlignUp(CeilDiv(kSelectionChunkRows, 8U), 32U); // 1024
maskUint16Elements = maskBytes / sizeof(uint16_t);           // 512
```

The two compare outputs are combined in place into one mask buffer. The exact
CANN 9.1 `And` overload must be confirmed to permit destination/source aliasing.
If that overload does not permit it, allocate a third 1024-byte result mask;
the resulting 208,384-byte total still fits UB with 12,800 bytes remaining.

## Vector Selection Flow

For each peer and each resident `dstSlot` chunk:

1. Copy `chunkElements` entries from GM to the 32 KiB `dstSlot` UB buffer.
2. Establish the MTE2-to-Vector dependency.
3. Compute `peerBase` and `peerEnd` in `uint64_t` scalar arithmetic.
4. Produce the inclusive lower-bound mask with `CMPMODE::GE`.
5. Produce the exclusive upper-bound mask with `CMPMODE::LT`.
6. Combine the packed masks with bitwise `And`.
7. Use `GatherMask` to compact the reusable `int16_t slotIndex` tensor into
   `selectedIndex`.
8. Establish the Vector-to-Scalar dependency before reading `selectedCount`.
9. Establish the Vector-to-SIMT dependency before the WQE builder reads
   `selectedIndex` and `dstSlot`.

Conceptually:

```cpp
Compares(lowerMask, dstSlot, static_cast<int32_t>(peerBase),
    CMPMODE::GE, chunkElements);
Compares(upperMask, dstSlot, static_cast<int32_t>(peerEnd),
    CMPMODE::LT, chunkElements);
PipeBarrier<PIPE_V>();

And(lowerMask, lowerMask, upperMask, packedMaskElements);
PipeBarrier<PIPE_V>();

uint64_t selectedCount64 = 0U;
GatherMask(selectedIndex, slotIndex,
    lowerMask.ReinterpretCast<uint16_t>(), true, chunkElements,
    {1U, 1U, 0U, 0U}, selectedCount64);
SyncFunc<HardEvent::V_S>();
const uint32_t selectedCount = static_cast<uint32_t>(selectedCount64);
```

`packedMaskElements` is the fixed full-buffer count of 512 `uint16_t`
elements, not `chunkElements`. `GatherMask` still receives the actual
`chunkElements`, so stale or padding bits beyond a short tail cannot produce a
selected index. The implementation does not depend on either comparison
clearing bytes outside its logical result.

The implementation must use the exact CANN 9.1 signatures and supported mask
tensor view. The existing repository has prior compiled usage of
`Compares<int32_t>`, `GatherMask<int16_t>`, and a packed mask reinterpreted as
`uint16_t`; `GE`, `LT`, packed-mask `And`, aliasing, and tail count remain
compile-validation gates for the selected CANN 9.1 installation.

### Integer Boundary Handling

`dstSlot` elements are `int32_t`, while peer bounds are calculated in
`uint64_t`. Bounds must not be narrowed before their range is checked:

- if `peerBase > INT32_MAX`, the selected set is empty;
- otherwise the lower comparison uses `static_cast<int32_t>(peerBase)`;
- if `peerEnd <= INT32_MAX`, perform the upper `LT` comparison and `And`;
- if `peerEnd > INT32_MAX`, every `int32_t` value passing the lower comparison
  is already below `peerEnd`, so the lower mask is the final mask.

For the normal route domain where both bounds fit `int32_t`, the hot path uses
the requested two consecutive comparisons. `GE` is required at the lower
bound; `GT` would incorrectly drop `targetSlot == 0`.

No explicit invalid-entry comparison is needed because every `peerBase` is
non-negative and the lower-bound comparison rejects `-1`.

## Direct SIMT WQE Construction

Delete `MoonEpCombineV2SelectPeerRoutesVf`. Retain a single 128-thread payload
builder and change its inputs from `MoonEpCombineV2RouteEntry` to:

- resident `dstSlot`;
- compacted `selectedIndex`;
- `chunkStart`;
- `peerBase`;
- `batchOffset` and `batchCount`;
- the existing local/remote row bases, row size, sequence phase, and SQ heads.

Each thread creates at most one payload WQE:

```cpp
const uint32_t task = static_cast<uint32_t>(threadIdx.x);
if (task >= context->batchCount) {
    return;
}

const uint32_t densePosition = context->batchOffset + task;
const uint32_t relativeIndex = static_cast<uint16_t>(
    selectedIndex[densePosition]);
const uint32_t sourceSlotIndex = context->chunkStart + relativeIndex;
const int32_t encoded = dstSlot[relativeIndex];
const uint32_t targetSlot = static_cast<uint32_t>(
    static_cast<uint64_t>(encoded) - context->peerBase);
```

The builder then derives addresses directly:

```cpp
localAddr = localRowBase +
    static_cast<uint64_t>(sourceSlotIndex) * rowBytes;
remoteAddr = remoteRowBase[lane] +
    static_cast<uint64_t>(targetSlot) * rowBytes;
```

The existing 3:1 QP assignment and global sequence phase remain unchanged.
`sequenceBase` is the peer-lifetime number of payload rows already issued;
`sequencePhase = sequenceBase & 3U`. The builder's `task` remains batch-local.

There is no builder loop with a stride of 128. A second batch advances
`batchOffset`; it never rebuilds the first 128 entries.

## Payload Batching and Doorbells

For one compacted chunk:

```cpp
for (uint32_t batchOffset = 0U;
    batchOffset < selectedCount;
    batchOffset += kPayloadBatchRows) {
    const uint32_t batchCount = Min(
        selectedCount - batchOffset, kPayloadBatchRows);
    // Build and submit this batch.
}
```

`Compares`, `And`, and `GatherMask` run once per peer/chunk, not once per WQE
batch.

One batch contains at most 128 payload WQEs. Because 128 is divisible by the
four-position 3:1 sequence period, the maximum payload counts are exactly:

```text
six-port payload: 96
two-port payload: 32
```

The final batch may append at most grant and done/end controls to each QP:

```text
six-port issue capacity: 96 + 2 = 98
two-port issue capacity: 32 + 2 = 34
total issue capacity: 132 WQEs = 8448 bytes
```

Operator-constant and peer-constant WQE prefill loops must use these new
payload capacities. They prefill exactly 96 six-port and 32 two-port payload
entries; the two control slots per QP remain scalar-built on the final batch.

After MTE3 has published each QP's active WQEs to its SQ, ring that QP's
doorbell exactly once if its active count is nonzero. A batch can therefore
ring zero, one, or two doorbells. There is no single doorbell shared by both
QPs.

The implementation must preserve the existing order:

```text
SIMT/scalar WQE fields complete
  -> scalar/SIMT-to-MTE3 dependency
  -> MTE3 copy to each SQ
  -> MTE3 completion dependency
  -> publish SQ head/count state
  -> st_dev doorbell for each non-empty QP
```

## Mandatory Final Remote Submission

Exactly one submission is marked `finalBatch` for every Remote peer, after all
selection chunks and all payload batches for that peer have been consumed.

- If the final chunk has selected entries, its last payload batch is final.
- If the final chunk has no selected entries, issue a synthetic zero-payload
  final batch.
- If the peer has no payload in any chunk, the synthetic final batch is still
  mandatory.

The final submission preserves the existing control protocol for each QP:

1. append or locally publish the grant according to the existing successor
   rule;
2. append the peer done/end WQE with ordered completion;
3. include these controls in the SQ copy and head update; and
4. ring each now-non-empty QP once.

Thus a zero-token Remote peer still sends its done/end signal and grant signal.
Skipping this final submission can leave another rank waiting indefinitely.

The final decision must not be based only on `selectedCount`. It is:

```cpp
finalBatch = isLastChunk &&
    (batchOffset + batchCount == selectedCount);
```

The zero-selection case is handled explicitly because it has no natural loop
iteration.

## Self Migration

Self uses the same vector comparison, mask combination, and compacted
`selectedIndex` representation. It no longer reads `MoonEpCombineV2RouteEntry`.

For each selected Self row:

```cpp
relativeIndex = static_cast<uint16_t>(selectedIndex[densePosition]);
sourceSlotIndex = chunkStart + relativeIndex;
encoded = dstSlot[relativeIndex];
targetSlot = static_cast<uint64_t>(encoded) - peerBase;

source = workspace_ + sourceSlotIndex * rowBytes_;
target = scratch_ + targetSlot * rowBytes_;
```

The existing row-batched and tiled double-buffered Self MTE pipelines remain.
They may consume all compacted indices for the current chunk through their
existing internal copy batches; the 128-row payload-WQE limit does not apply to
Self because Self creates no payload WQEs.

After all Self chunks are copied, preserve the existing `SubmitSelfGrant()`
behavior. Self does not send a Remote done WQE: `WaitInboundDone()` already
treats the local source as complete. This design changes the Self route
representation, not its control protocol.

## Removed State and Code

Delete all of the following from the active Combine V2 implementation:

- `MoonEpCombineV2RouteEntry`;
- `MoonEpCombineV2SelectState`;
- `MoonEpCombineV2SelectPeerRoutesVf`;
- `routeEntryBuf_` and its capacity/size constants;
- `threadMaxSlotIdxBuf_` and its cursor helpers;
- `selectStateBuf_`;
- `curWqeNum`, `batchBase`, `batchSelected`, and `pausedThreadCount`;
- the `simt_api/device_atomic_functions.h` include when no other use remains;
- selector first-pass/resume loops; and
- source guards and tests that require atomic route reservation.

Add or restore:

- the reusable `slotIndexBuf_` initialized once per core;
- `selectedIndexBuf_`;
- two packed compare-mask buffers;
- vector peer selection using `GE`, `LT`, `And`, and `GatherMask`; and
- a direct-index SIMT WQE builder with `batchOffset` and `batchCount`.

There must be no inactive `#if 0` copy of the deleted atomic selector in the
active kernel header.

## Synchronization Contract

This kernel mixes MTE, Vector, scalar, and SIMT execution. The implementation
must provide explicit handoffs for:

1. MTE2 completion before Vector compares read `dstSlot`;
2. compare completion before `And` reads both masks;
3. `And` completion before `GatherMask` reads the final mask;
4. `GatherMask` completion before scalar reads `selectedCount`;
5. `GatherMask` completion before SIMT reads `selectedIndex`;
6. SIMT WQE completion before MTE3 reads the issue region; and
7. MTE3 SQ publication before any doorbell store.

Use the dependency mechanisms supported by the selected CANN 9.1 programming
model. Do not assume that a Vector pipeline barrier establishes SIMT visibility
or that a SIMT barrier establishes MTE visibility.

The original `dstSlot` buffer and compacted index buffer remain resident until
all Remote WQE batches or all Self copies for that peer/chunk are complete.
MTE2 must not overwrite `dstSlot` with the next chunk earlier.

## Profiling Semantics

Retain the current metric categories with updated meanings:

- `selection_load`: GM-to-UB `dstLocal` movement and its handoff;
- `selection_select`: the two compares, packed-mask `And`, `GatherMask`, and
  selected-count handoff;
- `remote_wqe_build`: direct-index SIMT construction for all 128-row batches;
- `remote_submit`: control WQEs, MTE3 SQ publication, state updates, and
  doorbells; and
- `self_copy`: direct-index Self MTE copies.

Performance comparison must report both `selection_select` and total Combine
V2 time. Removing atomics can move the bottleneck into vector selection,
builder launch/barriers, MTE3 submission, or the increased number of 128-row
batches.

## Validation Plan

Implementation validation must cover:

1. Source guards rejecting `MoonEpCombineV2RouteEntry`,
   `MoonEpCombineV2SelectPeerRoutesVf`, `asc_atomic_add`, selector cursors, and
   selection state.
2. Source guards requiring one-time `CreateVecIndex`, two bound comparisons,
   packed-mask `And`, `GatherMask`, and direct-index WQE construction.
3. CANN 9.1 compilation of the exact `Compares`, `And`, and `GatherMask`
   overloads, including supported comparison modes, tensor types, aliasing,
   counts, and tails.
4. UB compile-time assertions for the normal two-mask layout and any selected
   non-alias fallback.
5. No-match, one-match, all-match, and mixed invalid `-1` chunks.
6. Lower-bound matches where `targetSlot == 0` and upper-bound exclusion where
   `encoded == peerEnd`.
7. A final tail smaller than 8192 and stale bits outside `chunkElements`.
8. Exactly 128, 129, and more than 256 selected entries without duplicates or
   omissions across `batchOffset` values.
9. Multiple chunks proving `sourceSlotIndex = chunkStart + relativeIndex`.
10. Non-power-of-two `NvS`, including the existing `NvS = 2040` shape.
11. Boundary handling when `peerBase` or `peerEnd` exceeds `INT32_MAX`.
12. Correct 3:1 QP assignment and sequence phase across chunk and batch
    boundaries.
13. Zero-payload Remote peers still publishing grant and done/end controls on
    both QPs.
14. Final payload and control capacities of 98 and 34 WQEs.
15. Self row-batch and large-row tile paths using direct selected indices.
16. Doorbells occurring only after MTE3 and exactly once per non-empty QP per
    batch.
17. Hardware correctness for Remote and Self before performance comparison.
18. Hardware profiling against the atomic-selector baseline with identical
    shape, rank count, warmup, iteration count, and profiling configuration.

Host and source tests do not prove Vector/SIMT handoff correctness or hardware
performance. Performance conclusions remain pending until the CANN 9.1 A5
kernel is built and exercised on hardware.

## Superseded Decisions

This document supersedes the route-selection and payload-builder portions of
`2026-08-12-moonep-combine-v2-simt-wqe-design.md` and the RouteEntry reuse
portions of `2026-08-12-moonep-combine-v2-self-copy-design.md`.

The following earlier decisions are explicitly reversed:

- route selection is Vector rather than SIMT;
- no peer-lifetime atomic WQE counter is retained;
- no per-thread selector cursor or overflow-resume protocol is retained;
- the compacted representation is `int16_t chunkIndex`, not RouteEntry; and
- payload selection for one chunk completes before its 128-row WQE batches
  begin.

Grant, done/end, CQ, Self relay, 3:1 QP, registered-memory, SQ publication, and
doorbell-ordering contracts remain unchanged except for the smaller per-batch
issue capacities defined here.
