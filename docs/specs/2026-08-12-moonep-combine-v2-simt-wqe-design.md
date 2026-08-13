# MoonEP Combine V2 SIMT WQE Construction Design

Status: Implemented; final `/home/pkg/b150` validation pending

Baseline: `80f184e perf: checkpoint combine v2 with self-copy disabled`

The implementation has passed the focused Host tests, a CANN b131 target
compile, and bounded 8P hardware runs on `141.61.49.223`. The requested b150
compile remains pending because `/home/pkg/b150` is not installed or mounted on
that host as of 2026-08-12. The b131 evidence must not be treated as b150 proof.

## Goal

Reduce the WQE preparation cost in the MoonEP Combine V2 send path. Replace
the current vector selection, scalar route decoding, scalar descriptor append,
and descriptor-driven WQE construction with the following staged flow:

1. Prefill operator-constant payload WQE fields once after send UB allocation.
2. Prefill peer- and QP-constant fields once before sending to each peer.
3. Load up to 64 KiB of `dstLocal` from GM to UB.
4. Select and compact routes for the current peer with a SIMT function.
5. Build the selected payload WQEs evenly with a second SIMT function.
6. Append grant and done control WQEs with the existing scalar path.
7. Copy the active WQEs from UB to SQ with MTE3, then ring the doorbells.

The first implementation keeps the existing `dstLocal` encoding and keeps
self-copy disabled. It is a payload WQE preparation optimization, not a change
to the public Combine V2 interface or to the Planner forward-route ABI.

## Scope

The implementation will change the Combine V2 kernel send path and its focused
tests. It will:

- introduce four SIMT functions for operator prefill, peer prefill, route
  selection, and payload WQE construction;
- allocate a 64 KiB `dstLocal` UB buffer;
- allocate fixed six-port and two-port issue buffers for 256 and 128 WQEs;
- allocate a route buffer for at most 256 payload entries;
- allocate one 32-byte selection state and 128 per-thread cursors;
- preserve the existing 3:1 six-port/two-port payload distribution;
- preserve scalar construction of grant and done WQEs;
- preserve MTE3 publication to SQ and `st_dev` doorbell ordering;
- retain the replaced selection and descriptor code under `#if 0` instead of
  deleting it; and
- make `SendSelfStep()` return success immediately.

This design does not:

- change `dstLocal` to a packed bit-field format;
- change Planner's public `dst` route encoding;
- optimize the remaining integer division and modulo;
- restore or optimize self-copy;
- change CQ generation, grant/done semantics, or QP ownership; or
- expose a new runtime or host parameter.

## Existing Route Contract

Planner's public forward route remains:

```text
dst[token * K + topk] = expertRank * NvS + expertRecvSlot
```

Combine V2 consumes the internal inverse route table:

```text
dstLocal[expertRecvSlot] = srcRank * NvS + token * K + topk
```

For peer `p`, the Combine V2 send path precomputes the encoded interval:

```cpp
peerBase = static_cast<uint64_t>(p) * slots;
peerEnd = peerBase + slots;
```

The selector accepts a non-negative entry when
`peerBase <= encoded < peerEnd`, then obtains `targetSlot` with
`encoded - peerBase`. `-1` remains the invalid or padding value. This removes
per-entry division and per-match modulo without requiring `NvS` to be a power
of two, so shapes such as `NvS = 2040` retain the original encoding contract.

The compacted route contains:

```cpp
struct MoonEpCombineV2RouteEntry {
    uint32_t sourceSlotIndex;
    uint32_t targetSlot;
};
```

`sourceSlotIndex` is the absolute row in the local registered workspace that
contains the expert output. It is also the index of the selected `dstLocal`
entry. `targetSlot` is the row in the peer's receive scratch that must receive
that output, normally the peer's `token * K + topk` route index.

The payload builder derives the addresses as:

```cpp
localAddr = localRowBase +
    static_cast<uint64_t>(route.sourceSlotIndex) * rowBytes;
remoteAddr = remoteRowBase +
    static_cast<uint64_t>(route.targetSlot) * rowBytes;
```

The selector computes modulo only after an entry's decoded peer matches the
current peer. Division remains necessary for every valid entry examined for a
peer. Its cost must be measured separately after this refactor; changing the
route encoding is explicitly deferred.

## Single Tuning Constant

The selector threshold and both SIMT launch widths are derived from one
compile-time constant:

```cpp
constexpr uint32_t kMoonEpCombineV2PayloadBatchRows = 128U;

constexpr uint32_t kMoonEpCombineV2SelectorThreads =
    kMoonEpCombineV2PayloadBatchRows;
constexpr uint32_t kMoonEpCombineV2BuilderThreads =
    kMoonEpCombineV2PayloadBatchRows;
constexpr uint32_t kMoonEpCombineV2MaxSelectedPayloadWqes =
    2U * kMoonEpCombineV2PayloadBatchRows;
```

The constant is not exposed through the public API, host launch arguments, an
environment variable, or a benchmark option. A later experiment changes only
this definition from 128 to 64 or 32 and recompiles the operator.

The active design supports values up to 128:

| Batch rows | Selector threads | Builder threads | Maximum payload WQEs |
| ---: | ---: | ---: | ---: |
| 128 | 128 | 128 | 256 |
| 64 | 64 | 64 | 128 |
| 32 | 32 | 32 | 64 |

The UB capacities remain allocated for the maximum 128-row configuration when
the tuning constant is reduced. Compile-time assertions must reject zero and
values above 128 and must prove that both QP lane buffers can hold the maximum
payload distribution plus two control WQEs.

## Why Selection and WQE Construction Are Separate

The uniform hardware probe creates destinations with:

```cpp
targetRank = slot % world;
```

A selector that scans with:

```cpp
index = threadIdx.x + iteration * selectorThreads;
```

has highly uneven matches for a fixed peer:

- at 128P with 128 selector threads, one thread sees all entries for a peer;
- at 16P, eight threads see entries for a peer; and
- at 8P, sixteen threads see entries for a peer.

Constructing a WQE immediately in the matching selector thread would therefore
make the 128P uniform case a single-thread WQE builder. The first SIMT function
only selects and compacts route information. The second assigns compacted
entries by route-buffer index, so WQE construction is balanced even when route
selection is not.

The user-visible correctness contract permits WQEs for the same peer to be
issued in a different order. Each route must retain its original
`sourceSlotIndex` and `targetSlot`; grant and done controls must still be
published after all payload WQEs for the peer.

## UB Layout

The active send-path allocations are:

| Buffer | Size at batch rows 128 | Purpose |
| --- | ---: | --- |
| `dstSlotBuf_` | 64 KiB | 16,384 `int32_t` inverse-route entries loaded from GM |
| `sixPortIssueBuf_` | 16 KiB | 256 complete 64-byte WQEs |
| `twoPortIssueBuf_` | 8 KiB | 128 complete 64-byte WQEs |
| `routeEntryBuf_` | 2 KiB | 256 `sourceSlotIndex/targetSlot` pairs |
| `threadMaxSlotIdxBuf_` | 512 B | One `uint32_t` cursor for each of 128 threads |
| `selectStateBuf_` | 32 B | Peer counter and per-invocation selector state |
| prefill/build context | implementation-sized, aligned | Operator, peer, QP, head, and row fields |

The issue-buffer sizes are fixed capacities, not the number copied to SQ. Each
submission copies only its active payload and control WQE counts.

The following old buffers and their active allocations are disabled but
retained under `#if 0` for comparison during this optimization:

- `dstRankBuf_`;
- `slotIndexBuf_`;
- `selectedIndexBuf_`;
- `compareMaskBuf_`;
- `descriptorBuf_`; and
- the old relay/self-copy buffer when it has no other active consumer.

Keeping those old allocations active while expanding the issue buffers would
exceed the current 216 KiB UB budget. The new active allocation must retain a
compile-time total-UB assertion after all context and alignment sizes are
finalized.

## Selection State

The 32-byte core-local UB state is shared by the selector threads on one AI
Core. It is not shared across AI Cores:

```cpp
struct alignas(32) MoonEpCombineV2SelectState {
    uint32_t curWqeNum;
    uint32_t batchBase;
    uint32_t batchSelected;
    uint32_t pausedThreadCount;
    uint32_t reserved[4];
};
```

The exact field access may be scalar arguments plus UB words, but the active
storage must remain 32 bytes and preserve these semantics:

- `curWqeNum` is cumulative for one peer and is cleared only before processing
  that peer, not before every selector invocation or GM chunk;
- `batchBase` snapshots `curWqeNum` before one selector invocation;
- `batchSelected` is `curWqeNum - batchBase` after the SIMT function returns;
- `pausedThreadCount` is reset before each selector invocation and counts
  threads that stopped at the overflow threshold while more entries remain in
  their strided range.

`curWqeNum` is the only per-route atomic counter. According to the target
`asc_atomic_add` contract, the API returns the value that existed before the
addition. The route-buffer index is therefore:

```cpp
const uint32_t old = asc_atomic_add(&state->curWqeNum, 1U);
const uint32_t routeIndex = old - state->batchBase;
```

The implementation must compile this exact use against the `/home/pkg/b150`
CANN headers before relying on it. The official API reference used for this
contract is `asc_atomic_add`:

<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/zh/api/SIMT-API/%E5%8E%9F%E5%AD%90%E6%93%8D%E4%BD%9C/asc_atomic_add.md>

## Per-Thread Resume Cursors

`threadMaxSlotIdxBuf_` contains 128 `uint32_t` entries. Active selector thread
`tid` exclusively owns entry `tid`, so updating the entry does not require an
atomic max. A thread keeps its last examined absolute `dstLocal` index in a
register and writes it to UB once before returning from the selector.

The cursor is the last examined slot, not only the last matching slot. This is
required to prevent rescanning non-matching entries and to ensure that a resumed
thread neither skips nor duplicates an entry.

For the first invocation over a newly loaded chunk:

```cpp
index = chunkStart + threadIdx.x;
```

Every active thread writes this first candidate to its cursor before the first
invocation returns, even when the candidate is already at or beyond the chunk
end. This initializes cursor entries for short tail chunks in which some
threads have no valid slot to examine. Such an out-of-range cursor remains
exhausted on later invocations over the same chunk.

For later invocations over the same chunk:

```cpp
index = threadMaxSlotIdx[threadIdx.x] +
    kMoonEpCombineV2SelectorThreads;
```

Each thread then advances by `kMoonEpCombineV2SelectorThreads`. The first-pass
flag is explicit; the implementation must not manufacture an initial cursor by
subtracting the stride from an unsigned `chunkStart`, because the first chunk
would underflow.

The cursor and `sourceSlotIndex` are absolute GM indices. `dstSlots` points to
the beginning of the current UB chunk, so the selector reads the encoded value
with the chunk-relative index:

```cpp
encoded = dstSlots[index - chunkStart];
```

An absolute index must never be applied directly to the 64 KiB UB buffer.

When a thread reaches the end of its strided range, it records its last scan
position and returns. For a thread with at least one valid entry this is the last
examined index; for an initially out-of-range tail thread it is the initialized
first candidate. When a thread stops at the overflow threshold and its next
strided index is still inside the chunk, it atomically increments
`pausedThreadCount`. A nonzero `pausedThreadCount` causes another selector
invocation over the same GM-loaded UB chunk. Exhausted threads return
immediately in that invocation, while paused threads resume from their own
cursors.

The `dstLocal` chunk remains resident in UB until every thread has exhausted
its strided range. Only then may MTE2 overwrite the buffer with the next chunk.

## Selector Bound

For every matching route in one invocation, a thread obtains `routeIndex` with
the atomic add, stores the complete `MoonEpCombineV2RouteEntry`, and then tests
the overflow threshold:

```cpp
if (routeIndex >= kMoonEpCombineV2PayloadBatchRows) {
    // The current route has already been stored.
    // Record this thread's cursor, mark it paused if work remains, and exit.
}
```

Before the threshold is reached, exactly `payloadBatchRows` entries can obtain
indices `[0, payloadBatchRows - 1]`. After that point, each selector thread can
store at most one additional entry because its first index at or above the
threshold makes it exit. Therefore:

```text
batchSelected <= payloadBatchRows + selectorThreads
              == 2 * payloadBatchRows
```

At the default value this is at most 256 payload routes. The bound applies to
payload only; grant and done control WQEs are not included.

The single-thread 128P uniform case usually produces 129 selected routes per
full selector invocation: the matching thread stores indices 0 through 128 and
then exits. This is expected. The 256 value is a strict capacity bound, not a
target count that every input must reach.

## SIMT Interfaces

The implementation uses four SIMT functions. Names may be adjusted to match
local style, but their ownership and data flow must remain as specified.

### Operator-Level Prefill

```cpp
__simt_vf__ __aicore__
LAUNCH_BOUND(kMoonEpCombineV2BuilderThreads)
void MoonEpCombineV2PrefillOperatorWqesVf(
    __ubuf__ uint8_t *sixPortWqes,
    __ubuf__ uint8_t *twoPortWqes,
    __ubuf__ const MoonEpCombineV2OperatorFields *fields);
```

This function runs once after the send UB buffers are initialized and fills
fields that do not change between peers or batches:

- zero/default bits for each 64-byte WQE;
- payload opcode;
- payload flag and `nf` defaults;
- `inlineMsgLen` and `sgeNum`;
- SGE length from `rowBytes`; and
- the local SGE token ID default.

It prefills all 256 six-port entries and all 128 two-port entries, independent
of the active payload batch constant.

### Peer-Level Prefill

```cpp
__simt_vf__ __aicore__
LAUNCH_BOUND(kMoonEpCombineV2BuilderThreads)
void MoonEpCombineV2PrefillPeerWqesVf(
    __ubuf__ uint8_t *sixPortWqes,
    __ubuf__ uint8_t *twoPortWqes,
    __ubuf__ const MoonEpCombineV2PeerFields *fields);
```

This function runs once at the beginning of `SendRemoteStep()` after the two
QP-specific remote memory records for the current peer have been resolved. It
fills fields that are constant for the peer and lane:

- token enable;
- remote jetty type;
- target hint;
- TP ID;
- remote jetty or segment ID;
- remote token value; and
- remote EID low and high words.

It also restores the operator-constant payload fields in every issue entry that
may have been overwritten by the previous peer's scalar control WQEs. In the
initial implementation the peer-level function writes those payload constants
for all fixed issue-buffer entries, rather than tracking the previous control
indices. In particular, it must restore payload `flag`, SGE length, and local
SGE token ID; updating only the remote QP fields is incorrect.

The remote row address is not peer-constant at WQE level because it includes
`targetSlot`. SQ basic-block index, owner, and local SGE address are also
batch- or route-dependent and are left to the payload builder.

Peer prefill covers the complete fixed issue-buffer capacities. Operator-level
prefill establishes the initial template once; peer-level prefill both applies
the new peer fields and makes every entry a valid payload template again before
reuse.

### Route Selection

```cpp
__simt_vf__ __aicore__
LAUNCH_BOUND(kMoonEpCombineV2SelectorThreads)
void MoonEpCombineV2SelectPeerRoutesVf(
    __ubuf__ const int32_t *dstSlots,
    uint32_t chunkStart,
    uint32_t chunkElements,
    uint32_t peer,
    uint64_t slots,
    bool firstPass,
    __ubuf__ MoonEpCombineV2SelectState *state,
    __ubuf__ uint32_t *threadMaxSlotIdx,
    __ubuf__ MoonEpCombineV2RouteEntry *routes);
```

The selector:

1. obtains the first absolute index from `firstPass` or the thread cursor;
2. walks that thread's strided entries in the resident UB chunk;
3. skips `-1` entries;
4. checks whether the encoded destination is in the peer's precomputed
   `[peerBase, peerEnd)` interval;
5. skips entries outside that interval;
6. obtains `targetSlot` by subtracting `peerBase` for a match;
7. atomically reserves the route-buffer index;
8. stores `sourceSlotIndex` and `targetSlot` before testing the threshold; and
9. records its private cursor before returning.

The selector does not write WQEs and does not access SQ GM.

### Payload WQE Construction

```cpp
__simt_vf__ __aicore__
LAUNCH_BOUND(kMoonEpCombineV2BuilderThreads)
void MoonEpCombineV2BuildPayloadWqesVf(
    __ubuf__ uint8_t *sixPortWqes,
    __ubuf__ uint8_t *twoPortWqes,
    __ubuf__ const MoonEpCombineV2RouteEntry *routes,
    uint32_t selectedCount,
    uint32_t sequenceBase,
    __ubuf__ const MoonEpCombineV2BuildContext *context);
```

The builder assigns compacted routes as:

```cpp
for (uint32_t route = threadIdx.x;
     route < selectedCount;
     route += kMoonEpCombineV2BuilderThreads) {
    // Build this route's WQE.
}
```

Since `selectedCount <= 2 * builderThreads`, every thread builds at most two
payload WQEs. The builder uses `sequenceBase + route` to preserve the existing
3:1 six-port/two-port sequence across selector invocations and GM chunks.
`sequenceBase` is the peer-cumulative `batchBase`; it is not reset per batch.

For each active WQE the builder writes only dynamic fields:

- lane-local output index;
- SQ basic-block index;
- owner bit;
- remote address low and high words; and
- local SGE virtual address.

All other payload fields come from operator and peer prefill.

## SendRemoteStep Flow

`SendRemoteStep(peer, step)` follows this sequence:

1. Clear `curWqeNum` once for the peer.
2. Resolve both QP lane records for the peer.
3. Run peer-level prefill for both fixed issue buffers.
4. For each 64 KiB `dstLocal` chunk:
   1. copy the chunk from GM to `dstSlotBuf_` with MTE2;
   2. establish the MTE2-to-SIMT dependency;
   3. invoke the selector with `firstPass = true`;
   4. snapshot `batchSelected` after the selector completes;
   5. build and submit the selected payload WQEs;
   6. while `pausedThreadCount != 0`, repeat selection over the same UB chunk
      with `firstPass = false`; and
   7. load the next GM chunk only after the current one is exhausted.
5. Mark only the last payload submission for the peer as final.
6. If the final selector result contains zero payload routes, still issue the
   required final grant/done control-only submission.

The implementation may avoid a look-ahead by treating a batch as final when
the selector reports no paused thread and the current chunk is the last GM
chunk. Empty non-final selector results are not submitted.

## QP Distribution and Capacity

Payload sequence position continues to select the two-port lane once every four
WQEs; the other three positions use the six-port lane. `sequenceBase` preserves
that phase across all batches for one peer.

With 256 payload WQEs, the maximum active payload counts are:

```text
six-port: 192
two-port:  64
```

The final batch appends at most two control WQEs per lane. The maximum active
counts are therefore 194 and 66, within the fixed 256 and 128 issue capacities.
The 256-payload limit does not include these controls.

## Control WQEs and SQ Publication

`AppendControlWqe()` already constructs a complete WQE in the supplied issue
UB buffer. It does not write directly to SQ GM. Grant and done WQEs therefore
remain scalar and are appended after the actual payload count in each lane's
existing issue buffer.

No dedicated control-WQE UB allocation is added. A separate four-WQE region
would consume another 256 bytes and would complicate contiguous MTE3 publication
without solving a capacity problem.

The required publication order remains:

1. finish operator/peer/dynamic WQE writes in UB;
2. append scalar grant and done WQEs for the final batch;
3. establish the scalar/SIMT-to-MTE3 dependency;
4. copy only active six-port and two-port WQEs from UB to their SQ rings;
5. wait for MTE3-to-scalar completion;
6. update SQ head and completion-count device fields; and
7. ring both SQ doorbells with `st_dev`.

No SIMT function may store a WQE directly into SQ GM or ring a doorbell.

## Self Step and Retained Code

Self-copy stays disabled during this optimization. `SendSelfStep()` becomes:

```cpp
__aicore__ inline bool MoonEpCombineV2::SendSelfStep(uint32_t peer)
{
    (void)peer;
    return true;

#if 0
    // Previous selection, route decode, relay, and self-copy implementation.
#endif
}
```

This immediate return intentionally performs no self selection, route decode,
or local copy. Correctness checks that include self-routed rows are expected to
fail until self-copy is optimized separately. Performance results from this
state must be labeled as excluding self-copy.

The old `dstRank`, `slotIndex`, `selectedIndex`, compare mask, descriptor, and
self-copy code remains in the source under `#if 0`. Its active UB allocations
must be disabled; merely making the execution path unreachable does not recover
the UB capacity.

## Synchronization Contract

This kernel mixes MemBase MTE operations, scalar code, and SIMT functions. The
implementation must preserve explicit handoffs between these models:

- MTE2 completion before the selector reads `dstSlotBuf_`;
- selector completion before scalar code reads selection state or launches the
  payload builder;
- operator and peer prefill completion before the builder reuses their WQEs;
- builder completion before scalar control append or MTE3 publication;
- scalar control append completion before MTE3 publication; and
- MTE3 completion before SQ head updates and doorbells.

The initial implementation may retain the existing broad barriers around
`Simt::VF_CALL` while correctness is established. Any later narrowing of those
barriers requires target-version evidence and hardware validation. A MemBase
pipeline event must not be assumed to provide SIMT thread visibility unless the
target CANN programming model documents that handoff.

## Failure and Boundary Handling

The implementation must preserve or add checks for the following when safety
checks are enabled:

- `encoded == -1` is skipped;
- decoded peer is less than `rankSize`;
- decoded target slot is less than `slots`;
- `routeIndex` is less than
  `kMoonEpCombineV2MaxSelectedPayloadWqes`;
- `batchSelected` is no greater than that same limit;
- lane payload plus control count fits the fixed issue buffer;
- SQ outstanding entries remain below the existing limit; and
- the thread cursor never resumes outside the resident chunk.

The current trusted-input performance configuration may compile these checks
out, but the bounds remain part of the design and must be covered by host-side
or reference tests.

## Profiling

Existing metric meanings should be retained where possible:

- selection load measures GM-to-UB `dstLocal` movement and its handoff;
- selection select measures SIMT peer selection and route compaction;
- remote WQE build measures dynamic payload WQE construction;
- remote submit measures scalar controls, MTE3 SQ copy, and doorbells.

Operator and peer prefill must be distinguishable during focused profiling,
either with dedicated metrics or clearly defined inclusion in prepare/send
timelines. The old scalar descriptor metric must not silently continue to claim
that the removed descriptor path is active.

## Validation

Before performance testing, validation will cover:

1. Compile-time UB and lane-capacity assertions for batch rows 128.
2. A host reference model of per-thread cursor progression and atomic ticket
   allocation.
3. No-match, one-match, sparse, all-match, and invalid-entry chunks.
4. The 128P uniform pattern where one selector thread owns every match.
5. Exactly 128 matches, more than 128 matches, and the 256-entry upper bound.
6. Chunk tails smaller than the selector width.
7. Multiple selector invocations without skipped or duplicate source indices.
8. Multiple 64 KiB chunks with one peer-lifetime counter and preserved QP
   sequence phase.
9. Builder distribution proving at most two WQEs per thread.
10. Six-port/two-port counts and final control capacity.
11. Source guards proving WQEs are copied from UB to SQ through MTE3 before
    doorbells.
12. Source guards proving self step returns immediately and old code is
    retained but inactive.
13. Compilation with `/home/pkg/b150` CANN for the target 3510 build.
14. Hardware profiling against baseline commit `80f184e` with self-copy still
    excluded from both measurements.

Correctness output involving self-routed rows is not an acceptance criterion
for this intermediate state. Remote destination-address stability, bounds,
completion behavior, and absence of hangs remain required.

The hardware probe therefore classifies output as `passed`,
`self_only_failed`, or `failed`. The temporary
`--allow-correctness-failure` mode accepts only `self_only_failed`; any mismatch
whose source rank differs from the local rank rejects the run before timing.

## Review Decisions Recorded

- Keep the current multiply/divide/modulo `dstLocal` encoding.
- Store `sourceSlotIndex` and decoded `targetSlot` in each route entry.
- Use one private 32-bit maximum/last-scanned slot ID for each of 128 threads.
- Do not use atomic max for per-thread cursors.
- Keep one peer-lifetime atomic WQE counter in a 32-byte UB state.
- Default the single compile-time payload batch constant to 128.
- Bound payload selection at twice that constant, initially 256.
- Keep fixed issue capacities of 256 six-port and 128 two-port WQEs.
- Treat the 256 limit as payload only; controls are additional active WQEs.
- Keep grant and done construction scalar in the existing issue buffers.
- Keep old selection/descriptor/self-copy code under `#if 0`.
- Make `SendSelfStep()` return success immediately.
