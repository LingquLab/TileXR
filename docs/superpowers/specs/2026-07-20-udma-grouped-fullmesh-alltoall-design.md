# UDMA Grouped Fullmesh AllToAll Design

## Status

Approved for implementation planning on 2026-07-20.

## Objective

Add a new, standalone `alltoall_group` kernel that preserves global AllToAll
semantics while scheduling peers in symmetric groups of at most 16. The first
version prioritizes a small and auditable correctness model over registered
memory efficiency.

The existing experimental 35-core big-data fullmesh kernel remains unchanged.
The new kernel must not inherit its local/remote role split, copy-ready
aggregation, ACK protocol, profiling stages, or experimental mode branches.

## Scope

The first version supports:

- `8 <= rankSize <= 128` and `rankSize % 8 == 0`, matching the current
  `TILEXR_MAX_RANK_SIZE` and `CommArgs` peer-array limit.
- A fixed peer-group width of 16: eight forward and eight backward circular
  distances.
- Any payload whose complete double-buffered registered receive layout,
  signals, and control data fit below the per-rank 1 GiB UDMA registration
  limit.
- One complete AllToAll operation per kernel invocation.
- Host-side warmup and repeat launches on the same ordered stream.
- Chunked transfers within an invocation when one peer payload is divided into
  multiple passes.

The first version does not support:

- Arbitrary rank sizes that are not multiples of eight.
- Reusing only 16 receive lanes.
- ACK or credit flow control.
- Concurrent grouped AllToAll invocations on multiple streams for the same
  communicator and registered buffer.
- Dynamic group width.

## Data Semantics

Each rank owns source-major input and output arrays:

```text
input[dstRank][elementsPerPeer]
output[srcRank][elementsPerPeer]
```

Rank `src` sends `input[dst]` to rank `dst`. Rank `dst` stores it in
`output[src]`. Self data is copied locally without UDMA.

## Peer Group Schedule

For rank `r`, group `g`, and local distance index `i` in `[0, 7]`:

```text
distance = g * 8 + i + 1
forwardPeer  = (r + distance) % rankSize
backwardPeer = (r - distance + rankSize) % rankSize
```

Worker lanes map as follows:

- Lane `0..7` selects the forward peer.
- Lane `8..15` selects the backward peer with `i = lane - 8`.
- A lane is inactive when `distance > rankSize / 2`.
- When `distance == rankSize / 2`, only the forward lane is active because the
  forward and backward peers are identical.

The group count is:

```text
groupCount = ceil((rankSize - 1) / 16)
```

This schedule covers every non-self rank exactly once and is symmetric: if
rank A schedules rank B in group `g`, rank B schedules rank A in the same
group, although their local lane numbers can differ.

Examples:

- `rankSize=16`: one group with 15 active peers.
- `rankSize=24`: groups with 16 and 7 active peers.
- `rankSize=64`: groups with 16, 16, 16, and 15 active peers.

## Core Assignment

The kernel launches exactly 32 AIV blocks:

```text
core 0..15   send workers
core 16..31  receive workers
```

Send lane `L` and receive lane `L` independently evaluate the peer schedule
for each group. They do not share progress state and do not require
`SyncAll()`.

The 16 receive workers shard the self payload and copy it directly from input
to output. This local copy can overlap the first group's remote sends.

Inactive lanes skip UDMA and signal waits but continue their normal loop and
return cleanly.

## Registered Memory Layout

Each rank allocates two complete receive planes:

```text
payloadPlane[2][rankSize][elementsPerPeer]
signalPlane[2][rankSize][128 bytes]
debug/control area
```

The ping-pong plane for invocation `I` is:

```text
slot = I & 1
```

The payload slot is indexed by global source rank, not by worker lane:

```text
payloadOffset(slot, sourceRank, chunkOffset) =
    payloadBase[slot] +
    sourceRank * bytesPerPeer +
    chunkOffset

signalOffset(slot, sourceRank) =
    signalBase[slot] + sourceRank * 128
```

For example, rank 37 sending to rank 5 writes rank 5's payload slot 37 and
signal slot 37. Rank 5 later copies payload slot 37 into output slice 37.
The sender's local worker lane has no effect on the remote address.

Input and output remain ordinary GM allocations. Only the double-buffered
receive planes, signal planes, and control region are registered.

The host rejects a plan unless:

```text
2 * aligned(outputBytes) +
2 * aligned(rankSize * 128) +
aligned(debugControlBytes) <= 1 GiB
```

## Token Encoding

Signal values are nonzero, monotonically ordered 64-bit tokens:

```text
bits 63..32  invocationId + 1
bit  31      ping-pong slot
bits 30..16  group
bits 15..0   pass + 1
```

The host validates that group and pass dimensions fit their fields. All ranks
must use the same invocation ID and collective call order.

Receivers accept `observedToken >= expectedToken`, not equality. A sender may
publish a later pass before the receiver samples an earlier token. Because
passes write non-overlapping offsets and use the same ordered QP, a later token
also proves completion of the earlier payload writes.

Signal polling uses an MTE-to-UB load helper with a cycle timeout. It must not
use an unqualified ordinary GM pointer spin loop.

## Data Path

For each active send worker, group, and pass:

1. Compute the target peer and chunk range.
2. Select the peer's explicit route/QP.
3. Issue one `UDMAPutSignalNbiOnQp` that transfers
   `input[targetPeer][chunk]` to
   `target.payloadPlane[slot][sourceRank][chunk]` and publishes
   `target.signalPlane[slot][sourceRank]`.
4. Immediately call `UDMAQuietStatusOnQp` with the same peer and QP.

For each active receive worker, group, and pass:

1. Compute the source peer.
2. Poll `signalPlane[slot][sourcePeer]` until it reaches the expected token.
3. Copy `payloadPlane[slot][sourcePeer][chunk]` to
   `output[sourcePeer][chunk]` using the local MTE relay helper.

Payload, signal, and quiet must use the same explicit QP. The signal path must
not fall back to the generic QP0 route.

## Progress and Buffer Safety

There is no per-group or per-pass `SyncAll()`. Every worker owns independent
source or target slices and progresses according to its own signal dependency.
Kernel completion naturally waits for all 32 blocks to return.

There is no cross-rank Host barrier between invocations. Invocations are
submitted in identical order on each rank and execute serially on one stream.
Ping-pong is sufficient under this collective contract:

1. Invocation `I` writes plane `I & 1`.
2. A fast rank can enter invocation `I+1` and write the other plane.
3. It cannot complete `I+1` until it receives `I+1` data from every rank.
4. A slower rank cannot send `I+1` until its invocation `I` kernel has
   completed, which means it has consumed all invocation `I` receive data.
5. Therefore plane `I & 1` is safe before any rank can enter invocation `I+2`
   and reuse it.

This proof does not hold if ranks skip or reorder calls, or if the same
communicator and buffers are used concurrently on multiple streams. Those uses
are out of scope.

## QP Selection

The kernel selects a deterministic valid QP for each peer using the current
max-weight route policy. The selected `qpIdx` is reused for payload plus signal
and immediate quiet.

The implementation should expose QP selection as a small helper independent of
the old big-data fullmesh worker functions. This keeps route policy separate
from group scheduling and buffer addressing.

## Failure Handling

Host-side validation rejects invalid rank size, dimensions, memory capacity,
or missing UDMA registry before launch.

Each device worker records only its first failure:

```text
stage
group
pass
peer
qpIdx
quietStatus
expectedToken
observedToken
```

A quiet failure is recorded immediately. A receive wait records a timeout and
returns instead of spinning forever. Peer failure recovery is not attempted in
the first version; other ranks may also reach their timeouts.

All early configuration exits must be uniform across blocks. The kernel does
not contain a block-divergent barrier.

## Instrumentation

The new kernel has a compact trace vocabulary rather than the old experimental
stage profiler:

- kernel
- self-copy
- send-put-signal
- send-quiet
- receive-wait
- receive-copy

Trace records include invocation, ping-pong slot, group, pass, lane, peer, and
QP. Instrumentation is optional and must not change buffer ownership or
synchronization.

## Host Integration

The demo adds a dedicated grouped-fullmesh mode and launch wrapper. Host code:

1. Validates `rankSize`, payload dimensions, token dimensions, and the 1 GiB
   registered-memory limit.
2. Allocates and registers the two payload planes plus signals and controls.
3. Initializes signals to zero once.
4. Launches one AllToAll operation per invocation with a globally consistent
   invocation ID.
5. Uses Host-side warmup and repeat loops on the same stream.
6. Validates the full source-major output after the measured runs.

No Host rank barrier is inserted between invocations.

## Verification

### Unit tests

Peer-schedule tests cover rank sizes 8, 16, 24, 32, 40, and 64 and verify:

- Every non-self peer appears exactly once.
- No peer appears twice.
- Pair scheduling is symmetric by group.
- The diameter peer appears once for even rank sizes.
- Last-group inactive lanes do not create peers.

Layout tests verify:

- Two payload planes do not overlap.
- Source-rank payload offsets and 128-byte signal slots are correct.
- Every remote range lies inside the registered allocation.
- Plans at and above the 1 GiB boundary are accepted or rejected correctly.
- Token fields do not overflow.

Source/structure tests verify:

- Exactly 32 blocks are launched.
- Payload plus signal uses `UDMAPutSignalNbiOnQp`.
- Immediate quiet uses the same `qpIdx`.
- No `SyncAll()` or generic signal route exists in the grouped kernel.

### Physical validation

The first physical target is 2x8 ranks. Validation includes:

- Full output correctness for all 16 ranks.
- The 15-peer final-group boundary.
- Host warmup 5 and measured repeat 50.
- Alternating ping-pong planes across invocations.
- Loop 49 trace extraction for all 32 cores.
- Per-peer put, quiet, wait, and receive-copy timing.
- No timeout, quiet error, or stale-token acceptance.

Larger topologies use the same correctness checks when available; host mapping
tests cover multi-group scheduling before 64-rank hardware validation.

## Success Criteria

The first version is complete when:

- The new grouped kernel passes all mapping and layout unit tests.
- A physical 2x8 run passes full output validation on all ranks for warmup 5
  and repeat 50.
- Trace data shows at most 16 concurrent peers per rank and no global wait over
  all peers.
- Payload, signal, and quiet for every send use one explicit QP.
- No registered-memory plan exceeds 1 GiB per rank.
- The old fullmesh kernel remains behaviorally unchanged.
