# UDMA AllToAll Bigdata Multi-Node Design

## Purpose

This design extends the existing `testType=7` UDMA AllToAll bigdata demo to support
multi-node rank sizes, where one server owns 8 contiguous ranks and `rankSize > 8`
means multiple servers participate.

The 8-rank single-server path must remain unchanged. When `rankSize <= 8`, the
current bigdata kernel layout, block dispatch, pass protocol, correctness behavior,
and performance profile stay as they are today. When `rankSize > 8`, `testType=7`
switches to a multi-node worker dispatch that keeps local-server traffic parallel
and serializes all remote-server traffic through one additional 5-core worker group.

## Confirmed Assumptions

- Each server has 8 ranks.
- Global rank ids are contiguous by server:

  ```text
  nodeId = rank / 8
  localRank = rank % 8
  localNodeBegin = nodeId * 8
  localNodeEnd = localNodeBegin + 8
  ```

- `rankSize > 8` implies multi-node mode.
- Multi-node mode requires `rankSize` to be a positive multiple of 8.
- Remote traffic means all peers outside `[localNodeBegin, localNodeEnd)`, not only
  peers with the same `localRank`.
- The path is strict UDMA. It must not fall back to IPC.

## Scope

In scope:

- Keep the feature under `testType=7`.
- Add a multi-node branch for `rankSize > 8`.
- Keep `rankSize <= 8` behavior bit-for-bit equivalent at the dispatch level.
- Use one registered UDMA relay buffer per rank, as in the current bigdata path.
- Preserve the existing pass slicing, ping-pong relay reuse, token protocol, and
  host-side repeat launch model.
- Use 8 local peer worker groups plus 1 remote peer worker group in multi-node mode.
- Make the remote worker group cover every remote peer on every remote server.

Out of scope:

- Supporting non-contiguous or non-8-rank-per-server topology.
- Changing the TileXR UDMA registry model.
- Registering multiple UDMA memory regions.
- Introducing IPC fallback for multi-node bigdata alltoall.
- Optimizing remote traffic with multiple remote worker groups in the first version.

## Existing Bigdata Baseline

The current `testType=7` implementation uses:

- full `input` and `output` as ordinary GM buffers, not registered;
- one registered UDMA relay buffer;
- `passCount` kernel-internal passes based on the relay data capacity;
- two ping-pong data slots per peer;
- five logical roles per peer:

  ```text
  role 0: copy input shard 0 to local UDMA send slot
  role 1: copy input shard 1 to local UDMA send slot
  role 2: wait for copyDone then UDMA send to peer
  role 3: wait for ready, copy recv slot shard 0 to output
  role 4: wait for ready, copy recv slot shard 1 to output, then ACK peer
  ```

- monotonic pass tokens, with all waits using `observed >= token`;
- copy-side ACK checks before reusing the same ping-pong slot;
- one `SyncAll()` at kernel exit so the next host-launched kernel can reuse the
  same registered UDMA memory from ping-pong slot 0 again.

The multi-node design must reuse these worker functions where possible. It changes
which peer list a worker group processes, not the data-transfer protocol itself.

## Multi-Node Worker Model

The selected model is a deterministic task-queue dispatch rather than
`peer = blockIdx / 5` for all global peers.

For `rankSize <= 8`:

```text
blockDim = 5 * rankSize
workerGroup = blockIdx / 5
role = blockIdx % 5
peer = workerGroup
```

For `rankSize > 8`:

```text
localRankSize = 8
workerGroupCount = localRankSize + 1
blockDim = 5 * workerGroupCount = 45
workerGroup = blockIdx / 5
role = blockIdx % 5
```

Worker groups `0..7` process local-server peers:

```text
peer = localNodeBegin + workerGroup
```

Worker group `8` processes all remote-server peers serially. Its five cores keep
the same role meaning as the existing 5-core per-peer design, but each role loops
over the same remote peer queue.

The remote peer queue is deterministic and circular. It starts from `rank + 8`,
wraps at `rankSize`, skips every rank on the local server, and stops after all
`rankSize - 8` remote peers have been collected:

```text
remoteCount = 0
for step in 0..rankSize-1:
    peer = (rank + 8 + step) % rankSize
    if peer is outside [localNodeBegin, localNodeEnd):
        process peer
        remoteCount += 1
        if remoteCount == rankSize - 8:
            break
```

Examples:

```text
rankSize=16, rank=3  -> remote peers: 11, 12, 13, 14, 15, 8, 9, 10
rankSize=16, rank=10 -> remote peers: 2, 3, 4, 5, 6, 7, 0, 1
rankSize=24, rank=5  -> remote peers: 13..23, 8..12
```

This order preserves the original requirement that remote communication starts at
`rank + 8`, while covering every card on every remote server.

## Kernel Dispatch

The existing bigdata kernel can split into two dispatch paths:

```text
if rankSize <= 8:
    BigDataSingleNodeDispatch()
else:
    BigDataMultiNodeDispatch()
```

Single-node dispatch should keep the current code shape.

Multi-node dispatch validates:

```text
rankSize > 8
rankSize % 8 == 0
blockIdx < 45
```

Then it assigns:

```text
workerGroup = blockIdx / 5
role = blockIdx % 5

if workerGroup < 8:
    peer = localNodeBegin + workerGroup
    process one local peer with the existing role worker
else:
    for peer in RemotePeerQueue(rank, rankSize):
        process peer with the existing role worker
```

The `loop` and `pass` structure remains:

```text
for loop in 0..loopCount-1:
    for pass in 0..passCount-1:
        dispatch local or remote peer work
```

Remote role workers must use the same peer order for all five roles. This is
important because copyDone, ready, recvCopyDone, and ACK tokens are all indexed by
the real peer rank and ping-pong slot.

## Memory Layout

The registered UDMA relay layout remains indexed by global peer rank, not by
worker-group index. Multi-node mode does not compress remote peers into one shared
slot, because each remote peer needs independent ping-pong reuse and ACK state.

The existing plan remains valid:

```text
networkPeerCount = rankSize - 1
dataSlotCount = networkPeerCount * pingPongSlots * 2
controlGroupBytes = pingPongSlots * rankSize * localCopyShards * controlSlotBytes
signalBytes = 3 * controlGroupBytes
dataBytes = registeredBytes - controlGroupBytes - signalBytes
chunkElements = floor(dataBytes / (dataSlotCount * sizeof(int32_t)))
passCount = ceil(elementsPerPeer / chunkElements)
```

Data slots:

```text
sendSlot[slot][peerIndex(peer, rank)]
recvSlot[slot][peerIndex(peer, rank)]
```

Control slots:

```text
copyDone[slot][peer][copyShard]
recvCopyDone[slot][peer][recvShard]
ready[slot][peer][0]
ack through peer IPC memory indexed by [slot][rank]
```

All offsets must continue to pass `UDMARegisteredRangeValid`. The plan must fail
early if `chunkElements <= 0`.

## Synchronization Protocol

For each `(loop, pass, peer)`:

1. Copy role 0 and role 1 wait for the old ACK before overwriting a reused
   ping-pong slot when `globalPass >= pingPongSlots`.
2. Copy roles copy `input[peer][chunkOffset + shardRange]` to the local UDMA
   `sendSlot[peer]` using the existing GM-UB-GM ping-pong copy helper.
3. Copy roles write `copyDone[slot][peer][shard] = token`.
4. Send role waits until both copyDone slots are `>= token`.
5. Send role posts `UDMAPutSignalNbi` from local `sendSlot[peer]` to remote
   `recvSlot[rank]`, with remote ready signal set to `token`.
6. Receive roles wait until local `ready[slot][peer] >= token`.
7. Receive roles copy local `recvSlot[peer]` to `output[peer][chunkOffset]` using
   the existing GM-UB-GM ping-pong copy helper.
8. Receive roles write `recvCopyDone[slot][peer][recvShard] = token`.
9. The last receive role waits until both recvCopyDone shards are `>= token`, then
   writes the remote ACK token to the peer-visible ACK slot.
10. Kernel exit performs `SyncAll()` once, after all local and remote peer queues
    are complete.

No per-pass `SyncAll()` is required. Slot reuse is protected by ACK tokens, and
using a per-pass global barrier would serialize unrelated local and remote work.

All token checks must compare `observed >= token`, not `observed == token`, because
producer roles can advance and write a later token before the consumer polls.

## Ascend C API Constraints

The local copies remain GM-UB-GM transfers. There is no direct GM-to-GM copy path.

Implementation rules:

- Continue using the existing 2 x 64 KiB UB ping-pong buffer for relay copies.
- Use `DataCopyPad` for copy-in and copy-out, including tails that are not known
  to be 32-byte aligned.
- Keep MTE2/MTE3 synchronization in the copy helper through explicit flags or an
  equivalent queue-based synchronization point.
- Do not introduce production `GlobalTensor::GetValue()` or `SetValue()` for data
  movement. Scalar token load/store helpers are acceptable only for control words
  where the current implementation already uses them.
- Keep `chunkBytes` within `uint32_t`, matching the current UDMA API usage.

## Host-Side Changes

`PlanAllToAllBigDataUdma(rankSize, elementsPerPeer)` remains global-rank based and
does not need a separate remote-worker layout.

`AllToAllBigDataBlockDim(rankSize)` should preserve single-node behavior and return
the multi-node worker count when `rankSize > 8`:

```text
rankSize <= 8: 5 * rankSize
rankSize > 8 : 5 * (8 + 1)
```

The helper should use explicit names in implementation to avoid confusing global
rank size with local node rank size, for example:

```text
kAllToAllBigDataRanksPerNode = 8
kAllToAllBigDataMultiNodeWorkerGroups = kAllToAllBigDataRanksPerNode + 1
```

Before allocating and registering memory, the host should reject:

```text
rankSize > 8 && rankSize % 8 != 0
```

The host repeat model stays unchanged:

- `TILEXR_DEMO_ALLTOALL_REPEAT` launches repeated kernels from host;
- each kernel uses `loopCount=1`;
- the stream is synchronized once after the host launch loop;
- `kernelLoopBase = iter` keeps tokens globally monotonic across kernels.

## Debug And Error Handling

The existing debug array should keep the current first-pass fields. Multi-node mode
should add or reuse fields for:

- multi-node mode enabled flag;
- `localNodeBegin`, `localNodeEnd`, and active `blockDim`;
- remote peer count;
- first remote peer and last remote peer seen by the remote worker;
- timeout status per peer where debug capacity allows.

Timeout behavior must remain strict: a timeout or non-zero UDMA quiet status should
fail the test rather than falling back to IPC.

If `rankSize > 8` and topology validation fails, the demo should print a clear
host-side error before `TileXRUDMARegister`.

## Testing Plan

Unit/source tests:

- `AllToAllBigDataBlockDim(8) == 40`.
- `AllToAllBigDataBlockDim(16) == 45`.
- `AllToAllBigDataBlockDim(32) == 45`.
- Remote peer queue examples:
  - `rankSize=16, rank=3` gives `11,12,13,14,15,8,9,10`.
  - `rankSize=16, rank=10` gives `2,3,4,5,6,7,0,1`.
- Source guard verifies `rankSize <= 8` keeps the existing dispatch.
- Source guard verifies `rankSize > 8` uses the multi-node remote peer queue.
- Source guard verifies token waits use `>= token`.

On-board validation:

1. Re-run existing 8P `testType=7` correctness and performance. The kernel name
   and path stay the same, but the single-node dispatch must match prior behavior.
2. Run a 16P small payload correctness test across two servers.
3. Run a 16P per-peer 2 MiB payload test with `TILEXR_DEMO_ALLTOALL_REPEAT=100`
   and msprof kernel timing.
4. Run a larger multi-pass payload and confirm `passCount > 1`, correctness passes,
   and no ACK/ready timeout appears.

Success criteria:

- 8P results do not regress beyond normal measurement noise.
- 16P and larger multi-node runs finish without IPC fallback.
- All ranks pass output validation.
- No UDMA range validation failure, CQ error, ready timeout, ACK timeout, or data
  mismatch appears in logs.

## Performance Expectations

The multi-node first version intentionally serializes remote peers through one
5-core worker group. Therefore:

- local-server traffic remains parallel across 8 peer groups;
- remote traffic time scales roughly with the number of remote peers;
- the implementation prioritizes correctness and controlled resource usage over
  maximum cross-server bandwidth;
- future optimization can add more remote worker groups after this protocol is
  validated.

This design is expected to be slower than a fully parallel `5 * rankSize` remote
dispatch, but it avoids launching one 5-core worker group for every remote rank and
keeps synchronization behavior tractable.
