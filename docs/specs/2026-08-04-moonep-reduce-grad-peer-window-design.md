# MoonEP ReduceGrad Hybrid Peer-Window/UDMA Design

## Goal

Implement a native, high-throughput MoonEP ReduceGrad stage for CANN 9.1 and
Ascend 910A5/Ascend950. The implementation reuses the existing TileXR peer
windows in `CommArgs::peerMems[]` for gradient rows up to 1 MiB and uses TileXR
registered-memory UDMA for larger rows. It must work across nodes, integrate
with the current MoonEP Torch facade and benchmark, and keep the caller's
stream asynchronous.

The operator reduces gradients produced for remotely prefetched expert slots
back into the owner rank's local expert gradients. It then clears each source
rank's consumed slot gradients for reuse by the next microbatch.

## Fixed Decisions

1. Transport is selected independently for gate, up, and down from the logical
   bytes in one expert/slot row. `rowBytes > 1 MiB` requires UDMA;
   `rowBytes <= 1 MiB` uses the PR #90 peer-window DataAsFlag path. Equality is
   deliberately assigned to peer memory. A selected UDMA path never silently
   falls back to peer memory.
2. Window capacity is never a ReduceGrad literal. On PR #90 the authoritative
   per-window data capacity is `IPC_BUFF_MAX_SIZE = 512 MiB`; the communicator
   allocation is 1028 MiB for two `(2 MiB flags + 512 MiB data)` partitions.
   When issue #92 decouples memory-mode capacity from legacy communicator
   defaults, only the shared capacity resolver changes.
3. Full expert gradients never reside in the peer window. Small rows stream
   bounded DataAsFlag chunks through two internal data halves. Large rows and
   the bounded UDMA staging workspace reside in one ordinary device allocation
   registered once before warmup and timing.
4. The peer path pushes slot chunks directly to the owner rank's peer window.
   Each 512-byte DataAsFlag record carries 480 payload bytes and 32 ready bytes;
   owner ranks clear the ready bytes after consumption. The UDMA path pushes
   bounded payload chunks plus signals into the owner rank's registered staging
   lanes. Neither path uses an all-rank barrier for every chunk.
5. Gate, up, and down gradients are reduced by one fused V2 launch. They share
   plan inspection, sender/receiver scheduling, and slot cleanup.
6. The V1 ReduceGrad entry point remains available with its existing stub
   behavior. The real fused contract is introduced as V2 so the existing V1
   binary layout is not silently reinterpreted.
7. The Host launch packs arguments and calls `rtKernelLaunchWithFlagV2`
   directly. The kernel translation unit contains device code only.
8. Cross-node behavior is required. Small rows use cross-node peer windows and
   large rows use registered UDMA. A missing required transport, invalid layout,
   failed registration, or bounded progress timeout is an error; there is no
   local copy fallback that reports native success.
9. The same `TileXRMoonEPBuffer` remains single-stream and non-reentrant. This
   gives ReduceGrad exclusive use of the peer data window for the duration of
   its launch and lets later native stages reuse the same window serially.

## Scope

### In Scope

- A5/Ascend950 direct-launch kernel and Host validation.
- Contiguous fp32 gate, up, and down gradients.
- Rank-2 through rank-4 tensor descriptors with first dimension `E + B`.
- Different per-expert element counts for gate, up, and down.
- PR #90 DataAsFlag packing, capacity derivation, and two-half reuse protocol.
- TileXR UDMA registration, registered staging, signals, and completion for
  expert rows strictly larger than 1 MiB.
- Arbitrary 480-byte payload blocks and a bounded final tail.
- Same-node and cross-node operation, including mixed peer/UDMA projections in
  one ReduceGrad launch.
- Torch facade, native capability reporting, correctness checks, and benchmark
  timing for the real ReduceGrad stage.

### Non-Goals

- Registered custom operators, OpDef, ACLNN packaging, or graph integration.
- Native Dispatch, PrefetchWeight, or Combine implementations.
- SDMA, HCCL, or automatic fallback from the transport selected by row size.
- Supporting UDMA through `TileXRCommInitThread`; the existing UDMA registry
  requires a live cross-process socket exchange.
- Concurrent MoonEP operations sharing one communicator peer window.
- Claims about cross-node correctness or performance before a real two-node A5
  run has completed.

## Semantic Contract

Let:

- `R = plan.world`: ranks in the MoonEP planner group.
- `E = plan.e`: total experts in the group.
- `B = plan.b = E / R`: experts owned by each rank and available remote slots
  per source rank.
- `Pq`: contiguous fp32 elements per expert for projection `q`, where `q` is
  gate, up, or down.
- `rowBytes(q) = Pq * sizeof(float)`: the logical bytes transferred for one
  non-local slot contribution of projection `q`.

Transport selection is deterministic and does not depend on topology or
runtime timing:

```text
rowBytes(q) <= 1 * 1024 * 1024  -> peer-window DataAsFlag
rowBytes(q) >  1 * 1024 * 1024  -> registered UDMA
```

The decision is made once per projection, before launch. UDMA rows may be split
into bounded WQEs and their final WQE may be smaller than 1 MiB; all chunks of a
projection stay on the transport selected from the complete logical row. A
slot whose owner is the same rank needs no inter-rank transfer and is consumed
directly, but it follows the same deterministic accumulation order.

Each gradient tensor has logical shape `[E + B, ...]`:

- rows `[0, E)` contain expert gradients;
- rows `[rank * B, (rank + 1) * B)` are owned by the current rank;
- rows `[E, E + B)` contain gradients for the current rank's prefetched slots.

`plan.expertsToCopy` is a contiguous int32 table `[R, B]`. Entry `[src, b]`
contains the global expert id whose gradient is stored in source rank `src`'s
slot row `E + b`, or a negative value for an unused slot.

For every projection and every expert owned by rank `r`, ReduceGrad computes in
place, in ascending `(src, b)` order:

```text
grad[expert] = grad[expert] + sum(
    source_slot[src, b] for expertsToCopy[src, b] == expert
)
```

After a source rank has completed every push from one used slot row, it clears
that local row `[E, E + B)`. The destination record remains valid until its
owner has accumulated the payload and cleared the embedded ready bytes. Unused
slot rows remain unchanged. Expert rows not owned by the current rank remain
unchanged.

Each output element is owned by exactly one AIV block. Contributions for that
element are added serially in `(src, b)` order, preserving deterministic fp32
rounding relative to the CPU/device oracle.

## Public ABI

The library continues to export all V1 structures and functions. It adds:

- `TILEXR_MOONEP_ABI_VERSION_V2`;
- `TileXRMoonEpReduceGradWorkspaceQueryV2` and
  `TileXRMoonEpReduceGradGetWorkspaceSizeV2`;
- `TileXRMoonEpReduceGradArgsV2`;
- `TileXRMoonEpReduceGradV2`;
- a V2 capability query that distinguishes the native V2 ReduceGrad from the
  legacy V1 stub.

`TileXRMoonEpReduceGradArgsV2` contains:

- `structSize` and `abiVersion`;
- `TileXRCommPtr comm` and `const TileXRMoonEpPlanV1 *plan`;
- gate, up, and down gradient tensor descriptors;
- a caller-owned byte workspace and its size;
- a caller-owned int32 status tensor of shape `[1]`;
- a bounded wait-iteration budget and flags.

The workspace query returns the required alignment, registered workspace
bytes, selected transport for each projection, and the UDMA chunk size from the
plan dimensions and tensor descriptors. If any projection selects UDMA, Host
requires the workspace, but not the user gradient tensors, to be the base of
the one active TileXR UDMA registered region. The one-region requirement
matches `TILEXR_UDMA_MAX_REGIONS = 1`; Host verifies that every gathered region
contains the same required workspace layout. Peer-only cases neither require
nor create a UDMA registration.

The call validates and enqueues work but does not synchronize the stream.
Validation, missing-transport, or launch failures are returned synchronously.
A device progress or UDMA completion failure is written to the status tensor
and surfaced by the Torch facade at its existing stream synchronization
boundary.

The highest ABI version query returns V2, while V1 entry points remain callable.
The V2 capability query reports Planning and ReduceGrad as native and keeps
Dispatch, PrefetchWeight, and Combine marked as stubs.

## PR #90 Window Dependency

PR #90 changes the current constants as follows:

```text
IPC_BUFF_MAX_SIZE       100 MiB -> 512 MiB
TILEXR_COMM_BUFFER_SIZE 200 MiB -> 1024 MiB
TILEXR_BUFF_BYTES       204 MiB -> 1028 MiB
```

It also defines the memory-mode model rooted at:

```text
peerMems[rank] + IPC_DATA_OFFSET
  1 MiB state window
  two internal data halves
  optional operator workspace reservation
```

Issue #92 may replace the enlarged global defaults with an explicit
memory-mode capacity. ReduceGrad therefore calls one TileXR-owned capacity
resolver. It does not infer usable bytes from `TILEXR_COMM_BUFFER_SIZE`, copy a
512 MiB literal, or assume that all bytes exported by `rtIpcSetMemoryName` are
owned by one operator. Pure layout tests accept an injected window size so both
the current main capacity, the PR #90 512 MiB capacity, and later issue #92
capacity can be checked with the same formulas.

The implementation branch remains based on `main`; unrelated PR #90 EP sources
are not copied or merged into ReduceGrad. Target performance validation uses a
base containing PR #90 or its issue #92 successor.

## Cross-Node Evidence Boundary

PR #90 has a technically valid review finding against its cross-node claim: its
Host path accepts `localRankSize < rankSize` when every `peerMems[]` entry is
non-null, but its kernel uses the same direct MTE path for every rank and the PR
contains no cross-node runtime proof. Resolving that review thread does not make
the missing evidence disappear.

ReduceGrad uses that direct peer-window path only for projections whose logical
row is at most 1 MiB. TileXR communicator initialization configures remote
SuperPod PID/SDID access and opens exported peer allocations with
`rtIpcOpenMemory`, but pointer presence alone is not accepted as proof. For
larger rows ReduceGrad uses TileXR's registered UDMA transport, which is the
required cross-node data path rather than a failure fallback.

The delivered benchmark uses projection/rank/row/element-distinct payloads and
derives cross-node ReduceGrad validity only from the structured correctness
results produced by the actual peer-only, UDMA-only, or mixed case. An
environment variable cannot self-certify cross-node support. ReduceGrad retains
bounded waits and status propagation so a failed or unsupported mapping returns
a device status instead of hanging. UDMA registration occurs only with
`TileXRCommInitRankLocal` or an equivalent communicator that retains its live
socket exchange; `InitThread` is rejected for every UDMA-selected case.

Single-node success, non-null remote pointers, an environment flag, or a
resolved PR thread cannot satisfy cross-node acceptance. A two-node matrix must
pass the peer-only, UDMA-only, and mixed-transport ReduceGrad oracle cases before
the implementation is reported as cross-node capable.

## Peer-Window Layout

Let `windowBytes` be the authoritative memory-mode capacity and reserve the
same 1 MiB state prefix used by PR #90. Host computes the worst-case layout:

```text
windowBase       = peerMems[rank] + IPC_DATA_OFFSET
recordBase       = windowBase + stateWindowBytes
halfBytes        = align_down((windowBytes - stateWindowBytes) / 2, 512)
maxIncomingSlots = R * B
minSlotStride    = align_down(halfBytes / maxIncomingSlots, 512)
minChunkPayload  = (minSlotStride / 512) * 480
```

Every rank has the same `[R, B]` plan table. The active kernel deliberately uses
the plan-independent global slot index `src * B + b`:

```text
slot(stage, src, b) = recordBase
    + stage * halfBytes
    + (src * B + b) * minSlotStride
chunkPayload = (minSlotStride / 512) * 480
```

This fixed address derivation lets source and owner agree without a Host D2H
copy or a timed synchronization on the asynchronous Planner table. It reserves
the worst-case `R * B` slots even for a lightly loaded owner, so plan-aware
owner-local compaction remains a measured follow-up optimization if green-zone
results show peer chunking or flag scans are material after PR #90 increases the
window capacity.

Host rejects layouts where state reservation, `R * B`, 512-byte alignment,
minimum payload, tensor chunk count, or a final address range is invalid. It
also verifies every `peerMems[r]` pointer, the A5 topology flag,
plan/communicator agreement, and all tensor byte counts.

Planner and ReduceGrad may reuse the same memory-mode bytes because the MoonEP
Buffer orders them on one stream and Planner outputs live in caller-owned
tensors once Planner has completed.

Only projections selected for the peer path reserve records in this layout.
The layout remains valid when all three projections use UDMA, but no peer data
records are touched in that case.

## DataAsFlag Streaming Protocol

The kernel obtains one magic value from `TileXRCommNextMagic` and uses the two
internal halves as chunk stages. Sender and receiver AIV groups run concurrently
inside the same launch; they must not be separated by an all-AIV barrier that
would prevent streaming progress.

For each projection:

1. Each sender AIV owns a subset of the current rank's valid source slots.
   It derives the destination owner, compact slot index, stride, and chunk size.
2. Before every stage write, including chunk zero of a later invocation, the
   sender bounded-polls the destination slot's ready bytes until the previous
   payload has been cleared.
3. The sender packs local slot-gradient bytes as PR #90 DataAsFlag records and
   writes them directly to `peerMems[owner]`. Payload and ready state become
   visible in the same MTE3 transfer.
4. Receiver AIVs own disjoint local expert/tile ranges. For each contribution
   in ascending `(src,b)` order, they bounded-poll the local record, copy its
   payload to UB, add fp32 values to the local accumulator, then clear only the
   record flags they consumed.
5. Clearing all flags in a stage is the acknowledgement that allows its sender
   to reuse that stage two chunks later. No per-chunk all-rank round trip is
   required.
6. A source slot row is zeroed only after all three projections and every peer
   or UDMA transfer for that row has been remotely consumed and acknowledged.
   A receiver exits only after every expected contribution has been accumulated
   and acknowledged.

Gate, up, and down run inside the same launch. Their chunk counts may differ.
The plan table and compact owner/slot metadata are built once per AIV group and
reused across all three projections.

## Registered UDMA Layout and Protocol

UDMA is selected for a projection before launch when its logical row is
strictly larger than 1 MiB. Registration is a collective Host operation and is
never placed in the measured ReduceGrad loop:

1. query the workspace size and alignment;
2. allocate and zero one ordinary device byte workspace per rank;
3. call `TileXRUDMARegister` once on that workspace before warmup;
4. reuse the same handle for every ReduceGrad launch;
5. quiesce the bound stream, unregister, then destroy the communicator.

The registered region starts at the workspace pointer. All ranks use the same
layout and required byte count:

```text
registered workspace base (remote byte offset 0)
  remote-written ready state (one cache line per source and stage)
  receiver-written completion state (a separate cache line per source and stage)
  source-side UDMA GET scratch (a third pair of cache lines per target)
  two outbound payload stages per target rank
  two inbound payload stages per source rank
```

`udmaChunkBytes` is aligned, at least 1 MiB for non-tail chunks, no larger than
`UINT32_MAX`, and bounded by the workspace query. It defaults to a conservative
multi-MiB value and is a benchmark tuning parameter. Workspace growth is
`O(R * udmaChunkBytes)`, not `O(R * B * rowBytes)`. A source serializes its
valid `(projection, b, chunk)` sequence for one owner into that owner's two
outbound stages; different owners progress in parallel. The corresponding
receiver lane is indexed by source rank, so source and owner derive all remote
offsets without exchanging per-slot addresses.

Each target peer has exactly one persistent UDMA control AIV. It is the sole
writer of that peer's QP 0 state and performs outgoing data submissions plus
completion GET polling. This is required because the current public UDMA
wrappers hard-code QP 0 and update a shared SQ head. Receiver AIVs consume local
inbound stages and write local completion state, but never post WQEs.

For each UDMA chunk:

1. the peer control AIV copies source bytes through UB into its registered local
   outbound stage, flushes the stage, and calls
   `UDMAPutRegisteredSignalNbi` to copy the payload to the owner's inbound
   stage while atomically publishing the exact sequence;
2. it checks UDMA CQ completion before treating the local outbound stage as
   submitted;
3. a receiver AIV bounded-polls ready, invalidates the inbound payload cache
   lines, accumulates the payload into its disjoint owned output chunk in
   ascending `(src, b)` order, and publishes completion in a separate cache
   line;
4. the source control AIV uses checked UDMA GETs on the same target QP to poll
   that exact completion sequence before reusing the stage; missed polls use a
   bounded exponential scalar backoff to avoid a high-fan-in CQ polling storm;
5. two stages stay in flight: before posting chunk `n`, the source only drains
   chunk `n-2`, then drains the final two chunks at the end of the row.

Dedicated receiver AIVs remain active while control AIVs wait, so a sender wait
does not prevent the peer from making receive progress. Invocation magic plus a
monotonically ordered per-peer sequence prevents stale ready or completion
values from satisfying a later launch. Separating remote-written ready,
receiver-written completion, and GET scratch into distinct cache lines prevents
clean/invalidate operations from overwriting a concurrently updated field.

The current `UDMAQuiet` discards the return value from `UDMAPollCQ`. ReduceGrad
therefore adds a backward-compatible device helper that returns CQ completion
status and uses it on every data submission and completion GET; the existing
void helper remains unchanged for current callers. A nonzero CQ status becomes
a deterministic ReduceGrad device error rather than silent success.

## Kernel Data Path

The A5 kernel uses a persistent grid with a validated block dimension no larger
than the target's 64 AIV limit. The default and environment override follow the
Planner's bounded configuration style.

Peer sender AIVs perform:

1. source slot GM -> UB DataAsFlag packing;
2. bounded acknowledgement polling before stage reuse;
3. UB -> destination `peerMems[owner]` MTE3 writes;
4. local used-slot cleanup after every send has completed.

Peer receiver AIVs perform, for each owned expert/chunk tile:

1. load the local expert gradient tile into an fp32 accumulator in UB;
2. scan the UB-resident plan table in ascending `(src, b)` order;
3. bounded-poll and load matching local DataAsFlag records through MTE2;
4. use vector fp32 `Add` into the accumulator;
5. clear consumed embedded flags;
6. store the accumulator to the owned expert row with bounded tail handling.

For UDMA-selected projections, producer and receiver vector AIVs use the same
fp32 tiling and deterministic reduction order, while per-peer control AIVs own
all UDMA posting and acknowledgement state. Mixed projections execute in one
fused launch and share the plan scan, status, magic, and final used-slot cleanup.

All element and byte offsets use checked 64-bit arithmetic. UB tile size,
producer/consumer block allocation, and block dimension are performance tuning
parameters selected by target compile and A5 measurements, not hard-coded from
the CUDA implementation.

Sender/receiver core counts, UB record batches, fp32 tile size, and block
dimension are tuned together. Every MTE/vector dependency has an explicit event
boundary.

## Timeout and Error Handling

The unbounded wait inside the current generic DataAsFlag receive helper is not
used directly. ReduceGrad provides bounded producer acknowledgement and
consumer arrival polling, using the caller's wait budget and target-appropriate
cache maintenance.

Status values distinguish success, invalid device state, peer timeout, UDMA CQ
failure, and the offending peer/phase/slot. If progress fails, the affected
sender, receiver, or control AIV stops reusing the unsafe stage. Other ranks
eventually reach their bounded timeout. The caller treats the case as failed
and preserves communicator and registered-workspace lifetime until the existing
host completion protocol terminates or releases every rank.

No timeout path switches transports, performs a local-only reduction, clears a
possibly observed slot gradient, unregisters memory with work in flight, or
reports native success.

## Torch Integration

The Torch facade keeps `ProjectionBuffers` as the high-level tensor container.
Native V2 ReduceGrad validates gate, up, and down as contiguous fp32 tensors
with first dimension `E + B`; deprecated `*_reduce` fields are not used by V2.

The runtime creates one fused V2 argument structure and one native call. The
Plan owns a ReduceGrad status tensor alongside Planner status so both lifetimes
extend through asynchronous completion. The Buffer queries, allocates, zeros,
and collectively registers one UDMA workspace lazily when any projection
crosses the threshold. Registration is cached for compatible shapes, completed
before benchmark warmup, and never performed from a timed `reduce_grad()` call.
`quiesce()` checks both statuses after stream synchronization. `close()`
quiesces, unregisters the workspace, releases it, then destroys the communicator.

Because registration is collective, every rank must present the same projection
shapes and enter preparation in the same order. Shape changes that alter the
required workspace require a quiesce plus collective unregister/register cycle;
they are rejected while work is pending.

The caller may continue to hold complete `[E + B, ...]` tensors. Native code
only accesses the current rank's owned expert slice and the final `B` slot rows,
so V2 does not require global expert rows to be copied through the peer window.

## Benchmark Integration

The existing `tools/moonep` benchmark remains the end-to-end entry point.
ReduceGrad changes from a stub copy check to native distributed correctness and
performance checks.

Case configuration gains projection-shape inputs while retaining a small
default derived from `hidden_size`. Cases can use rank-2 flat rows or rank-3
MoonEP-style expert matrices.

The benchmark initializes deterministic values from `(rank, projection, row,
element)`. After ReduceGrad it verifies:

- every owned expert equals its local seed plus all matching source slots;
- non-owned expert rows are unchanged;
- used local slot rows are zero;
- unused local slot rows are unchanged;
- ReduceGrad status reports success on every rank.

Reports retain per-rank device-event samples and cross-rank maximum latency.
They add streamed bytes, chunk count, slot stride, block dimension, effective
bandwidth, same-node/cross-node identity, and a ReduceGrad-specific validity
flag. They also report the exact 1 MiB threshold, logical row bytes and selected
transport per projection, UDMA chunk/workspace bytes, registration status, and
confirm that registration time is excluded. Overall `transport_performance_valid`
remains false while other required MoonEP stages are stubs.

Performance work includes warmup, stable repeated measurements, block/tile
sweeps supported by bounded configuration, and retention of the fastest valid
configuration for representative small, tail, fan-in, and production-dimension
cases. Compile or simulator timing is not performance evidence.

## Verification

### Host and Source

- ABI layout and C header compatibility for V1 and V2.
- Layout tests for current-main, PR #90 512 MiB, and injected future capacities;
  1 MiB state reservation; compact incoming counts; record stride; 480-byte
  tails; chunk counts; and 64-bit expert offsets.
- Transport-boundary and UDMA workspace tests for 1 MiB minus one fp32 element,
  exactly 1 MiB, 1 MiB plus one fp32 element, mixed gate/up/down selections,
  aligned state/lanes, `UINT32_MAX` WQE bounds, and registry containment.
- Validation tests for missing peers, wrong topology, invalid plan/tensor/status,
  unsupported dtype, zero wait budget, missing or mismatched UDMA registration,
  `InitThread`, and runtime launch failure.
- CPU streaming oracle for ring, fan-in, duplicate, empty, asymmetric, and
  random plans.
- Source checks requiring Runtime V2 launch in Host, PR #90 DataAsFlag format,
  one capacity resolver, no literal window size, no unbounded wait, no launch
  code in the kernel, no HCCL dependency, one QP-0 owner per peer, and checked
  UDMA completion.
- Torch fake-runtime tests for one fused call, stream forwarding, status
  and workspace lifetime, collective registration outside timing, capability
  reporting, and benchmark correctness metadata.

### A5 Same Node

- CANN 9.1 Host/kernel build and installed dependency/RPATH inspection.
- 1-rank and 8-rank correctness for full chunks, tails, empty plans, high
  fan-in, repeated launches, mixed peer/UDMA projections, timeout injection,
  and CQ failure injection where supported.
- Device-event performance sweeps with retained shape, dtype, block/tile,
  chunk, build, driver, and SoC metadata.

### A5 Cross Node

- At least two A5/Ascend950 nodes in one TileXR communicator.
- Every cross-node `peerMems` pointer must be non-null.
- Peer-only correctness cases must reuse both halves across repeated launches,
  so they detect the same overwrite risk identified in PR #90 review.
- UDMA-only correctness cases must cover both stages, checked CQ completion,
  and repeated sequence reuse before their timing is marked valid.
- Ring, fan-in, asymmetric, random, repeated-magic, and tail cases must match
  the oracle on every rank for peer-only, UDMA-only, and mixed projections.
- At least one production inner shape such as `7168 x 3072` must exercise
  64-bit offsets and multi-chunk streaming with a memory-feasible `E/B`.
- Cross-rank maximum latency and effective bandwidth must be reported from
  stable warm measurements.

Cross-node support is not considered verified until this hardware matrix runs.

## Acceptance Criteria

1. `codex/moonep-reducegrad` builds the V2 Host and A5 kernel with CANN 9.1.
2. V1 ABI remains present and V2 capability reporting is unambiguous.
3. Native V2 selects peer memory for rows up to 1 MiB and registered UDMA for
   larger rows; active code contains no HCCL path, transport fallback, or
   hard-coded 100/512 MiB peer capacity.
4. Gate/up/down are reduced by one asynchronous Runtime V2 launch.
5. Host, oracle, ABI, source, and Torch tests pass.
6. Same-node A5 correctness and performance evidence is recorded separately.
7. Two-node A5 correctness passes before cross-node support is claimed.
8. The MoonEP benchmark reports real ReduceGrad correctness and timing while
   honestly retaining stub status for the other unimplemented stages.
9. The same layout code works with current main, PR #90's 512 MiB window, and
   the future issue #92 memory-mode capacity contract.
10. Cross-node support is claimed only after peer-only, UDMA-only, and
    mixed-transport ReduceGrad correctness matrices pass on real two-node A5
    hardware; the report derives this from case artifacts, never an environment
    assertion.
