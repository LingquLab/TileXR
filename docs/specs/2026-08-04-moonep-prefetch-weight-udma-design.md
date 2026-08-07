# MoonEP Fused PrefetchWeight over Direct UDMA

## Goal

Replace the MoonEP PrefetchWeight stub with one native operator whose embedded
AIV binary is registered and launched through the runtime API and fetches the
three expert projections through TileXR registered-memory UDMA.
The operator must use one Host call, one kernel launch, one Plan scan, and one
completion phase for gate, up, and down weights. Integrate the native path into
the Torch facade and `tools/moonep` benchmark so its correctness and performance
are reported as real PrefetchWeight evidence.

## Scope

- Add one fused native PrefetchWeight Host/Kernel path under `src/moonep`.
- Extend the TileXR UDMA queue image so one fused kernel may use multiple
  independently owned QP/SQ/CQ workers without queue-state races.
- Replace the old `TileXRMoonEpPrefetchWeightV1` stub implementation in place.
- Add the packed registered-weight lifecycle to the MoonEP Torch facade.
- Exercise and time the fused operator in the existing MoonEP benchmark.
- Preserve C++14, CANN 9.1.0, and the A5 / Ascend950 UDMA boundary.
- Keep Dispatch, Combine, and ReduceGrad as their current stubs.

The active implementation must not include or link code from `reference/MoonEP`.
That checkout is comparison-only.

## Upstream Semantics Retained

The implementation keeps the observable MoonEP PrefetchWeight behavior:

- the Planner-owned `expertsToCopy[rank, slot]` selects the remote expert for
  each local prefetch slot;
- a negative expert id leaves the corresponding slot untouched;
- PrefetchWeight is separate from Dispatch and does not add an inter-rank
  barrier;
- all work is asynchronous on the caller's stream;
- gate, up, and down weights are prefetched together before expert execution.

MoonEP's CUDA kernel uses a multi-SM TMA pipeline over globally mapped NVLink
memory. TileXR instead submits large reads to device-managed UDMA queues.
Multiple AIV blocks must not concurrently update the same SQ/CQ state. The
TileXR kernel therefore supports multiple AIV workers only when every worker
owns a distinct QP/SQ/CQ tuple; no blockDim value is assumed to be optimal
before target-hardware measurement.

## Public ABI

Keep `TileXRMoonEpPrefetchWeightArgsV1` and
`TileXRMoonEpPrefetchWeightV1` while the project remains under development.
The V1 entry point is replaced in place with fixed gate, up, and down inputs:

```text
TileXRMoonEpPrefetchWeightV1(args, stream)
  args.comm
  args.plan
  args.gate
  args.up
  args.down
  args.flags
```

Each tensor is an in-place packed projection. The source rows and destination
slots live in the same tensor; no separate output tensor is accepted. The
three descriptors may have different expert shapes but must use the same
supported dtype and registered arena.

No release ABI has been published, so this development change does not advance
the MoonEP ABI version or shared-library SOVERSION. Planning, Dispatch,
PrefetchWeight, Combine, and ReduceGrad retain their V1 symbols and struct
contracts. Capability reporting marks Planning and PrefetchWeight native, and
leaves only Dispatch, Combine, and ReduceGrad in the stub mask. The Torch loader
continues to load `libtilexr-moonep.so.1` and calls the fused PrefetchWeight V1
symbol.

## Registered Weight Layout

Let `E` be the global expert count, `R` the Planner world size, and
`B = E / R` the experts owned by each rank. Each projection uses:

```text
[2 * B, ...expert shape...]
 rows [0, B)     locally owned source experts
 rows [B, 2 * B) local prefetch slots
```

For global expert `e`:

```text
owner rank       = e / B
owner-local row  = e % B
destination row  = B + slot
```

This compact UDMA layout replaces MoonEP's `[E + B, ...]` globally mapped
virtual layout. It does not replicate all global weights on every rank.

Gate, up, and down are aligned views into one flat packed allocation. Their
base offsets and expert-row byte sizes are 64-byte aligned. The full allocation
is registered once with `TileXRUDMARegister` after source weights are ready and
before measured iterations. Registration, socket AllGather, MR import,
unregistration, allocation, and packing are outside the PrefetchWeight hot
path. The current one-region TileXR UDMA registry remains unchanged.

The caller must construct the same packed layout on every rank. The Host path
derives each projection's byte offset from the local registry base and verifies
that every peer region contains the required source range. Re-registering an
unrelated TileXR region invalidates the weight arena and makes the next
PrefetchWeight call fail validation.

## Host Contract

Before launch, the Host path validates:

- communicator, Plan, stream, and all three projection descriptors;
- A5 / Ascend950 UDMA capability and a valid registered-memory registry;
- Plan rank/world/B consistency with `CommArgs`;
- supported BF16 or FP16 dtype, contiguous rank 2-4 tensors, and first
  dimension exactly `2 * B`;
- nonzero, checked expert-row byte counts no larger than `UINT32_MAX`;
- 64-byte projection offsets and expert-row alignment;
- all local and remote registered-region bounds;
- exact in-place source/slot ownership, with no separate output allocation.

The Host path chooses `blockDim` from `B`, the provisioned UDMA worker count,
and the measured target default. A bounded tuning override may select a smaller
worker count but may never exceed the independently provisioned QPs. The Host
does not synchronize or copy `expertsToCopy` merely to count active slots;
sparse workers exit in the Kernel.

The Host embeds the linked AIV binary, registers it once with
`rtDevBinaryRegister` and `rtFunctionRegister`, and launches it with
`rtKernelLaunchWithFlagV2`. No generated `<<< >>>` Host wrapper is linked.

Validation or launch errors are returned synchronously. UDMA completion errors
are written to the Plan status tensor and are surfaced by the existing
event/stream synchronization boundary. PrefetchWeight never synchronizes the
caller's stream internally.

## UDMA Worker Queues

The existing `UDMAInfo.qpNum` and `qpIdx` contract is extended from its current
single-QP image to a configurable worker count. For every local route/EID, the
transport provisions an independent QP, SQ, and CQ for each worker. Device
images flatten queue and remote-memory information as `[peer][qpIdx]`.

Existing UDMA callers continue to use `qpIdx = 0` and retain their behavior.
PrefetchWeight assigns one `qpIdx` to each submitting AIV block. Queue head,
tail, WQE count, CQ consumer state, and remote TPN/token data are never shared
by two producer blocks.

The worker count is selected before communicator initialization and is bounded
by target resource limits. Supported tuning candidates are 1, 2, 4, and 8.
The final default is chosen from A5 / Ascend950 benchmark evidence, not from a
source-level assumption. Queue creation and MR import remain outside the
measured PrefetchWeight interval.

## Kernel Data Flow

The fused runtime-registered kernel runs on one or more AIV blocks:

1. Initialize PrefetchWeight status to success.
2. Each block reads its strided subset of the local
   `expertsToCopy[rank, B]` row.
3. For every active slot, derive owner rank and owner-local expert row.
4. Submit one registered-memory `UDMAGetNbi` per gate/up/down expert row through
   the block's exclusive `qpIdx`.
5. Retain a used-peer set and submit the block's reads before completion
   polling.
6. Call status-returning UDMA quiet operations on the same exclusive QP.
7. Complete the required cache-visibility and status publication protocol only
   after every worker's reads have completed.

The large expert-row transfer is one WQE. Shapes whose single projection row
exceeds the UDMA `uint32_t` SGE length are rejected instead of silently split
into another algorithm. Sparse and duplicate slots remain valid. An unused
slot is not written.

No IPC peer-window, SDMA, HCCL, host memcpy, UDMA PUT, transport fallback, or
registration call appears in the PrefetchWeight kernel path. If direct UDMA is
unavailable or the packed arena is not the active registered region, the call
fails explicitly.

## Torch Integration

The Torch facade adds a packed projection owner that retains the flat backing
allocation, the three aligned tensor views, and the UDMA handle. It supports
loading local source rows directly so steady-state PrefetchWeight performs no
packing or local copy.

`TileXRMoonEPBuffer.prefetch_weight`:

- requires the packed registered projection owner;
- launches one fused native call;
- mutates the three slot ranges in place;
- returns the same projection object for expert execution;
- retains the Plan, backing allocation, and views until stream completion.

Buffer close first reaches the existing quiescence boundary, then unregisters
the weight arena, and only then destroys the communicator. Registration is
collective and must be called by every rank in identical order.

## Benchmark Integration

`tools/moonep` allocates and registers the packed arena once per benchmark
case, fills each rank's owned source rows deterministically, and leaves slot
rows at a sentinel value. Every coordinated iteration calls the fused operator
exactly once.

Correctness checks compare each active slot with the selected owner's source
row for all three projections and verify that negative slots retain the
sentinel. Reports identify PrefetchWeight as native and include:

- expert shape, dtype, active slot count, and transferred bytes;
- per-rank PrefetchWeight latency samples and cross-rank maxima;
- p50/p90/p99 latency and effective GB/s;
- UDMA registration state and target SoC evidence.

The benchmark covers small and production-shaped rows, including MoonEP's
representative `512x512`, `1024x3072`, and `3584x3072` BF16 expert shapes.
Warmup, stream synchronization, and timing follow the existing benchmark
boundary. Registration and deterministic initialization are excluded from the
measured interval.

Overall transport performance remains invalid while Dispatch, Combine, and
ReduceGrad are stubs. PrefetchWeight's own native correctness and performance
are reported independently rather than being hidden by that aggregate flag.

## Verification

1. Host-only ABI, layout, overflow, registry-bound, sparse-slot, duplicate-slot,
   capability, and lifecycle tests.
2. Python fake-runtime tests proving one V1 call, one registration lifecycle,
   in-place results, Plan lifetime, and removal of the old stub behavior.
3. CANN 9.1 target compilation of Host and A5 kernel artifacts, plus symbol,
   SONAME, dependency, and RPATH inspection.
4. Bounded A5 / Ascend950 correctness on representative full, sparse, unused,
   and duplicate slot plans for BF16 and FP16.
5. Warmed hardware sweeps of 1, 2, 4, and 8 UDMA workers and valid blockDim
   choices for the benchmark shapes above. Retain raw samples and profiler
   evidence, choose the default from stable cross-rank maxima, and verify that
   extra workers do not regress sparse cases disproportionately.

Host tests and target compilation do not prove UDMA data-plane correctness or
performance. Same-node hardware does not prove cross-node UDMA behavior; those
claims require the corresponding topology to be exercised.

## Non-Goals

- Implementing Dispatch, Combine, or ReduceGrad.
- Expanding TileXR UDMA to multiple simultaneously registered regions.
- Allowing multiple AIV blocks to share one mutable SQ or CQ.
- Falling back to peer memory, SDMA, or local-copy stubs.
- Reproducing MoonEP's CUDA TMA implementation on Ascend.
- Claiming full MoonEP transport performance while other stages remain stubs.
