# TileXR MoonEP Torch Benchmark and Peer-Memory Planner Design

## Status

Approved for implementation on 2026-08-02 through the user's successive
Planner, peer-memory, no-HCCL, forward/backward, and no-padding decisions.

## Decisions

1. The existing A5 direct-launch Planner is integrated into this repository and
   used by the Torch benchmark. Planning is not a Python stub.
2. Cross-node Planner metadata uses TileXR `CommArgs::peerMems[]` memory
   semantics. It does not register memory with UDMA and has no UDMA fallback.
3. Dispatch, PrefetchWeight, Combine, and ReduceGrad are versioned native C ABI
   stubs in this phase. Their public contracts and stream lifetimes are real;
   their transport and math bodies are replaced by later operator work.
4. The benchmark uses TileXR bootstrap and communicators directly. It does not
   create an HCCL process group or use HCCL for data or control.
5. The benchmark covers the complete forward and backward protocol and keeps a
   Planner result alive until both backward branches finish.
6. Current hardware acceptance is bounded to the hardware actually available.
   Interfaces and topology metadata must not prevent later 256P, 512P, or
   1024P hierarchical implementations.

## Evidence and Target

- Active repository target: C++14, CANN 9.1.0, driver 25.1.rc1 or later.
- Temporary Planner source: the implementation developed in worktree `9a27`.
  The unpadded direct-launch interface uses new
  `TileXRMoonEpPlannerGetWorkspaceSizeV2` and `TileXRMoonEpPlannerV2`
  symbols so an already deployed library with the historical incompatible
  signatures cannot be called accidentally.
- Planner semantic baseline: `MoonshotAI/MoonEP` commit
  `0f385f038fc33bec22e3bcf5a07a8a22693e754c` under comparison-only
  `reference/MoonEP`.
- Development host: `root@141.61.49.192`, with 8 visible Ascend950PR devices,
  CANN 9.1.0, and deployed Planner libraries under
  `/home/d00520898/TileXR-MoonEP-Planner/install/lib64`.
- The development host reports driver `25.1.rc1.b188`, which meets the
  repository's supported 25.1.rc1 baseline. Results from it remain development
  evidence rather than a release-validation claim.
- The host's default Python 3.13 environment has neither `torch` nor
  `torch-npu`. A matching isolated environment is a prerequisite for real
  Torch-NPU validation.

## Scope

### Planner

- Bring the A5 Planner source, build option, public header, Host launcher,
  kernel, CPU oracle, and focused tests into the active tree.
- Preserve MoonEP tie-breaking while returning exact, unpadded route groups.
- Remove the Host-only cross-node rejection.
- Require every `peerMems[rank]` entry used by the Planner to be non-null.
- Use the peer window for TPE publication, ready flags, and remote TPE reads on
  both same-node and cross-node paths.
- Add bounded cross-rank waiting and device-visible status reporting so a
  missing peer does not spin forever.
- Fix dispatched capacity at `S * K`. Logical route layout does not have an
  alignment parameter; only hardware-required workspace and DMA alignment is
  retained internally.

### Torch Integration

- Add a lightweight Python package under
  `integrations/moonep_torch/tilexr_moonep`.
- Reuse the repository's `ctypes` pattern: load `libtile-comm.so` with
  `RTLD_GLOBAL`, load the MoonEP libraries, pass `Tensor.data_ptr()`, and use
  the current Torch NPU stream.
- Own communicator, workspace, Plan, completion, and cleanup lifetimes in one
  `TileXRMoonEPContext`/`TileXRMoonEPBuffer` facade.
- Keep native providers replaceable per operator without changing benchmark
  calls.

### Native Stub Operator ABI

- Add an installed public header and `libtilexr-moonep.so`.
- Define versioned POD descriptors for context, Plan tensors, projections, and
  optional completion/status storage.
- Export Dispatch, PrefetchWeight, Combine, and ReduceGrad functions using raw
  device pointers, checked 64-bit sizes, `TileXRCommPtr`, and caller-owned
  `aclrtStream`.
- Stub implementations validate the complete contract and enqueue only bounded
  local copy/zero work needed to keep forward and backward runnable. They must
  never report themselves as real communication implementations.

### Benchmark

- Add a JSON-driven benchmark under `tools/moonep`.
- Support deterministic inputs, case selection, warmup, measured iterations,
  correctness checks, per-stage timing, end-to-end timing, per-rank artifacts,
  and cross-rank maximum percentiles.
- Use an external/local process launcher with `RANK`, `WORLD_SIZE`,
  `LOCAL_RANK`, and a common `TILEXR_COMM_ID`. Do not call
  `torch.distributed.init_process_group`.
- Record the implementation of every stage (`native` or `stub`) and mark the
  current full-flow result as transport-performance-invalid while any required
  stage is a stub.

## Non-Goals

- Real Dispatch, PrefetchWeight, Combine, or ReduceGrad communication kernels.
- Registered custom-operator packaging, OpDef, or ACLNN lifecycle.
- HCCL bootstrap, HCCL collectives, or HCCL communicator extraction.
- UDMA for Planner metadata, including fallback use.
- Claims that a single-node, host, simulator, or virtual-rank run validates a
  cross-node data plane.
- Removing the current 128-rank TileXR ABI limit in this phase.

## Invocation Path

The authoritative invocation path is direct launch.

The Torch caller owns:

- current NPU device and current stream selection;
- tensor shape, dtype, contiguity, device, and storage-offset validation;
- Planner workspace allocation and lifetime;
- output allocation and Plan lifetime;
- communicator creation/destruction;
- asynchronous error observation at event/stream synchronization;
- identical collective call order on all ranks.

No component in this path calls `aclInit`, `aclFinalize`, `aclrtSetDevice`, or
`aclrtResetDevice` behind Torch. Those lifetimes remain owned by Torch-NPU.

## Public Tensor Contract

Notation:

```text
S     input tokens per rank
K     routed experts per token
E     global experts in the Planner group
R     ranks in the Planner group
B     E / R for the training path
NvS   dispatched capacity returned by the Planner workspace query
H     hidden size
```

Planning inputs:

```text
topk_experts       int32 [S, K]
tokens_per_expert  int32 [E]
```

Planning outputs:

```text
dst                int32 [S, K]
cu_seqlens         int32 [E + B]
experts_to_copy    int32 [R, B]
remote_stats       int32 [2]
planner_status     int32 [1]
```

`NvS` is exactly `S * K`. Each `cu_seqlens` entry is the real cumulative end
for its group; empty groups repeat the preceding end. No route-group padding,
padding capacity, or padding cleanup output exists in the ABI, Plan, Python
objects, benchmark configuration, report schema, or CPU oracle.

Host calculations use checked 64-bit arithmetic. `dst` encoding remains signed
int32 and is cast only after checking `R * NvS <= INT32_MAX + 1`.

The Python `MoonEPPlan` owns all output tensors, dimensions, epoch, backend
identity, and opaque native payload. Workspace is scratch and may be cached per
static configuration after the kernel completes.

## Peer-Memory Cross-Node Planner Protocol

### Host Preconditions

For the temporary A5 Planner, cross-node execution is accepted only when:

1. the communicator reports the A5/Ascend950 topology capability;
2. `rank`, `rankSize`, `localRank`, and `localRankSize` are valid;
3. every `CommArgs::peerMems[rank]` required by the Planner is non-null;
4. `E * sizeof(int32_t)` fits inside the data portion beginning at
   `IPC_DATA_OFFSET`;
5. all ranks use identical `S/K/E`, call count, and stream ordering.

The Planner does not inspect `udmaInfoPtr`, `udmaRegistryPtr`, or any registered
memory handle. A missing peer window returns a clear initialization/not-support
error before launch.

The existing TileXR communicator currently attempts to open every global peer
window. Therefore a successfully initialized communicator already establishes
the non-null address precondition. Cross-node hardware validation must still
prove that the mapped address provides the required memory semantics on the
selected product and driver.

### Window Layout

Each rank publishes its local `[E]` TPE row at:

```text
peerMems[rank] + IPC_DATA_OFFSET
```

The first `IPC_DATA_OFFSET` bytes remain owned by reusable magic-tagged TileXR
flags. Planner ready uses a dedicated, collision-checked event id and a magic
from `TileXRCommNextMagic`; shared flag memory is never reset between rounds.

### Device Ordering

The publishing block performs:

1. GM-to-UB copy of local TPE;
2. UB-to-local-peer-window copy;
3. explicit MTE3 completion;
4. all-AIV `SyncAll`;
5. ready flag publication;
6. ready waits for every rank;
7. direct GM-to-UB reads from each `peerMems[sourceRank]` TPE row.

The ready flag is the release/acquire boundary: payload completion must precede
the flag write, and the flag observation must precede the remote TPE read. Any
cache maintenance required by the exact CANN 9.1 peer-memory mapping is added
only from target-header and two-node hardware evidence; it is not replaced by
UDMA operations.

### Timeout and Status

The current generic `SyncCollectives::WaitSyncFlag` spins without a bound. The
temporary Planner uses a Planner-specific bounded poll and caller-owned
`planner_status`. The direct-launch ABI accepts a wait budget; its synchronous
return reports validation/launch errors, while status reports asynchronous
peer timeout after the caller reaches its stream synchronization boundary.

On timeout:

- no later Planner phase runs;
- one deterministic status identifies timeout and peer rank;
- Torch surfaces the error after event/stream synchronization;
- all ranks are expected to abort the current benchmark case;
- there is no UDMA retry or fallback.

## Forward and Backward Flow

```text
Forward
  Router
    -> Planning (real, once)
    -> Dispatch (stub in this phase)
    -> PrefetchWeight (stub in this phase)
    -> Expert forward
    -> Combine (stub in this phase)

Backward
  grad_output
    -> Dispatch(saved Plan; no Planning and no PrefetchWeight)
    -> Expert backward
    -> Combine(saved Plan) for input-hidden gradient
    -> ReduceGrad(saved Plan) for duplicate expert gradients
```

Combine sums K hidden paths but does not multiply route weights. The benchmark
routes optional weights through a second Dispatch invocation (`[S,K] -> [NvS]`),
applies them to Expert forward output before Combine, and reuses the same
dispatched weights on the Expert-backward input. A second Combine invocation
restores optional weights to `[S,K]`. The dtype/shape pair distinguishes hidden
and weight invocations without another ABI revision.

The Torch Buffer binds asynchronous work to one current NPU stream at a time,
retains Plan/input/output/workspace objects through completion, and synchronizes
plus checks every pending Planner status before destroying the communicator.
After an explicit synchronization the Buffer may bind a new current stream.

The Plan remains alive until both backward branches and ReduceGrad complete.
The same Buffer is not reentrant; calls sharing its communicator and peer
window are stream ordered.

## No-HCCL Launch and Reporting

The local A5 launcher spawns one process per device. All processes receive the
same reachable `TILEXR_COMM_ID`; TileXR's socket exchange builds the
communicator directly. Rank files and launcher-side aggregation replace HCCL
reductions for benchmark statistics.

After every rank reaches its local NPU synchronization boundary, benchmark
workers join a separate TCP completion rendezvous before destroying the TileXR
communicator. Arrivals carry a public launch id, case id, rank,
local-quiescence state, and case outcome, and are authenticated with an
HMAC-SHA256 secret that is never sent or written to artifacts. The coordinator
releases communicator destruction only when every rank reports
`quiesced=true`, and propagates any case failure as a global abort. A rank that
cannot prove quiescence keeps its communicator alive and signals the managed
launcher to terminate all workers together. Artifact/sentinel failures cannot
bypass that hold. This host-only lifecycle barrier keeps peer windows alive
until all Planner kernels have terminated. It carries no Planner payload, is
outside the measured interval, and is not a communication fallback for any
MoonEP stage.

Reports include:

- TileXR and MoonEP ABI versions;
- Git SHA, CANN, driver, Torch, Torch-NPU, and SoC evidence;
- global/local rank metadata and endpoint;
- case dimensions and seed;
- warmup and measured iterations;
- raw per-rank samples and per-iteration cross-rank maxima;
- p50/p90/p99 and failure reason;
- per-stage `native`/`stub` identity;
- `peer_memory_cross_node`, `cross_node_validated`, and
  `transport_performance_valid` flags.

## Scale-Compatible Topology

The current `CommArgs` and communicator reject more than 128 ranks. The public
Torch context nevertheless records separate global and group metadata:

```text
global_rank, global_world_size
node_rank, node_count
local_rank, local_world_size
planner_group_rank, planner_group_size
lane_group_rank, lane_group_size
```

The intended later topology for 8 devices per node is:

```text
local group: 8 ranks within a node
lane group:  same local rank across nodes
```

At 1024P each lane group has 128 ranks. The current direct-launch Planner uses
at most 64 ranks per planner group because `blockDim <= 64`; a later hierarchy
can therefore split one lane group into two planner groups without changing the
public global/lane metadata. The formal hierarchical Planner and relay protocol
are not implemented in this phase. Static tests cover 256P/512P/1024P metadata,
planner groups bounded to 64 ranks, and int32 capacity limits without allocating
global payload buffers.

## Build and Installation

- Preserve C++14 and CANN 9.1 compatibility.
- Add `TILEXR_BUILD_MOONEP_PLANNER` and `TILEXR_BUILD_MOONEP`, both default
  `OFF`.
- Keep the Planner independent of `src/ep` and all active targets independent
  of `reference/`.
- Install Host libraries with embedded pure-AICore kernels plus public headers through
  normal CMake install rules. Do not install Bisheng Host-wrapper kernel SOs.
- Runtime RPATH is `$ORIGIN`; no toolkit `devlib` directory appears in RPATH or
  RUNPATH.
- Planner build remains A5-only. Torch/source tests remain runnable without an
  NPU where their dependencies are available.

## Verification

### Source and Host

- Planner layout, overflow, status, peer-window capacity, and cross-node Host
  validation tests.
- Fake `CommArgs` matrices for same-node, cross-node-all-peers-ready, missing
  remote peer, invalid locality, and `R/NvS` overflow.
- Public C ABI compile and source-ownership guards.
- Torch fake-runtime tests for dtype/device/contiguity/current-stream checks,
  output allocation, Plan reuse, cleanup, and exact operator call order.
- Forward/backward benchmark smoke with deterministic stub providers.
- Static 256P/512P/1024P topology and encoding tests.
- Static and launcher tests for mapping 16 logical ranks onto 8 physical
  devices with two processes per device.

### A5 Single Node

- CANN 9.1 target compile.
- Existing Planner CPU-oracle matrix at 1/4/8 ranks.
- Torch 8-rank native Planning plus four native stubs.
- Torch and C++ Planner runs with 16 logical ranks oversubscribed onto the 8
  physical devices. Reports must identify this as oversubscription, not 16P
  physical-hardware performance.
- Full forward/backward repeated rounds with Plan reuse.
- Current-stream/event timing and cross-rank maximum aggregation.
- `readelf`/`ldd` checks for `$ORIGIN`, no `devlib` RPATH, and no HCCL
  dependency in the benchmark path.

### A5 Cross Node

- At least two A5/Ascend950 nodes using one TileXR communicator and the same
  `TILEXR_COMM_ID`.
- Explicit assertion that every remote Planner `peerMems` address is non-null.
- A bounded peer-window payload and ready-flag probe before the full Planner.
- Five Planner outputs compared with the CPU oracle for balanced, biased,
  duplicate, and repeated-magic cases.
- Peer loss or skipped rank test proving bounded timeout and error propagation.

Cross-node behavior remains unvalidated until this matrix runs on actual
hardware. Single-node results cannot satisfy it.

Remote validation uses a new, isolated directory on `root@141.61.49.192`; it
does not overwrite `/home/d00520898/TileXR-MoonEP-Planner` or its install tree.

## Acceptance Criteria

1. The active repository builds and installs the real temporary A5 Planner and
   four versioned native stub operators.
2. Torch calls the real Planner on its current NPU stream through TileXR with
   no HCCL process group.
3. The benchmark runs the defined forward and backward protocol and reuses the
   same Plan in backward.
4. Same-node 8-rank Planner outputs remain byte-for-byte equal to the oracle.
5. Cross-node Planner uses only peer-memory payload/flags and contains no UDMA
   registration or transfer calls.
6. Missing peer windows and ready timeouts fail explicitly rather than hang or
   fall back.
7. Results state exactly which operators are stubs and which hardware boundary
   was validated.
8. The public context and static tests preserve a path to hierarchical
   256P/512P/1024P without claiming those scales are implemented.
