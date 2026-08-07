# TileXR Planner V2 Design

## Status

Implemented by PR #96 and extended by the local Planner V2 delivery change.
This document records the Planner architecture, metadata and downstream
contract, synchronization protocol, source ownership, and the hardware
validation completed for that combined baseline. The detailed field-level
contract is in `moonep_planner_v2_downstream_contract.md`.

## Decisions

1. Planner V2 is an A5/Ascend950 direct-launch Planner implemented in TileXR.
   It does not depend on active sources under `reference/`.
2. The public API separates local algorithm scratch from caller-owned metadata
   scratch so their sizes and lifetimes are explicit.
3. Cross-rank Planner metadata uses the communicator's existing
   `CommArgs::peerMems[]` windows. Planner V2 does not register or transfer
   metadata through UDMA.
4. Every rank runs the same deterministic planning algorithm after publishing
   inputs and verifying a common call header.
5. Reusable magic-tagged barriers provide bounded data, status, ready, and
   final-consensus phases. Shared flag memory is not reset between calls.
6. Device-kernel implementation remains under `kernels/`; Runtime V2 launch
   and compiler-generated launch stubs are Host responsibilities under
   `host/`.
7. The MoonEP compatibility API remains available and adapts its single
   workspace contract to the explicit Planner V2 workspaces.
8. Metadata V2 adds destination-oriented `remoteExperts[R,B]` and owner-oriented
   `expertTargets[E/R,ceil(R/64)]` without changing the legacy optimized Plan
   descriptor or the MoonEP V1 ABI.
9. Planner owns placement metadata only. Expert Migration performs weight
   movement, Dispatch writes the final `tokenRemap`, and Combine consumes that
   remap; Planner does not implement those downstream data planes.

## Evidence and Target

- Repository baseline: C++14, CANN 9.1.0, and driver 25.5.0 or later.
- Planner kernel target: Ascend910A5/Ascend950 vector cores, selected through
  `TILEXR_MOONEP_PLANNER_SOC_TYPE`.
- Planner build is rejected for non-A5 SoCs.
- Planner behavior is checked against an independent CPU reference and focused
  Host, layout, algorithm, ABI, source, and multi-rank tests.
- The final local Host/source suite passes 12 of 12 focused Planner/EP tests.
- The A5/Ascend950 Planner data path has been validated on actual hardware at
  single-node 2 Rank, single-node 8 Rank, cross-cabinet 4-node 32 Rank, and
  cross-cabinet 16-node 128 Rank scales.
- Every validated Rank exited with code zero, emitted `PLAN_VALIDATION_PASS`,
  agreed on the committed epoch and global digests, and left no Planner process
  behind after validation.

## Scope

### Public Planner API

- `TileXRMoeEpPlanV2GetWorkspaceSize` validates static dimensions and returns
  local and metadata workspace sizes.
- `TileXRMoeEpPlanV2` validates the communicator, buffers, Plan identity,
  workspace capacity, peer windows, and stream before launching asynchronously.
- Planner launch APIs reject communicators that do not advertise
  `TOPO_910A5`; a kernel built for `dav-c310-vec` is never enqueued on 910B or
  another incompatible target.
- `TileXRMoeEpPlanV2WithMetadata` adds an ABI-versioned, counted metadata output
  descriptor while preserving the legacy optimized entry point.
- `EncodeMoonEPGlobalTokenId`, `DecodeMoonEPGlobalTokenId`, `DecodeMoonEPDst`,
  and `BuildMoonEPRouteDescriptor` provide one production implementation of the
  route encoding contract for Dispatch and Combine consumers.
- `TileXRMoonEpPlannerGetWorkspaceSizeV2` and `TileXRMoonEpPlannerV2` preserve
  the existing MoonEP-facing compatibility contract.

### Planning Algorithm

- Count routed tokens per expert and rank.
- Construct deterministic rank affinity from global rank ids.
- Allocate same-server moves before cross-server moves.
- Enforce prefetch-slot, destination-capacity, and optional per-rank-pair token
  limits.
- Append remaining local token segments and construct `dst`, `cuSeqlens`,
  destination-oriented `remoteExperts`, owner-oriented `expertTargets`, legacy
  `expertsToCopy`, `remoteStats`, and status outputs.
- Preserve the affinity order in metadata for reuse only after an epoch commits
  successfully with the same topology.

### Cross-Rank Protocol

- Publish one mailbox row per source rank into every target rank's owner peer
  window with MTE3, then consume all source rows from the current rank's local
  owner `peerMems[rank]` window with MTE2.
- Exchange the call header, tokens-per-expert row, global rank id, and status.
- Use four bounded collective phases: data, status, ready, and consensus.
- Re-publish the ready outcome and reduce final rank-local status before
  committing the epoch.

### Compatibility Layer

- Derive the historical MoonEP configuration from communicator rank count and
  `S/K/E`.
- Carve the optimized local workspace, metadata workspace, generated global
  rank ids, duplicate buffers, and status from one caller-owned workspace.
- Preserve legacy optimized `expertsToCopy[B]` as the current destination Rank
  row while copying the MoonEP V1 outward compatibility buffer as `[R,B]`.
- Validate the complete runtime before copying the generated global-rank map;
  use a synchronous H2D copy because its source is temporary Host storage.
- Copy compatibility outputs asynchronously on the caller's stream.

### Downstream Ownership

- Expert Migration may consume `remoteExperts[dstRank,B]` to build destination
  copy lists or `expertTargets[localExpert,W]` to discover owner-side targets.
- Dispatch decodes local `dst[S,K]`, performs the actual token placement, and
  writes `tokenRemap[NvS]` with encoded global token ids at final receive slots.
- Combine consumes Dispatch's `tokenRemap`, skips `UINT64_MAX`, decodes the same
  `R/S/K` tuple, and routes expert output back to the originating token route.
- Planner does not claim that metadata publication moved expert weights or token
  payloads. The native Dispatch, Combine, and weight-transfer implementations
  remain outside this component.

## Non-Goals

- Running Planner V2 on 910B or other non-A5 products.
- Using HCCL for Planner metadata or barriers.
- Registering Planner metadata with UDMA or adding a UDMA fallback.
- Using `creditmem`, producing `zeroFillRanges`, or registering `peerMems[]` as
  UDMA transfer targets.
- Producing `tokenRemap`, executing real expert-weight copies, or implementing
  Dispatch/Combine token data movement inside Planner.
- Resetting shared peer flag memory between calls.
- Increasing the active communicator ABI beyond
  `TileXR::TILEXR_MAX_RANK_SIZE`.
- Claiming cross-node correctness from Host, source-only, simulator, or
  single-node tests.
- Owning stream synchronization or device-runtime initialization inside the
  Planner API.

## Component Ownership

```text
src/include/tilexr_ep_plan.h
  Public configuration, Plan descriptor, status, and V2 C ABI

src/moonep/planner_v2/common/
  Shared POD types, workspace/mailbox layout, and Host/device algorithm

src/moonep/planner_v2/reference/
  Independent CPU oracle used by tests

src/moonep/planner_v2/host/
  Validation, API adaptation, Runtime V2 argument packing, and kernel launch

src/moonep/planner_v2/kernels/
  AscendC device kernel, peer publication, barriers, and algorithm execution
```

`host/planner_kernel_launch.cpp` is compiled by Bisheng together with the
device kernel so CANN 9.1 can keep using its compiler-generated registration
stub. Its source ownership is nevertheless Host-side: the device-kernel file
contains no `rtKernelLaunchWithFlagV2` call and no launch wrapper.

## Public Contract

Notation:

```text
S      tokens on each rank
K      experts selected per token
R      ranks in the Planner communicator
E      global experts; E must be divisible by R
B      prefetch slots per rank
W      expert-target bitmap words; W = ceil(R / 64)
Cap    token capacity per rank; required to equal S * K
NvS    encoded destination stride; NvS >= Cap
```

Inputs are caller-owned device buffers:

```text
topkExperts       int32 [S, K]
tokensPerExpert   int32 [E]
globalRankIds     int32 [R]
```

The legacy optimized Plan descriptor uses caller-owned device buffers:

```text
dst               int32 [S, K]
cuSeqlens         int32 [E + B]
expertsToCopy     int32 [B]      current destination Rank row
remoteStats       int32 [2]
status            int32 [8]
```

Metadata V2 adds counted caller-owned outputs without changing that descriptor:

```text
remoteExperts     int32  [R, B]  destination-oriented expert ids
expertTargets     uint64 [E/R,W] owner-oriented destination bitmap
```

Every metadata count is an element count, not a byte count. Metadata V2 does
not contain `tokenRemap`; Dispatch owns and fills that mapping after actual
receive-slot placement. The MoonEP V1 compatibility layer continues to copy a
complete `[R,B]` remote-expert matrix into its historical outward
`expertsToCopy` buffer.

The Plan descriptor also carries duplicate-group buffers used by the public
and compatibility ABI. All Plan pointers must be non-null and int32-aligned.
The caller owns every input, output, workspace, Plan object, communicator, and
stream until asynchronous work has completed.

The Plan identity must match the invocation:

```text
plan.s             == S
plan.k             == K
plan.r             == R
plan.e             == E
plan.b             == config.prefetchSlots
plan.cap           == config.rankTokenCapacity
plan.nvS           == config.nvS
plan.tokenPadding  == config.tokenPadding
plan.epoch         != 0
```

## Configuration and Limits

The optimized API requires:

- positive `R/S/K/E/B/Cap/NvS/tokenPadding`;
- `K <= 32`;
- `E % R == 0`;
- `Cap == S * K`;
- `NvS >= Cap`;
- `0 <= tokenRouteLimitPerPair <= Cap`;
- `cardsPerServer == 8`;
- `cardsPerCabinet == 64`;
- `crossCandidateCount == 3`;
- `R * NvS <= INT32_MAX` for destination encoding.

Workspace arithmetic is checked in uint64 before conversion to addressable
sizes. Layout-only tests may exercise up to 512 logical ranks, while an active
TileXR communicator remains limited to `TILEXR_MAX_RANK_SIZE` (currently 128).

## Workspace Layout

Both workspaces are aligned to 64 bytes.

Local workspace contains:

```text
expertCount
rankLoad
remainingTpe
alloc
remoteExpertSet
srcExpertCursor
dstExpertCursor
expertPhysicalBase
localExpertOrdinal
tokenSegments
routedPairTokens        optional
scratchStatus
```

Metadata workspace contains:

```text
planCallHeaders
tokensPerExpert matrix
globalRankIds
epochState
affinityOrder
localStatusByRank
barrierFlags
```

The `registeredMetaWorkspace` API name describes the ABI region. Planner V2
itself does not call `TileXRUDMARegister` and does not treat this buffer as a
UDMA transfer target.

## Peer Mailbox Layout

Each source rank owns one mailbox row in the data portion of every peer window:

```text
peerMems[target] + IPC_DATA_OFFSET + sourceRank * rowBytes

row:
  PlanCallHeader       128-byte stride
  globalRankId         sizeof(int32_t)
  status               64-byte stride
  tokensPerExpert      E * sizeof(int32_t)
  unused capacity      through a fixed 4 MiB row stride
```

The 4 MiB stride is `IPC_BUFF_MAX_SIZE / TILEXR_MAX_RANK_SIZE`, so header and
payload addresses do not depend on locally supplied dimensions. This lets
ranks compare mismatched call headers without first deriving conflicting row
offsets. Transfers still cover only the 512-byte-aligned prefix containing the
actual `E` entries; the fixed stride does not turn each publication into a
4 MiB transfer.

The Host validates that all `peerMems[0..R)` entries are non-null, that the
actual input prefix fits one fixed row, and that `R * rowBytes` fits the full
IPC data window. A source rank first writes its row under
`peerMems[sourceRank]`, loads through MTE2 into a UB relay, and uses MTE3 to push
the valid prefix into the same `sourceRank` slot under every target owner
window. A destination consumes all rows only from its local owner
`peerMems[rank]` mapping.

No Planner path reads `udmaInfoPtr`, `udmaRegistryPtr`, or a registered remote
memory handle.

## Barrier Protocol

Planner V2 reserves four phase families in the IPC flag region:

```text
data    publish and validate call inputs
status  publish and reduce rank-local algorithm status
ready   keep peer windows and the completed Plan mutually ordered
consensus re-publish ready outcomes and reduce final status before commit
```

For each phase and source rank, the slot address is derived from event id 4096,
the phase, rank count, and a 512-byte stride. A static assertion keeps the
maximum slot range below `IPC_DATA_OFFSET`.

Each call obtains a fresh magic through `TileXRCommNextMagic`. The published
64-bit value combines magic and phase. For each source slot, MTE3 writes the
flag directly to every target owner's `peerMems[targetRank]`; each rank polls
all source slots through MTE2 only from `peerMems[rank]`. Explicit event
ordering and cache maintenance surround both directions. Every wait is bounded
by `waitIterations`.

On timeout, status word 0 is `1000 + timedOutPeer`. Remaining status words may
contain the phase, peer, observed value, and address evidence needed for device
debugging. A timeout prevents epoch commit.

## Call Consistency and Epoch Commit

Every rank publishes a `PlanCallHeader` containing ABI version, dimensions,
configuration, epoch, and topology hash. Rank 0's header is the comparison
baseline. Any mismatch returns `PLAN_ERROR_CONFIG_MISMATCH` through the
device-visible status protocol.

The topology hash is an FNV-style hash of `globalRankIds`. A cached affinity
order is reusable only when:

- requested and committed epochs agree;
- the committed and requested headers match;
- the affinity-valid bit is set; and
- the saved topology hash matches.

The kernel writes `committedEpoch = epoch` only after all four phases complete
and the consensus reduction returns `PLAN_OK`. In particular, a rank that
times out during `ready` republishes that failure before peers evaluate the
commit condition.

## Algorithm Stages

The shared implementation is compiled for both Host tests and the AscendC
kernel through address/function macros. Its deterministic stages are:

1. Validate dimensions, ids, token counts, capacities, and workspace bounds.
2. Build expert counts, rank loads, and rank affinity.
3. Move feasible expert-token segments within each eight-card server.
4. Move remaining feasible segments across server groups using the configured
   candidate count.
5. Append tokens that remain on their home ranks.
6. Build destination-oriented remote expert sets, owner-oriented expert target
   bitmaps, cumulative sequence lengths, and encoded destinations.
7. Fill status and statistics, including partial-plan reasons where constraints
   prevent a complete move set.

The CPU reference is intentionally separate and checks output invariants and
exact results for representative balanced, biased, duplicate, capacity, and
partial-plan cases.

## Host Invocation Path

```text
TileXRMoeEpPlanV2
  -> PreparePlanLaunchContext
     -> TileXRGetCommArgsHost / TileXRGetCommArgsDev
     -> ValidatePlanHostArguments
  -> LaunchPlanKernel
     -> TileXRCommNextMagic
     -> launch_tilexr_ep_plan_kernel
        -> compiler-generated CANN launch stub or Runtime V2 branch
        -> tilexr_ep_plan_kernel
```

The public API returns validation and launch-enqueue errors synchronously.
Algorithm, mismatch, partial-plan, and timeout results are asynchronous and are
observed in `plan.status` after the caller synchronizes its stream or waits on
an appropriate event.

Planner V2 does not call `aclInit`, `aclFinalize`, `aclrtSetDevice`,
`aclrtResetDevice`, or `aclrtSynchronizeStream` behind the caller.

## Build and Installation

- Enable through `TILEXR_BUILD_MOONEP_PLANNER`.
- Bisheng compiles the kernel and Host launch translation units as GNU++17
  because CANN 9.1 AscendC headers require newer language features.
- The normal Host shared library remains C++14.
- `libtilexr-moonep-planner.so` links the generated
  `libtilexr_moonep_planner_kernel.so`, `tile-comm`, `ascendcl`, and `runtime`.
- Both libraries and the public Planner headers are installed through CMake.
- Installed runtime RPATH is `$ORIGIN`; the toolkit `devlib` directory is used
  only at link time and is not placed in RPATH/RUNPATH.

## Failure Model

Synchronous API failures cover:

- null or misaligned pointers;
- invalid dimensions/configuration/Plan identity;
- undersized workspaces;
- unavailable Host or device `CommArgs`;
- a communicator without `TOPO_910A5`;
- missing peer windows;
- a peer-mailbox payload or complete footprint outside the IPC data window;
- failure to obtain a new magic; and
- kernel enqueue failure.

Asynchronous status covers:

- cross-rank call-header mismatch;
- tokens-per-expert mismatch;
- encoded layout overflow;
- move-record overflow;
- internal algorithm invariant failure;
- constrained partial plans; and
- bounded peer timeout.

There is no automatic retry, UDMA fallback, or silent downgrade. All ranks must
use the same dimensions, configuration, epoch, call order, and stream ordering.

## Verification

### Source and Host

The final local validation command set is:

```text
cmake --build work/tests-ep -j 8
ctest --test-dir work/tests-ep --output-on-failure
git diff --check
```

It builds successfully and passes all 12 focused Planner/EP tests. Coverage
includes public ABI compilation, workspace layout and overflow, Host argument
validation, independent CPU-reference comparison, downstream route helpers,
metadata counts and compatibility semantics, no-UDMA/no-zero-fill source
guards, barrier ownership, epoch commit, and the 2/8/32/128 Rank harness.

The installed production library exports:

```text
TileXRMoeEpPlanV2
TileXRMoeEpPlanV2GetWorkspaceSize
TileXRMoeEpPlanV2WithMetadata
EncodeMoonEPGlobalTokenId
DecodeMoonEPGlobalTokenId
DecodeMoonEPDst
BuildMoonEPRouteDescriptor
```

### A5/Ascend950 Hardware Matrix

Validation used the current installed Planner, kernel, and `tile-comm` artifacts
from `/home/l00929943/TileXR-planner-v2-pr96-20260805`. Each Rank compared
legacy and metadata outputs with the CPU oracle and participated in epoch and
digest agreement.

| Scale | Placement | Result | Evidence phase |
| --- | --- | --- | --- |
| 2 Rank | one node, devices 0-1 | 2/2 pass | `2rank-singlehost-20260805-194320-1621428` |
| 8 Rank | one node, devices 0-7 | 8/8 pass | `8rank-singlehost-20260805-194514-1747897` |
| 32 Rank | four nodes across two cabinets | 32/32 pass | `32rank-crosscabinet-20260805-195207-48360` |
| 128 Rank | sixteen nodes across two cabinets | 128/128 pass | `128rank-crosscabinet-20260805-201251-53128` |

The 128-Rank run covered all eight requested nodes in cabinet C02-A07 and all
eight requested nodes in cabinet C02-A04. All 128 exit files contained zero,
all logs contained `PLAN_VALIDATION_PASS`, Rank 0 emitted `ALL_RANKS_PASS`, and
the requested and committed epochs matched. Post-run inspection found no
residual validation Planner process on any node.

### Synchronization and Evidence Boundary

Local-to-remote source synchronization used 16 Mutagen sessions only. The 29
executable-delivery files used by hardware validation (that set excludes this
canonical design document) had the same combined SHA-256 manifest on the local
workspace, Mutagen alpha mirror, and every remote node:

```text
723dadee76c15e23692c4d1a681d9c444c49bbfdee26f590ca777a45032685b4
```

Hardware logs are retained under
`/home/l00929943/tilexr-plan-evidence-c02-20260805`. The hardware claim covers
the Planner metadata and planning path exercised by that harness; it does not
claim that downstream Dispatch, Combine, or expert-weight movement kernels have
been implemented or validated.

## Acceptance Criteria

1. Public and compatibility APIs validate inputs and enqueue Planner work on the
   caller's stream without hidden synchronization.
2. Host launch code is outside the device-kernel source while retaining the
   CANN 9.1 compiler-generated registration path.
3. Same inputs and topology produce deterministic outputs matching the CPU
   reference.
4. All ranks agree on call headers and final status before epoch commit.
5. Peer publication and barriers use only `peerMems[]`, bounded magic-tagged
   waits, and explicit ordering.
6. Missing peers, mismatched calls, capacity errors, and timeouts are reported
   explicitly rather than hanging or falling back.
7. Build/install preserves C++14 Host compatibility, A5-only kernel targeting,
   `$ORIGIN` runtime lookup, and no active dependency on `reference/`.
8. Metadata V2 exposes complete destination and owner views without changing
   legacy ABI semantics; route helpers define the shared Dispatch/Combine
   encoding contract while leaving `tokenRemap` Dispatch-owned.
9. The validated 2/8/32/128 Rank matrix passes on actual supported hardware and
   validation claims remain scoped to the Planner path actually exercised.
