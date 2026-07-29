# EP Dispatch Memory Reference Port Design

## Goal

Port the complete reachable non-quantized A5 full-mesh dispatch behavior from
`reference/moe_distribute_dispatch_v2_full_mesh_3510_simplified.h` into
`src/ep/kernels/tilexr_ep_dispatch_memory_kernel.cpp`.

The TileXR kernel is launched directly through
`launch_tilexr_ep_dispatch_memory_kernel`. It does not expose the reference
implementation's separate `Init()` and `Process()` calling convention.

## Supported Scope

The first version supports:

- FP16 and BF16 input/output with identical input and output types;
- normal MoE experts;
- shared experts, including multiple ranks per shared expert;
- no active mask, token mask `[bs]`, or expert mask `[bs, topK]`;
- count and prefix-sum forms of `expertTokenNumsOut`;
- the reference AIV split between dispatch and count/cumsum work;
- the reference 512-byte DataAsFlag token slots and state-window protocol;
- the reference local output compaction order.

The first version rejects:

- all quantization modes;
- smooth-scale input and scale outputs;
- differing input and output types;
- TP execution;
- elastic rank remapping, zero-compute experts, and performance recording,
  which are already removed from the simplified reference source.

## Kernel Organization

`tilexr_ep_dispatch_memory_kernel.cpp` contains the ported implementation.
The obsolete `tilexr_ep_dispatch_memory_helpers.h` chunk/source-slot protocol
is deleted and is not used as an implementation reference.

The kernel constructs a `TPipe` and invokes one internal `Run()` entry. `Run()`
contains the reference initialization work followed by the reference process
work. The following reference functions retain their algorithms, AIV ownership,
and call relationships:

- `SetTilingDataAndCal`
- `SetDataStatus`
- `TokenToExpert`
- `SplitToCore`
- `SendToSharedExpert`
- `SendToMoeExpert`
- `CalcSendTokenBufNum`
- `AllToAllDispatch`
- `AllToAllDispatchA5`
- `CalTokenSendExpertCnt`
- `CalAndSendCntByRank`
- `BufferInit`
- `WaitDispatchClearStatus`
- `GatherSumRecvCnt`
- `GetCumSum`
- `WaitDispatch`
- `CalRecvAndSetFlag`
- `CalCumSum`
- `WaitCumSumFlag`
- `SetValidExpertInfo`
- `CheckDataArriveWithFlag`
- `CopyInAndOut`
- `WaitAndFormatOutput`
- `RunPosRecord`
- `LocalWindowCopy`
- token-mask and expert-mask calculation helpers

Quantization-only functions and branches are removed rather than stubbed.

## Communication Context Mapping

The reference A5 context places a one-MiB state area before the data window.
TileXR reproduces that relationship inside each peer's IPC data region:

```text
peerWindowBase(rank) = commArgs.peerMems[rank] + IPC_DATA_OFFSET
stateWindow(rank)    = peerWindowBase(rank)
dataWindow(rank)     = peerWindowBase(rank) + 1 MiB
statusDataSpace      = stateWindow(selfRank)
```

The first one MiB therefore keeps the reference layout:

- ping state 0: `[0, 384 KiB)`;
- ping state 1: `[384 KiB, 768 KiB)`;
- per-AIV run state from `768 KiB`;
- cumsum exchange from `868 KiB`;
- cumsum completion flags from `876 KiB`.

The mapping does not use or overwrite TileXR's flag region before
`IPC_DATA_OFFSET`.

The data window starts after the one-MiB state area. Its two halves use the
reference `dataState * (totalWinSize / 2)` selection and retain the combine
reserve prefix and the dispatch DataAsFlag layout.

## Local Workspace

The reference cumsum path requires `workspaceGM`, replicated once per AIV.
TileXR reserves this workspace at the tail of the local IPC data region:

```text
workspaceStatusNum = epWorldSize * moeExpertNumPerRank
workspaceBytes = align32(aivNum * workspaceStatusNum * sizeof(int32_t))
totalWinSize    = IPC_BUFF_MAX_SIZE - 1 MiB - workspaceBytes
workspaceGM     = local dataWindow base + totalWinSize
```

Only the local rank accesses this workspace. The reservation nevertheless uses
the communicator-wide maximum receive-status count rather than the local
`rscvStatusNum`. This keeps `totalWinSize` and both dispatch-half offsets
identical on shared-expert and MoE-expert ranks, so remote payload addresses
match the receiver's polling addresses. Host validation checks that each data
state half can contain the combine reserve and every reference expert segment
before launch.

## Launch Interface

The internal launch interface becomes:

```cpp
void launch_tilexr_ep_dispatch_memory_kernel(
    uint32_t blockDim,
    void *stream,
    GM_ADDR commArgs,
    GM_ADDR x,
    GM_ADDR expertIds,
    GM_ADDR xActiveMask,
    GM_ADDR expandXOut,
    GM_ADDR expertTokenNumsOut,
    GM_ADDR epRecvCountsOut,
    GM_ADDR assistInfoForCombineOut,
    int64_t bs,
    int64_t h,
    int64_t topK,
    int64_t moeExpertNum,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t globalBs,
    int64_t expertTokenNumsType,
    int64_t activeMaskType,
    int64_t dtype,
    int64_t magic);
```

`CommArgs` supplies rank, world size, and peer addresses. `blockDim` supplies
the reference `aivNum`. Window sizes, UB sizes, core-group sizes, and offsets
are derived by the exact reference formulas and validated on the host.

`magic` is the communicator-wide invocation sequence. Its low bit selects the
reference ping-pong state half consistently across ranks; TileXR IPC buffers do
not provide the pre-synchronized state bit assumed by the HCCL/MC2 context.

The old route-count, chunk-count, offset, source-slot, payload, and total-size
arguments are deleted. `magic` is retained solely for communicator-wide state
selection.

Each receive-count status occupies one 32-byte block. The last two `int32`
fields store the count followed by the `float(1.0)` arrival flag. This adapts
the reference status block to TileXR's existing payload-before-tail-flag
ordering. The sender copies the complete block, or strided complete blocks,
with one `DataCopy`; the receiver polls the tail flags before consuming the
adjacent counts. The
memory dispatch path must not add a separate `SyncCollectives` ready message,
because that would split one reference transaction into a count write followed
by another remote flag write.

## Active Mask Contract

The public API adds an active-mask type because a pointer alone cannot identify
the reference tiling flags:

```text
NONE   = xActiveMask must be null
TOKEN  = xActiveMask points to [bs]
EXPERT = xActiveMask points to [bs, topK]
```

Token and expert mask modes are mutually exclusive, matching the supported
reference tiling cases.

## Shared Expert Contract

Shared expert ranks are the leading EP ranks, as in the reference code.
Validation requires:

- `sharedExpertNum == 0` iff `sharedExpertRankNum == 0`;
- `sharedExpertRankNum % sharedExpertNum == 0` when shared experts exist;
- `sharedExpertRankNum < rankSize`;
- `moeExpertNum % (rankSize - sharedExpertRankNum) == 0`.

The port retains `rankNumPerSharedExpert`, `idInSharedGroup`, the shared-expert
AIV allocation formula, and the reference destination-rank formula.

## Output Compatibility

`expandXOut`, `expertTokenNumsOut`, and `epRecvCountsOut` keep the reference
ordering and values. Reference `sendCountsOut` maps to TileXR
`epRecvCountsOut`.

TileXR keeps its four-int `assistInfoForCombineOut` record:

```text
[sourceRank, sourceTokenIndex, topKIndex, expertId]
```

The first three fields are the reference `expandIdxOut` triple. The fourth
field is the TileXR extension required by the current combine path. Shared
expert records use the same `topK + sharedExpertIndex` convention as the
reference code.

## Host Validation

The memory-dispatch host path:

- rejects quantization and scale-related inputs;
- rejects TP;
- validates EP rank/world values against `CommArgs` when explicitly supplied;
- derives `globalBs` as `bs * rankSize` when the public API passes zero;
- validates the active-mask pointer/type pair;
- validates all reference UB and IPC-window size formulas;
- obtains the A5 vector-core count and launches that exact block dimension.

## Testing

Tests are added or updated before implementation to cover:

- the reduced launch signature and removal of chunk/source-slot parameters;
- deletion of `tilexr_ep_dispatch_memory_helpers.h`;
- presence of the reference dispatch, count/cumsum, DataAsFlag, compaction,
  shared-expert, token-mask, and expert-mask paths in the target kernel;
- active-mask pointer/type validation;
- general shared-expert rank grouping;
- rejection of quantization and TP;
- reference state/data/workspace layout calculations and overflow rejection;
- compatibility of the four-field assist tuple;
- host and source-guard unit suites;
- A5 kernel compilation when the CANN environment is available.
