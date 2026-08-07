# MoonEP Upstream-Compatible V1 and Native Weight Stages Design

## Status

Approved direction from the 2026-08-04 design discussion. This specification defines
the implementation boundary for replacing the unstable TileXR MoonEP V1 ABI, aligning
its default tensor-level behavior with `3rdparty/moonep`, and adding native
PrefetchWeight and ReduceGrad operators.

## Goals

- Replace the generic single-input/single-output stage ABI with stage-specific tensor
  arguments matching upstream MoonEP's default data path.
- Expand Plan V1 and Planning outputs so Dispatch, PrefetchWeight, Combine, and
  ReduceGrad can share an upstream-shaped saved plan.
- Support configurable prefetch slot count `B` and route padding through
  `tokenPadding`.
- Preserve exact BF16/FP32 tensor dtypes, in-place weight/gradient ownership, optional
  route weights, and saved-plan forward/backward reuse.
- Implement PrefetchWeight and ReduceGrad as native Ascend C direct-launch operators.
- Keep all five stages on the explicit pure-AICore registration path and verify the
  resulting flow on physical NPU 4-7.

## Non-Goals

- `zero_copy=True` is not implemented in this change. The Torch facade accepts the
  option only to reject it clearly; exposing peer-window storage as a Torch tensor view
  needs a separate allocation and lifetime API.
- CUDA VMM objects are not reused on Ascend. TileXR provides the same logical ownership
  through its communicator and peer windows.
- Cross-node UDMA, FP16 hidden tensors, performance tuning, and framework-specific
  group GEMM integration are outside this change.
- Active targets must not compile or link code from `3rdparty/moonep` or `reference/`.

## Authoritative Upstream Contract

The compatibility target is the checked-in comparison source:

- `3rdparty/moonep/moonep/api.py`
- `3rdparty/moonep/moonep/planning.py`
- `3rdparty/moonep/moonep/dispatch.py`
- `3rdparty/moonep/moonep/combine.py`
- `3rdparty/moonep/moonep/prefetch.py`
- `3rdparty/moonep/moonep/grad_reduce.py`
- `3rdparty/moonep/README.md`
- `3rdparty/moonep/tests/test_e2e.py`

These files define expected behavior only. TileXR owns all active implementation code.

## Dimensions

```text
S             input tokens per rank
K             routed experts per token
N             S * K
R             EP rank count
E             global expert count
epn           E / R
B             prefetch slots per rank, 1 <= B <= epn
tokenPadding  logical segment alignment, positive
NvS           N + (tokenPadding - 1) * 2 * epn
H             hidden elements per token row
H'            projection-specific second matrix dimension
```

Training callers use `B == epn`. Smaller B is accepted for inference-style prefetch.
All Host arithmetic is checked in 64 bit. `K <= 32`, `E % R == 0`, and signed `dst`
encoding bounds remain mandatory.

## Public ABI

### Tensor Descriptor

`TileXRMoonEpTensorV1` remains the common contiguous device-tensor descriptor. Rank 3
and rank 4 descriptors are required by the new weight/gradient stages. No strides or
storage offsets are added; all tensors must be contiguous at storage offset zero.

### Plan V1

`TileXRMoonEpPlanV1` is replaced rather than extended compatibly because V1 is not yet
stable. Its scalar and tensor fields are:

```text
N, R, E, B, NvS, K
dst             int32 [N]
expertsToCopy   int32 [R, B]
zeroFillRanges  int32 [E+B, 2]
remoteStats     int32 [2]
dupGroups       int32 [NvS, 3]
dupLoffs        int32 [NvS]
dupCounts       int32 [2]
status          int32 [1]    TileXR asynchronous status extension
```

`cuSeqlens` is not plan-owned in upstream MoonEP. It moves to the Planning arguments as
an explicit `int32 [E+B]` output and is returned separately by the Torch Dispatch API.

Each valid `dupGroups[g]` stores `(primary_loff, dup_start, dup_count)`.
`dupLoffs[dup_start:dup_start+dup_count]` stores the duplicate local offsets.
`dupCounts[0]` and `dupCounts[1]` are the valid group and duplicate-prefix sizes.

### Planning

The workspace query accepts `S`, `K`, `E`, `B`, and `tokenPadding`, and returns the
required workspace size plus `NvS`. Planning V1 accepts:

```text
topkExperts       int32 [S, K]
tokensPerExpert   int32 [E]
workspace         uint8 [workspaceBytes]
cuSeqlens         int32 [E+B]
plan              mutable Plan V1 outputs
waitIterations
flags
```

Planning populates `dst`, `expertsToCopy`, `zeroFillRanges`, `remoteStats`,
`cuSeqlens`, and status. Scratch needed by fresh Dispatch to construct duplicate groups
stays in the plan-retained workspace.

### Dispatch

```text
hiddenSh          BF16 [S, H]
routeWeightsSk    optional FP32 [S, K]
hiddenNvsh        BF16 [NvS, H]
routeWeightsNvs   optional FP32 [NvS]
plan              saved Plan V1
flags
```

The optional route-weight input and output must either both be null or both be valid.
Fresh planning uses `TILEXR_MOONEP_FLAG_BUILD_DEDUP`; saved-plan redispatch omits it.
The fresh path materializes the plan-owned duplicate structures. Reuse never mutates
them.

### PrefetchWeight

```text
gate              FP16/BF16 [2B, ...]
up                FP16/BF16 [2B, ...]
down              FP16/BF16 [2B, ...]
plan              saved Plan V1
flags
```

PrefetchWeight follows the PR93 registered-memory layout. `B` must equal `E/R`.
Each projection may have different trailing dimensions, must be contiguous with rank
2 through 4, and must have a 64-byte-aligned base and expert-row size. The three
non-overlapping projection views reside in one flat ordinary-device-memory allocation
that is registered once with `TileXRUDMARegister`; `peerMems[]` is not used.

Rows `[0,B)` hold the current rank's local experts. For destination slot `b`,
`expertsToCopy[rank,b]` selects a global expert, its owner row is `expert % B`, and
UDMA writes it into row `B+b`. A `-1` slot remains unchanged. Cleanup synchronizes
outstanding work, unregisters the allocation, and only then destroys the communicator.

### Combine

```text
hiddenNvsh        BF16 [NvS, H]
routeWeightsNvs   optional FP32 [NvS]
hiddenSh          BF16 [S, H]
routeWeightsSk    optional FP32 [S, K]
plan              saved Plan V1
flags
```

The hidden path performs the K-route BF16 gather with FP32 accumulation and one final
BF16 conversion. Route weights are gathered bit-exactly without arithmetic. Duplicate
groups are reduced according to the saved plan before primary routes are gathered.

### ReduceGrad

```text
fullGateGrad      FP32 [E+B, ...]
fullUpGrad        FP32 [E+B, ...]
fullDownGrad      FP32 [E+B, ...]
gateReduceBuffer  FP32 [R, B, ...]
upReduceBuffer    FP32 [R, B, ...]
downReduceBuffer  FP32 [R, B, ...]
plan              saved Plan V1
flags
```

Trailing dimensions must match within each full-grad/reduce-buffer pair. Only the local
expert range `[rank*epn, (rank+1)*epn)` in each full gradient may change. Contributions
are accumulated in ascending `(sourceRank, slot)` order. After a cross-rank consumption
barrier, each rank zeros the live slots in its locally owned reduce-buffer slice;
unused `-1` slots remain unchanged.

## Flags and Stream Semantics

- Calls enqueue work on the supplied `aclrtStream` and return without Host
  synchronization.
- `async_finish` is implemented by the Torch facade's stream/event orchestration, not
  as a C ABI field.
- Inter-rank synchronization is required by the current peer data protocol. The
  skip-sync flag remains reserved in the unstable V1 ABI, but the Torch facade and
  runtime reject `inter_rank_sync=False` before native launch until a separate optional
  entry barrier exists.
- A zero-copy flag is reserved but rejected with `TILEXR_MOONEP_ERROR_NOT_SUPPORTED`.
- A fresh magic value comes from `TileXRCommNextMagic` for every native stage call.
- Device completion or timeout is reported through `plan->status` after caller stream
  synchronization.
- The Torch facade compares the synchronized status with the success marker of the last
  stage enqueued for each pending Plan; Planning expects 0 and the four native data
  stages expect 2000, 3000, 4000, or 5000 as applicable.

## Transport and Kernel Protocols

Dispatch, Combine, and ReduceGrad use same-host
`CommArgs::peerMems[] + IPC_DATA_OFFSET` on the A5/Ascend950 path. Their first
implementation is correctness-oriented and uses bounded magic-tagged synchronization.
PrefetchWeight instead uses PR93 registered-memory UDMA and does not require local and
global rank counts to match.

Dispatch and Combine choose a hidden-row chunk such that `NvS * chunkBytes` fits in the
100 MiB peer data window. They iterate chunks rather than rejecting a tensor merely
because the complete `[NvS,H]` payload exceeds the window.

PrefetchWeight submits one UDMA GET for each projection and live slot, using the
requesting worker's QP. Each worker waits for the peers it used before the stage
publishes success status `4000`.

ReduceGrad processes one projection and row chunk at a time. Every rank publishes all
of its local `B` reduce-slot chunks into its own peer window. Owner ranks consume sources
in ascending rank/slot order, accumulate into owned full-gradient rows, synchronize,
then clear their locally owned live slots.

All peer-window byte formulas are checked. A selected chunk must contain at least one
element and must obey the Ascend C data-movement alignment contract.

## Direct Launch and Build

Add sibling operator trees:

```text
src/moonep/prefetch_weight/
src/moonep/reduce_grad/
```

Planner, Dispatch, PrefetchWeight, Combine, and ReduceGrad use the same architecture:

1. Bisheng compiles a pure AICore ELF with `--cce-aicore-only`.
2. CMake embeds the binary into the Host library.
3. Host code calls `rtDevBinaryRegister` and `rtFunctionRegister` with a unique stable
   synthetic signature.
4. Host code launches only through `rtKernelLaunchWithFlagV2`.

No Host source may contain `kernel<<<...>>>` or a wrapper that hides that syntax. Runtime
RPATH remains `$ORIGIN`; CANN `devlib` is link-time-only and must not appear in
RPATH/RUNPATH.

## Torch API

The TileXR facade adopts upstream-facing method shapes:

- `dispatch(hidden_sh, route_weights_sk=None, topk_experts_sk=None,
  tokens_per_expert=None, plan=None, async_finish=False, *,
  inter_rank_sync=True, zero_copy=False)`
- `prefetch_weight(plan, projections, async_finish=False)` after constructing a
  compact owner with `ProjectionBuffers.from_local_weights(...)` and registering it
  with `register_projection_buffers(...)`
- `combine(plan, hidden_nvsh, route_weights_nvs=None, async_finish=False,
  inter_rank_sync=True, *, zero_copy=False)`
- `reduce_grad(plan, async_finish=False, full_gate_grad, full_up_grad,
  full_down_grad, gate_reduce_buffer, up_reduce_buffer, down_reduce_buffer)`

When Dispatch receives no plan, the facade allocates Planning outputs, runs Planning,
then sets the fresh-dedup flag. With a supplied plan it skips Planning and returns
`cu_seqlens=None`. `zero_copy=True` raises a precise not-supported error before native
launch.

## Validation

Validation evidence is layered:

1. ABI layout and C-header compilation.
2. CPU references for padding, duplicate groups, dispatch/combine, prefetch, and
   gradient reduction/clearing.
3. Host validation and exact launch-argument tests with fake runtime functions.
4. Source guards for direct registration, forbidden wrapper syntax, peer-window bounds,
   and active-source independence from upstream/reference trees.
5. CANN 9.1 target compilation, installation, symbol inspection, dependency inspection,
   and RPATH checks on `141.61.49.223` in Conda environment `ai_moe_test`.
6. Single-rank smoke followed by four-rank correctness on physical NPU 4-7, including
   saved-plan redispatch, optional route weights, three in-place weight projections,
   multi-source FP32 gradient accumulation, consumed-slot clearing, and native stage
   status markers on every rank.

The final report must distinguish Host/static evidence, target compilation, single-rank
execution, and multi-rank peer-data movement. It must not claim cross-node UDMA or
zero-copy validation.
