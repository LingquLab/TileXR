# MoonEP Native Five-Stage API

TileXR provides native Planning, Dispatch, PrefetchWeight, Combine, and ReduceGrad
stages for same-host Ascend 910A5/Ascend950 communicators. The implementation targets
CANN 9.1, uses the unstable V1 C ABI in `tilexr_moonep.h`, and enqueues work on the
caller's stream without Host synchronization.

## Plan Contract

`TileXRMoonEpPlanV1` stores `N/R/E/B/NvS/K` and device pointers for:

```text
dst             int32 [N]
expertsToCopy   int32 [R,B]
zeroFillRanges  int32 [E+B,2]
remoteStats     int32 [2]
dupGroups       int32 [NvS,3]
dupLoffs        int32 [NvS]
dupCounts       int32 [2]
status          int32 [1]
```

`N=S*K`, `E%R==0`, `1<=B<=E/R`, and `1<=K<=32`. Planning accepts explicit
`tokenPadding` and returns
`NvS=N+(tokenPadding-1)*2*(E/R)`. `cuSeqlens` is a separate Planning output with
shape `[E+B]`; it is not owned by Plan V1.

All Plan tensors and the planner workspace must remain alive through saved-plan reuse.
Dispatch with `TILEXR_MOONEP_FLAG_BUILD_DEDUP` materializes dedup metadata. A saved-plan
Dispatch omits that flag and does not mutate the metadata.

## Stage Tensors

| Stage | Inputs | Outputs or mutation |
| --- | --- | --- |
| Planning | `topkExperts` int32 `[S,K]`, `tokensPerExpert` int32 `[E]` | Plan tensors and `cuSeqlens` |
| Dispatch | hidden BF16 `[S,H]`, optional weights FP32 `[S,K]` | hidden BF16 `[NvS,H]`, optional weights FP32 `[NvS]` |
| PrefetchWeight | three BF16 `[E+B,H,H']` tensors | requested rows copied in place to rows `[E,E+B)` |
| Combine | hidden BF16 `[NvS,H]`, optional weights FP32 `[NvS]` | hidden BF16 `[S,H]`, optional weights FP32 `[S,K]` |
| ReduceGrad | three full FP32 `[E+B,H,H']` tensors and buffers FP32 `[R,B,H,H']` | owned expert rows accumulated; local live buffer slots cleared |

Each PrefetchWeight projection may have different trailing dimensions. Each ReduceGrad
buffer must match its corresponding full gradient's trailing dimensions. Optional
Dispatch/Combine route-weight input and output pointers are both present or both null.

PrefetchWeight leaves `expertsToCopy[rank,b]==-1` slots unchanged. ReduceGrad accumulates
in ascending `(sourceRank,slot)` order, changes only locally owned expert rows, clears
only consumed slots in the local `[rank,B]` buffer slice, and preserves unused and
nonlocal buffer rows.

## Peer Protocol

All five stages are same-host IPC-only. They use
`CommArgs::peerMems[rank] + IPC_DATA_OFFSET`, never select or depend on the UDMA fields
in `CommArgs`, and reject cross-node communicators before launch. Dispatch and Combine
select a 32-byte-aligned hidden chunk such that `NvS*chunkStride` fits in the 100 MiB
peer window, then iterate chunks. PrefetchWeight processes one projection, slot, and
row chunk at a time. ReduceGrad publishes the local rank's B slot chunks and owners
consume ranks in ascending order.

Core communicator initialization may probe optional UDMA capability. That probe is
best-effort: an unavailable transport, nonzero initialization result, or standard
exception disables UDMA and continues with the established IPC communicator. A
successful probe does not change the MoonEP data path.

Every call obtains a fresh `TileXRCommNextMagic` value. Peer waits are bounded and
magic-tagged; shared flag memory is never reset to start a round. Device completion is
reported through `plan->status` after stream synchronization:

| Stage | Success | Invalid/input failure | Remote failure base | Timeout base |
| --- | ---: | ---: | ---: | ---: |
| Planning | 0 | planner-specific | planner-specific | planner-specific |
| Dispatch | 2000 | 2001 | 2100 | 2200 |
| Combine | 3000 | 3001 | 3100 | 3200 |
| PrefetchWeight | 4000 | 4001 | 4100 | 4200 |
| ReduceGrad | 5000 | 5001 | 5100 | 5200 |

The same status word is reused by later stages, so inspect it at the desired completion
boundary. The Torch facade tracks the last enqueued stage for each pending Plan and
checks its corresponding success marker after stream synchronization.

## Launch and Installation

All five kernels are compiled as pure AICore ELF files, embedded in their Host
libraries, registered with `rtDevBinaryRegister` and `rtFunctionRegister`, and launched
with `rtKernelLaunchWithFlagV2`. Host `kernel<<<...>>>` wrappers are prohibited.

The installed public library depends on all five Host libraries. Its runtime RPATH is
`$ORIGIN`; CANN `${ARCH}-linux/devlib` is link-time-only and must not appear in RPATH or
RUNPATH. Active targets do not include or link `3rdparty/moonep` or `reference/`.

## Torch Facade

The Torch facade follows the upstream parameter shape. Fresh Dispatch accepts routing
inputs, runs Planning, and returns `(hidden_nvsh, route_weights_nvs, cu_seqlens, plan)`.
Saved-plan Dispatch skips Planning and returns `cu_seqlens=None`. `async_finish=True`
appends a recorded NPU event. PrefetchWeight and ReduceGrad mutate caller tensors in
place. `zero_copy=True` raises `NotImplementedError` before any native launch because
the facade does not expose external ownership for peer-window views.
`inter_rank_sync=False` also raises `NotImplementedError` before any native launch:
the current kernels require every peer data-ready and drained wait, and do not yet have
a separate optional entry barrier that can be skipped safely.

The untimed correctness runner has a built-in `tilexr_native` adapter that maps the
same facade into the normalized five-stage protocol. Between PrefetchWeight and Combine,
both reference and candidate execute the mandatory BF16 Torch-NPU Expert Forward
(`npu_grouped_matmul`, `npu_swiglu`, `npu_grouped_matmul`), producing a sixth comparison
artifact without changing the native ABI. The runner preserves native Plan/tensor
ownership, exposes dedup metadata only after the first successful Dispatch, and rejects
non-native capabilities, mismatched launcher dimensions, or cross-node topology before
Planning. There is no precomputed or pure-Python Expert Forward fallback.
Use `--mode correctness` without a candidate option for this adapter; an explicit
`--candidate-backend MODULE:FACTORY` remains an override.

## Validation Evidence

On 2026-08-04, the CANN 9.1 target build and all 22 configured CTests passed on
`141.61.49.223`, including 16 MoonEP/Planner tests. The installed public library
exported all five V1 symbols, used `$ORIGIN`, and resolved `libascend_hal.so` from the
real driver directory.

A single-rank run passed on physical NPU 4. The final four-rank run used physical NPU
4-7 with `S=64 K=4 E=32 H=512 B=8 NvS=256`. Every rank reported native success for
all five stages, including fresh and saved-plan Dispatch, and passed deterministic
hidden, optional route-weight, three-projection prefetch, FP32 reduction, and live-slot
clearing checks. Logs are retained at
`/tmp/tilexr_moonep_4rank_verified_20260804` on that host.

The real Torch facade also passed the padded `planning-small` correctness case on
physical NPU 4. That run exercised Planning, fresh and saved-plan Dispatch/Combine,
PrefetchWeight, ReduceGrad, per-stage status tracking, synchronization, and teardown.
Artifacts are retained at `/tmp/tilexr_moonep_torch_status_correct_20260804`.

This is same-host IPC functional correctness evidence. Cross-node MoonEP, including
MoonEP-over-UDMA, is unsupported. Core TileXR UDMA remains a separate optional
capability, and this evidence does not validate its data plane. The evidence also does
not validate FP16, throughput, the full precision matrix, or profiling. Accordingly,
`transport_performance_valid` remains false.

On 2026-08-05, the Torch-NPU Expert Forward extension passed a one-rank
`planning-small` differential and a four-rank `skewed-padding` differential on physical
NPU 4-7. Every rank emitted passing Planning, Dispatch, PrefetchWeight, ExpertForward,
Combine, and ReduceGrad artifacts. The GMM call boundary converts the Plan's int32
`cuSeqlens` to the int64 `groupList` required by Torch-NPU 2.10.0.post2 and owns clones
of native-written hidden/projection inputs. Four-rank artifacts are retained at
`/tmp/tilexr-moonep-expert-forward-gmm-4r-20260805`.
