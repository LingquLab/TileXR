# Performance-First ReduceGrad Owner-Pull Design

Date: 2026-08-11

## Status

Approved by the user on 2026-08-11. The owner-pull implementation and exact
ordered-FP32 validation are complete. On latest `main`, the fixed shared-QP
domain reduced TileXR P50 to `1536-1562 us` in three `20 x 50` runs, versus
the retained pinned-native P50 of `2724-2737 us`. This clears the numerical
3% gate against the retained baseline, but the native artifact used
`torch_npu 2.7.1.post5.dev20260730` while the final TileXR runs used
`2.7.1.post4.dev20260417`; a strict same-runtime native rerun remains open.
Sender-PUT is therefore a conditional next stage, not part of this patch.

## Goal

Replace the current TileXR MoonEP ReduceGrad implementation with an Ascend950
implementation whose steady-state latency is reproducibly lower than native
Ascend MoonEP at commit `a49538a45e5c5bdc82aa6ae02548f99e72ec67eb`.

The replacement is performance-first and A5/Ascend950-specific. The existing
sender/receiver ReduceGrad kernel, peer-window transport, UDMA push protocol,
per-chunk acknowledgements, and transport fallback are removed. They are not
kept as a runtime fallback.

The upstream Python `Buffer.reduce_grad` behavior remains compatible:

- gate, up, and down FP32 gradients are reduced in place;
- only experts owned by the local rank are updated;
- contributions are accumulated in ascending `(source_rank, slot)` order;
- only live slots in the local rank's reduce buffers are cleared;
- unused slots and non-local storage remain unchanged;
- asynchronous stream ordering remains valid.

## Baseline and Measured Workload

The comparison source is the ignored checkout
`reference/ascend-moonep-dev` at commit
`a49538a45e5c5bdc82aa6ae02548f99e72ec67eb`.

Native MoonEP's kernel in `kernels/moonep_grad_reduce.cpp` uses:

- a fixed 12,288-element FP32 tile, or 48 KiB per remote GET;
- `aclshmemx_mte_get_nbi` directly from remote symmetric memory to UB;
- one 48 KiB accumulator and 48 KiB ping/pong input tiles;
- one output load and store per UB tile;
- source-rank/slot ordered accumulation;
- 32 AIV workers by default;
- a global vector barrier before clearing local live slots.

The complete native `Buffer.reduce_grad` launches this kernel three times,
once for gate, up, and down. TileXR can fuse those projections and pay one
cross-rank completion barrier.

The relevant matrix sizes are large even though native MoonEP issues small
individual GETs:

| Shape | One FP32 expert gradient | Native GETs per gradient |
| --- | ---: | ---: |
| 3584 x 3072 | 42 MiB | 896 |
| 7168 x 2048 | 56 MiB | about 1,195 |

For three contributors, one 56 MiB expert causes 168 MiB of remote reads per
projection. This is the large-transfer regime in which UDMA must be measured
against UB-memory rather than rejected based on the 48 KiB native tile size.

## Confirmed Target Constraints

The target hosts use CANN 9.1.T560 and Ascend950DT. The installed header
`include/pto/npu/comm/async/urma/urma_async_intrin.hpp` declares the public
URMA GET path with GM source and destination pointers:

```cpp
uint64_t __urma_get_async(
    __gm__ uint8_t *dst,
    __gm__ uint8_t *src,
    uint64_t transferSize,
    const UrmaExecContext &execCtx);
```

The WQE SGE contains a 32-bit length, a local MR token, and a GM virtual
address. Therefore the design does not assume that UDMA can write directly to
UB. UDMA reads land in registered, non-cacheable GM staging and the AIV vector
pipeline consumes that staging through MTE2.

The current TileXR UDMA registration model exposes one active region. Its
per-QP device state already contains the two pieces needed for a better model:

- `UDMAWQCtx.localTokenId` selects the local SGE memory region;
- the per-peer, per-QP `UDMAMemInfo` selects the remote target memory region.

The new design uses these per-QP fields to bind a local staging MR to one of
three remote gradient MRs without switching registration on the hot path.

## Why the Current Implementation Is Replaced

The current kernel performs the following sequence for every chunk:

1. Copy an unregistered gradient chunk into registered outbound GM.
2. UDMA PUT the chunk into the owner's inbound GM.
3. Quiet the QP.
4. Accumulate the inbound chunk.
5. Poll an 8-byte acknowledgement with UDMA GET plus quiet.
6. Reuse one of two stages and clear the source.

It also assigns receiver work by local expert only, serializing projections and
chunks for an expert, and uses broad pipeline barriers in dense copy/add/clear
loops.

There is a separate host-side problem: the Python hot path performs a device
synchronize before re-confirming ReduceGrad registration. Dispatch,
PrefetchWeight, and ReduceGrad use different active memory regions, so a full
flow can repeatedly switch UDMA registration profiles. Stage event timing then
includes preceding work and registration coordination. Both the kernel
protocol and this hot-path profile switching are removed.

## Chosen Architecture

### 1. Persistent UDMA Profiles

Extend the UDMA runtime with persistent, handle-addressed profiles. A profile
owns one or more local memory registrations, their imported remote regions, a
device registry, and a device `UDMAInfo` image. Multiple profiles may coexist
and share the same hardware QPs. A kernel receives the profile's device
pointers explicitly; selecting a profile does not mutate `CommArgs`, register
memory, synchronize a device, or exchange host metadata.

The existing single-region registration API remains available to unrelated
operators. ReduceGrad does not use its replace-active behavior.

The ReduceGrad profile contains four region roles on every rank:

| Region | Local purpose | Remote purpose |
| --- | --- | --- |
| staging | UDMA READ destination and lane state | none |
| gate source | local live gate reduce-buffer slice | remote gate source |
| up source | local live up reduce-buffer slice | remote up source |
| down source | local live down reduce-buffer slice | remote down source |

Each QP binding chooses the staging region as its local SGE token and exactly
one projection region as its remote token. With eight QPs, the initial mapping
is 3/3/2 lanes across gate/up/down; the measured mapping may be changed when
projection byte sizes differ.

Registration happens during `prepare_reduce_grad`, before warmup and timed
iterations. The preparation signature includes tensor addresses, local slice
addresses, byte sizes, rank/plan dimensions, chunk size, and QP mapping. A hot
call with mismatched pointers or sizes fails validation. It does not
implicitly synchronize or re-register.

The internal native ReduceGrad ABI is advanced to carry the three reduce
buffer source pointers and the persistent profile handle/device view. The
public upstream-shaped Python signature is unchanged. Existing exported C ABI
symbols may adapt to the new engine, but no exported entry may launch the old
kernel or old protocol.

### 2. Large-Chunk Owner Pull

The owner pulls contributions. No sender AIV, sender notification, completion
GET, or sender acknowledgement exists.

For each projection, local expert, and MiB-scale row chunk, the owner scans the
small `[R, B]` plan in ascending flattened order and identifies contributors.
It posts deferred UDMA READ WQEs for remote contributors into that lane's
registered staging area, rings the QP doorbell once for the batch, and polls
one ordered completion frontier.

The initial chunk candidates are 1, 2, 4, and 8 MiB. A target-hardware sweep
selects the checked-in default. The choice must include transfer setup, CQ
polling, GM staging consumption, and vector accumulation; UDMA link bandwidth
alone is insufficient.

A work item normally has no more than one slot per source rank. To preserve the
general contract, a plan with more contributors than one staging wave can hold
is processed in consecutive waves. Waves and contributions retain flattened
`source_rank * B + slot` order. This is a new-path slow case, not a fallback to
the deleted implementation.

### 3. QP Lanes and AIV Helper Groups

One leader AIV exclusively owns each QP, eliminating concurrent SQ head/tail
updates. The remaining AIVs are divided into helper groups, one group per QP
lane. With eight QPs and 64 AIVs, the initial shape is eight leaders and seven
helpers per lane.

Each lane has two GM staging banks. A leader performs this pipeline:

1. Batch all remote reads for work item N into bank ping.
2. Publish ping readiness to its helper group after CQ completion.
3. Batch work item N+1 into bank pong while helpers compute N.
4. Wait for helper completion before reusing ping.
5. Alternate banks until the projection lane's work list is empty.

Each bank reserves `rank_size * chunk_bytes` for one normal contribution wave.
At 16 ranks, eight lanes, two banks, and a 2 MiB chunk, payload staging is 512
MiB. Workspace sizing is checked for overflow and reported explicitly.

Helpers partition the staged chunk into UB tiles. For each UB tile a helper:

1. Loads the owned output tile once into FP32 UB.
2. Loads each staged or local contribution in ordered sequence.
3. Accumulates with FP32 vector add.
4. Stores the owned output tile once.

The UB layout keeps an accumulator plus ping/pong input tiles. MTE2-to-vector
and vector-to-MTE dependencies use precise events. `PIPE_ALL` is reserved for
boundaries whose target headers require it, not used as the default dependency.

### 4. Completion and Clear

Every lane first completes all UDMA reads and all helper work. A local AIV
barrier then ensures no worker can still read a source slot.

One rank leader publishes one magic-tagged TileXR completion flag and waits for
the corresponding flag from every rank. A rank with non-zero device status
publishes the adjacent failure step; peers still satisfy the same wait, then
detect the non-success value and set their local status. Thus every rank skips
source clearing if any rank fails, without a second barrier or a new mailbox.
This replaces per-AIV global participation. The magic comes from
`TileXRCommNextMagic`; shared flag storage is never reset.

After the global barrier, all AIVs partition the three local reduce-buffer
slices by projection, live slot, and tile and clear them in parallel. Unused
local slots and every non-local slice are untouched.

## API and Ownership Rules

- The ReduceGrad V2 ABI is an unreleased development interface in this branch.
  The owner-pull replacement changes it in place; no compatibility wrapper for
  the deleted ReduceGrad protocol is retained.
- Multi-rank ReduceGrad requires A5/Ascend950 UDMA and at least one QP per
  projection. Missing capability returns `NOT_SUPPORT`; there is no old
  ReduceGrad fallback.
- ReduceGrad requires at least four ranks. Rank counts from one through three
  return `NOT_SUPPORT` during preparation and have no Kernel path. Eight and
  larger rank counts remain supported when the hardware provides usable peer
  windows and at least one UDMA QP per projection.
- Registered source slices and staging allocations must remain alive and
  unchanged until the last ReduceGrad event completes.
- Re-preparation requires a quiescent communicator and is outside steady-state
  timing.
- Profile registration remains unsupported in `InitThread` mode.
- UDMA WQEs are assembled in UB and published through MTE3. Doorbells use
  `st_dev` only after WQE publication completes.
- Runtime RPATH and pure-AICore binary registration/launch rules remain
  unchanged.

## Implementation Scope

Expected affected areas are:

- `src/comm/udma`: persistent profile registration, cleanup, imports, and
  per-QP local/remote region binding;
- `src/include`: profile descriptors and device accessors while preserving
  C++14 and existing single-region consumers;
- `src/moonep/reduce_grad`: replacement layout, host validation, kernel args,
  kernel, binary launch, and removal of old transport code;
- `integrations/moonep_torch/tilexr_moonep`: preparation cache keyed by pointer
  identity, direct use of reduce buffers, and removal of hot-path synchronize,
  tail copy, and registration switching;
- `tools/moonep` and `tests/moonep`: isolated baseline harness, correctness
  checks, timing hygiene, and source guards proving the old protocol is absent;
- documentation describing the new UDMA-only multi-rank contract.

`reference/` remains comparison-only and is never compiled or linked into an
active target.

## Validation Strategy

### Correctness

Validate four ranks on physical devices 0-3 of the target host. Include empty, sparse,
mixed, heavy, and full plans; aligned and tail chunks; and all three
projections. Check:

- bitwise equality with the source-rank/slot ordered FP32 reference where the
  same operation order is used;
- only locally owned expert rows change;
- all and only local live reduce slots become zero;
- unused and non-local slots remain byte-identical;
- repeated magic rounds do not consume stale flags;
- asynchronous launch and explicit synchronization report device failures.

### Performance

Build native MoonEP from the pinned baseline commit and TileXR from the same
test directory on both hosts. Use identical dimensions, plan tensors, rank
count, stream synchronization, warmup, iterations, and device-event timing
boundaries.

Primary cases include:

- MoonEP's dedicated 3584 x 3072 suite;
- E=384, H=7168, H'=2048, K=8, S=8192, B=48;
- 4 ranks on physical devices 0-3 of one host;
- 20 warmup iterations and at least 50 measured iterations.

Report the cross-rank maximum per iteration, then P50 and P99. Run each primary
case at least three times. One-time allocation, MR registration/import, and
kernel binary registration are reported separately and excluded from
steady-state latency.

Completion requires all of the following:

1. TileXR P50 and P99 are both lower than native MoonEP for the primary 4-rank
   cases in every repeated run.
2. The median advantage is at least 3%, so measurement noise is not presented
   as a win.
3. No correctness condition above fails.
4. Device-event timelines show no host/device synchronize or registration in
   the timed ReduceGrad path.
5. `msprof` confirms batched MiB-scale UDMA reads, overlap between the next
   staging bank and current vector work, and no per-chunk acknowledgement GET.

### Tuning Sequence

Tune one variable group at a time and retain raw JSON/CSV results:

1. UDMA versus native UB-memory transfer sweep from 48 KiB through 16 MiB.
2. Chunk size with one QP, including staging and accumulation.
3. QP count and projection-to-QP allocation.
4. Helper count per lane and UB tile size.
5. Ping/pong overlap and WQE batch depth.
6. End barrier and local clear partitioning.

## Stop Conditions and Risks

The first implementation milestone is a target-hardware proof that several
simultaneously registered MRs can supply a staging local token and
projection-specific remote tokens on shared QPs. If CANN/HCCP rejects that
profile, or if independent profiles cannot safely share the QPs, implementation
returns to design review. It must not silently substitute the old protocol.

Other material risks are:

- MR/import resource limits with four regions per rank;
- a UDMA/GM staging crossover larger than the real contribution chunks;
- staging workspace pressure at 16 ranks;
- helper-group flag/cache overhead erasing pipeline overlap;
- a low-QP environment that cannot allocate one lane per projection;
- an apparent win caused by different timing boundaries or by excluding work
  that native MoonEP includes.

No performance claim is made until the 4-rank target-hardware evidence
satisfies the stated gate. The physical-card mapping and excluded faulty
devices are retained with the raw results.

## Implementation Outcome and Experiment Ledger

### Implemented Owner-Pull Optimizations

The delivered implementation replaces the previous ReduceGrad protocol with:

- persistent, handle-addressed multi-MR UDMA profiles, avoiding hot-path
  registration switching and device synchronization;
- batched deferred owner GETs with one doorbell and completion frontier per
  participating source/QP;
- one fused gate/up/down launch, deterministic projection-to-QP allocation,
  ping/pong registered GM staging, and leader/helper AIV groups;
- separation of the fixed 32-QP shared transport profile from three active
  ReduceGrad lanes, mapped to physical QPs `{0, 1, 16}` to preserve the
  measured six-port/six-port/two-port route mix without allocating workspace
  for inactive QPs;
- an 8 MiB production default selected by the retained chunk sweep, while
  preserving the explicit chunk-size override for targeted validation;
- ordered FP32 accumulation in flattened `(source_rank, slot)` order;
- one magic-tagged cross-rank completion barrier followed by parallel clearing
  of local live slots, with the same flag carrying a failure step so any rank
  failure prevents clearing on every rank;
- retained peer-memory `SyncCollectives`; the ReduceGrad data plane itself has
  no legacy sender/receiver, PUT/ACK, or peer-packed-record fallback.

Cold failure paths were tightened without changing the owner-pull data loop: a
failed kernel launch drains the already-enqueued asynchronous status memset
once; a data, profile, or workspace configuration failure skips lane work but
still reaches the existing collective barrier, whose adjacent failure step
prevents every rank from clearing source data; and ReduceGrad device statuses
`1..5` poison the Python context. These are correctness boundaries, not
performance optimizations, and add no stream synchronization to a successful
launch or another cross-rank barrier.

These changes passed exact ordered-FP32 correctness on four physical cards.
The pre-shared-domain control runtime is:

`/tmp/TileXR-reducegrad-20260811-44dc37b/install-control-2bank-runtime-20260812`

For four ranks, one expert per rank, one slot, mixed plan, FP32
`3584 x 3072`, and an 8 MiB transfer setting, owner-pull measured
P50/P99 `2764.82/2843.38 us`. Native MoonEP measured approximately
`2720-2737 us`; the required 3% gate was approximately `<2655 us`.
That explicit 3-QP owner-pull control was close to native but did not pass the
acceptance gate.

Latest `main` initializes MoonEP with a fixed 32-QP shared domain. The first
delivery build rejected that domain during workspace query with
`TileXR ret=-6` because transport QPs and active lanes were incorrectly the
same count. The final integration keeps all 32 QPs in the persistent profile,
uses only logical lanes `{0,1,2}`, and maps them to physical QPs `{0,1,16}`.
Inactive QPs have valid harmless bindings and are never scheduled. This keeps
the 4-rank, 8 MiB workspace at `203423744` bytes rather than scaling it to 32
lanes.

Final artifacts are on `141.61.53.106` under:

`/tmp/TileXR-reducegrad-delivery-20260812-8d22775/artifacts`

| Run | Warmup x measured | P50 | P99 | Result |
| --- | ---: | ---: | ---: | --- |
| `final-perf-shared32-run1` | `20 x 50` | `1562.40 us` | `1662.54 us` | Passed |
| `final-perf-shared32-run2` | `20 x 50` | `1550.89 us` | `1632.76 us` | Passed |
| `final-perf-shared32-run3` | `20 x 50` | `1536.22 us` | `1595.37 us` | Passed |
| pinned native `perf-20x50` | `20 x 50` | `2723.68 us` | `2755.00 us` | Passed |
| pinned native interleaved A | `20 x 50` | `2736.30 us` | `2768.92 us` | Passed |
| pinned native interleaved B | `20 x 50` | `2737.48 us` | `2800.86 us` | Passed |

The worst final TileXR P50 is `42.6%` below the best retained native P50; the
worst TileXR P99 is `39.7%` below the best retained native P99. The final
`final-exact-4card-shared32-guarded` run passed exact ordered-FP32 validation
on all four ranks with P50/P99 `1565.46/1567.53 us` over its three smoke
samples. Those three samples are correctness evidence, not a replacement for
the three formal performance runs.

After the final review fix, the default-parameter regression
`final-post-review-default8m-exact` again passed exact ordered-FP32 validation
on four physical cards, including live-source clearing. It intentionally
omitted `--chunk-bytes`; the reported layout selected `8388608` bytes and the
same `203423744`-byte workspace. Its three smoke samples measured P50/P99
`1614.31/2004.99 us`; these samples are correctness evidence, not a formal
performance run. The summary SHA256 is
`019f7c4a41d86544a1cdefe008dee7fa4c7935272c0b4d95d062d51d244762bc`.
The final ReduceGrad shared-library SHA256 is
`33764d7e8149c4c7f23a1f58916e8484e3b6b74906e58893d98c092651fa82c0`
on both validation hosts.

The native comparison source remains pinned at
`a49538a45e5c5bdc82aa6ae02548f99e72ec67eb`. A same-runtime rerun was attempted
on the final host. The old extension required the system `GLIBCXX_3.4.30`,
PyTorch and torch-npu library search paths, and then still failed under
`torch_npu 2.7.1.post4.dev20260417` with duplicate NPU backend registration.
No reference source or server environment was modified to bypass that
incompatibility. Therefore the large measured advantage is actionable, but a
strict same-runtime A/B remains a release-quality follow-up.

### UDMA GET Calibration

The retained probe is under `/tmp/TileXR-get-probe-20260812-44dc37b` with
artifacts in `artifacts/get-calibration-run1`.

- Probe SHA256:
  `37aead24de783550dbe61859f5a345651030b4d6bb5a307324b0d894d752be7a`.
- `libtile-comm.so` SHA256:
  `2615b4103dfdbe3c047c7c40da6bb525afa1e260e09a5334ec6826db4b51fdb4`.
- High-bandwidth-route QP GET reached approximately `273 GB/s`.
- Low-route QP GET reached approximately `95.6 GB/s`.
- GET plus staging consumption reached only approximately `28-35 GB/s`.
- Byte-exact validation passed.

This isolates the remaining owner-pull bottleneck to staging consumption and
owner-side accumulation/synchronization rather than raw UDMA GET bandwidth.

### Performance Experiments Not To Repeat

The following owner-pull tuning directions were tested and rejected because
they produced no repeatable improvement or regressed the control:

| Experiment | Observation | Decision |
| --- | --- | --- |
| Chunk-size sweep | 8 MiB control: `2764.82/2843.38 us`. 9, 10, 10.5, 11, 12, 14, and 16 MiB gave P50 `2801`, `2765`, `2867`, `2804`, `2790`, `2861`, and `2890 us`; 2 MiB was `2860 us`. | Keep 8 MiB. Do not repeat a blind chunk sweep. Artifacts: `control-chunk-*`, `reducegrad-async-two-bank-chunk*`, and `reducegrad-accum-pipeline-v1/perf-screen-chunk16m`. |
| Three or twelve staging banks | Three banks reached `2762.27/2792.89 us`, statistically flat against two banks; twelve banks regressed to `2861.70/3034.28 us` and consumed more workspace/state. | Keep two banks. Artifacts: `reducegrad-async-three-bank-v1` and `reducegrad-async-twelve-bank-v1`. |
| Six or eight active QPs | Six-QP 7 MiB: `2872.42/2914.68 us`; six-QP 8 MiB: `2900.12/3391.89 us`; eight-QP 4 MiB: `2947.77/3049.88 us`. Four-QP `6/3/1` was `2772.12/2866.72 us`. | Keep three active lanes. Do not add QPs without a new data-flow hypothesis. |
| All-six-port or all-two-port routing | All-six-port three-QP was `2762.69/2879.98 us`, flat/noisier than control; all-two-port was `2787.76/2922.89 us`. | Link route alone is not the old owner-pull bottleneck. Keep `{0,1,16}` in shared mode. |
| QP rotation/interleaving | Rotating QPs per work item produced no stable gain in the local sweep and was excluded from production. | Keep stable single-owner QP assignment. Revisit only with a different data-flow model. |
| 32 AIV and helper-shape tuning | 32 AIV at 8 MiB regressed to `2806.93/2929.62 us`; 48 AIV output-prefetch was `2784.40/2901.04 us`. | Keep 64 AIV. Do not retune helpers in isolation. |
| Accumulator/data-copy micro-pipelines | Accumulator pipeline `2809.00/2900.88 us`; aligned DataCopy `2810.22/2954.08 us`; bank skew `2785.19/2960.88 us`. | Rejected. The extra pipeline/cache bookkeeping did not improve end to end. |
| Output/input prefetch variants | Output-prefetch variants ranged from P50 `2767.53` to `2790.40 us`; input+output prefetch was `2796.69/2931.12 us`. | Keep the checked-in two-bank consume loop; do not layer more prefetch without profiling a new bottleneck. |
| Inline contributor cache | Helper contributor cache was `2752.98/2852.92 us`, not repeatably better; fully inline contributors regressed to `2898.88/3064.55 us`. | Rejected and reverted. |
| Immediate doorbell, WQE batch, polling backoff | Immediate doorbell `2783.25/3008.42 us`; alternate WQE batch `2910.37/3009.26 us`; polling backoff `2819.27/2986.26 us`. | Retain deferred batch publication and current completion polling. |
| Leader overlap or single coordinator | Leader overlap `2866.05/2967.99 us`; one coordinator `3301.58/3359.10 us`. | Retain one leader per lane with distributed helpers. |
| Early helper release | Produced vector-core exception `507035` from out-of-bounds internal-buffer access; the reverted control was still `2825.93/3262.03 us`. | Unsafe and rejected. Do not retry without a redesigned ownership/lifetime proof. |
| Contributor-ready overlap | Exact ordered FP32 passed, but P50/P99 regressed to `2763.22/2903.76 us` and helper wait rose to `295-323 us`. | Rejected and reverted. Artifacts: `/tmp/TileXR-reducegrad-source-ready-20260812`. |

The rejected contributor-ready artifacts remain under
`/tmp/TileXR-reducegrad-source-ready-20260812`. Local QP-rotation and chunk
sweep helper scripts were experimental only and are intentionally excluded
from this patch.

### Debugging History Not To Repeat

These were correctness investigations, not viable optimization candidates:

- The first sparse/mixed implementations showed staged data was correct while
  helper-visible metadata or done tokens were stale. The final protocol uses
  cache-line-separated magic-tagged tokens and explicit cache maintenance.
- Increasing `waitIterations` did not fix missing helper completion; it only
  prolonged the timeout. Do not treat this class of failure as a timeout-tuning
  problem.
- MTE barriers around the bank item did not substitute for the required GM
  visibility/cache-line protocol. The `correctness-8r-empty-*` and
  `reducegrad-4r-debug-v10` through `v22` artifacts are diagnostic history.
- Exact correctness initially stopped before launch at the default 512 MiB
  inspection cap: the tested tensors require `660602880` bytes. Use at least
  `805306368` for this `3584 x 3072`, four-rank case.
- Latest-main initially failed before launch with `ret=-6`. The cause was not
  workspace overflow: shared initialization exposes 32 transport QPs while
  the old owner-pull layout allowed at most eight lanes. Keep transport-QP
  count separate from active-lane count and require the `UDMA_SHARED_QP`
  capability before applying the fixed `{0,1,16}` physical mapping.
- UDMA completion must also remain separate from both counts. Capture each
  queue's completion frontier before posting a batch, advance it only for
  successful submissions, and wait for that explicit target. Do not reread a
  shared published WQE count when completing the batch, and do not infer CQE
  length from SQE contents that hardware may already have consumed.

## Recommended Next Stage: Sender-PUT Fan-In

This section is a conditional design recommendation and stop gate only. It is
not part of the implemented owner-pull patch. Because shared-QP owner-pull now
measures below the `<1.9 ms` sender-PUT exploration gate, sender-PUT should not
replace it on the current workload without first demonstrating a material,
repeatable advantage under a strict same-runtime A/B.

Use three non-owner ranks as senders and one rank as owner. Each sender PUTs
gate, up, and down directly into the owner's registered staging. Use routes
`6,6,6,2` and distribute payload approximately 3:1 between the aggregate
six-port paths and the two-port path. Allocate whole-row, non-reused staging so
senders need no per-chunk acknowledgement. Each source publishes one
magic-tagged ready indication only after all three projections are visible.
The owner then accumulates strictly in `(source_rank, slot)` order and retains
the existing peer-memory `SyncCollectives` completion path.

For the validated `3584 x 3072` FP32 case, one projection row is 42 MiB. One
sender transfers 126 MiB for three projections, and the owner receives
`3 projections x 3 contributors x 42 MiB = 378 MiB` per ReduceGrad round.
This is firmly a large-transfer regime, so UDMA is preferred over UB-memory for
the data plane even though owner-side consumption must still be optimized.

Before production ReduceGrad is redesigned, build an isolated four-physical-
card exploratory probe with three senders and one owner. It must pass exact
ordered-FP32 validation and achieve P50 `<1.9 ms` for the 378 MiB fan-in. Stop
and return to design review if either condition fails; do not integrate a
sender-PUT production path based only on raw link-bandwidth results.

The immediate next validation step is instead to rebuild pinned native MoonEP
for the exact final `torch_npu` environment, then repeat three interleaved
`20 x 50` runs on the same physical cards. Only pursue sender-PUT if that strict
comparison invalidates the shared-QP owner-pull gain or a broader workload
matrix exposes a new bottleneck.
