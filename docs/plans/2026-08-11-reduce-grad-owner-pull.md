# ReduceGrad Owner-Pull Implementation Plan

Date: 2026-08-11

## Classification

Durable plan. Retain this document with the implementation because it records
cross-module UDMA ownership, compatibility boundaries, hardware gates, and the
performance evidence required for future maintenance.

Execution status on 2026-08-12: Tasks 1 through 11 produced a correct four-card
owner-pull implementation and reproducible benchmark/probe tooling. The first
explicit 3-QP control measured `2764.82/2843.38 us`, but latest `main` uses a
fixed 32-QP shared domain. Separating that transport domain from three active
lanes and mapping them to physical QPs `{0, 1, 16}` reduced P50 to
`1536-1562 us` in three `20 x 50` runs, versus retained pinned-native P50
`2724-2737 us`. This clears the numerical 3% gate against the retained
baseline. The final TileXR and retained native artifacts used different
`torch_npu` builds, so a strict same-runtime native rebuild and interleaved A/B
is the immediate next validation step. Sender-PUT is conditional future work
and is not implemented in this patch.

## Goal and Scope

Implement the approved design in
`docs/specs/2026-08-11-reduce-grad-owner-pull-design.md`:

- replace TileXR MoonEP ReduceGrad with persistent-profile, large-chunk UDMA
  owner-pull on Ascend950;
- fuse gate, up, and down in one kernel and one cross-rank barrier;
- remove hot-path synchronization, reduce-buffer tail copies, registration
  switching, sender/receiver roles, UDMA push, per-chunk acknowledgement GETs,
  and the old peer transport from ReduceGrad;
- preserve upstream Python behavior and ordered FP32 accumulation;
- prove steady-state P50 and P99 are reproducibly lower than native Ascend
  MoonEP commit `a49538a45e5c5bdc82aa6ae02548f99e72ec67eb`.

The initial workspace is `D:/1-项目/TileXR/TileXR`. Remote validation uses
both `root@141.61.53.106` and `root@141.61.53.110` under the same unique path:

`/tmp/TileXR-reducegrad-20260811-44dc37b`

## Non-Goals

- Do not change CANN 9.1 or C++14 compatibility.
- Do not compile or link anything from `reference/` into TileXR.
- Do not redesign Dispatch, PrefetchWeight, Combine, EP, collectives, SDMA, or
  ordinary peer-memory transports except for the minimum profile API needed to
  let their existing registration behavior coexist with ReduceGrad.
- Do not add a 910B, peer-window, UDMA-push, or host collective ReduceGrad
  fallback.
- Do not claim UDMA data-plane validation from local unit tests, simulator, or
  910B hardware.
- Do not include one-time allocation, MR registration/import, or binary
  registration in steady-state latency.

## Authoritative References

- Approved design:
  `docs/specs/2026-08-11-reduce-grad-owner-pull-design.md`
- Repository validation guidance: `docs/BUILD_VERIFICATION.md`
- Existing registered-memory design:
  `docs/specs/2026-08-06-udma-registered-memory-multi-qp-design.md`
- UDMA Host lifecycle: `src/comm/udma/tilexr_udma_context.cpp` and
  `src/comm/udma/tilexr_udma_transport.cpp`
- Device WQE path: `src/include/tilexr_udma.h` and
  `src/include/tilexr_udma_types.h`
- Public communicator ABI: `src/include/tilexr_api.h`
- MoonEP public ABI: `src/include/tilexr_moonep.h`
- Current code to replace: `src/moonep/reduce_grad/`
- Torch integration: `integrations/moonep_torch/tilexr_moonep/`
- Native baseline kernel:
  `reference/ascend-moonep-dev/kernels/moonep_grad_reduce.cpp`
- Native baseline benchmark:
  `reference/ascend-moonep-dev/benchmarks/bench_grad_reduce.py`
- Installed target evidence:
  `/usr/local/Ascend/cann-9.1.T560/include/pto/npu/comm/async/urma/`

## Global Constraints

- Preserve ordinary `TileXRUDMARegister` behavior for existing consumers.
- Profile registration is collective, forbidden in `InitThread`, and must have
  explicit quiescent-lifetime rules.
- UDMA WQEs are built entirely in UB, copied to SQ with MTE3, and followed by a
  precise MTE3-to-scalar dependency before an `st_dev` doorbell.
- One AIV leader owns each QP. No two blocks update the same SQ/CQ head or tail.
- The new kernel applies contributions in ascending `source_rank * B + slot`
  order. Prefetch may finish out of order; FP32 additions may not.
- Shared synchronization flags use `TileXRCommNextMagic`; they are never reset.
- Keep all workspace arithmetic in checked 64-bit Host code and keep each WQE
  transfer length within the 32-bit SGE field.
- Preserve unrelated user files and changes in the working tree.

## Task 1: Add Persistent UDMA Profile Contracts and Host Unit Tests

**Objective and role:** Define the smallest reusable contract that can keep a
ReduceGrad staging MR and three source MRs registered simultaneously while
binding per-QP local and remote region roles. This is the foundation for every
later task.

**Background and prerequisites:** The current runtime has one active
`RegistrationState`, one device registry, and a handle fixed to zero. Existing
`UDMAWQCtx.localTokenId` and per-peer/per-QP `UDMAMemInfo` already permit
different local and remote registrations per QP.

**Modification scope:**

- `src/include/tilexr_api.h`
- `src/include/tilexr_udma_reg.h`
- `src/comm/tilexr_comm.h`
- `src/comm/tilexr_comm.cpp`
- `src/comm/comm_wrap.cpp`
- `src/comm/udma/tilexr_udma_context.h`
- `src/comm/udma/tilexr_udma_context.cpp`
- `src/comm/udma/tilexr_udma_transport.h`
- `src/comm/udma/tilexr_udma_transport.cpp`
- focused files under `tests/udma/unit/`

**Work:**

1. Add bounded profile descriptors: local regions, region count, and one
   `(local_region, remote_region)` binding per QP.
2. Add collective register/unregister/query APIs returning nonzero persistent
   handles and explicit device `UDMAInfo`/registry views.
3. Generalize registration state so legacy active registration and persistent
   profiles can coexist and be cleaned independently.
4. Register every local MR on each needed HCCP context, exchange every remote
   MR descriptor/key, import remote regions, and build per-QP device metadata
   from the declared bindings.
5. Make partial failure rollback deterministic and preserve cleanup-pending
   state when HCCP cleanup itself fails.
6. Keep legacy handle-zero behavior and `CommArgs` publication unchanged.

**Constraints and non-goals:** Do not make profile selection a mutable global
operation. Do not add dynamic allocation in device code. Do not widen unrelated
registry consumers to multi-region semantics.

**Acceptance and verification:**

- Unit tests cover invalid region counts, invalid QP bindings, overflow,
  partial registration/import failure, independent cleanup, legacy coexistence,
  and exact per-QP token/remote metadata selection.
- Existing UDMA registration and source-guard tests remain green.
- Run:

  `cmake -S . -B build -DTILEXR_BUILD_TESTS=ON`

  `cmake --build build -j`

  `ctest --test-dir build -R 'udma' --output-on-failure`

**Artifacts and downstream interfaces:** A profile handle plus immutable Host
and device views consumed by Tasks 2, 4, and 6.

## Task 2: Add Deferred UDMA READ Batching

**Objective and role:** Let one QP leader post all contributor GET WQEs, ring
one doorbell, and wait one ordered completion frontier.

**Background and prerequisites:** Depends on Task 1's profile device view. The
existing PUT path has deferred publication helpers, while GET always rings its
doorbell immediately.

**Modification scope:**

- `src/include/tilexr_udma.h`
- `tests/udma/unit/test_tilexr_udma_device_api.cpp`
- `tests/udma/unit/test_tilexr_udma_source_guard.cpp`

**Work:**

1. Add a profile-aware registered-range and remote-address accessor.
2. Add deferred `UDMAGetNbiOnQp` that publishes a READ WQE without ringing.
3. Return or expose the submitted completion frontier needed for one bounded
   CQ poll after a batch.
4. Reuse the existing doorbell and quiet primitives without scalar SQ stores.
5. Add exact validation for zero length, null pointers, 32-bit length, QP
   ownership, and registered local/remote ranges.

**Constraints and non-goals:** Do not change legacy immediate GET semantics.
Do not allow UB addresses as local SGE addresses.

**Acceptance and verification:** Unit tests prove N deferred GETs advance the SQ
without ringing, one flush rings the final head, and one quiet consumes the
expected CQ frontier. Source guards verify MTE3 publication precedes `st_dev`.

**Artifacts and downstream interfaces:** Device functions used only by a
single owner block per QP in Task 5.

## Task 3: Build the Multi-MR Hardware Proof

**Objective and role:** Prove the approved architecture on CANN 9.1.T560 and
Ascend950 before replacing the operator.

**Background and prerequisites:** Depends on Tasks 1 and 2. This is the design's
mandatory stop gate.

**Modification scope:**

- a focused hardware probe under `tests/udma/demo/` or the existing UDMA demo
  location;
- minimal CMake/script wiring for that probe;
- no ReduceGrad production code yet.

**Work:**

1. Allocate four independent 2 MiB-aligned device regions per rank: staging
   plus three patterned sources.
2. Register one persistent profile with at least three QPs, binding staging as
   every local region and one distinct remote source region per QP.
3. On two ranks, issue 48 KiB, 256 KiB, 1 MiB, 2 MiB, 4 MiB, 8 MiB, and 16 MiB
   deferred READ batches through every binding.
4. Verify byte-exact destination data and independent profile/legacy-region
   lifetime.
5. Record transfer and combined staging-consumption timing in machine-readable
   output.

**Remote execution:** Create the common directory on both hosts, synchronize
the same source snapshot, source `scripts/common_env.sh`, and build with
CANN `/usr/local/Ascend/cann-9.1.T560`. Do not exclude cards solely because
`npu-smi` reports `Alarm`.

**Acceptance and verification:** All three remote region bindings transfer
correct data through one shared QP set, a legacy registration remains usable,
and unregistering either handle does not invalidate the other.

**Stop condition:** If HCCP rejects simultaneous MRs, the target cannot bind
the selected local/remote tokens, or shared QP state is not independently safe,
stop implementation and return to design review. Do not start Task 4 and do not
restore or optimize the old ReduceGrad.

**Artifacts and downstream interfaces:** Raw JSON/CSV size sweep, target logs,
and the checked-in chunk-size hypothesis used by Task 4.

## Task 4: Replace ReduceGrad Layout, Host Contract, and Launch ABI

**Objective and role:** Remove the old transport model from Host code and
describe only the owner-pull workspace/profile/kernel contract.

**Background and prerequisites:** Depends on a passing Task 3. The public
upstream Python signature stays stable, but the native internal ABI must carry
the three reduce-buffer sources and persistent profile view.

**Modification scope:**

- `src/include/tilexr_moonep.h`
- `src/moonep/reduce_grad/common/reduce_grad_common.h`
- `src/moonep/reduce_grad/host/reduce_grad_layout.h`
- `src/moonep/reduce_grad/host/reduce_grad_layout.cpp`
- `src/moonep/reduce_grad/host/reduce_grad_host.h`
- `src/moonep/reduce_grad/host/reduce_grad_host.cpp`
- `src/moonep/reduce_grad/host/reduce_grad_launch.cpp`
- ReduceGrad Host/layout/ABI unit tests under `tests/moonep/unit/`

**Work:**

1. Delete peer/UDMA transport selection, sender control counts, outbound and
   inbound offsets, ACK state, and the 1 MiB threshold from ReduceGrad layout.
2. Define checked workspace layout for per-QP lane state and two banks of
   `rank_size * chunk_bytes` payload staging.
3. Require A5 UDMA and at least one QP per projection for multi-rank launch.
4. Allocate QPs to projections deterministically, initially proportional to
   projection bytes with stable tie-breaking.
5. Add the three local reduce-buffer slice descriptors and persistent profile
   handle/device pointers to preparation and launch context.
6. Keep one-time preparation separate from hot launch. Validate pointer identity
   and reject stale preparation without synchronizing or registering.
7. Launch only the replacement pure AIV binary through
   `rtKernelLaunchWithFlagV2`.

**Constraints and non-goals:** Existing C symbols may delegate to the new
engine for compatibility, but no entry can launch or contain the old protocol.
Do not add topology-only transport selection.

**Acceptance and verification:** Unit tests cover the minimum-rank rejection and workspace arithmetic at 4/8/16
ranks, QP counts, all projection sizes, alignment, pointer mismatch, unsupported
capability, profile mismatch, and launch argument layout. Source tests assert
old protocol identifiers and acknowledgement paths are absent.

**Artifacts and downstream interfaces:** One immutable launch context and
kernel argument block for Task 5; Torch-facing preparation inputs for Task 6.

## Task 5: Replace the Ascend C ReduceGrad Kernel

**Objective and role:** Implement fused gate/up/down owner-pull with batched
large UDMA reads, helper-group accumulation, one global barrier, and parallel
clear.

**Background and prerequisites:** Depends on Tasks 2 and 4. The current
`tilexr_moonep_reduce_grad_kernel.cpp` is replaced rather than incrementally
retained.

**Modification scope:**

- `src/moonep/reduce_grad/kernels/tilexr_moonep_reduce_grad_kernel.cpp`
- minimal common headers owned by ReduceGrad;
- kernel source/compile guards in `tests/moonep/unit/`.

**Work:**

1. Implement deterministic projection-to-lane work indexing over
   `(local_expert, row_chunk)`.
2. Let one leader block own each QP and batch remote contributors in flattened
   plan order into ping/pong GM banks.
3. Support contributor waves when a plan contains more contributors than one
   rank-sized bank can represent.
4. Pipeline next-bank UDMA with current-bank helper computation using
   magic/epoch-tagged lane flags and bounded cache maintenance.
5. Partition UB subtiles across helper blocks. Load output once, add all
   contributions in order, and store once per subtile and wave.
6. Use accumulator plus ping/pong input UB buffers and precise MTE2/vector/MTE3
   events. Remove broad barriers from inner copy/add loops.
7. Locally synchronize, execute a one-leader-per-rank TileXR barrier, locally
   release workers, and clear three local live slot sets in parallel.
8. Publish the first device failure to status and ensure every block still
   reaches required local barriers to avoid deadlock.

**Constraints and non-goals:** No sender role, UDMA PUT, per-chunk ACK, 8-byte
poll GET, peer packed-record path, or scalar/direct-GM SQ/doorbell writes. No
early return before a collective participant boundary.

**Acceptance and verification:**

- Bisheng compiles the kernel for Ascend950 with CANN 9.1.T560.
- Source guards prove only leaders touch QPs and the deleted protocol is absent.
- Hardware correctness passes sparse, mixed, heavy, full, empty-local, tail,
  repeated-magic, and multi-wave plans at 4 ranks.
- Device status is zero and no CQ error, timeout, vector exception, hang, or
  stale flag appears.

**Artifacts and downstream interfaces:** Embedded replacement AIV binary with
the Task 4 argument contract.

## Task 6: Remove Torch Hot-Path Copies and Registration Switching

**Objective and role:** Ensure the measured Python `reduce_grad` interval
contains only stream-ordered preparation already enqueued by the caller and the
new kernel launch.

**Background and prerequisites:** Depends on Tasks 1 and 4. Current code copies
`reduce_buffer[rank]` into full-gradient tail rows and synchronizes before
re-confirming active registration on every call.

**Modification scope:**

- `integrations/moonep_torch/tilexr_moonep/abi.py`
- `integrations/moonep_torch/tilexr_moonep/runtime.py`
- `integrations/moonep_torch/tilexr_moonep/torch_api.py`
- compatibility adapter only where required by the unchanged public signature;
- focused Python tests under `tests/moonep/python/`.

**Work:**

1. Pass the three reduce buffers directly through the native ABI.
2. Key preparation by allocation identity, local slice pointer, size, shape,
   plan dimensions, QP mapping, and chunk size.
3. Register the persistent ReduceGrad profile once during preparation and retain
   all source/staging owners until completion or explicit re-prepare/close.
4. Remove the full-gradient tail copy and post-launch legacy zeroing from the
   native path.
5. Remove device synchronize and active-region registration calls from every
   prepared hot launch.
6. Require quiescence only when pointer identity changes, a profile is destroyed,
   or the runtime closes.
7. Preserve asynchronous event behavior, in-flight ownership, idempotent close,
   and device-status propagation.

**Constraints and non-goals:** Do not hide first-use registration inside timed
benchmark iterations. Do not weaken tensor/device/contiguity validation.

**Acceptance and verification:** Python unit tests prove one preparation causes
one profile registration, repeated launches cause none, changed pointers require
re-prepare, hot launch does not synchronize, and all ownership/error paths
release resources exactly once.

**Artifacts and downstream interfaces:** Stable upstream `Buffer.reduce_grad`
behavior with an explicitly preparable steady-state path.

## Task 7: Add Correctness and Native-Baseline Benchmark Harnesses

**Objective and role:** Provide identical, auditable correctness and timing
boundaries for TileXR and the pinned native baseline.

**Background and prerequisites:** Can begin after Task 4's ABI is stable and is
completed after Tasks 5 and 6.

**Modification scope:**

- focused additions under `tools/moonep/`;
- focused tests under `tests/moonep/python/`;
- `scripts/README.md` or MoonEP validation documentation for commands;
- no modifications under `reference/ascend-moonep-dev` beyond ignored build
  products.

**Work:**

1. Add an isolated ReduceGrad runner that can invoke native or TileXR with the
   same generated plan and tensors.
2. Prepare and register before warmup; use device events around only the three
   native launches or one fused TileXR launch.
3. Gather every iteration's latency and compute the cross-rank maximum before
   P50/P99 aggregation.
4. Emit raw JSON including commit IDs, CANN/driver/device metadata, dimensions,
   plan density, warmup, iterations, chunk, QPs, block layout, and one-time
   preparation latency.
5. Add exact mutation checks for owned rows, live local slots, unused slots,
   non-local slots, and ordered FP32 results.
6. Cover native dedicated 3584x3072 cases and the 7168x2048 E=384/B=48 primary
   case.

**Constraints and non-goals:** No host wall-clock comparisons for the pass gate.
No rank-zero-only timing. No different plan, initialization, or synchronization
between implementations.

**Acceptance and verification:** Unit-test generation/statistics locally, then
run correctness at 4 ranks. Raw results must make timing boundaries and
excluded one-time work explicit.

**Artifacts and downstream interfaces:** Reproducible benchmark command and raw
result schema consumed by Task 9.

## Task 8: Run Local Build, Unit, ABI, and Source Verification

**Objective and role:** Catch host, layout, lifecycle, source-rule, and Python
regressions before consuming hardware time.

**Background and prerequisites:** Depends on Tasks 1, 2, 4, 5, 6, and 7.

**Modification scope:** Tests and fixes within the files already owned by prior
tasks. No unrelated refactor.

**Verification:**

1. Initialize submodules and source `scripts/common_env.sh` where the local
   environment supports CANN.
2. Configure with MoonEP and tests enabled.
3. Build and install.
4. Run all UDMA and MoonEP unit/source/ABI tests.
5. Run focused Python unit tests for FFI, Torch preparation, compatibility, and
   benchmark statistics.
6. Run `git diff --check` and inspect the complete scoped diff.

If the Windows workspace lacks CANN, run host-independent Python/source tests
locally and perform the full build on both target hosts. Report that boundary
without treating it as hardware proof.

**Acceptance and verification:** All runnable tests pass; any test requiring
unavailable local CANN is listed and then passed remotely in Task 9.

**Artifacts and downstream interfaces:** A single synchronized source snapshot
ready for remote validation.

## Task 9: Build and Validate on Both Ascend950 Hosts

**Objective and role:** Establish target-matched correctness before tuning.

**Background and prerequisites:** Depends on Task 8. Use the exact same source
snapshot and remote path on both machines.

**Remote work:**

1. Resolve and verify `/tmp/TileXR-reducegrad-20260811-44dc37b` on each host,
   then create it and synchronize the workspace while excluding `.git`, local
   build output, ignored native build output, and unrelated untracked archives.
2. Initialize submodules or synchronize their required contents.
3. Source `scripts/common_env.sh`; verify CANN 9.1.T560, Bisheng, driver, and NPU
   visibility.
4. Configure, build, install, and run focused UDMA/MoonEP tests on both hosts.
5. Run the hardware profile probe.
6. Run isolated correctness at four ranks on physical devices 0-3 of host 106.
   Retain the physical device mapping with the results.

**Constraints and non-goals:** Preserve both hosts' unrelated `/tmp` contents.
Do not treat `Alarm` alone as an unhealthy card. Do not claim 8/16-rank proof
from the required 4-rank run.

**Acceptance and verification:** Both hosts build the same source; all focused
tests pass; 4-rank exact correctness passes; repeated runs do not hang or
consume stale flags.

**Artifacts and downstream interfaces:** Build logs, environment metadata,
correctness JSON, and a stable candidate for performance tuning.

## Task 10: Tune, Profile, and Enforce the Faster-Than-Native Gate

**Objective and role:** Select measured defaults and prove the user-visible
performance requirement rather than merely improving the old implementation.

**Background and prerequisites:** Depends on Task 9 correctness.

**Work:**

1. Build native MoonEP at the pinned baseline commit in a separate ignored
   remote comparison directory.
2. Sweep 48 KiB through 16 MiB transfer sizes, then tune chunk size, QP count,
   projection allocation, helper count, UB tile, bank overlap, batch depth,
   barrier, and clear partitioning one group at a time.
3. Re-run correctness after every kernel-affecting tuning change.
4. For each primary 4-rank case, run 20 warmups and at least 50
   device-event iterations, repeated three times.
5. Use `msprof` on representative native and TileXR runs. Verify MiB-scale
   batched reads, next-bank/current-compute overlap, no ACK GETs, and no hidden
   registration/synchronize inside the measured stage.
6. Check in only defaults justified by the retained raw evidence.

**Pass gate:**

- TileXR cross-rank-max P50 and P99 are lower than native in every repeated
  primary 4-rank run.
- Median advantage is at least 3%.
- All correctness checks remain green.
- One-time preparation is separately reported.

**Failure handling:** Continue targeted tuning while evidence identifies an
in-scope bottleneck. If the correct new architecture cannot meet the pass gate,
report the measured limitation and do not label the task complete. Do not
restore the deleted implementation.

**Artifacts and downstream interfaces:** Raw JSON/CSV, profiler outputs,
comparison summary, and final checked-in tuning constants.

## Task 11: Final Documentation and Completion Verification

**Objective and role:** Align maintained documentation with the replacement and
verify the exact final tree after the last tuning edit.

**Modification scope:**

- `docs/moonep/DISPATCH_COMBINE.md`
- `docs/BUILD_VERIFICATION.md`
- relevant script documentation;
- approved spec and this plan only for factual implementation notes.

**Work:**

1. Remove descriptions of the old ReduceGrad publish/ack protocol.
2. Document preparation lifetime, UDMA-only multi-rank requirement, workspace
   sizing, and benchmark commands.
3. Record validation scope as Ascend950/CANN 9.1.T560 on the actual 4-rank
   runs.
4. Run the complete focused local and remote verification again after the final
   edit, including `git diff --check` and status review.
5. Use `superpowers-neo-verification-before-completion` before describing the
   work as complete or faster than native.

**Acceptance and verification:** Documentation matches behavior, no old
ReduceGrad source path remains, all final tests pass, and the retained evidence
satisfies the faster-than-native gate.

## Dependency Order

`Task 1 -> Task 2 -> Task 3 (hardware stop gate) -> Task 4 -> Task 5 -> Task 6`

Task 7 may start after Task 4 and finishes after Tasks 5 and 6. Task 8 depends
on all implementation and harness work. Tasks 9, 10, and 11 are sequential.

## Handoff Checklist

- Approved design and this plan are retained with the implementation.
- Old ReduceGrad transport code and source guards are removed, not disabled.
- Persistent profiles are independently owned and cleaned.
- Hot launches perform no synchronize, MR registration, or legacy tail copy.
- Remote directories on both hosts contain the same source snapshot.
- Correctness evidence covers 4 ranks on physical devices 0-3.
- Performance evidence compares the pinned native commit with identical timing
  boundaries and contains cross-rank-max P50/P99.
- No completion claim is made unless the reproducible performance gate passes.

## Retained Tuning Conclusions

Do not restart owner-pull tuning from the original variable sweep. The
implementation outcome, exact measurements, GET calibration hashes, rejected
chunk/bank/QP/route/helper/contributor-ready experiments, and sender-PUT stop
gate are recorded in
`docs/specs/2026-08-11-reduce-grad-owner-pull-design.md` under
"Implementation Outcome and Experiment Ledger". Treat that section as the
starting evidence for the next phase.
