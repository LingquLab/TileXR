# MoonEP ReduceGrad Hybrid Peer-Window/UDMA Implementation Plan

## Goal

Implement the proposed native MoonEP ReduceGrad V2 contract from
`docs/specs/2026-08-04-moonep-reduce-grad-peer-window-design.md`. Rows up to
1 MiB use existing TileXR peer windows; larger rows use TileXR registered UDMA.
Integrate both paths, including mixed projections, into the current MoonEP
benchmark with real correctness and performance measurement.

## Scope and Non-Goals

The work covers the public V2 ABI, Host layout and validation, Runtime V2
launch, A5 kernel, Torch facade, benchmark, focused tests, and staged hardware
validation. It preserves C++14 Host code, CANN 9.1 compatibility, the V1 ABI,
and the existing Planner.

It does not implement native Dispatch, PrefetchWeight, or Combine; add SDMA or
HCCL; create a registered custom operator; support UDMA from `InitThread`; or
claim cross-node support without real two-node evidence.

## Authoritative References

- `docs/specs/2026-08-04-moonep-reduce-grad-peer-window-design.md`
- `src/include/tilexr_moonep.h`
- `src/include/comm_args.h`
- `src/include/tilexr_data_as_flag.h`
- `src/include/tilexr_udma.h`
- `src/include/tilexr_udma_reg.h`
- `src/ep/host/ep_launch_context.cpp` (registered-workspace precedent)
- `tests/udma/demo/tilexr_udma_demo_kernel.cpp` (device API precedent)
- `src/moonep/planner/host/planner_launch.cpp`
- `src/moonep/planner/kernels/tilexr_moonep_planner_kernel.cpp`
- PR #90 (`codex/ep-memory-dispatch-combine-patch`) window/layout changes
- PR #90 cross-node and half-reuse review threads at head `b40c66e`
- issue #92 memory-mode capacity follow-up
- `integrations/moonep_torch/tilexr_moonep/`
- `tools/moonep/benchmark.py`
- `reference/MoonEP/moonep/grad_reduce.py` (comparison only)
- `reference/MoonEP/tests/test_grad_reduce.py` (comparison only)

Active targets must not include or link anything from `reference/`.

## Task 1: Define the V2 ABI and Host Contract

**Objective and role:** Add an unambiguous fused ReduceGrad ABI while keeping
the shipped V1 layout and behavior callable.

**Background and prerequisites:** The design fixes three fp32 gradient
tensors, in-place owner accumulation, local slot cleanup, caller-owned status,
the strict 1 MiB transport boundary, bounded wait, and one asynchronous launch.

**Modification scope:**

- `src/include/tilexr_moonep.h`
- `src/moonep/host/tilexr_moonep.cpp`
- focused ABI/Host tests under `tests/moonep/unit/`

**Constraints and non-goals:**

- Do not reinterpret or remove `TileXRMoonEpReduceGradArgsV1`.
- Do not mark the V1 function as the native fused implementation.
- Do not synchronize the caller stream.
- Validate with checked 64-bit arithmetic before narrowing or pointer offsetting.

**Implementation:**

1. Add ABI V2 constants, workspace query, fused args, entry point, and V2
   capability query.
2. Define status values, including peer timeout and UDMA CQ failure, and require
   a contiguous int32 `[1]` device tensor.
3. Compute `rowBytes` with checked arithmetic and select peer for `<= 1 MiB`
   and UDMA for `> 1 MiB`, independently for all three projections.
4. Validate plan/communicator agreement, A5 target flags, required peer
   pointers, gradient ranks/dtypes/shapes, workspace/registry containment,
   communicator mode, byte counts, wait budget, and flags.
5. Preserve V1 tests and add exact C/C++ layout coverage for V2.

**Acceptance and verification:**

- V1 layout tests remain unchanged and pass.
- V2 accepts valid rank-2/rank-3/rank-4 tensors with different per-expert sizes.
- Boundary tests select peer at exactly 1 MiB and UDMA at 1 MiB plus one fp32
  element; mixed projection selections are preserved in the kernel arguments.
- Every invalid descriptor, overflow, missing required transport, registry
  mismatch, or unsupported topology returns a deterministic synchronous error
  without launching.

**Artifacts and downstream interfaces:** A stable `TileXRMoonEpReduceGradV2`
contract consumed by the Host launcher and Python ctypes definitions.

## Task 2: Implement Hybrid Layout and CPU Oracle

**Objective and role:** Make peer-window offsets, UDMA registered-workspace
offsets, transport selection, chunk counts, work partition, and expected math
independently testable before device code.

**Background and prerequisites:** Task 1 fixes the public tensor/plan contract.
PR #90 provides a 512 MiB memory-mode window after `IPC_DATA_OFFSET`, reserves a
1 MiB state prefix, and splits the remainder into two data halves. Issue #92 may
move that capacity behind an independent contract.

**Modification scope:**

- new `src/moonep/reduce_grad/common/` and `host/` layout files
- new or extended `tests/moonep/unit/` layout/oracle tests

**Constraints and non-goals:**

- No device API calls in the pure layout/oracle layer.
- No fabricated one-element tile when capacity is invalid.
- Do not hard-code either current main's 100 MiB or PR #90's 512 MiB value.
- Keep units explicit in names: window/state/half/record/payload bytes and chunk
  count.
- Keep every UDMA WQE byte count within `uint32_t` and every region offset in
  checked `uint64_t` arithmetic.

**Implementation:**

1. Add one capacity resolver whose current PR #90 source is
   `IPC_BUFF_MAX_SIZE`; isolate the later issue #92 change to this boundary.
2. Compute the 1 MiB state reservation, two halves, worst-case `R*B` record
   capacity, and 512/480 DataAsFlag overhead with checked arithmetic.
3. Define deterministic fixed `source*B+slot` peer addresses from `[R,B]` so
   source/owner agreement needs no Host readback of the asynchronous plan;
   retain owner-local compaction as a benchmark-driven follow-up.
4. Compute a cache-line-aligned registered workspace with two outbound and two
   inbound payload stages per peer plus sequence, ack, and completion state.
   Keep growth `O(R * udmaChunkBytes)` and require the workspace pointer at
   registered region offset zero.
5. Define flattened peer sender/receiver work, one control owner per peer that
   rotates over all negotiated QPs, UDMA sequence/ack work, and final-tail
   bounds per projection.
6. Implement a CPU oracle that reduces in `(src, b)` order, preserves non-owner
   rows, zeros used local slots, and preserves unused local slots.

**Acceptance and verification:**

- Inject and cover current-main, PR #90 512 MiB, and future capacities, plus
  `B=1`, representative/max `B`, fixed incoming slots, exact chunks, 480-byte
  tails, zero/invalid capacity, and large 64-bit offsets.
- Cover the transport threshold immediately below/at/above 1 MiB, mixed
  projections, UDMA stage alignment, region containment, WQE limits, and
  workspace overflow.
- Cover ring, duplicate, fan-in, empty, asymmetric, and seeded random plans.

**Artifacts and downstream interfaces:** A checked `ReduceGradLayout` shared by
Host validation/launch and tests, plus an oracle for demo/benchmark comparison.

## Task 3: Add Direct Runtime V2 Launch and Build Wiring

**Objective and role:** Build and launch one A5 device-only kernel artifact from
the fused Host entry point.

**Background and prerequisites:** Tasks 1 and 2 provide validated parameters and
an exact layout. Follow the existing Planner Host/kernel separation.

**Modification scope:**

- new `src/moonep/reduce_grad/host/*launch*`
- new `src/moonep/reduce_grad/kernels/` source
- `src/moonep/CMakeLists.txt` and/or nested ReduceGrad CMake
- install and source-ownership tests

**Constraints and non-goals:**

- Host remains C++14; device compilation may use the repository's current
  Bisheng CCE standard and A5 architecture flags.
- Put argument packing and the seven-argument `rtKernelLaunchWithFlagV2` call
  directly in the Host launch function.
- Kernel source must not contain Runtime includes, `rtKernelLaunch`, or `<<<`.
- Runtime RPATH/RUNPATH must not contain toolkit `devlib`.

**Implementation:**

1. Define one contiguous kernel-argument POD with exact pointer/width ordering,
   selected transport bits, peer layout, registered workspace layout, and UDMA
   chunk/state offsets.
2. Obtain `commArgs` and one magic value before launch.
3. Build/install the A5 kernel shared object and link it into the MoonEP Host
   library without linking `reference/` or adding HCCL; use the existing TileXR
   UDMA device header and core registration API.
4. Add source checks for launch placement, kernel purity, A5 flags, install
   paths, and forbidden dependencies.

**Acceptance and verification:**

- Host fake tests observe one launch with exact block dimension, stream, args
  size, and error translation.
- Target build emits both Host and kernel libraries with `$ORIGIN` install RPATH.

**Artifacts and downstream interfaces:** A callable launcher used by Task 4.

## Task 4: Implement the Hybrid Streaming A5 Kernel

**Objective and role:** Perform deterministic, multi-chunk cross-node reduction
and cleanup through PR #90-style DataAsFlag records for rows up to 1 MiB and
registered UDMA lanes for larger rows, with bounded progress on both paths.

**Background and prerequisites:** Tasks 2 and 3 fix layout and launch ABI. The
algorithm and synchronization invariants are in the design spec.

**Modification scope:**

- `src/moonep/reduce_grad/kernels/`
- small device-only helpers under `src/moonep/reduce_grad/common/` when needed
- kernel source and hardware tests under `tests/moonep/`

**Constraints and non-goals:**

- Peer projections use `peerMems`, `IPC_DATA_OFFSET`, the resolved window
  capacity, and PR #90's 512-byte/480-byte DataAsFlag format.
- UDMA projections stage source chunks through the registered workspace and use
  `UDMAPutRegisteredSignalNbiOnQp`; they never target `peerMems` or unregistered
  remote memory.
- Exactly one persistent control AIV owns every negotiated QP for each target
  peer and selects QPs by sequence ordinal. No producer or receiver AIV may
  mutate that peer's WQ state.
- Do not call an unbounded DataAsFlag receive or generic flag wait.
- Do not touch non-owned expert rows or clear unused slot rows.
- Preserve ascending `(src, b)` accumulation order for each output element.
- Sender and receiver AIV groups must make progress concurrently; do not insert
  an all-AIV barrier between producing and consuming streamed chunks.

**Implementation order:**

1. Initialize status, peer addresses, registered-workspace addresses, selected
   transports, UB plan table, per-owner incoming counts, and deterministic
   compact slot indices.
2. Partition the persistent AIV grid into one sender/control AIV per peer and
   receiver AIVs with bounded ready/completion state.
3. Implement sender-side bounded clear acknowledgement, GM -> UB DataAsFlag
   packing, and direct MTE3 push to `peerMems[owner]`. Check for cleared flags
   before every write, including the first use of a half in a later launch.
4. Implement receiver-side bounded arrival polling, DataAsFlag payload unpack,
   64-bit fp32 accumulation, deterministic `(src,b)` ordering, and flag clear.
5. Alternate the two data halves by chunk index; cleared embedded flags are the
   per-slot acknowledgement, so no all-rank chunk barrier is added.
6. Add two local registered outbound and two inbound stages per peer. Pipeline
   GM -> UB -> outbound GM preparation with UDMA payload+signal PUTs, checked CQ
   completion, local receiver accumulation, and remote acknowledgement.
7. Keep receiver AIVs active while each control AIV pipelines two outgoing
   stages, checks CQ status, and polls exact remote completion sequences on its
   exclusively owned QP. Reject stale invocation/sequence values, separate
   remote-written and local-written state by cache line, and bound every wait.
8. Add a backward-compatible checked UDMA quiet helper that returns
   `UDMAPollCQ` status; retain the existing void `UDMAQuiet` API for current
   callers.
9. Fuse projection loops and clear each used local source row only after peer
   acknowledgements and UDMA acknowledgements for all three projections.
10. Expose bounded sender/receiver/control, record-batch, block, tile, and UDMA
   chunk tuning inputs needed by performance validation.

**Acceptance and verification:**

- Target compiler accepts all selected DataAsFlag, vector, cache, and UDMA API
  overloads and resource usage.
- Same-node hardware matches the oracle for full/tail, duplicate/fan-in, empty,
  asymmetric, random, repeated, and large-offset cases.
- Timeout injection exits with status rather than hanging or clearing unsafe
  slots.
- Repeated launches cannot reuse either half until every destination record for
  that half has been cleared.
- Multiple AIVs never race any peer QP, UDMA stage reuse requires the exact
  ack sequence, and injected CQ failure reaches status instead of being ignored.

**Artifacts and downstream interfaces:** The native fused kernel used by V2.

## Task 5: Integrate the Torch Facade

**Objective and role:** Route high-level MoonEP backward through one fused native
V2 call while preserving current device, stream, Plan, tensor, and status
lifetimes.

**Background and prerequisites:** Task 1 defines ctypes ABI; Task 4 implements
the launched semantics.

**Modification scope:**

- `integrations/moonep_torch/tilexr_moonep/abi.py`
- `integrations/moonep_torch/tilexr_moonep/runtime.py`
- `integrations/moonep_torch/tilexr_moonep/torch_api.py`
- `tests/moonep/python/`

**Constraints and non-goals:**

- Do not initialize `torch.distributed` or HCCL.
- Require current-device NPU tensors, contiguity, zero storage offset for the
  exported full tensor, fp32 dtype, and first dimension `E+B`.
- Keep asynchronous references alive until `quiesce()` or close.
- Keep deprecated `*_reduce` fields source-compatible but do not route V2
  through three legacy calls.
- Never register or unregister memory while a bound-stream operation is in
  flight, and never include registration latency in a timed stage sample.

**Implementation:**

1. Add ctypes V2 query/argument structures, registration symbols, and
   highest-version handling.
2. Allocate one ReduceGrad status tensor per Plan and retain it through launch.
3. Add an explicit ReduceGrad preparation step. It queries the hybrid layout,
   lazily allocates and zeros an aligned byte workspace, and collectively calls
   `TileXRUDMARegister` only when a projection exceeds 1 MiB. Compatible shapes
   reuse the registration; incompatible shapes require a quiesced collective
   unregister/register cycle.
4. Replace the runtime loop of three V1 calls with one fused V2 call.
5. Check ReduceGrad status at synchronization and cleanup boundaries. On close,
   quiesce, unregister, release the workspace, and only then destroy the
   communicator.
6. Update capability identity so benchmark reports ReduceGrad native without
   misreporting the V1 stub.

**Acceptance and verification:**

- Fake runtime records exactly one V2 call on the bound current stream.
- Fake runtime proves collective registration happens before warmup/timing,
  stays cached across compatible calls, and unregisters after quiescence.
- Invalid dtype/device/shape/contiguity/storage/plan/status cases fail before
  native invocation.
- Plan and tensors remain referenced through asynchronous completion.

**Artifacts and downstream interfaces:** A native `buffer.reduce_grad()` path
used unchanged by end-to-end backward orchestration.

## Task 6: Integrate Correctness and Performance Benchmarking

**Objective and role:** Turn the existing ReduceGrad benchmark stage from stub
copy timing into honest native distributed correctness and performance evidence.

**Background and prerequisites:** Task 5 provides the high-level call. Overall
full-flow performance remains invalid while other stages are stubs.

**Modification scope:**

- `tools/moonep/config.py`
- `tools/moonep/benchmark.py`
- `tools/moonep/cases/`
- reporting helpers only where new fields require them
- `tests/moonep/python/`

**Constraints and non-goals:**

- Do not compare native ReduceGrad against the old copy stub as a speedup.
- Separate ReduceGrad-specific validity from overall transport validity.
- Use device events with warmup and an explicit synchronization boundary.
- Retain enough metadata to reproduce every performance result.

**Implementation:**

1. Extend case configuration with gate/up/down expert shapes while preserving a
   small legacy-derived default.
2. Initialize deterministic rank/slot/projection values that every rank can use
   to compute its expected owner rows from the shared Plan.
3. Replace copy-exact checks with owner-sum, non-owner-preservation, used-slot
   zero, unused-slot preservation, and status checks.
4. Record per-rank samples, cross-rank maxima, chunk/stride/block/tile metadata,
   streamed bytes, effective bandwidth, topology, and stage validity. Also
   report logical row bytes, the exact 1 MiB threshold, selected transport per
   projection, UDMA chunk/workspace bytes, registration state, and proof that
   registration time is excluded.
5. Add focused cases immediately below/at/above 1 MiB, mixed transports, tails,
   fan-in, and a memory-feasible `7168 x 3072` projection.
6. Add bounded block/tile/UDMA-chunk sweeps and retain the fastest configuration
   that first passes correctness.

**Acceptance and verification:**

- Host-only fake-provider tests cover schema, deterministic expected values,
  reports, and capability flags.
- A5 benchmark artifacts distinguish same-node, cross-node, oversubscribed, and
  unverified runs.

**Artifacts and downstream interfaces:** Reproducible ReduceGrad benchmark JSON
and human-readable summaries integrated into the current MoonEP workflow.

## Task 7: Final Validation and Performance Tuning

**Objective and role:** Establish evidence at each available layer and optimize
only against correct target-hardware runs.

**Background and prerequisites:** Tasks 1-6 are complete from one final source
state.

**Modification scope:** Test/build artifacts and benchmark outputs only; any
source tuning remains within the approved kernel/Host parameters.

**Constraints and non-goals:**

- Do not install packages or modify persistent CANN/driver state without new
  authority.
- Do not call same-node, compile, or simulator evidence cross-node proof.
- Do not retain a faster configuration that fails any correctness case.

**Validation order:**

1. Run focused C/C++ ABI, layout, Host, oracle, source, and Python tests.
2. Run `git diff --check` and inspect public symbols, kernel ABI, forbidden
   dependencies, PR #90 capacity/DataAsFlag agreement, RPATH/RUNPATH, and the
   changed benchmark schema.
3. Build Host and kernel with the CANN 9.1 A5 target toolchain.
4. Run bounded 1-rank then 8-rank same-node correctness for peer-only,
   UDMA-only, and mixed projections.
5. Sweep valid block/tile/UDMA-chunk configurations with warm device-event
   timing; registration remains outside the measured interval.
6. Run the two-node all-peer DataAsFlag push/consume/clear preflight on a PR #90
   or issue #92 capacity base. Verify unique payloads and reuse both halves
   before any timed ReduceGrad case.
7. Run an all-peer registered UDMA payload/signal/checked-completion/ack
   preflight. Verify both stages and repeated sequence reuse.
8. Run cross-node peer-only, UDMA-only, and mixed correctness cases, then stable
   cross-node performance cases.
9. Re-run the focused local/static suite after the final tuned source edit.

**Acceptance and verification:**

- All available layers pass from the final source state.
- Reports state exact hardware, topology, CANN, driver, build SHA, shapes,
  configuration, correctness, latency, and bandwidth.
- Missing two-node hardware remains an explicit unverified requirement rather
  than a support claim.
- A failed required peer or UDMA preflight fails cross-node acceptance and does
  not authorize a transport switch.

**Artifacts and downstream interfaces:** Final source, tests, installed build
artifacts where authorized, and benchmark reports ready for Git delivery.

## Final Review Checklist

- V1 compatibility and V2 capability semantics are clear.
- Host and kernel argument ordering match exactly.
- No active target includes `reference/`.
- Peer/UDMA selection is exactly `<= 1 MiB` versus `> 1 MiB` per projection.
- UDMA uses only registered workspace offsets and one control AIV per peer that
  rotates over all negotiated QPs; SDMA and HCCL remain absent.
- Every peer-window offset stays within the resolved memory-mode capacity.
- Every registered-workspace offset and WQE length stays within validated
  region and integer bounds.
- Current-main, PR #90 512 MiB, and injected future capacities use one formula.
- Every ready/clear wait is bounded and every error path preserves unsafe slots.
- Sender and receiver AIV groups cannot deadlock behind an all-AIV barrier.
- Owner/non-owner/used/unused tensor invariants are tested.
- Benchmark timing is device-side, warmed, synchronized, and reproducible.
- Registration/unregistration is collective, lifetime-safe, and outside timed
  ReduceGrad samples.
- Same-node and cross-node claims match the hardware actually exercised.
