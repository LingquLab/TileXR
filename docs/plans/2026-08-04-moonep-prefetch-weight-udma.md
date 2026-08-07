# MoonEP Fused PrefetchWeight UDMA Implementation Plan

## Goal

Implement the approved fused MoonEP PrefetchWeight operator over direct TileXR
UDMA, replace the development PrefetchWeight stub in place, integrate a packed
registered-weight lifecycle into the Torch facade, and measure the native stage
through the existing MoonEP benchmark.

The authoritative architecture and acceptance boundary are in
`docs/specs/2026-08-04-moonep-prefetch-weight-udma-design.md`.

## Scope And Non-Goals

- Preserve C++14 and CANN 9.1.0 Host compatibility.
- Target the A5 / Ascend950 `__NPU_ARCH__ == 3510` Kernel path.
- Keep `reference/MoonEP` comparison-only.
- Keep Dispatch, Combine, and ReduceGrad as stubs.
- Do not add peer-memory, SDMA, HCCL, or local-copy fallback behavior.
- Do not add multiple simultaneously registered UDMA regions.

## Task 1: Add Independently Owned UDMA Worker Queues

**Objective and role:** Turn the existing `qpNum/qpIdx` device contract into a
real configurable multi-worker queue image so multiple Prefetch AIV blocks can
submit without sharing mutable SQ/CQ state.

**Background and prerequisites:** The current transport creates one QP/CQ per
route EID and flattens one queue context per peer. Existing device APIs already
accept `qpIdx` internally but public wrappers select zero.

**Modification scope:**

- `src/include/tilexr_udma_types.h`
- `src/include/tilexr_udma.h`
- `src/comm/udma/tilexr_udma_layout.{h,cpp}`
- `src/comm/udma/tilexr_udma_transport.{h,cpp}`
- focused UDMA unit tests

**Constraints and non-goals:**

- Provision independent QP, SQ, and CQ state for every worker.
- Flatten queue and remote-memory metadata as `[peer][qpIdx]`.
- Preserve existing qpIdx-zero wrappers and behavior.
- Bound the worker configuration to 1, 2, 4, or 8 and reject invalid values.
- Do not change registered-region count or public registration lifecycle.

**Acceptance and verification:**

- Pure layout tests cover one and multiple QPs, pointer arithmetic, dimensions,
  and invalid vector sizes.
- Existing UDMA registry and transport-layout tests remain green.
- Device helpers expose qpIdx-aware GET and status-returning quiet operations.
- Host-only compilation and `git diff --check` pass.

**Artifacts and interfaces:** A stable `UDMAInfo` image with `qpNum`, indexed
queue/memory helpers, and a transport-selected worker count consumed by Task 2.

## Task 2: Implement The Fused Native PrefetchWeight Operator

**Objective and role:** Replace the V1 local-copy stub with one native V1 Host
call and one runtime-registered Kernel that moves gate, up, and down slot rows
in place.

**Background and prerequisites:** Requires Task 1's exclusive qpIdx worker
contract and the existing Planner's `expertsToCopy[rank, B]` output.

**Modification scope:**

- `src/include/tilexr_moonep.h`
- `src/moonep/CMakeLists.txt`
- new Host layout/launch files under `src/moonep/host/`
- new Kernel files under `src/moonep/kernels/`
- focused MoonEP C/C++ tests

**Constraints and non-goals:**

- Keep `TileXRMoonEpPrefetchWeightArgsV1`, its function, and MoonEP SOVERSION 1
  while the project is still under development.
- Replace the V1 implementation in place with the fixed three-projection ABI.
- Accept compact `[2 * B, ...]` BF16/FP16 projections in one registered arena.
- Validate checked byte arithmetic, 64-byte alignment, per-peer registry bounds,
  and one-WQE `uint32_t` row length.
- Launch one fused Kernel with blockDim bounded by B and worker count.
- Keep unused slots untouched and report CQ failures asynchronously.
- Do not synchronize the caller stream in the Host API.

**Acceptance and verification:**

- ABI, capability, invalid-input, overflow, registry-bound, and launch tests.
- Kernel/source tests cover one fused launch and absence of old stub behavior.
- Target CANN compilation reaches every supported dtype and worker path.
- Installed symbol and SONAME inspection confirms PrefetchWeight V1 and
  `libtilexr-moonep.so.1`.

**Artifacts and interfaces:** `TileXRMoonEpPrefetchWeightV1`, the fused Kernel
artifact, and capability metadata used by Task 3.

## Task 3: Integrate Packed Weights Into Torch And MoonEP Benchmark

**Objective and role:** Make the native operator usable without hot-path
registration or packing and produce honest stage-specific correctness and
performance evidence.

**Background and prerequisites:** Requires Task 2's exact V1 tensor contract,
library name, capabilities, and asynchronous status behavior.

**Modification scope:**

- `integrations/moonep_torch/tilexr_moonep/**`
- `tools/moonep/**`
- `tests/moonep/python/**`
- MoonEP demo and documentation references affected by the removed stub

**Constraints and non-goals:**

- Own one flat aligned backing allocation and three `[2 * B, ...]` views.
- Register once after deterministic/model weight initialization and unregister
  only after quiescence.
- Launch exactly one V1 Prefetch call and mutate slots in place.
- Keep packed allocation, views, Plan, and UDMA handle alive asynchronously.
- Benchmark registration and initialization outside the measured interval.
- Report PrefetchWeight correctness/performance independently while aggregate
  transport performance remains invalid due to the other stubs.

**Acceptance and verification:**

- Fake-runtime tests prove one registration, one fused call, and ordered cleanup.
- Correctness covers full, sparse, duplicate, and unused slots for all three
  projections.
- Config/report tests cover expert shapes, transferred bytes, latency, and GB/s.
- Representative MoonEP shapes include 512x512, 1024x3072, and 3584x3072 BF16.

**Artifacts and interfaces:** Packed projection owner, updated Buffer API,
updated benchmark case/report schema, and stage-specific validation output.

## Task 4: Validate And Tune On Target Hardware

**Objective and role:** Establish CANN 9.1 compile evidence, A5 / Ascend950 data
plane correctness, and a measured default worker/blockDim policy.

**Background and prerequisites:** Requires the final integrated source from
Tasks 1-3 and access to compatible target hardware. Host checks alone are not
runtime or performance proof.

**Modification scope:** Validation artifacts and only tuning constants or
heuristics justified by the measurements. No unrelated environment changes.

**Validation order:**

1. Run final host C++ and Python focused suites.
2. Configure, build, and install with the CANN 9.1 A5 toolchain.
3. Inspect exported symbols, SONAME, dependencies, and RPATH/RUNPATH.
4. Run bounded BF16/FP16 correctness with full, sparse, duplicate, and unused
   slot plans.
5. Sweep 1, 2, 4, and 8 workers plus valid blockDim choices on the benchmark
   shapes, with warmup and synchronized device timing.
6. Select the default from stable cross-rank maximum latency and effective
   bandwidth while checking sparse-case regressions.

**Acceptance and verification:** Retain commands, target/version metadata, raw
samples, and clear evidence boundaries. If target hardware is unavailable,
record the exact remaining build/run command and do not claim UDMA performance.

## Final Verification

- Run all focused UDMA, MoonEP, Torch facade, and benchmark tests from the final
  source state.
- Run `git diff --check` and inspect the final scoped diff.
- Search active code for the removed PrefetchWeight stub behavior and forbidden
  fallback transports.
- Report host, target compile, target runtime, cross-node, and performance
  evidence separately.
