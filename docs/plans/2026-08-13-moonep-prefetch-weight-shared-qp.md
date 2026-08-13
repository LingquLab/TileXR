# MoonEP PrefetchWeight Shared-QP Plan

## Goal And Scope

Implement the approved design in
`docs/specs/2026-08-13-moonep-prefetch-weight-shared-qp-design.md`, validate it
on CANN 9.1 and eight Ascend950 devices, and deliver a patch based on current
`origin/main`.

Only PrefetchWeight layout, private launch ABI, Kernel QP selection, focused
tests, benchmark tooling, and performance documentation are in scope. Public
ABI, registration, peer-memory, and other MoonEP stages are non-goals.

## Task 1: Implement And Test The Mapping Contract

**Objective and role:** Separate logical worker assignment from physical QP
selection without changing slot ownership.

**Background and prerequisites:** Use the fixed shared-domain profile from
`src/comm/udma/tilexr_udma_config.*` and the approved mappings in the design.

**Modification scope:**

- `src/moonep/prefetch_weight/host/prefetch_weight_layout.*`
- `tests/moonep/unit/test_tilexr_moonep_prefetch_weight_host.cpp`

**Constraints and non-goals:** Preserve supported worker counts and the current
block-dimension override. Non-shared QPs use identity mapping.

**Acceptance and verification:** The Host unit test proves four-worker shared
mapping `{0,1,2,16}`, eight-worker shared mapping
`{0,1,2,3,4,5,16,17}`, identity fallback, and invalid-input behavior.

**Artifacts and interfaces:** A packed private QP map stored in
`PrefetchWeightLayout` for Task 2.

## Task 2: Propagate And Consume The Physical QP

**Objective and role:** Pass the map through the registered direct-launch ABI
and use it for all PrefetchWeight WQ/CQ operations.

**Background and prerequisites:** Depends on Task 1. The logical worker remains
the slot scheduler.

**Modification scope:**

- `src/moonep/prefetch_weight/host/prefetch_weight_launch.cpp`
- `src/moonep/prefetch_weight/kernels/tilexr_moonep_prefetch_weight_kernel.cpp`
- focused launch/source tests under `tests/moonep/unit/`

**Constraints and non-goals:** Preserve WQE-in-UB, MTE3 publication, `st_dev`
doorbells, CQ accounting, status behavior, and public API. Do not introduce a
second implementation path.

**Acceptance and verification:** Focused CTest targets pass and source checks
show mapped QPs are used for queue lookup, submit, and quiet. A target CANN 9.1
build compiles and embeds the Kernel.

**Artifacts and interfaces:** Installed PrefetchWeight library and integration
package used by Task 3.

## Task 3: Run Ascend950 Correctness And Performance A/B

**Objective and role:** Decide from hardware evidence whether the mapping is a
real improvement.

**Background and prerequisites:** Depends on a passing Task 2 build. Reuse the
healthy HCCL AIV-only environment baseline and the existing EP8 benchmark case
on `141.61.49.195`.

**Modification scope:** Remote files under a new `/tmp` directory and retained
benchmark artifacts. Do not modify system CANN or driver installations.

**Constraints and non-goals:** Use eight physical ranks, BF16 exact slot
validation, NPU events, cross-rank maxima, and the same timing exclusions as
the baseline. Run a short diagnostic first; retain the production mapping only
if correctness passes and the large-transfer P50 improves materially.

**Acceptance and verification:** Run `3 x (5 + 20)` for the optimized TileXR
build. Compare against TileXR main and native MoonEP with P50, P99, effective
GB/s, per-repeat medians, and correctness status.

**Artifacts and interfaces:** Raw per-rank JSON, aggregate JSON, build and run
logs, and result paths for Task 4.

## Task 4: Record Results And Deliver The Patch

**Objective and role:** Make the optimization reproducible without the current
conversation and produce a patch based on latest main.

**Background and prerequisites:** Depends on Tasks 1-3 and their retained raw
artifacts.

**Modification scope:** A stable benchmark under `tools/moonep/`, focused
MoonEP documentation, this plan/spec, and the final Git diff.

**Constraints and non-goals:** Record ineffective experiments and failed test
launches that would otherwise be repeated. Keep temporary remote paths out of
runtime code.

**Acceptance and verification:** Documentation names commits, hardware,
dimensions, timing boundary, baseline, optimized results, validation limits,
and next-stage recommendations. Relevant tests pass from a clean build and
`git diff --check` is clean. Generate a patch against the fetched latest main.

**Artifacts and interfaces:** A scoped commit and `.patch` file; no push or PR
unless separately requested.
