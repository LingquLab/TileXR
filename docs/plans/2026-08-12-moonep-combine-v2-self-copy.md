# MoonEP Combine V2 Self-Copy Implementation Plan

Status: Implemented. B150 compile, six focused Host tests, and bounded 8P
correctness passed using `/home/pkg/910_B150/cann-9.1.0` on 9号柜 CPU1 for
BS=128 and BS=8192.

## Goal

Implement the approved Self-copy pipeline in
`docs/specs/2026-08-12-moonep-combine-v2-self-copy-design.md` without changing
the Planner route ABI, Remote transport protocol, or public launch interface.

## Scope

The implementation touches the Combine V2 kernel, its focused source guards and
Host reference tests, and the approved design/plan artifacts. It must preserve
C++14 and target CANN compatibility.

The experimental untracked `tilexr_moonep_combine_v2_kernel_back.h` is
read-only reference material and is not part of the build or implementation.

## Work

1. Replace the separate oversized issue buffers with one continuous 260-entry
   UB allocation. Expose logical six-port and two-port views at entries 0 and
   194, prefill only 192/64 payload templates, append up to two controls per
   lane, and retain the existing per-SQ MTE3 publication and ring-wrap split.
2. Delete obsolete compare/gather/descriptor declarations, allocations, and
   inactive implementation from the active kernel.
3. Allocate two dedicated 64 KiB Self relay buffers with `TQue`, while retaining
   a compile-time proof that all send buffers fit in 216 KiB UB.
4. Reuse `LoadSelectionChunk()` and `SelectPeerRoutes()` in `SendSelfStep()`.
   Consume each selected `RouteEntry` batch before invoking the selector again.
5. For rows at most 64 KiB, derive rows per group from aligned `rowBytes_`, cap
   the group at eight rows, and overlap the next MTE2 group with the pending
   MTE3 group. For larger rows, flatten routes into at-most-64-KiB tiles and use
   the same two-buffer pipeline.
6. Update source guards and Host reference tests for continuous WQE offsets,
   lane capacities, dynamic group sizing, legacy-code removal, RouteEntry-based
   Self addressing, and synchronization ordering.
7. Run focused local Host checks. Synchronize through Mutagen and compile with
   `/home/pkg/b150` on an available target server. Before any hardware run,
   apply the task's process-occupancy rules.

## Constraints

- Every WQE remains fully constructed in UB and is copied to SQ with MTE3.
- Doorbells remain `st_dev` operations after SQ MTE3 completion.
- Self does not modify SQ/CQ/control state.
- `sourceSlotIndex` is absolute and must not receive `chunkStart` twice.
- Copy exactly `rowBytes_`; aligned UB padding is never written to GM.
- Do not reuse WQE, route, cursor, or old selection storage as Self relay.
- Do not commit, push, or remove unrelated untracked files as part of this plan.

## Verification

- Focused Combine V2 schedule and source-guard tests pass.
- `git diff --check` passes.
- Target kernel compiles against `/home/pkg/b150`.
- Bounded hardware correctness covers Self routes and representative dynamic
  relay group sizes when a suitable target is available.
- Profiling distinguishes selector time from the complete Self copy pipeline.
