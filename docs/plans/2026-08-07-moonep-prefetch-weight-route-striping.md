# MoonEP PrefetchWeight Route Striping Plan

## Goal And Boundaries

Implement the approved per-row weighted route striping design in the PR #93
worktree. Do not change the public MoonEP ABI, Planner behavior, or stub stages.

## Tasks

1. Add Host-owned route parsing and 64-byte weighted slice construction in
   `src/moonep/host/prefetch_weight_layout.*`. Extend focused Host tests with
   valid, invalid, tail, empty-slice, and one-QP cases.
2. Extend the private embedded-kernel launch ABI in
   `src/moonep/host/prefetch_weight_launch.cpp` and its source guards so the
   packed route weights reach the kernel without public API changes.
3. Update `src/moonep/kernels/tilexr_moonep_prefetch_weight_kernel.cpp`: each
   AIV owns its QP, traverses every slot, submits only its weighted row slice,
   and quiets only peers with submitted work. Preserve status propagation and
   registered-region bounds.
4. Change the two-worker launcher route to `topology,port_count:6`, update Python
   tests and MoonEP documentation, and retain existing 1/4/8 configurations.
5. Run focused Host/Python tests, target CANN compilation, artifact checks,
   two-card correctness with one remote slot, route evidence, and an identical
   Clos-only versus striped performance comparison.

## Risks And Acceptance

- No two AIVs may post to the same QP.
- Slice arithmetic must not overflow or leave gaps/overlaps.
- A worker with an empty slice must not quiet an unused QP.
- Runtime and Host must agree on QP count and packed weights.
- Correctness plus QP count alone is insufficient route proof; both physical
  route contributions need observable evidence.
