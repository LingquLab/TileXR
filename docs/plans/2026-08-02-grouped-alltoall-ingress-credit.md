# Grouped AllToAll Ingress Credit Implementation Plan

## Goal And Scope

Implement the default-disabled `window=1` ingress-credit protocol approved in
`docs/specs/2026-08-02-grouped-alltoall-ingress-credit-design.md`. The change
must bound payload admission to one source per destination lane while
preserving the existing path exactly when credits are disabled.

This plan does not change route weights, cabinet topology interpretation,
quiet batching, copyout assignment, or multi-pass credit semantics. Initial
hardware acceptance is single-pass only.

## Task 1: Pure Layout And Schedule Contracts

**Objective and role:** Define host-testable helpers for credit enablement,
token construction, receive ownership, and next-group peer mapping.

**Background and prerequisites:** The approved spec is authoritative. Existing
peer mapping and plan helpers live in
`tests/udma/demo/tilexr_udma_alltoall_group_layout.h`.

**Modification scope:** The grouped layout header and
`tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`.

**Constraints and non-goals:** Preserve existing peer mapping and default
layout behavior. Do not add device-only dependencies to the host-testable
header.

**Acceptance and verification:** Unit tests cover 256 ranks, group zero,
inactive final lanes, owner selection, shared-route credit identity, invocation
separation, and default-disabled behavior. Run the grouped layout unit binary.

**Artifacts and interfaces:** Stable helper contracts consumed by Host and
Kernel work.

## Task 2: Registered Credit Plane And Host Interface

**Objective and role:** Reserve ping-pong credit storage and pass a validated
ingress-window value and credit offsets into kernel launch.

**Background and prerequisites:** Task 1 token and layout contracts are fixed.
The grouped plan already owns payload, signal, and control regions.

**Modification scope:** Grouped layout plan, demo environment parsing and
allocation/launch wiring, grouped kernel declaration and launch wrapper, plus
focused source-contract tests.

**Constraints and non-goals:** `INGRESS_WINDOW=0` remains the default. Only 0
and 1 are accepted. Existing payload and signal offsets remain stable where
practical; registered-size overflow checks remain mandatory.

**Acceptance and verification:** Unit tests prove valid/invalid environment
and launch wiring contracts, credit planes do not overlap, and max registered
size accounts for both planes.

**Artifacts and interfaces:** Kernel receives two credit offsets and the
ingress-window value.

## Task 3: Kernel Credit Wait And Publication

**Objective and role:** Gate each destination send once per group, publish a
local request after both route-ready tokens arrive, and let the primary send
core issue the next lane credit.

**Background and prerequisites:** Tasks 1 and 2 provide mapping, storage, and
arguments. Existing MTE token wait and UDMA NBI primitives are reused.

**Modification scope:**
`tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp`, existing UDMA helper
interfaces only if required, and grouped source-contract tests.

**Constraints and non-goals:** Do not add per-credit quiet. Only cores 32..47
publish local requests; only primary send cores publish UDMA credits, preserving
the shared-SQ single-producer rule. Local request slots are 64-byte isolated so
independent receive-core cache maintenance cannot lose another lane's request.
Primary and secondary workers wait on the same destination credit. The
default-disabled branch must not execute a credit wait or write. If source
lifetime cannot be made safe without per-credit quiet, stop and revise the
design instead of weakening this constraint.

**Acceptance and verification:** Static/source tests prove placement,
single-producer ownership, request isolation, and unique shared-SQ completion
reclamation. Compile checks validate Ascend C overloads and launch signatures.
The exact final source passes a clean b131 Bisheng build and 4x8 hardware
validation; 256P admission-bound validation remains pending.

**Artifacts and interfaces:** Trace-visible or debug-visible credit wait and
publish phases where trace capacity permits.

## Task 4: Verification And Delivery Readiness

**Objective and role:** Establish local correctness and document remaining
hardware evidence.

**Background and prerequisites:** Tasks 1 through 3 are complete.

**Modification scope:** Tests and documentation required to keep validation
commands and limitations accurate.

**Constraints and non-goals:** Run only the explicitly authorized NPU matrix.
Do not modify or stage unrelated existing worktree files.

**Acceptance and verification:** Run focused unit tests, relevant Python
parser/source tests, formatting/build checks available on Windows, inspect the
final diff, and record the exact 2x8 then 4x8 then 256P hardware matrix. The
implemented path has completed 4x8 functional and warmup-5/repeat-50
performance validation; 256P admission-bound validation remains pending.

**Artifacts and interfaces:** A reviewable implementation, verification
summary, and scoped Git commit after the final checks pass.

## Key Risks

- Credit NBI source storage may be reused before hardware consumes it.
- Two receive copy slices may publish duplicate credits.
- Primary and secondary send workers may wait on different tokens by mistake.
- A final inactive lane may wait for or publish a nonexistent peer.
- Registered-memory growth may cross a region boundary or maximum-size check.
- Trace storage may need expansion if new phases are recorded.
