# MoonEP Combine V2 Fullmesh Implementation Plan

## Goal

Implement the approved Combine V2 Fullmesh data path in
`docs/specs/2026-08-16-moonep-combine-v2-fullmesh-data-path-design.md` while
preserving the existing 32 shared CLOS QPs and public query semantics.

Same-server remote payload and done traffic must use one of eight logical
Fullmesh slots. A rank creates only the seven peer slots on a standard eight-card
server; its Self slot remains invalid. Fullmesh failure is fatal for Combine V2
and never falls back to the CLOS data path.

## Scope

- Extend the UDMA runtime with a second, versioned Fullmesh device view and host
  capability query.
- Reuse the current HCCP lifecycle to create one direct QP per same-server peer,
  and import the remote peer's slot for this rank.
- Publish CLOS and Fullmesh registered-memory views under one generation.
- Add Host fail-closed validation before magic allocation and Kernel launch.
- Route same-server payload and one done token through Fullmesh, then publish two
  CLOS grants only after the Fullmesh CQ succeeds.
- Make full-sync submissions request and consume their own CLOS CQ.
- Extend logic, Host, runtime, source-guard, and build tests.

## Non-Goals

- Do not change `TileXRUDMAGetQpCount()` or the 32 shared-QP layout.
- Do not expose Fullmesh QPs to PrefetchWeight, ReduceGrad, or profile bindings.
- Do not change MoonEP public Buffer/plan/Python contracts or the Combine V2
  output/workspace layout.
- Do not add a fallback from Fullmesh to CLOS for same-server data.
- Do not perform device, SSH, multi-host, correctness, or performance testing in
  this task. Hardware validation remains explicitly unproven.

## Contract

### Runtime and Host

- The primary `CommArgs::udmaInfoPtr` remains a 32-QP CLOS `UDMAInfo`.
- A tail field in `CommArgs` points to a versioned Fullmesh view containing an
  eight-slot `UDMAInfo`, local rank, valid peer mask, connected count, and
  registration generation.
- For local ranks `a` and `b`, local slot `b` imports remote slot `a`; Self and
  cross-server entries are invalid.
- Fullmesh capability is published only when all required same-server direct QPs
  and the current memory registration are complete on every rank.
- The CLOS registry generation and Fullmesh view generation are equal and become
  visible in one CommArgs publication.
- Combine V2 returns `TILEXR_MOONEP_ERROR_NOT_SUPPORTED` for a missing or malformed
  Fullmesh view, incomplete mask, wrong generation, or absent registration.

### Kernel

- Self: local MTE copy, no done; grant remains two logical lanes.
- Same server: all payload plus lane-0 done on Fullmesh, wait and validate its CQ,
  then publish two CLOS grants and consume grant-only CQs before advancing.
- Cross server: existing two-lane CLOS payload, grant, done, and CQ behavior.
- Receive-side done waits are locality-aware: Self 0, same-server remote 1,
  cross-server remote 2.
- Every Fullmesh CQ validates owner, status, substatus, entry index, and queue
  accounting independent of optional DFX checks.
- Every full-sync batch requests one CLOS completion and retires it before waiting
  for remote sync signals.

## Ordered Work

### 1. Runtime Fullmesh Domain

Objective: add the independent Fullmesh capability without changing the CLOS
resource or public 32-QP contract.

Modification scope: `src/include/comm_args.h`, UDMA public/internal types,
`src/comm/udma/`, `src/comm/tilexr_comm.*`, `src/comm/comm_wrap.cpp`, and focused
UDMA unit tests.

Work:

1. Define the versioned host/device Fullmesh view and query API.
2. Resolve only same-server topology EIDs and build seven direct peer QPs.
3. Exchange slot-indexed QP descriptors and import remote slot `localRank` into
   local slot `peerLocalRank`.
4. Build an eight-slot UDMA image with zero/invalid Self and cross-server entries.
5. Extend ordinary memory prepare/commit/abort/cleanup so CLOS and Fullmesh views
   are handled transactionally and share a monotonically increasing generation.
6. Publish and clear the Fullmesh pointer/capability together with CommArgs.

Acceptance: public QP count stays 32; masks contain exactly the active local peers;
all failure paths clear Fullmesh capability and preserve cleanup ownership.

### 2. Schedule and Host Gate

Objective: make locality and fail-closed behavior explicit before launch.

Modification scope: `src/moonep/common/moonep_combine_schedule.h`, Combine V2 Host,
and schedule/Host unit tests.

Work:

1. Add same-server, local-slot, logical-QP, and expected-done helpers.
2. Query the Fullmesh host view and validate version, slot count, local rank,
   connected count, exact peer mask, registration readiness, and generation.
3. Validate every registered workspace range before allocating magic.

Acceptance: missing/incomplete Fullmesh always returns NOT_SUPPORTED; invalid user
workspace remains INVALID_ARGUMENT; magic is unchanged on rejected launches.

### 3. Kernel State Machine

Objective: enforce Fullmesh CQ success before grant publication.

Modification scope: Combine V2 Kernel header/entry and profiling/failure helpers.

Work:

1. Add a one-lane Fullmesh queue state selected by `peer % localRankSize`.
2. Reuse the existing UB WQE staging capacity for up to 128 payload WQEs plus one
   done WQE, and publish through MTE3 followed by `st_dev`.
3. Split same-server steps into Fullmesh payload/done completion and deferred
   two-lane CLOS grant completion phases.
4. Make receive-side done polling locality-aware.
5. Record the actual logical Fullmesh QP (`32 + slot`) on failures.
6. Mark the final full-sync WQE completion-producing and consume its CQ locally.

Acceptance: control flow contains no path from a failed/unconsumed Fullmesh CQ to
grant publication; all touched SQ/CQ ledgers retire to their submitted targets.

### 4. Verification and Review

Objective: establish source, Host, logic, and compile correctness without making
hardware claims.

Modification scope: focused tests and maintained architecture/validation docs.

Checks:

1. Schedule tests for 2/8/16/32/64/128 ranks, slot masks, peer routes, successors,
   and 0/1/2 done expectations.
2. Host tests for every Fullmesh capability and generation failure.
3. Runtime tests for seven created peers, Self omission, asymmetric import indices,
   invalid cross-server slots, transactional publication, and unchanged QP count.
4. Source guards for UB construction, MTE3-before-doorbell, CQ-before-grant, and
   self-contained full-sync CQ consumption.
5. Build the core runtime, Combine V2 Host, and target CANN 9.1 Kernel target.
6. Review the final diff for C++14 compatibility, ABI impact, arithmetic/bounds,
   resource cleanup, queue ownership, and failure convergence.

Hardware behavior, ordering, correctness, and performance remain a residual risk
until the separate hardware gate in the approved design is run later.

## Risks

- Reusing one HCCP runtime for both domains is required to avoid conflicting RA
  ownership; cleanup ordering must not release shared contexts prematurely.
- The existing UDMA image is rank-by-QP. Fullmesh validity must be explicit so a
  zero-filled Self or cross-server entry is never treated as usable.
- CLOS and Fullmesh registration cannot be committed independently; partial
  prepare, publication, commit, or cleanup must restore or hide both views.
- Combine V2 is a direct registered-binary launch. Host arguments, embedded Kernel
  signature, and CANN 9.1 compilation must remain exactly aligned.
