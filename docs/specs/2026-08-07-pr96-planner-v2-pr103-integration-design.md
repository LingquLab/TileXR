# PR96 Planner V2 Integration on Current MoonEP

## Status

Approved direction, revised after the PR96-priority conflict audit and the
mainline PR106/PR107 integration. This document records the implementation
boundary for integrating PR96 Planner V2 on the current MoonEP runtime.

When PR96 and current main own the same Planner V2 responsibility, PR96 wins.
Planner V2 creates the installed planner target and owns the default Planner
tests. Current main remains authoritative for the public MoonEP ABI, PR107
Dispatch and Planner V3 semantics, and PR106 ReduceGrad V1/V2 semantics.

The user subsequently authorized fresh target-hardware validation after the
PR96-priority re-adaptation. Validation remains staged and claims are limited to
the hardware and rank configurations actually exercised.

## Goal

Make the optimized PR96 Planner V2 implementation the authoritative Planner V2
target and test contract in the current runtime. Preserve the current-main
Planner V3 backend for the native MoonEP Planning path without overriding the
mainline Dispatch, PrefetchWeight, Combine, or ReduceGrad contracts.

## Scope

- Add the PR96 `tilexr_ep_plan.h` public contract and `planner_v2` algorithm,
  layout, Host, reference, and kernel implementation.
- Export `TileXRMoeEpPlanV2GetWorkspaceSize`, `TileXRMoeEpPlanV2`, and
  `TileXRMoeEpPlanV2WithMetadata` from `libtilexr-moonep-planner.so`.
- Route the historical `TileXRMoonEpPlannerGetWorkspaceSizeV2` and
  `TileXRMoonEpPlannerV2` compatibility symbols to the PR96 optimized backend.
- Make `src/moonep/planner_v2` create `tilexr-moonep-planner` and own its
  install, SONAME, public include surface, and default Planner tests.
- Restore the PR96 legacy V2 layout/Host test sources. They remain part of the
  V2 contract even though optimized production entry points use the newer V2
  Host path.
- Keep `TileXRMoonEpPlannerGetWorkspaceSizeV3` and `TileXRMoonEpPlannerV3` on
  a namespaced current-main compatibility backend appended to the V2-owned shared
  library. `TileXRMoonEpPlanningV1` continues to call V3.
- Port PR96 Host, source, ABI, algorithm, reference, downstream-contract, and
  multi-rank tests to the current-main baseline.

## Architecture

`libtilexr-moonep-planner.so` is created and installed by the PR96 V2 build and
contains two Planner backends:

1. Planner V2 is the optimized deterministic placement planner from PR96. It
   publishes metadata through IPC `peerMems[]`, uses magic-tagged bounded
   barriers, and exposes the optimized and historical V2 APIs.
2. Planner V3 is the current-main compatibility planner. Its internal
   Host/Kernel namespace is distinct from the V2 namespace. It continues to produce
   the current Plan layout and metadata consumed by native MoonEP stages, while
   retaining PR107's cross-node and route semantics.

Both backends are built into the same shared library so public ABI ownership
and installation remain stable. V2 owns target creation; V3 contributes a PIC
object backend. Each backend has a distinct AICore kernel name, embedded binary,
registration state, and fixed runtime signature.

## Kernel Build and Launch

PR96's linked kernel shared-object and compiler-generated launch wrapper are
not retained. The Planner V2 kernel is migrated to the established MoonEP
kernel path:

- compile a pure AICore relocatable binary with CANN 9.1 Bisheng;
- link and embed that binary into the Host library;
- register it once through `EnsureMoonEpKernelRegistered`;
- launch it through `rtKernelLaunchWithFlagV2` using one packed Host argument
  structure and a Planner-V2-specific signature;
- keep Host compilation at C++14 and use GNU++17 only for AscendC compilation;
- keep runtime RPATH at `$ORIGIN` and never add CANN `devlib` to RPATH/RUNPATH.

The migration preserves PR96's IPC-only metadata protocol. It does not add
UDMA registration, UDMA data movement, scalar GM flag writes, or flag-memory
resets.

This is the only source-precedence exception: repository policy explicitly
forbids the PR96 Host `kernel<<<...>>>` wrapper and CANN `devlib` runtime paths.
The pure-AICore embedded Runtime V2 path preserves PR96 behavior while meeting
the active CANN 9.1 build and runtime contract.

## Compatibility Boundaries

- Current main is authoritative for the MoonEP ABI, including PR106's
  `TileXRMoonEpReduceGradArgsV1` contract and ReduceGrad V2 additions.
- The 120-byte `TileXRMoonEpPlanV1` layout remains unchanged.
- Planner V3 retains the output and cross-node semantics from PR107/current main.
- Planner V2 remains A5/Ascend950-only and rejects incompatible topology.
- The new `tilexr_ep_plan.h` API remains C++14 Host compatible.
- Active targets do not include or link implementation code from `reference/`.
- PR96 documentation that claims broader historical hardware evidence is not
  treated as evidence for this integrated revision.

## Error Handling

Synchronous validation and enqueue failures return through the existing TileXR
status contract. Device algorithm failures, mismatch, partial plans, and bounded
timeouts remain observable in Planner V2 status metadata after caller-managed
stream synchronization. No implicit runtime initialization, synchronization,
retry, or transport fallback is introduced.

## Verification

Host and build verification:

- build and install the current-main configuration with MoonEP and tests enabled;
- run the migrated PR96 Planner V2 Host/source/reference/ABI tests;
- run current-main MoonEP, Dispatch, ReduceGrad, and Planner tests;
- inspect exported symbols and runtime RPATH/RUNPATH;
- run `git diff --check` and review the final diff.

Current mock end-to-end verification:

- capture Planner V2 registration, fixed signature, packed launch arguments,
  magic handling, Runtime V2 enqueue errors, and metadata entry points with
  Host runtime stubs;
- compare Planner V2 algorithm and layout results with the independent CPU
  reference across the focused case matrix;
- verify downstream route helpers, metadata V2, and historical V2 compatibility;
- run the current-main Planner and MoonEP Host/source/ABI suites;
- run the current-main Python fake-runtime correctness and reference suites through
  Planning, Dispatch, PrefetchWeight, ExpertForward, Combine, and ReduceGrad.

The earlier successful CANN build/install establishes compile and link
compatibility only. The interrupted one-rank run and historical PR96 logs are
not accepted as current hardware correctness evidence.

## Non-Goals

- Removing the current-main Planner V3 compatibility ABI or changing the
  120-byte Plan V1 layout.
- Reverting or overriding PR106 MoonEP ABI/ReduceGrad or PR107 Dispatch/Planner
  behavior.
- Adding 910B support to Planner V2.
- Adding UDMA to Planner V2 or validating UDMA data-plane transfer.
- Reproducing PR96's historical cross-node 32-rank or 128-rank evidence.
- Pushing or modifying PR96, PR103, PR106, or PR107 remotely.

## Acceptance Criteria

1. PR96 Planner V2 owns the integrated target and default Planner tests; the
   library exports optimized V2, historical V2, and current-main V3 APIs
   without duplicate symbols.
2. PR96 Planner V2 outputs match its independent CPU reference in focused mock
   validation.
3. The current-main MoonEP workflow passes through its Host and Python
   fake-runtime correctness/reference layers.
4. The final installed artifacts preserve C++14 Host compatibility, CANN 9.1,
   A5 targeting, `$ORIGIN` runtime lookup, and IPC-only Planner metadata.
5. The original workspace and unrelated worktrees remain unchanged.
