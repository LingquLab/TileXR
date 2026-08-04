# A5 Direct SDMA Implementation Plan

Date: 2026-07-28
Status: Approved for execution
Design: `docs/specs/2026-07-28-a5-direct-sdma-design.md`
Updated: 2026-07-29 (Task 9 adds approved strided batch submission)

## Goal

Implement the approved Ascend950/A5 direct-SDMA backend inside TileXR while
preserving the existing A2/A3 PTO backend and public API. The production path
must use CANN's built-in system AICPU `ShmemSdmaStarsQuery`, map Host RTSQ
doorbells, publish a TileXR-owned 48-channel workspace, and let AIV submit the
single-copy two-SQE or strided-batch `N + 1` sequence directly.

The project-wide supported baseline becomes CANN 9.1.0 with driver
`25.1.rc1` or later. The validated hardware baseline is
`Ascend950PR_9589`, driver `25.1.rc1.b188`, device 5.

## Scope

- Add a fixed-width, versioned A5 workspace and SQE ABI shared by Host and AIV.
- Add a production A5 Host backend to `libtile-comm.so`, including dynamic
  built-in-op loading, 48 STARS streams, query compatibility, RTSQ mappings,
  workspace publication, and complete failure cleanup.
- Dispatch `SDMACopyNbi`, `SDMACopyStridedNbi`, and `SDMAWait` to A5 direct
  submission for dav-3510 kernels while keeping PTO single-copy behavior on
  A2/A3.
- Extend SDMA unit, integration, build, demo, dependency, and hardware checks.
- Change requirements and CI provisioning from driver `25.5.0+` to
  `25.1.rc1+`, including Huawei RC/build-suffix comparison.
- Replace the temporary preflight and probe with the production path, and
  retire obsolete A5 PoC documents.

## Non-Goals

- Do not build, install, embed, or load a TileXR custom AICPU kernel or OPP.
- Do not set or document `ASCEND_CUSTOM_OPP_PATH`.
- Do not include or link any source under `reference/`; it remains evidence
  only.
- Do not change `CommArgs`, public Host API signatures, the opt-in environment
  variable, workspace ABI, or event-zero semantics.
- Do not replace PTO SDMA on A2/A3, make SDMA mandatory for communicator init,
  add more than one outstanding event per channel, or claim Host/simulator
  checks prove the A5 data plane.
- Do not add the CANN `devlib` directory to runtime RPATH/RUNPATH.

## Authoritative References

- Approved behavior and acceptance: the design linked above.
- Existing Host ownership and fallback: `src/comm/sdma/tilexr_sdma_transport.*`
  and `src/comm/tilexr_comm.*`.
- Existing device API and PTO adapter: `src/include/tilexr_sdma.h` and
  `src/include/tilexr_sdma_compat.h`.
- Probe evidence to extract before deletion:
  `tests/sdma/a5_aicpu_probe/{main.asc,sdma_probe_kernel.asc,sdma_probe_types.h}`.
- Build/install wiring: `src/comm/CMakeLists.txt` and `tests/sdma/CMakeLists.txt`.
- Driver enforcement: `scripts/ci/provision/{common,cann,verify}.sh` and
  `tests/ci/test_control_contract.py`.
- Build and hardware expectations: `docs/BUILD_VERIFICATION.md` and
  `docs/SDMA_TRANSPORT.md`.

`reference/` may be consulted to verify layouts and runtime behavior, but it is
not an implementation dependency and must not appear in active target include
paths, sources, or link inputs.

## Dependency Order

Tasks 1 and 2 establish contracts needed by Tasks 3 and 4. Task 5 depends on
the production Host and device paths. Task 6 is independent of Tasks 2-5 at the
file level but must land before final validation. Task 7 follows functional
implementation and baseline migration. Task 8 is the original backend
acceptance gate. Task 9 depends on Tasks 1, 4, 5, and 8 and is the final batch
extension gate; PoC removal is not considered complete until its evidence is
covered there.

## Task 1: Define The A5 ABI And Test Seams

**Objective and role:** Define the only Host/AIV binary contract for the A5
backend and the pure validation/event helpers needed for deterministic unit
tests. This prevents the probe's diagnostic structs from becoming an implicit
production ABI.

**Background and prerequisites:** Follow the approved 48-channel,
64-byte-aligned workspace, two-SQE completion protocol, one-outstanding-event
rule, and event-zero contract. Reuse the validated 64-byte A5 SQE field layout
from the probe, with explicit static assertions for every relied-upon size and
offset.

**Modification scope:** Add an installed shared header such as
`src/include/tilexr_sdma_a5_types.h`; update `src/include/tilexr_sdma_types.h`,
`src/comm/CMakeLists.txt`, and focused SDMA metadata/header tests. Keep Host-only
runtime handles and ACL objects out of the installed ABI.

**Constraints and non-goals:** Use fixed-width integer fields, C++14-compatible
constructs, device-visible addresses, a nonzero magic/version, 48 channels, and
64-byte cache-line alignment. Do not expose a new public Host function or reuse
PTO workspace layout. Do not use implementation-defined event bitfields; use
mask/shift helpers with invalid/stale detection.

**Acceptance and verification:** Build Host and dav-3510 header compile tests;
unit-test structure sizes/offsets, queue wraparound, maximum length checks,
channel range, event encode/decode, generation wrap avoiding zero, malformed
events, and stale-generation rejection.

**Artifacts and interfaces:** A stable A5 workspace header, channel record,
completion payload/record, SQE definition, backend identifier, and pure helpers
consumed by the Host initializer, AIV adapter, and tests.

## Task 2: Add Dynamic CANN And Driver Runtime Adapters

**Objective and role:** Isolate all optional A5 runtime calls behind an owned,
injectable adapter so missing symbols or system OPP support become a clean SDMA
capability failure and Host logic can be tested without hardware.

**Background and prerequisites:** The built-in opapi lifecycle requires tensor
creation/destruction, `aclnnShmemSdmaStarsQueryGetWorkspaceSize`, and
`aclnnShmemSdmaStarsQuery`. Stream/SQ/CQ/logical-CQ/physical-die discovery,
HAL query, and `halResAddrMap`/`halResAddrUnmap` are also required. The current
core target already links ACL/runtime/real driver HAL; the built-in opapi entry
points must be resolved dynamically.

**Modification scope:** Add private files under `src/comm/sdma/` for the A5
loader/runtime function table and RAII resource wrappers; wire them into
`src/comm/CMakeLists.txt`; add fake-function-table unit tests under
`tests/sdma/unit/`.

**Constraints and non-goals:** Load only CANN/runtime shared libraries expected
from the active CANN environment, report the first missing boundary, and close
owned handles idempotently. Do not link a TileXR OPP, depend on probe binaries,
or fall back to a custom `.aicpu` kernel. Do not source symbols from
`reference/` or introduce `devlib` runtime search paths.

**Acceptance and verification:** Unit tests cover complete symbol resolution,
each required-symbol failure, idempotent close, and no-call behavior after a
failed load. `readelf -d/-l` and `ldd` later confirm no custom OPP dependency,
no CANN `devlib` RPATH/RUNPATH, and real-driver HAL resolution.

**Artifacts and interfaces:** A private A5 runtime operation table plus scoped
wrappers for contexts, streams, tensors, device allocations, op workspaces,
RTSQ mappings, and library handles. Task 3 consumes this adapter and tests can
replace every external operation with deterministic fakes.

## Task 3: Implement The 48-Channel A5 Host Backend

**Objective and role:** Create, validate, publish, and destroy the complete A5
SDMA resource set while containing all failures as best-effort capability
fallback.

**Background and prerequisites:** Use the Task 1 ABI and Task 2 runtime adapter.
Create 48 `ACL_STREAM_DEVICE_USE_ONLY` streams in the communicator's device and
context. Present all stream records to a built-in query in one isolated context
first. Accept a complete successful result, or accept the `25.1.rc1`
compatibility path only for exact sync status `507018`, zero register-base, and
otherwise complete validated data; then query each remaining channel in a
fresh isolated context.

**Modification scope:** Add private A5 backend/query implementation files under
`src/comm/sdma/`; refactor `src/comm/sdma/tilexr_sdma_transport.*` into a
runtime-SoC dispatcher; remove `PreflightAscend950SDMA`; update status values
only if diagnostics require a distinct A5 failure category. Preserve the
existing `TileXRComm::InitSDMA` publication flow.

**Constraints and non-goals:** Validate every returned status, ID, SQ base,
64-byte SQE size, depth, head/tail range, Host HAL tail/SQE-size cross-check,
and RTSQ mapping length before publication. Restore the communicator context
and run a lightweight allocation/memset/sync/copy health check after every
expected partial failure. Publish only after all 48 channels succeed. On any
failure, unmap/destroy/free only TileXR-owned resources in reverse order, leave
the workspace null and SDMA flag clear, return TileXR success to communicator
initialization, and support repeated `Shutdown()`.

**Acceptance and verification:** Fake-runtime tests cover full-query success,
expected partial batch plus 47 isolated queries, unexpected sync codes,
malformed/partial channel records, failed context restore or health check,
HAL mismatch, RTSQ failure, allocation/copy failure at every ownership stage,
reverse cleanup, reinitialization, and unchanged A2/A3 PTO selection. Hardware
validation later requires all 48 distinct streams/SQs/mappings alive together.

**Artifacts and interfaces:** An A5 backend object owned by
`TileXRSDMATransport::Impl`, a fully initialized device workspace address, and
structured first-failure diagnostics. The transport exposes only its existing
availability/workspace/status interface to `TileXRComm`.

## Task 4: Implement The AIV Direct Submission Backend

**Objective and role:** Make the existing device calls submit and wait for A5
SDMA directly on dav-3510 while leaving PTO calls unchanged on supported A2/A3
kernels.

**Background and prerequisites:** Use Task 1 layout and the approved two-SQE
protocol. Architecture selection follows the existing TileXR convention used
by UDMA: `__NPU_ARCH__ == 3510`, `CATLASS_ARCH == 3510`, plus a test-only force
macro if needed. The Host already selects and publishes the matching workspace.

**Modification scope:** Add an installed A5 device adapter such as
`src/include/tilexr_sdma_a5.h`; update `src/include/tilexr_sdma.h`, install
wiring, and dav-3510 compile tests. Keep PTO-specific code in
`tilexr_sdma_compat.h`.

**Constraints and non-goals:** Validate workspace magic/version/backend,
addresses, byte length, channel, depth, and queue capacity before writes.
Atomically claim the channel; a busy channel returns 0. Advance a nonzero
generation, update the completion payload, build data and 64-byte completion-copy SQEs with
type 11, `wrCqe=0`, credit 254, flush all touched cache lines, issue ordering,
and ring RTSQ only after both SQEs are ready. `SDMAWait(0)` succeeds; other
waits reject bad channel/generation/stale events, poll matching completion, and
release only their claimed channel. Never truncate a transfer length or ring a
doorbell after failed validation.

**Acceptance and verification:** Host-pure helper tests cover invalid/busy/stale
paths and wraparound; dav-3510 compilation covers actual AIV intrinsics and SQE
writes; source guards ensure A5 does not include PTO headers and A2/A3 still do.
Hardware demo later verifies completion generation and every destination byte.

**Artifacts and interfaces:** An A5 device adapter called by unchanged
`SDMACopyNbi`/`SDMAWait`, with unchanged event-zero semantics and explicit
compile-time routing between A5 and PTO implementations.

## Task 5: Promote The Existing Demo To Production Acceptance

**Objective and role:** Turn `tests/sdma/demo` into the single supported
data-plane validator for A2/A3 PTO and A5 direct SDMA, including boundary and
concurrency coverage.

**Background and prerequisites:** Depends on Tasks 1-4. Current A5 build-target
selection changes are retained, but A5 demo availability must not depend on PTO
headers or `libnnopbase.so`. The demo must consume only installed TileXR APIs
and headers.

**Modification scope:** Update `tests/sdma/CMakeLists.txt`, `tests/sdma/build.sh`,
`tests/sdma/run_tests.sh`, and `tests/sdma/demo/*`; add focused unit/source
tests for build selection and CLI validation. Integrate repeat/channel/block
options into the existing runner rather than creating a second probe target.

**Constraints and non-goals:** Preserve the Ascend910B default and explicit
Ascend950 selection. Reject unsupported SoCs, channel IDs, sizes, or zero
iterations. Use distinct channels per AIV block, bounded Host waits, full byte
comparison, and generation checks. A skipped device or unavailable SDMA is not
an A5 data-plane pass.

**Acceptance and verification:** On A5, run 64 B, 4 KiB, and 1 MiB on channel 0
and channel 47; run a multi-block copy with distinct channels concurrently;
and repeat init/use/shutdown while confirming communicator health. On A2/A3 or
Host-only environments, existing unit/build behavior remains valid and claims
remain scoped.

**Artifacts and interfaces:** One installed demo executable/kernel and runner
that prints SoC, driver, device, active backend, channel count, selected
channels, sizes, iterations, completion generations, and comparison results.

## Task 6: Migrate The Driver Baseline To 25.1.rc1

**Objective and role:** Make repository guidance and CI enforcement agree with
the validated supported baseline, including Huawei RC syntax.

**Background and prerequisites:** The minimum is `25.1.rc1`; build-suffixed
`25.1.rc1.b188` is supported. Comparison is semantic, not lexical: compare
major/minor first; for the same major/minor, RC numbers order before final
numeric patch releases; optional `.bN` is metadata and does not lower the base
version. A syntactically valid later major/minor release is supported even if
its final component is an RC.

**Modification scope:** Update `scripts/ci/provision/common.sh`, `cann.sh`, and
`verify.sh`; extend `tests/ci/test_control_contract.py`; update minimum-version
claims in `AGENTS.md`, `README.md`, `docs/BUILD_VERIFICATION.md`,
`docs/SDMA_TRANSPORT.md`, and both architecture diagram sources. Preserve any
`25.5.0` text that is explicitly labeled as a historical observed environment.

**Constraints and non-goals:** Accept case-insensitive `rc`, positive RC
numbers, optional numeric build suffixes, final numeric releases, and later
major/minor versions. Reject `25.1.rc0`, older releases, missing components,
extra arbitrary suffixes, and malformed strings. Do not use `sort -V` or make
driver validation exact-equality based.

**Acceptance and verification:** The comparator matrix accepts at least
`25.1.rc1`, `25.1.RC1`, `25.1.rc1.b188`, `25.1.rc2`, `25.1.0`, `25.2.rc1`, and
`26.0.0`; it rejects at least `25.1.rc0`, `25.0.99`, `24.99.99`, `25.1`,
`25.1.rc`, `25.1.rc1.bad`, and empty input. Provision and verify messages both
name `25.1.rc1`.

**Artifacts and interfaces:** A shared shell comparator with documented
ordering behavior, its Python contract tests, and consistent repository/diagram
minimum-version text.

## Task 7: Remove PoC Paths And Consolidate Documentation

**Objective and role:** Leave one production implementation and one accurate
set of operator/validation instructions after extracting all useful probe
evidence.

**Background and prerequisites:** Execute only after Tasks 3-6 have tests and
the production demo covers the PoC's successful data path. Preserve the
approved design and this durable implementation plan.

**Modification scope:** Delete `tests/sdma/a5_aicpu_probe/`,
`docs/plans/2026-07-27-a5-sdma-adaptation.md`, and
`docs/plans/2026-07-27-a5-aicpu-sdma-poc.md`; remove the temporary Host
preflight and its source-guard expectations; rewrite README/SDMA/build docs and
script catalog references around the production backend.

**Constraints and non-goals:** Do not delete historical validation facts that
remain accurate; relabel them as evidence and state exact hardware. Remove all
claims that A5 production support is pending, requires 25.5.0, or needs a
TileXR OPP/custom OPP path.

**Acceptance and verification:** Repository searches find no active reference
to `PreflightAscend950SDMA`, `a5_aicpu_probe`, custom TileXR AICPU artifacts,
`ASCEND_CUSTOM_OPP_PATH`, or a `25.5.0+` minimum. Installed manifests contain
only the production library and headers. Documentation commands match runnable
scripts.

**Artifacts and interfaces:** Updated user/developer documentation, retained
approved design and implementation plan, and no PoC build/install artifacts.

## Task 8: Execute Final Local And A5 Acceptance

**Objective and role:** Produce proportionate evidence that the implementation
is compatible, failure-safe, dependency-clean, and functional on the supported
A5 baseline.

**Background and prerequisites:** All earlier tasks complete. Synchronize a
clean workspace to the approved Ascend950 validation host, use physical device
5 and the validated CANN 9.1.0 installation, and confirm synchronization before
and after remote work.

**Modification scope:** No feature expansion. Test-driven fixes may touch only
the owning modules above; any architecture, API, or acceptance change returns
to design approval.

**Constraints and non-goals:** Do not reset unrelated devices, alter the CANN
installation, install an OPP, or treat a skipped/fallback run as a pass. Keep
remote temporary outputs outside tracked paths or remove them after evidence is
captured. Preserve unrelated local worktree changes.

**Acceptance and verification:** Run CI contract tests, SDMA unit/integration
tests, C++14 core build/install, and Ascend910B/dav-3510 header/kernel builds.
On device 5 require 48 simultaneously live validated channels; copies at 64 B,
4 KiB, and 1 MiB on channels 0 and 47; a distinct-channel concurrent
multi-block run; repeated init/use/shutdown; full byte and generation matches;
and a healthy communicator context after each loop. Run dependency checks with
`readelf` and `ldd`, confirming real driver HAL and no `devlib` RPATH/RUNPATH.
Record exact SoC, driver, CANN, device, channel count, sizes, concurrency, and
iterations in `docs/SDMA_TRANSPORT.md`.

**Artifacts and interfaces:** Test logs or concise recorded results, updated
validation documentation, and a final diff/status review demonstrating that
only scoped production, test, baseline, diagram, and cleanup files changed.

## Task 9: Promote Strided Batch Submission

**Objective and role:** Move the hardware-proven batch prototype into the
installed TileXR device API so multiple independent same-sized copies can share
one A5 doorbell and one completion event.

**Background and prerequisites:** Representative A5 measurements show
3.4-4.0x amortized improvement for 4-64 KiB, 2.2x at 1 MiB, and 1.39x at
4 MiB. The accepted design adds `SDMACopyStridedNbi` without changing Host or
workspace ABI. One batch uses `N` data SQEs plus one completion-copy SQE and
retains one outstanding event per channel.

**Modification scope:** Update `src/include/tilexr_sdma_a5_types.h` with pure
multi-entry queue helpers, implement the A5 strided batch in
`src/include/tilexr_sdma_a5.h`, expose the architecture-routed wrapper in
`src/include/tilexr_sdma.h`, and replace the private implementation in
`tests/sdma/demo/tilexr_sdma_demo_kernel.cpp`. Extend metadata, header/source,
benchmark, and transport documentation tests as appropriate.

**Constraints and non-goals:** Preserve C++14 and CANN 9.1, the A5 workspace
layout/version, Host APIs, event format, existing single-copy behavior, A2/A3
PTO behavior, and one outstanding event per channel. Validate all inputs before
claiming a channel. Require non-overlapping strides for multi-copy batches,
reject address arithmetic overflow and insufficient queue capacity, leave one
SQ entry unused, and ring exactly one doorbell. Do not add descriptor arrays,
begin/append/commit state, CQE polling, broadcast/overlap semantics, or a
multi-copy PTO implementation.

**Acceptance and verification:** Unit-test arbitrary-entry capacity, tail and
task-ID wrap, invalid count/stride/overflow inputs, and unchanged two-entry
single-copy helpers. Compile the installed header and benchmark kernel for
dav-3510. On A5 run counts 1, 2, 4, 8, 16, and 32 across queue/task-ID wraps,
verify every destination slice, repeat init/use/destroy, and rerun 4 KiB,
64 KiB, 1 MiB, and 4 MiB representative timings with equal 64 MiB backing and
a rotating working set. Run the eight focused SDMA test executables and inspect
the final dependency/RPATH state.

**Artifacts and interfaces:** Installed `SDMACopyStridedNbi`, reusable A5
multi-entry queue helpers, benchmark coverage using only the production API,
and recorded hardware semantics/performance evidence.

## Key Risks And Controls

- **Expected AICPU failure corrupts the active context:** every query runs in a
  disposable isolated context; restoration and a runtime health operation are
  mandatory before accepting partial data.
- **Partial workspace is accidentally trusted:** acceptance requires the exact
  status and every independent field/ID/HAL/mapping cross-check; all other
  partial states fail closed.
- **Queue overwrite or cross-block race:** reserve `N + 1` slots while leaving
  one unused, permit one outstanding event per channel, atomically claim it,
  and assign distinct channels to concurrent blocks.
- **Cache/order mismatch:** flush the payload and every written SQE; issue the
  validated barrier sequence before the RTSQ tail write.
- **ABI drift between Host and AIV:** one installed fixed-width header and
  static assertions are compiled in both modes.
- **Optional runtime becomes a hard dependency:** dynamically resolve the
  built-in opapi functions and keep all failures best-effort.
- **False hardware confidence:** distinguish compile/source/fallback evidence
  from the named A5 data-plane acceptance matrix.
- **Cleanup damages application state:** record ownership per resource, destroy
  in reverse order, restore the prior context/device, and never reset a device.

## Completion Gate

The change is complete only when all nine tasks pass, A2/A3 PTO selection is
preserved, A5 device 5 passes the full hardware matrix with 48 live channels,
driver `25.1.rc1.b188` is accepted by the shared comparator, dependency checks
are clean, and the repository contains neither the PoC nor a TileXR OPP path.
