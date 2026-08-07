# UDMA Registered-Memory And Generic Multi-QP Implementation Plan

## Goal And Authority

Implement the approved design in
`docs/specs/2026-08-06-udma-registered-memory-multi-qp-design.md` from the clean
`origin/main` baseline. The result is a standalone UDMA pull request that fixes
Ascend950 registered-memory reliability and provides an operator-neutral,
configurable one-through-eight QP transport.

Repository constraints in `AGENTS.md` are mandatory: preserve C++14 and CANN
9.1 compatibility, keep UDMA best effort, do not register `peerMems[]`, do not
enable registration in `InitThread`, and do not add a CANN `devlib` runtime
path.

## Scope Boundaries

In scope are the UDMA Host transport and lifecycle, deterministic RootInfo and
route-spec parsing, generic dynamic QP resources, public QP-count discovery,
device per-QP operations, focused tests, and the cross-host UDMA demo/runner.

Out of scope are MoonEP, EP algorithm policy, grouped AllToAll, SDMA behavior,
multi-region registration, device launch syntax, and performance-kernel changes.
Production identifiers and environment variables must remain operator-neutral.

## Authoritative References

- `AGENTS.md`
- `docs/specs/2026-08-06-udma-registered-memory-multi-qp-design.md`
- `docs/BUILD_VERIFICATION.md`
- `src/comm/tilexr_comm.{h,cpp}` and `src/comm/comm_wrap.cpp`
- `src/comm/udma/tilexr_udma_{context,transport,layout}.{h,cpp}`
- `src/include/tilexr_{api,udma,udma_types,udma_reg}.h`
- `tests/udma/`

The two received MoonEP patches and PR #88 are comparison inputs only. Do not
copy their MoonEP names, capability bits, route policy, grouped collective,
SDMA, or performance changes.

## Task 1: Isolate The Branch And Establish Testable Internal Contracts

**Objective and role:** Work only in `codex/udma-multi-qp`, based on
`origin/main@f951e8c`, and introduce the internal configuration/parser
boundaries needed by the later transport refactor.

**Background and prerequisites:** The approved spec fixes the public behavior.
The current transport keeps RootInfo parsing and route selection as anonymous
helpers in one large source file, which prevents meaningful behavioral unit
tests.

**Modification scope:**

- Add internal UDMA configuration and RootInfo parser headers/sources under
  `src/comm/udma/` when needed for direct unit testing.
- Update `src/comm/CMakeLists.txt` and `tests/udma/CMakeLists.txt` only for the
  new production sources and focused test targets.
- Add parser/config tests under `tests/udma/unit/`.
- Retain the approved design and this plan under `docs/`.

**Constraints and non-goals:** Keep types internal to `src/comm/udma`; do not
change the public ABI or transport behavior in this task. The parser must not
depend on C++17 or a third-party JSON library.

**Acceptance and verification:**

- Parse unset/empty, one, two, three, and eight route rules.
- Reject empty rules, nine rules, unknown selectors, zero and malformed values.
- Parse RootInfo whitespace, escaped strings, quoted/plain integers and arrays;
  reject malformed or truncated input.
- No `std::regex` remains in UDMA production sources.
- Parser/config unit targets compile and pass on a Host build.

**Artifacts and interfaces:** Normalized `UDMAQpConfig`, fixed-size wire
descriptor conversion/comparison helpers, parsed RootInfo/EID port-count data,
and deterministic route-selection inputs for Task 4.

## Task 2: Correct Ascend950 Registration And Make MR Cleanup Retryable

**Objective and role:** Remove the `528101` registration failure caused by an
incorrect HCCP registration contract and prevent failed rollback/unregister
from losing live HCCP handles.

**Background and prerequisites:** Chip identification is already available in
`TileXRComm`; one production `RaCtxLmemRegister` call exists. The exact byte
count must pass through unchanged, including 2,097,152 bytes.

**Modification scope:**

- Thread `localRankSize` and a chip-derived `nonPinRegistration` option through
  `src/comm/tilexr_comm.cpp`, `tilexr_udma_context.*`, and
  `tilexr_udma_transport.*`.
- Recognize both Ascend950PR and Ascend950DT in the existing chip detection
  layer without coupling the flag to QP count.
- Initialize all required `MrRegInfoT` flags before the sole
  `RaCtxLmemRegister` call and add complete failure diagnostics.
- Convert local/remote MR cleanup to first-error-returning, attempt-all helpers
  that erase only resources successfully released.
- Build every candidate MR/TPN device image in a separate allocation. Publish
  that image and the candidate registry together, then swap Host ownership;
  never overwrite the currently published `udmaInfoDev_` during prepare.
- Add a context lifecycle state that hides the device/Host registry during
  cleanup-pending and allows unregister to retry residual cleanup.
- Add HCCP-loader injection or an equivalent internal seam sufficient for
  behavioral fault-injection tests.

**Constraints and non-goals:** One active region and handle `0` remain the
contract. Do not impose size/alignment checks beyond non-null and positive
bytes. Registration remains unsupported in `InitThread`.

**Acceptance and verification:**

- Unit evidence proves all flags are set before the unique registration call.
- Partial local register/import and first unregister failures preserve handles
  for retry and do not publish a registry.
- A failed replacement preserves the previous published registration when
  restoration succeeds; otherwise it enters cleanup-pending.
- A successful retry returns the context to transport-ready.
- Unregister and cleanup retry do not enter a new socket collective after the
  registry has been hidden.

**Artifacts and interfaces:** Transactional Host/context memory state and
retryable transport-owned registration/import maps consumed by Task 4.

## Task 3: Generalize The Device Image Layout

**Objective and role:** Make `UDMAInfo::qpNum` authoritative while preserving
the binary layout and legacy QP0 indexing.

**Background and prerequisites:** `BuildUDMAInfoImage` currently hard-codes one
QP and assumes one vector entry per rank.

**Modification scope:**

- Change `BuildUDMAInfoImage` to accept `qpNum` explicitly.
- Validate non-zero QP count, equal vector lengths, divisibility by `qpNum`,
  and checked total-size/offset arithmetic.
- Keep rank-major/QP-minor ordering: `peer * qpNum + qpIdx`.
- Extend `tests/udma/unit/test_tilexr_udma_transport_layout.cpp` for QP counts
  1, 2, and 3 plus invalid and overflow cases.

**Constraints and non-goals:** Do not reorder or resize `UDMAInfo`,
`UDMAWQCtx`, `UDMACQCtx`, or `UDMAMemInfo` fields.

**Acceptance and verification:** The focused layout test proves pointer
offsets, copied metadata, QP count, repeated-EID entries, and rejection paths.

**Artifacts and interfaces:** A checked image builder used by the legacy and
configured paths in Task 4.

## Task 4: Implement Strict Generic N-QP Host Transport

**Objective and role:** Create independent `[peer][qp]` queues and route them
using `TILEXR_UDMA_QP_ROUTE_SPEC`, with distributed agreement and full rollback.

**Background and prerequisites:** Tasks 1-3 provide normalized route data,
RootInfo metadata, retryable MR ownership, and a dynamic image builder.

**Modification scope:**

- Read the route spec once per context and exchange a fixed-size normalized
  descriptor across all ranks before route construction.
- Pass `localRankSize` into transport options and distinguish same-node and
  cross-node peers.
- Store local and remote routes as checked rank-major/QP-minor matrices.
- Keep the unset/empty legacy single-QP path; use topology for same-node peers,
  prefer a deterministic aggregate EID across nodes, and retain first-EID fallback.
- In explicit mode, resolve `topology` or exact `port_count:N` selectors
  strictly and deterministically; allow duplicate selectors.
- Add `PerPeerQpState` ownership for channel, CQ, local/imported QP, scalars,
  doorbell/atomic addresses, TPN, and device queue images.
- Stage rank-wide agreement around local creation and remote import so one rank
  cannot enter a collective after another rank has abandoned the stage.
- Remove or replace the process-global sticky UDMA failure shortcut so one rank
  cannot skip initialization while peers enter a transport collective.
- Deduplicate MR imports by `(peer, localEid, remoteEid)` while copying the
  QP-specific TPN into every memory image entry.
- Publish availability and `CommArgs` only after the entire configured
  transport succeeds; attempt all cleanup and retain unreleased handles.

**Constraints and non-goals:** Support one through eight QPs. Explicit mode
must not reduce QP count or substitute EIDs. UDMA failure must leave IPC/SDMA
and the communicator usable. Do not add `ExtraFlag` bits.

**Acceptance and verification:**

- Behavioral tests cover descriptor match/mismatch and local parse failure.
- Route tests cover same-node topology, cross-node exact port count, missing
  routes, and two QPs sharing one EID.
- Fault tests cover partial CQ/QP creation and partial QP import cleanup.
- Layout tests show independent state and QP-specific TPN for every entry.
- Existing legacy single-QP Host tests remain green with the variable unset.

**Artifacts and interfaces:** A transport reporting dynamic `GetQpCount()` and
publishing a complete dynamic device image.

## Task 5: Add Generic Host And Device QP APIs

**Objective and role:** Expose multi-QP discovery and reusable data-plane
operations without changing existing callers.

**Background and prerequisites:** Task 4 provides the initialized QP count and
independent queue metadata.

**Modification scope:**

- Add `TileXRUDMAGetQpCount` to `src/include/tilexr_api.h` and implement it
  through `comm_wrap.cpp`, `TileXRComm`, context, and transport.
- Add `UDMAQpCount`, `UDMAQpValid`, and QP-aware remote-memory lookup to
  `src/include/tilexr_udma.h`.
- Add generic per-QP PUT/GET, flagged PUT, bounded deferred PUT, doorbell flush,
  and quiet-status helpers.
- Replace SQE flag literals with named completion and strong-order bits.
- Check rank/QP, registered range, and SQ capacity before mutating deferred
  queue state. Return a status for new checked operations.
- Track submitted completion count separately from the absolute SQ basic-block
  producer/reclaim positions. Set `sqeBbIdx`, reclaim using CQE `entryIdx` and
  the completed WQE's 1-BB/2-BB size, and reject invalid outstanding state.
- Require completion semantics for operations counted by quiet, so a caller
  cannot request a non-completing SQE and then wait forever for its CQE.
- Forward every existing no-QP wrapper to QP0 and preserve its source behavior.

**Constraints and non-goals:** No MoonEP names or assumptions. Do not change
`CommArgs` or `UDMAInfo` layout. Keep supported architecture guards intact.

**Acceptance and verification:**

- Host API tests cover invalid pointers, unavailable transport, QP1, and N-QP.
- Device compile/source tests cover all new helpers and QP0 forwarding.
- Deferred enqueue tests or a compilable Host model prove no state mutation on
  insufficient SQ capacity, correct 1-BB/2-BB reclaim, independent QP state,
  and explicit flush/quiet ordering.

**Artifacts and interfaces:** Stable public Host discovery API and generic
device API used by the demo in Task 6 and future operators.

## Task 6: Extend Focused Demo And Validation Automation

**Objective and role:** Make required behaviors observable on two physical
Ascend950PR hosts without depending on unrelated operator tests.

**Background and prerequisites:** Tasks 2, 4, and 5 must be integrated first.

**Modification scope:**

- Extend `tests/udma/demo/tilexr_udma_demo.{cpp,kernel.cpp}` to select QP count,
  register an exact byte count, transfer distinct slices on each QP, validate
  data, report quiet status, unregister, and optionally register a second time.
- Fix the demo barrier/runner to accept a reachable rank-0 address for
  cross-host execution rather than assuming loopback.
- Add or update a bounded runner under `tests/udma/demo/` for legacy, 2 MiB,
  two-QP, three-QP, missing-route, mismatched-config, and reuse cases.
- Update UDMA validation documentation only for commands and evidence actually
  supported by the runner.

**Constraints and non-goals:** IPC and SDMA are disabled for data-plane
acceptance. Do not run grouped AllToAll, MoonEP, or broad non-UDMA suites.

**Acceptance and verification:** Host and device demo artifacts build; source
guards verify ordinary `aclrtMalloc` registration; every hardware case has a
bounded timeout and unambiguous success/failure output.

**Artifacts and interfaces:** Reproducible cross-host validation commands and
logs suitable for the PR report.

## Task 7: Focused Verification And Delivery

**Objective and role:** Prove the standalone change locally and on
`141.61.49.195` / `141.61.49.198`, then deliver only the UDMA scope.

**Background and prerequisites:** All prior tasks are complete. Use the two
hosts and credentials already supplied by the user. Wait for free NPUs rather
than preempting other workloads.

**Modification scope:** Verification artifacts only; no scope expansion while
repairing failures.

**Acceptance and verification:**

- Initialize submodules; source `scripts/common_env.sh`; configure, build, and
  install core plus UDMA tests from a clean tree.
- Run focused parser/config/layout/registry/source/fault tests and the UDMA
  integration smoke test.
- Run `git diff --check`, inspect exported symbols, check no UDMA regex symbols,
  and confirm no `${ARCH}-linux/devlib` RUNPATH/RPATH in `libtile-comm.so`.
- Sync the same commit to both hosts and prove artifact hashes match.
- With `TILEXR_ENABLE_SDMA=0` and IPC unavailable across hosts, pass legacy
  QP0, exact 2 MiB, two-QP `port_count:6,port_count:2`, three-QP
  `port_count:6,port_count:6,port_count:2`, and unregister/re-register cases.
- Prove duplicate-selector QPs have independent SQ/CQ completion and distinct
  data slices.
- Prove missing-route and per-rank config mismatch fail without hanging or
  publishing partial UDMA.
- Confirm no case reports HCCP `528101`, data mismatch, fallback, or timeout.
- Request an independent code review, fix material findings, rerun affected
  checks, commit only scoped files, push `codex/udma-multi-qp`, and create the
  standalone PR against `LingquLab/TileXR:main`.

**Artifacts and interfaces:** Local test output, host logs, artifact hashes,
review findings/resolution, ordered buildable commits, and a PR description
that scopes hardware claims to Ascend950PR.

## Dependency And Delegation Map

Task 1 precedes route integration. Task 2 may proceed in parallel with Task 3.
Task 4 depends on Tasks 1-3. Task 5 can start after the dynamic image contract
is fixed but requires Task 4 for Host discovery. Task 6 depends on Tasks 4-5.
Task 7 depends on all implementation tasks.

Safe parallel ownership is limited to disjoint files: parser/config tests,
layout/device headers, and read-only review. Changes to `tilexr_udma_transport.*`,
`tilexr_udma_context.*`, communicator state, CMake target lists, repository
state, commits, pushes, and PR creation are serialized by the main agent.

## Commit Shape

Keep commits buildable and reviewable in this order:

1. Fix Ascend950 MR flags and diagnostics.
2. Make MR replacement and cleanup transactional and retryable.
3. Replace RootInfo regex parsing and add parser/config tests.
4. Add strict generic route configuration and distributed agreement.
5. Add dynamic per-peer N-QP resources and image construction.
6. Add Host discovery and generic per-QP device operations.
7. Add cross-host demo/runner, documentation, design, and plan.

Implementation detail may move between adjacent commits when needed to keep
each commit compiling, but the final diff must preserve the approved scope.
