# TileXR Checker Design

## Context

TileXR needs a no-Ascend-environment checker for collective communication operators. The primary purpose is multi-rank
functional validation, not single-kernel CPU debugging. The checker must accept the original operator surface and avoid
source edits to production kernels. It may use separate build targets, include-path overrides, compile-time shims, header
stubs, and fake runtime components.

This design is separate from the offline topology/performance simulator in
`docs/superpowers/specs/2026-06-20-collective-sim-design.md`. The simulator compares algorithms and bottlenecks. The
checker validates whether an existing operator implementation behaves correctly across ranks and whether its ordering
requirements are satisfied under adversarial but legal schedules.

Feasibility checks found these constraints:

- CANN CPU-domain twin debugging can execute some Ascend C kernel logic on CPU, but it is not a complete TileXR
  multi-rank communication backend.
- Current collective kernels depend on Ascend C address spaces, `GlobalTensor`/`LocalTensor`, block/core intrinsics,
  MTE copies, flags, barriers, and runtime-registered AICore objects.
- Existing collectives profiling is rank/core/stage aggregate data, not a pipe-level event log.
- `op-simulator` and CANN simulator traces are useful references for pipe categories, but the checker should not require
  them for normal no-NPU functional validation.
- EP dispatch can reuse the checker core later, but its routing/window semantics should be a plugin, not part of the
  first collective MVP.

## Goals

- Validate collective operators across multiple simulated ranks without an Ascend device, driver, or runtime.
- Preserve production operator source files. The checker may add wrapper targets and shim headers, but it must not require
  editing `src/collectives/kernels/*.cce`, collective kernel headers, or EP kernel sources for normal use.
- Accept the original host/API-level operator inputs where possible: send buffer, receive buffer, count, datatype,
  reduce op, root, communicator, and stream.
- Model TileXR communicator state: `rank`, `rankSize`, `localRankSize`, `extraFlag`, `peerMems`, `sendCountMatrix`,
  magic epochs, IPC flag regions, and IPC data regions.
- Detect multi-rank functional errors, missing synchronization, stale magic use, invalid communication-buffer access,
  and obvious deadlock patterns.
- Represent Ascend C asynchronous semantics explicitly instead of replacing everything with synchronous host copies.
- Produce structured findings with event and source evidence.
- Provide fast, friendly problem localization for broken operators: a user should see the likely failure class, affected
  rank/core/buffer, relevant source location, and next debugging action without reading raw traces first.
- Leave extension points for EP dispatch, UDMA/SDMA models, hardware trace comparison, and later cannsim integration.

## Non-Goals

- The first release does not prove real hardware performance.
- The first release does not emulate exact NPU cycle timing, cache behavior, or physical MTE/Vector contention.
- The first release does not require CANN CPU-domain twin debugging.
- The first release does not support every existing collective specialization.
- The first release does not validate HCCL, MC2, UDMA, SDMA, RA/HCCP, or CCU hardware behavior.
- The first release does not turn the checker into a browser algorithm editor or topology simulator.

## Selected Approach

Build a source-preserving, multi-rank checker under a new tool/test surface, for example:

```text
tools/checker/
tests/checker/
```

The checker compiles selected production kernel sources through a checker-specific include path. That include path
provides Ascend C and runtime shims for CPU execution. The shims implement typed memory objects, fake global/local
tensors, fake pipes, fake events, fake streams, fake communicator memory, and a scheduler that can interleave rank/core
execution.

The checker has three layers:

- **Source-preserving compile harness**: builds the original kernel translation units with checker headers first in the
  include path and with checker-specific compile definitions.
- **Multi-rank runtime model**: creates a simulated world containing ranks, comm args, peer memory windows, flag regions,
  user buffers, stream queues, and launch epochs.
- **Semantic and ordering checker**: compares final buffers against collective oracles and verifies happens-before
  constraints from recorded events.

CANN CPU-domain twin debugging remains optional. It can be used as a compile or execution probe when helpful, but the
checker does not depend on it.

## Architecture

```text
original host/API input
        |
        v
checker CLI / test fixture
        |
        v
RankWorld + FakeCommArgs + FakePeerMems + FakeStreams
        |
        +--> source-preserving kernel harness
        |       |
        |       v
        |   shimmed Ascend C API records events and mutates fake memory
        |
        +--> symbolic oracle for expected collective output
        |
        +--> happens-before and deadlock checks
        |
        v
checker_report.json + findings.json + report.html
```

Logical modules:

- `world/`: rank world, fake communicator, IPC memory, magic epochs, stream queues.
- `memory/`: typed user buffers, communication buffers, local buffers, ownership and lifetime tracking.
- `ascendc_shim/`: source-compatible stubs for the subset of Ascend C APIs used by TileXR kernels.
- `runtime_shim/`: fake launch, fake stream, fake ACL/runtime calls needed by host-level tests.
- `scheduler/`: deterministic and adversarial interleavings across rank/core/lane events.
- `collectives/`: AllGather and AllReduce SUM oracles, case generation, output comparison.
- `ordering/`: happens-before graph, flag matching, lane constraints, deadlock checks.
- `diagnostics/`: finding classification, source attribution, short failure summaries, and suggested next actions.
- `report/`: JSON/CSV/HTML outputs, findings, evidence links.
- `ep/`: reserved plugin namespace for later EP dispatch validation.

## Source Preservation

Production kernel sources are treated as read-only inputs. The checker must not require local edits such as adding
`#ifdef CHECKER` branches to kernel files.

Allowed mechanisms:

- A dedicated checker CMake target.
- Include-path precedence for shim headers such as `kernel_operator.h`, runtime launch headers, and selected TileXR
  compatibility wrappers.
- Compile definitions such as `TILEXR_CHECKER_RUNTIME=1`.
- Link-time substitution of runtime functions.
- Wrapper translation units that include production headers or `.cce` files from the checker target.
- Source-level policy tests that fail if production kernels start depending on unsupported APIs without a shim.

Disallowed mechanisms:

- Rewriting production collective or EP kernel source to make the checker pass.
- Adding checker-only branches to hot-path production kernel logic unless the branch also protects a real production
  portability issue and is reviewed separately.
- Treating hardware profiling output as required input for the no-NPU checker.

## Multi-Rank World

The world model should reflect the communicator shape expected by TileXR kernels:

- One `RankContext` per simulated rank.
- One host-side `TileXR::CommArgs` image per rank.
- One fake device-side `CommArgs` pointer per rank.
- `peerMems[]` pointing to fake peer-visible IPC windows.
- Per-rank IPC layout matching TileXR constants: flag space before data space, ping-pong selected by magic.
- A monotonic `magic` epoch per launched collective.
- Optional `extraFlag` routing metadata.
- Optional `sendCountMatrix` for future AllToAll/AllToAllV.

The first implementation should support `rankSize=2` and `rankSize=4` in one process. The design should not prevent
larger rank counts, but first tests should stay small enough to make exhaustive schedule exploration practical.

## Buffer Model

Buffers are role-aware:

- `user_input`: source buffer passed by the original API.
- `user_output`: destination buffer passed by the original API.
- `comm_flag`: synchronization region in an IPC window.
- `comm_data`: communication data region in an IPC window.
- `local_ub`: fake local tensor or UB allocation.
- `metadata`: arguments, counters, assist data, or report-only state.
- `registered_comm_buffer`: reserved for future UDMA-like paths.

The checker enforces TileXR's communication-buffer rule: cross-rank data movement must pass through a communication
buffer or an explicitly modeled registered communication buffer. Directly treating a user tensor as a peer-visible
network endpoint is a checker finding unless the operator's mode explicitly allows it.

## Ascend C Shim Semantics

The shim layer should model the subset of Ascend C APIs needed by the selected collective kernels. The important rule is
that stubs are evented. A stub may execute memory effects on CPU, but it must also record ordering events and respect
asynchronous dependencies.

Initial API categories:

- `GlobalTensor<T>` and `LocalTensor<T>` with bounds-checked typed access.
- `SetGlobalBuffer`, `GetValue`, and `SetValue` for functional compatibility and diagnostics.
- `DataCopy` and `DataCopyPad` as MTE events with source/destination, byte count, alignment metadata, and completion.
- `PipeBarrier<PIPE_ALL>` as a full-lane ordering edge.
- `SetFlag` / `WaitFlag` / `SyncFunc` / hard-event helpers as event edges.
- `TQue` `AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` as buffer-lifetime and producer-consumer edges.
- `TBuf` for local scratch allocation.
- `GetBlockIdx`, `GetBlockNum`, and core type selectors in a controlled rank/core execution context.
- Vector or scalar arithmetic operations used by the first AllReduce path.

The shim should not silently flatten asynchronous APIs into synchronous memcpy. It can execute the copy immediately for
functional state, but it must leave an event record that allows the ordering checker to catch consumers that read before
the copy is synchronized.

## Scheduling Model

The scheduler design supports three modes:

- `serial`: ranks and cores run in a deterministic order. Useful for debugging.
- `round_robin`: alternates runnable ranks/cores and flushes simple ordering mistakes.
- `adversarial`: delays selected copy, flag, or barrier completions to expose missing synchronization.

The MVP should implement `serial` and `round_robin`. `adversarial` mode is a post-MVP requirement because it needs
more complete API coverage and stronger blocked-state diagnostics to avoid noisy false positives.

Each operation emits events:

- `copy_start`, `copy_complete`
- `compute_start`, `compute_complete`
- `flag_store`, `flag_wait_start`, `flag_wait_success`, `flag_wait_blocked`
- `barrier`
- `buffer_alloc`, `buffer_ready`, `buffer_consume`, `buffer_free`
- `rank_launch`, `rank_complete`
- `finding`

The checker constructs a happens-before graph from program order, queue semantics, hard events, barriers, flag
publish/observe pairs, and buffer dependencies.

Deadlock checks should report:

- flag wait with no possible producer
- wait graph cycle with no runnable event
- stale magic mismatch
- all ranks blocked on a condition no rank can satisfy
- send/receive or copy-in/copy-out dependency missing

Ordering checks should report:

- read before copy completion
- local buffer reuse before all consumers complete
- output validation before rank completion
- cross-rank read before producer published data
- direct user-buffer cross-rank transfer
- impossible or ambiguous flag matching

## Fast Problem Localization

The checker should be useful when an operator is wrong. It should not only print `PASS` or `FAIL`, and it should not
force users to inspect raw event logs before seeing the likely root cause.

Every failed run should produce a short first-screen summary:

```text
FAIL allgather rank_size=4 count=16 dtype=int32 scheduler=round_robin
Top finding:
  [READ_BEFORE_COPY] rank=2 core=0 buffer=comm_data[rank1]
  peer_ipc_to_output read can run before producer copy completes.
  Evidence: event e42 reads bytes 0..63; producer e17 has no HB edge to e42.
  Source: src/collectives/kernels/kernels/lcal_allgather_big_data.cce:...
  Next: check missing WaitFlag/PipeBarrier/queue synchronization before peer read.
Artifacts:
  findings.json
  events.jsonl
  summary.txt
```

Finding categories should be stable and grep-friendly:

- `OUTPUT_MISMATCH`: final user output differs from the collective oracle.
- `UNSUPPORTED_API`: shim coverage is missing for a used Ascend C or runtime API.
- `READ_BEFORE_WRITE`: a buffer read has no completed producer.
- `READ_BEFORE_COPY`: a consumer can observe a `DataCopy` result before the copy is synchronized.
- `BUFFER_LIFETIME`: local buffer is reused or freed before all consumers complete.
- `FLAG_NO_PRODUCER`: a wait has no possible matching flag store.
- `FLAG_STALE_MAGIC`: wait/store magic does not match the current launch epoch.
- `DEADLOCK`: all runnable ranks are blocked on unsatisfied waits.
- `DIRECT_PEER_USER_BUFFER`: cross-rank access bypasses a communication buffer.
- `AMBIGUOUS_SOURCE`: the checker found a problem but cannot attribute it precisely enough.

Localization should combine dynamic event evidence and static source evidence:

- Dynamic evidence: event ids, rank, core, lane, buffer id, byte range, expected value, actual value, and HB path.
- Static evidence: source file, line, enclosing function or macro expansion when available.
- Operator evidence: op, dtype, count, rank size, scheduler, seed, magic epoch, and `extraFlag`.
- Suggested next action: one concise instruction such as checking a missing flag producer, adding a barrier in the real
  algorithm, or adding checker shim coverage for an unsupported API.

For output mismatches, the checker should report the smallest useful diff, not a full tensor dump:

- rank and output index
- expected symbolic value and actual symbolic value
- producing rank/chunk if known
- last writer event for that byte range
- nearest missing or suspicious HB edge when available

For deadlocks, the checker should report the wait cycle:

- blocked ranks and cores
- wait condition for each blocked execution context
- expected flag or producer event
- current magic epoch and observed flag value

The HTML report is optional post-MVP, but the text and JSON diagnostics are MVP requirements.

## Functional Oracles

The first collective oracles:

- AllGather, with deterministic rank/index input patterns.
- AllReduce SUM, initially for INT32.

The oracle consumes the original API-level case definition:

```yaml
op: allgather
rank_size: 4
count: 16
datatype: int32
input_pattern: rank_index
```

AllGather expected output for every rank is the concatenation of all ranks' input blocks in rank order.

AllReduce SUM expected output for every rank is the elementwise sum of all rank inputs.

The checker should reuse semantics already documented for TileXR collectives:

- AllGather, AllReduce, and Broadcast byte size is `count * dtype_size`.
- AllToAll and ReduceScatter byte size is `count * rankSize * dtype_size`.
- AllReduce and ReduceScatter MVP should support only SUM.
- Unsupported dtype, rank count, topology flag, or reduce op should fail clearly before execution.

## MVP Scope

MVP cases:

- AllGather INT32, `rankSize=2`, small aligned and non-aligned counts.
- AllGather INT32, `rankSize=4`, small counts.
- AllReduce SUM INT32, `rankSize=2`, small counts.
- Schedule modes: `serial` and `round_robin`.
- Findings for missing flag producer, stale magic, read-before-copy, and direct peer access without comm buffer.
- JSON reports plus a text summary with top finding and suggested next action.

MVP compile/runtime gates:

- Compile-only probe for the checker shim target against selected collective kernel headers.
- Source guard test proving production kernels were not modified for checker-only branches.
- Unit tests for fake `CommArgs`, fake IPC layout, and magic ping-pong selection.
- Unit tests for DataCopy/DataCopyPad event recording and synchronization edges.
- Unit tests for flag publish/wait matching.
- Unit tests for diagnostic classification and first-screen summaries for representative failures.

Post-MVP:

- adversarial schedule mode
- HTML timeline report
- AllReduce FP16/FP32 where arithmetic behavior is agreed
- Broadcast and ReduceScatter
- cannsim trace import for pipe-lane comparison
- hardware profile import for measured aggregate comparison
- EP dispatch plugin

## CPU-Domain Twin Debugging Position

CANN CPU-domain twin debugging is not a dependency of this checker. It remains useful as a reference and optional probe:

- It demonstrates that some Ascend C kernel launches can be translated to CPU execution.
- It provides `cpu_debug_launch.h` and cpudebug libraries that can guide shim design.
- It can be used as a compile or execution experiment for isolated kernels.

The checker should not depend on cpudebug because:

- TileXR needs multi-rank communication validation, not only single-kernel execution.
- Current kernels rely on fakeable but nontrivial device constructs: `peerMems`, IPC flags, magic epochs, MTE copies,
  hard events, and runtime registration.
- cpudebug does not provide true cross-machine communication or true hardware pipe contention.
- Some TileXR compile guards may not match cpudebug's predefined macros.

If later evidence shows cpudebug handles a specific TileXR kernel cleanly, the checker may add `cpudebug_backend` as an
optional execution backend behind the same report and oracle interfaces.

## EP Extension

EP should reuse the checker core but not enter the collective MVP.

Reusable pieces:

- rank world
- typed buffers
- fake `CommArgs`
- fake peer memory windows
- event trace
- happens-before checker
- source-preserving compile harness

EP-specific plugin pieces:

- parse `expertIds[bs, topK]`
- compute `dstRank = expertId / localExpertNum`
- model source-local peer-visible windows
- model `EpWindowHeader`, `EpSrcSlotHeader`, payload area, and assist tuples
- model `kEpStepWindowCleared` and `kEpStepDispatchReady`
- validate `expandX`, `expertTokenNumsOut`, `epRecvCountsOut`, and `assistInfoForCombineOut`

The core schemas should reserve buffer roles and op kinds for EP:

- roles: `route_table`, `window_slot`, `assist_metadata`
- ops: `route`, `slot_write`, `slot_read`, `count_expert`, `compact`

## Reports

The checker emits machine-readable output first:

```text
checker_report.json
findings.json
events.jsonl
summary.txt
```

Recommended `checker_report.json` fields:

- `schema`: version string
- `case`: op, rank size, count, datatype, reduce op, scheduler mode
- `environment`: checker version, source commit, backend, shim set
- `events`: event file references and counts
- `hb_edges`: edge counts by type
- `findings`: finding references by severity
- `result`: pass, fail, unsupported, or inconclusive
- `top_finding`: the highest-priority finding id for first-screen display

Each finding should include:

- `finding_id`
- `severity`
- `check_id`
- `message`
- `rank`, `core`, and event ids when known
- buffer id and byte range when known
- source file and line when statically attributable
- shortest relevant happens-before path or wait cycle
- suggested next action
- confidence: `high`, `medium`, or `low`

The first HTML report can wait until after the JSON/text report is useful. When added, it should clearly label event
sources as `static_inferred`, `stubbed_runtime`, `simulator_event`, or `measured_aggregate`.

## Error Handling

The checker should distinguish:

- `fail`: operator output or ordering is wrong.
- `unsupported`: the checker does not yet model a required API, dtype, op, or topology flag.
- `inconclusive`: a backend ran but did not provide enough evidence for a requested check.
- `internal_error`: checker bug or inconsistent model state.

Unsupported APIs should produce a concise capability error naming the missing shim symbol and the source file that used
it. They should not be reported as operator correctness failures.

## Testing Strategy

Tests should be layered:

- Unit tests for rank world, fake memory, `CommArgs`, magic, and IPC window layout.
- Unit tests for evented shims: copy, barrier, flags, queue, local buffer lifetime.
- Unit tests for AllGather and AllReduce oracles.
- Negative tests that inject missing flag producers, stale magic, read-before-copy, and direct user-buffer peer access.
- Compile-only tests for checker include shims against selected collective kernel headers.
- End-to-end checker cases for AllGather and AllReduce SUM.
- Golden-output tests for `summary.txt` and representative `findings.json` entries, so diagnostics stay friendly and
  stable over time.
- Source guard tests ensuring production kernel files are not modified for checker-specific branches.

When hardware is available later, real collectives tests remain separate validation gates. The checker is a no-NPU
preflight, not a replacement for real hardware correctness and performance runs.

## Open Risks

- Some production kernel sources may use Ascend C features that are expensive to stub faithfully. The first compile-only
  probe should identify those symbols before implementation commits to broad coverage.
- A too-synchronous stub could hide exactly the ordering bugs the checker should find. Event recording and adversarial
  scheduling must be part of the core, not a later report feature.
- Matching original host/API input while avoiding real ACL runtime may require a fake host API layer or a checker-only
  adapter around existing public APIs.
- Large kernels may instantiate many templates or macros, so compile time and diagnostic quality matter.
- Source attribution may be imprecise when the failure comes from a macro-heavy include chain. Low-confidence findings
  must say so and still provide the nearest event and source evidence.
- AllReduce arithmetic for non-INT32 types needs explicit precision rules before it becomes a correctness target.

## Approval Criteria

The design is ready to implement when the following are accepted:

- The checker's main promise is multi-rank no-NPU functional validation.
- Production operator sources remain unmodified for checker support.
- Header/runtime stubbing is allowed.
- CANN CPU-domain twin debugging is optional, not a dependency.
- Failed checks produce fast, friendly diagnostics with a top finding and next action.
- MVP starts with AllGather and AllReduce SUM, not EP.
- EP is reserved through plugin-friendly schemas and shared core abstractions.
