# TileXR Checker

The checker is an opt-in, no-NPU verification surface for TileXR collective and
EP dispatch correctness. It can run generic software collective models, selected
production collective headers through checker-only AscendC shims, and an
`ep_dispatch` CPU oracle, then emits deterministic reports for output
mismatches, communication ordering hazards, flag misuse, and source-level fault
location.

## Purpose

- Verify selected collective invariants without CANN runtime setup or NPU hardware.
- Verify `ep_dispatch` route payloads for multi-rank/multi-server CPU cases.
- Keep production operator headers unmodified; checker support is implemented by
  checker-only include paths, macro trace wrappers, and header probes.
- Report fast source locations for suspicious reads, copies, writes, and sync
  waits/stores.
- Preserve an output oracle so a run can separate functional output failures
  from ordering/timing findings. Production-header oracle materializers are
  gated by traced user-output write coverage, so a runner cannot pass by
  materializing expected output without first executing a production output
  write path.

## Boundary

- The checker does not replace hardware validation, CANN runtime validation, or
  collectives performance testing.
- Production-header trace runs instantiate and execute the C++ control flow in
  the original AscendC header under CPU-side shims. EP dispatch currently uses
  a source-aligned CPU oracle instead of including the `.cpp` kernel body. These
  modes do not execute AICore instructions, issue real DMA, or model
  cycle-accurate pipe timing. Pipe checks are abstract ordering checks over
  `SetFlag`, `WaitFlag`, and `PipeBarrier` trace events.
- CANN CPU-domain twin debugging is optional future input. The checker remains
  usable without cpudebug.

## Build

The checker is off by default. Enable it explicitly:

```bash
source scripts/common_env.sh
cmake -S . -B build-checker -DTILEXR_BUILD_CHECKER=ON -DBUILD_TESTING=ON
cmake --build build-checker --target tilexr_checker tilexr_checker_all -j"$(nproc)"
```

The CLI binary is produced at `build-checker/tools/checker/tilexr_checker`.
On a host without the CANN runtime, build checker targets explicitly; the
repository default target may include CANN-linked production libraries.

## CLI

Usage:

```text
tilexr_checker --op <allgather|allreduce> --rank-size <n> --count <n> \
  --datatype int32 [--reduce-op sum] [--scheduler <serial|round_robin>] \
  [--algorithm <default|allgather_hierarchy_double_ring|allreduce_big_data>] \
  [--server-count <n>] [--output-dir <dir>] [--inject-read-before-copy] \
  [--inject-peer-user-read] [--inject-pipe-wait] [--inject-ep-window-read]

tilexr_checker --generate-trace-adapter --adapter-name <name> \
  --source-file <path> --target-header <include> --output <checker-shim.h> \
  [--manifest-output <manifest.json>] \
  [--runner-output <runner.cpp> --runner-function <name> --runner-materializer <name>] \
  [--probe-output <probe.cpp>] \
  [--onboarding-output <onboarding.md>] [--strict-source-observation]

tilexr_checker --generate-trace-adapter --source-file <path> \
  --output-dir <generated-checker-dir>

tilexr_checker --scaffold-trace-bundle --source-file <path> \
  --output-dir <generated-checker-dir> [--adapter-name <name>] \
  [--target-header <include>] [--repo-root <repo>] \
  [--verification-report <file>] [--strict-source-observation]

tilexr_checker --analyze-trace-source --source-file <path> \
  [--adapter-name <name>] [--target-header <include>] [--repo-root <repo>] \
  [--trace-analysis-output <file>] \
  [--event-trace-template-output <events.jsonl>]

tilexr_checker --validate-event-trace --event-trace <events.jsonl> \
  --output-dir <dir> [--repo-root <repo>] [--adapter-name <name>]

tilexr_checker --verify-trace-bundle --adapter-name <name> \
  --output-dir <generated-checker-dir> [--repo-root <repo>] \
  [--verification-report <file>]

tilexr_checker --verify-installed-trace --adapter-name <name> \
  [--repo-root <repo>] [--verification-report <file>]

tilexr_checker --list-installed-traces [--repo-root <repo>]

tilexr_checker --verify-all-installed-traces [--repo-root <repo>] \
  [--verification-report <file>]

tilexr_checker --list-capabilities [--repo-root <repo>] \
  [--capability-report <file>]

tilexr_checker --op <ep_dispatch|ep_combine> --rank-size <n> --bs <n> --h <n> \
  --top-k <n> --moe-expert-num <n> --datatype <fp16|bfp16> \
  [--server-count <n>] [--output-dir <dir>] [--inject-ep-window-read]
```

Generic smoke:

```bash
build-checker/tools/checker/tilexr_checker \
  --op allgather --rank-size 4 --count 16 --datatype int32 \
  --scheduler round_robin --server-count 2 \
  --output-dir /tmp/tilexr-checker-smoke
```

`--server-count` defaults to `1`. It must be positive, not exceed
`--rank-size`, and divide `--rank-size`. The checker maps contiguous rank
groups onto servers and populates `CommArgs.localRank`,
`CommArgs.localRankSize`, event `server`, and event `peer_server` fields.

Production header trace for `src/collectives/kernels/91093/allgather_hierarchy_double_ring.h`:

```bash
build-checker/tools/checker/tilexr_checker \
  --op allgather --algorithm allgather_hierarchy_double_ring \
  --rank-size 10 --count 16 --datatype int32 \
  --scheduler round_robin --output-dir /tmp/tilexr-checker-hdb
```

Production header trace for `src/collectives/kernels/allreduce_big_data.h`:

```bash
build-checker/tools/checker/tilexr_checker \
  --op allreduce --algorithm allreduce_big_data \
  --rank-size 4 --count 524288 --datatype int32 --reduce-op sum \
  --scheduler round_robin --output-dir /tmp/tilexr-checker-arbd
```

Diagnostic injection:

```bash
build-checker/tools/checker/tilexr_checker \
  --op allgather --rank-size 2 --count 16 --datatype int32 \
  --scheduler serial --inject-peer-user-read \
  --output-dir /tmp/tilexr-checker-fail
```

The failure injection flags are for checker diagnostics only:

- `--inject-read-before-copy` inserts a read-before-copy hazard.
- `--inject-peer-user-read` inserts a peer-visible user-buffer read hazard.
- `--inject-pipe-wait` inserts an AICore pipe wait without a matching producer.
- `--inject-ep-window-read` inserts a combine-side
  `RegisteredCommBuffer` read without a covering dispatch window write.

EP dispatch CPU oracle:

```bash
build-checker/tools/checker/tilexr_checker \
  --op ep_dispatch --rank-size 2 --server-count 2 \
  --bs 3 --h 4 --top-k 2 --moe-expert-num 4 --datatype fp16 \
  --output-dir /tmp/tilexr-checker-ep
```

EP combine is supported by a checker-side source-aligned CPU oracle. It consumes
the same deterministic dispatch-shaped payload and assist metadata, reconstructs
the original `[bs, top_k, h]` route layout for each source rank, and records
combine-side `RegisteredCommBuffer` reads plus user-output writes with source
`src/ep/kernels/tilexr_ep_combine_kernel.cpp`:

```bash
build-checker/tools/checker/tilexr_checker \
  --op ep_combine --rank-size 2 --server-count 2 \
  --bs 3 --h 4 --top-k 2 --moe-expert-num 4 --datatype fp16 \
  --output-dir /tmp/tilexr-checker-ep-combine
```

To check the EP window localization path, inject a missing combine-side window
producer:

```bash
build-checker/tools/checker/tilexr_checker \
  --op ep_dispatch --rank-size 2 --server-count 2 \
  --bs 3 --h 4 --top-k 2 --moe-expert-num 4 --datatype fp16 \
  --inject-ep-window-read --output-dir /tmp/tilexr-checker-ep-window-fail
```

The top finding should be `READ_BEFORE_WRITE` with source
`src/ep/kernels/tilexr_ep_combine_kernel.cpp`, buffer
`RegisteredCommBuffer`, and the offending slot/offset/byte range in
`summary.txt` and `checker_report.json`.

To validate an abstract event model directly, use `--validate-event-trace`.
The input is the checker JSONL event schema emitted as `events.jsonl`; it can be
produced by a checker runner, a generated trace adapter, a separate source
instrumentation pass, or a hand-written minimized reproducer:

```bash
build-checker/tools/checker/tilexr_checker \
  --validate-event-trace --event-trace /tmp/events.jsonl \
  --output-dir /tmp/tilexr-checker-event-trace --repo-root .
```

This mode does not execute an operator or compare payload oracles. It runs the
same ordering diagnostics used by normal checker execution and writes the same
`summary.txt`, `findings.json`, `events.jsonl`, and `checker_report.json`
artifacts. Use it when the abstract event model has been generated
semi-automatically, or when you want to reduce a multi-rank/multi-pipe timing
bug to a small event file before debugging source code.
When `--adapter-name <name>` is also provided, validation reads
`tools/checker/installed_traces/<name>_trace_manifest.json` under `--repo-root`
and checks `required_event_coverage`. Missing `{kind, source_file, source_line}`
entries become source-located `EVENT_COVERAGE_GAP` findings, which catches
broken or incomplete generated traces even if the remaining event ordering
would pass.

The EP checker requires positive `bs`, `h`, `top-k`, and `moe-expert-num`;
`moe-expert-num` must be divisible by `rank-size`; datatype must be `fp16` or
`bfp16`. It fills deterministic per-rank payloads, generates deterministic
expert routes, and checks the same output classes as the EP demo: expanded
payload rows, assist tuples, per-source receive counts, and local expert token
counts. Dispatch emits source-aligned events for payload copy, assist tuple
writes, receive counts, and local expert token counts. Combine emits
source-aligned events for return-window header reads, payload reads, assist
reads, and restored output writes. The combine path is still a CPU oracle; it is
not direct execution of a production AscendC combine kernel body.

## Support Matrix

| path | op | datatype | ranks | reduce-op | event source |
| --- | --- | --- | --- | --- | --- |
| `default` | AllGather | int32 | 2, 4 | n/a | checker software model |
| `default` | AllReduce | int32 | 2 | sum | checker software model |
| `allgather_hierarchy_double_ring` | AllGather | int32 | even `rank_size > 2`; normally selected for large data or `rank_size > 8` | n/a | production header trace |
| `allreduce_big_data` | AllReduce | int32 | `rank_size > 2` and data size >= 2 MiB | sum | production header trace |
| `default` | EP Dispatch | fp16, bfp16 | `moe_expert_num % rank_size == 0` | n/a | source-aligned CPU oracle |
| `default` | EP Combine | fp16, bfp16 | `moe_expert_num % rank_size == 0` | n/a | source-aligned CPU oracle |

Reportable unsupported cases, such as a valid op/data shape that lacks a
checker executor, fail before model execution and are written as
`UNSUPPORTED_API` findings. Pure CLI argument errors, such as an unknown op name
or missing required shape argument, return from argument parsing and do not
create report artifacts.

## Trace Model

Production-header trace uses checker-only shims under `tools/checker/shim`.
The shims provide AscendC types, fake block context, IPC virtual address ranges,
and trace wrappers for copy/sync helper calls. Macro wrappers are applied only
around checker-owned includes of the production headers, so the production source
files remain unchanged.

The trace event stream contains:

- `READ`, `COPY`, and `WRITE` events with buffer role, rank, peer rank, slot,
  offset, byte count, source file, and source line.
- `FLAG_STORE` and `FLAG_WAIT` events with the merged magic/value payload.
- Mutual `FLAG_WAIT` cycles without a matching producer are reported as
  `DEADLOCK`; the individual missing producer waits are still kept as lower
  priority findings for detail.
- `PIPE_SET`, `PIPE_WAIT`, and `PIPE_BARRIER` events with rank, core, pipe,
  event id, source file, and source line. These events validate abstract AICore
  pipe ordering; they do not prove instruction-level latency or exact overlap.
- `server` and `peer_server` on communication, sync, and pipe events so
  cross-machine ordering findings can be separated from same-server findings.
- `allow_future_producer` on blocking comm-data reads that are protected by a
  sync wait in the production control flow.

If a trace hook sees a GM copy whose source or destination is not registered in
`TraceRuntime`, the checker records a diagnostic event and reports
`UNSUPPORTED_API` instead of silently passing. Real backing buffers also fail
when a non-virtual range is copied beyond its registered storage. Explicit IPC
virtual windows registered by the production-header runners are sparse by
design: they provide address identity for event/source coverage without forcing
the checker to allocate a full hardware-sized IPC buffer.

`allreduce_big_data` and `allgather_hierarchy_double_ring` reports should show
events sourced from their production headers, including real copy call-site line
numbers. Pipe events may point to included production helpers such as
`src/collectives/kernels/collectives.h`.

`ep_dispatch` reports should show events sourced from
`src/ep/kernels/tilexr_ep_dispatch_kernel.cpp`, including the payload copy,
assist tuple, receive-count, expert-token-count, window-header, and source-slot
header call sites. The checker validates deterministic payload and metadata
oracles, reports cross-server route events, records the production-style
intermediate EP window writes as `RegisteredCommBuffer` events, and writes
`checker_report.json.ep_dispatch_layout` with the output buffer layout plus the
production IPC window layout:

- `payload_bytes`: expanded payload region size;
- `assist`: offset and bytes for packed `{srcRank, tokenId, topKId, expertId}`;
- `recv_counts`: offset and bytes for per-source receive counts;
- `expert_token_nums`: offset and bytes for per-local-expert counts.
- `window`: production-style EP communication window values aligned with
  `src/ep/host/ep_layout.cpp`, including `max_routes_per_src`, `row_bytes`,
  aligned payload/assist bytes per source slot, `slot_bytes`, and `total_bytes`.

The EP window header and per-source slot header events are intentionally kept
separate from final user-output metadata. They give stale-window, route-bound,
and future combine-side checks a production-shaped intermediate state without
requiring real IPC memory or AICore execution.
Reads from `RegisteredCommBuffer` are checked against earlier covering writes on
the same rank/peer/slot/offset range. A future combine-side checker can
therefore reuse dispatch window events directly: missing or stale window headers
show up as source-located `READ_BEFORE_WRITE` findings instead of silent output
oracle drift.

This is not an instruction-level execution of the AscendC kernel. For installed
production-header traces, the final CPU oracle materializer writes the expected
output only after the trace has shown user-output write coverage for every rank
and output span; it validates that the production control flow reached the
output write path, but it still does not prove instruction-level data movement
or exact AICore arithmetic behavior.

## Adding a Header Trace

Most new collective header support should use the thin adapter pattern:

1. Add a checker-only shim under `tools/checker/shim/tilexr/checker/`.
2. Define `TILEXR_CHECKER_TRACE_SOURCE_FILE` to the production header path shown
   in reports.
3. Define `TILEXR_CHECKER_TRACE_TARGET_HEADER` to the include path passed to the
   compiler.
4. Include `tilexr/checker/collective_trace_adapter.h`.

The shared adapter injects the trace wrappers for `CpGM2GMPingPong`,
`SetSyncFlag`, and `WaitSyncFlag`; `kernel_operator.h` records AscendC pipe
primitives when the checker trace hook is enabled. Production headers stay
checker-free.

Before generating files, you can run the semi-automatic trace source analyzer
to see what the checker can hook automatically and what still needs an explicit
checker-side decision:

```bash
build-checker/tools/checker/tilexr_checker \
  --analyze-trace-source \
  --source-file src/collectives/kernels/allreduce_big_data.h \
  --adapter-name allreduce_big_data \
  --repo-root . \
  --trace-analysis-output /tmp/tilexr-checker-allreduce-big-data-trace-analysis.json \
  --event-trace-template-output /tmp/tilexr-checker-allreduce-big-data-events.jsonl
```

This command does not compile the operator or execute a checker case. It reads
the production source through `--repo-root`, preserves the production-relative
`source_file` in the report, and emits:

- `semi_automatic_trace.header_stub`: the checker-only include/macro shim shape
  that should wrap the production header.
- `auto_hook_candidates`: source lines that map to checker trace events such as
  `COPY`, `FLAG_STORE`, `FLAG_WAIT`, or `PIPE_BARRIER`.
- `manual_review_actions`: raw `DataCopy`/`DataCopyPad`/`SetAtomic*` or similar
  sites that need a wrapper, op-specific oracle/materializer, waiver, or
  explicit unsupported gate before the adapter can be treated as installed.
- `--event-trace-template-output`: a checker JSONL event template containing
  the auto-hookable source lines. It is intentionally conservative and should
  be edited or replaced with instrumented rank/slot/offset/bytes values before
  trusting timing results, but it can be passed directly to
  `--validate-event-trace` to verify schema and report plumbing.

Use this as the first no-NPU feasibility pass for a newly written header. A
clean analyzer report means the generic header shim can probably observe the
communication/sync skeleton; it does not mean the runner schedule, output
oracle, or installed event coverage has been implemented.

The CLI can generate the thin adapter skeleton:

```bash
build-checker/tools/checker/tilexr_checker \
  --generate-trace-adapter \
  --source-file src/collectives/kernels/alltoall_demo.h \
  --output-dir /tmp/tilexr-checker-alltoall-demo \
  --strict-source-observation
```

With only `--source-file` and `--output-dir`, the checker infers the adapter
name from the header basename and the target include relative to
`src/collectives/kernels/`. It writes:

- `<name>_trace_shim.h`: checker-only adapter header.
- `<name>_trace_manifest.json`: source path, target include, generated outputs,
  covered trace hooks, source-preservation contract, runner integration points,
  validation commands, and an `installed_trace_seed.required_event_coverage`
  list inferred from recognized source hook call sites.
- `<name>_trace_runner.cpp`: runner skeleton with inferred `Run<Name>Trace` and
  materializer review placeholders.
- `<name>_trace_probe.cpp`: header probe skeleton that includes the generated
  shim, audits `TraceAdapterMetadata`, `Metadata()`, and `Audit()`, and checks
  the exact adapter/source/target/include-guard metadata.
- `<name>_trace_onboarding.md`: concrete follow-up checklist for moving the shim
  into the checker include tree, adding the runner to `tools/checker/CMakeLists.txt`,
  declaring it in `collective_trace_runner.h`, moving the header probe under
  `tests/checker/unit/`, wiring `CollectiveExecutor`, adding
  `test_tilexr_checker_*` coverage, and proving source preservation.

The explicit flags remain available when the generated paths or names need to
match an existing checker target. The generator rejects outputs under production
source roots such as `src/collectives/` and `src/ep/`. This keeps the
semi-automatic step auditable: the tool generates the header/macro shim and
records the remaining runner work, but it does not silently modify production
sources or add an unreviewed algorithm path.

Generated adapter headers also publish a small header-only metadata API under
`tilexr::checker::trace_adapter_<name>`:

- `Metadata()` returns the adapter name, production source path, target include,
  include guard, observed hook symbols, and observed source lines.
- `Audit(const char **reason)` checks that the metadata is internally
  well-formed.

This metadata is for static audit and onboarding only. It proves that the source
scan found hookable call sites and that the generated shim carries those facts,
but it does not prove the installed runner actually executes those call sites.
Installed algorithms must still pin runtime `required_event_coverage` in
`tools/checker/installed_traces/<adapter>_trace_manifest.json`.

`--strict-source-observation` rejects generation when the source scan finds
known manual-review candidates such as raw `DataCopy`, `DataCopyPad`, or
`SetAtomic*` calls. Those primitives may be valid production code, but they are
not automatically covered by the generic `CpGM2GMPingPong`/sync wrappers. Use
the non-strict manifest only as an onboarding report, then add an explicit shim,
runtime model, or op-specific unsupported gate before treating the algorithm as
installed. The source scan is intentionally lightweight, but it strips comments
and string/character literals before matching symbols so documentation text does
not trigger strict-mode failures. Symbol matching also uses identifier
boundaries, so names such as `DataCopyCounter` do not count as raw `DataCopy`
calls.

In non-strict mode, the manifest still records those candidates as
`manual_review_required:true` with `required_actions` entries. Each action starts
as `status:"pending"` and names the symbol, source line, and expected follow-up:
add a trace wrapper, add an op-specific oracle/materializer, or add an explicit
unsupported gate. `--verify-trace-bundle` treats pending actions as incomplete.
Only change an action status to `handled`, `waived`, or `unsupported_gate` after
the checker-side runner has made that decision visible in code and tests.
If the source scan can read the header but observes no recognized checker trace
hook at all, the generator also emits a pending `coverage_decision` action. This
is intentional fail-closed behavior: an empty hook scan means the generic macro
adapter has not proved that it can see the production communication path. In
strict mode, the same condition is rejected immediately.

The generated `installed_trace_seed.required_event_coverage` block is not a
completed installed manifest. It is a review aid: copy selected entries into
`tools/checker/installed_traces/<adapter>_trace_manifest.json` after the runner
is wired and the call sites are known to be required. The installed verifier then
uses those entries as a runtime coverage gate.

For a newly written `src/collectives/kernels/allreduce_big_data.h`, the normal
checker workflow is:

1. Run `--analyze-trace-source` to inspect hookable source lines and manual
   actions without generating code.
2. Run `--scaffold-trace-bundle` to generate the checker-only shim, manifest,
   runner skeleton, probe, and onboarding plan outside production source roots.
3. Move reviewed shim/probe/manifest pieces into `tools/checker/` and
   `tests/checker/`; keep `src/collectives/kernels/allreduce_big_data.h`
   unchanged.
4. Implement the checker-side runner schedule and `Init`/`Process` call under
   `TraceRuntime::SetKernelContext`.
5. Review the oracle materializer and add representative smoke cases plus
   `required_event_coverage` entries.
6. Run `--verify-installed-trace --adapter-name allreduce_big_data --repo-root .`
   and a normal checker execution with `--op allreduce --algorithm
   allreduce_big_data`.

The verification claim at the end is intentionally narrow: it confirms the
checker selected the production algorithm path, observed the abstract trace
events it expects, compared output against the installed oracle, and can report
source locations for failures. It still does not execute AscendC AICore
instructions or prove cycle-accurate pipe timing.

For day-to-day onboarding of a newly written production header, prefer the
scaffold mode over manually composing every output path:

```bash
build-checker/tools/checker/tilexr_checker \
  --scaffold-trace-bundle \
  --source-file src/collectives/kernels/allreduce_big_data.h \
  --output-dir /tmp/tilexr-checker-allreduce-big-data-scaffold \
  --repo-root . \
  --verification-report /tmp/tilexr-checker-allreduce-big-data-scaffold/verification.json
```

This command infers `adapter_name=allreduce_big_data` and
`target_header=allreduce_big_data.h`, writes the checker-only shim, manifest,
runner skeleton, header probe, onboarding plan, and then immediately runs
`--verify-trace-bundle`. A normal first scaffold is usually `incomplete`: the
probe can already report `probe_compile: PASS`, while the runner checklist still
names `block_schedule_defined`, `operator_init_process_wired`, and
`oracle_materializer_reviewed` as pending. That is the expected handoff point:
the production header has not been edited, the macro/header trace layer is
compile-checked, and the remaining work is explicit checker-side runner/oracle
integration.

`--source-file` stays as the production-relative path recorded in metadata and
reports. When `--repo-root` is provided, scaffold mode uses it to read relative
sources during the hook scan, so the same command can be run from the repository
root, a build directory, or CI working directory without changing the production
path stored in the manifest.

Use `--strict-source-observation` with scaffold mode when you want the command
to fail immediately on raw `DataCopy`/`DataCopyPad`/`SetAtomic*` candidates or
on a source file with no recognized trace hook. Without strict mode, those items
are recorded as `required_actions` in the manifest and as checklist guidance in
the onboarding plan.

When `--runner-output` is set, the checker also writes a runner skeleton. The
skeleton includes the generated shim, creates `TraceRuntime`, registers virtual
peer windows, installs/restores `CommArgs.peerMems`, and emits a
`RunnerIntegrationChecklist`. The onboarding plan points to the same checklist
and the files that normally need checker-side edits. Generated runners return
`UNSUPPORTED` until the unchecked items are made true:

- `block_schedule_defined`: choose `block_num`/`block_idx` traversal for the
  production kernel.
- `operator_init_process_wired`: instantiate the production class from the shim
  header and call its `Init`/`Process` path under `TraceRuntime::SetKernelContext`.
- `oracle_materializer_reviewed`: confirm or implement the output materializer
  before comparing against the checker oracle.

Only after those items are reviewed should the runner be added to
`CollectiveExecutor`. This prevents a generated file from looking like a
validated production algorithm path before the production control flow and
oracle are actually wired.

Installed production-header runners must use the `TraceRuntime` materializer
registry explicitly, for example
`runtime.ApplyMaterializer("allreduce_big_data_sum_int32")`. A materializer
entry records the op, algorithm, datatype, reduce op, and output span used by
the oracle. Unknown materializer names return `UNSUPPORTED`, and repeated
application is idempotent. This keeps output correctness review visible in the
runner instead of hiding it in ad hoc writeback code.

Installed runners must also expose a `TraceScheduleSpec` such as
`allreduce_big_data_block_major` or
`allgather_hierarchy_double_ring_stage_major`. The spec records the production
block count, block/rank loop order, peer-window mode, and expected visit count.
Runner code should use that spec to drive `TraceRuntime::SetKernelContext`,
rather than duplicating literal block counts in the execution loop.

Before wiring a generated bundle into the checker, run the static bundle
verifier:

```bash
build-checker/tools/checker/tilexr_checker \
  --verify-trace-bundle \
  --adapter-name alltoall_demo \
  --output-dir /tmp/tilexr-checker-alltoall-demo \
  --repo-root . \
  --verification-report /tmp/tilexr-checker-alltoall-demo/verification.json
```

The verifier checks that the shim, manifest, runner skeleton, and onboarding
plan exist, that the generated header probe audits `Metadata()`/`Audit()`, that
the generated shim exposes `TraceAdapterMetadata`, `Metadata()`, and `Audit()`,
that the manifest contains the source-preservation, metadata API, and runner
checklist sections, and that the runner no longer has unchecked
`RunnerIntegrationChecklist` items. When `--repo-root` is provided, it also
compiles and runs `<name>_trace_probe.cpp` with include paths for the generated
bundle, `tools/checker/shim`, `tools/checker/include`, `src/include`, and
`src/collectives/kernels`. A complete generated bundle summary includes
`probe_compile: PASS`; failures include `probe_compile:<log>` so the first
triage step is the generated compile log, not a manual search through the
bundle. Exit code `2` means the bundle is still a source-preserving generated
scaffold and should not be treated as a completed checker algorithm path yet.

For algorithms that have already been wired into the checker, use the installed
trace verifier:

```bash
build-checker/tools/checker/tilexr_checker \
  --verify-installed-trace \
  --adapter-name allreduce_big_data \
  --repo-root .
```

To discover and verify every installed trace manifest in one no-NPU pass:

```bash
build-checker/tools/checker/tilexr_checker \
  --list-installed-traces --repo-root .

build-checker/tools/checker/tilexr_checker \
  --verify-all-installed-traces --repo-root . \
  --verification-report /tmp/tilexr-checker-installed-traces.json
```

The batch verifier scans `tools/checker/installed_traces/*_trace_manifest.json`,
runs the same installed verifier for each adapter, prints one summary line per
adapter, and writes a CI-friendly inventory JSON with per-adapter status,
probe/runtime smoke results, missing artifacts, unchecked items, and event
counts. The installed inventory currently includes the two production-header
collective traces and the `ep_dispatch` source-aligned CPU oracle.

To audit the checker support surface without executing smoke cases, use the
capability inventory:

```bash
build-checker/tools/checker/tilexr_checker \
  --list-capabilities --repo-root . \
  --capability-report /tmp/tilexr-checker-capabilities.json
```

The text output lists user-facing `op/algorithm` pairs such as
`allreduce/allreduce_big_data: installed`,
`ep_dispatch/default: installed`, and
`ep_combine/default: supported`. The JSON report records the same support
surface with `mode`, `source_file`, and `evidence`, plus the global checker
boundary: no NPU is required, production sources are not modified, and
`instruction_level_execution` is `false`. Treat this inventory as a coverage
map; use `--verify-all-installed-traces` when you need proof that installed
paths still compile, smoke, and satisfy manifest-pinned event coverage.

This mode checks the checker-side integration rather than a generated bundle:
shim header, runner declaration, runner source, `CollectiveExecutor` dispatch,
shim metadata/audit API, manifest source-file agreement, registered
materializer name, registered schedule spec, header probe test, and
`tests/checker/CMakeLists.txt` registration. The entry point is the installed
manifest at
`tools/checker/installed_traces/<adapter>_trace_manifest.json`, not a static
verifier table. To add a production algorithm, add that manifest with the
adapter name, production source file, shim path, runner function/source, probe
test, materializer name, schedule name/function, algorithm enum, and
representative smoke matrix.
`op` and `algorithm` describe the installed path. For collective traces,
`smoke_cases` contains one or more `{rank_size, count, server_count}` cases that
should cover the production algorithm's selection boundary, such as big-data
thresholds or multi-server rank placement. For EP traces, each smoke case should
use `{rank_size, server_count, bs, h, top_k, moe_expert_num, datatype}` instead.
The legacy top-level `rank_size`, `count`, and `server_count` fields are kept
only as a fallback for single-case collective manifests. Use
`required_event_coverage` entries to pin important production-header call sites,
for example `COPY`/`FLAG_WAIT`/`PIPE_WAIT` events at specific source lines. The
verifier runs every no-NPU smoke case through `CollectiveExecutor` and reports
`runtime_smoke: PASS` only when the selected production trace path emits events
without findings or output mismatches. The installed check is still incomplete
if a manifest-pinned event kind/source line is missing, which catches broken
macro/header shims even when the output oracle would otherwise pass.
`required_event_coverage` is required for installed trace manifests; static
adapter metadata is not accepted as a replacement. The installed verifier also
compiles and runs the manifest's `probe_test` directly, so a complete summary
contains both `probe_compile: PASS` and `runtime_smoke: PASS`.
It also rescans the current `source_file` and checks that installed shim
metadata still lists observed hook symbols and line numbers. If the production
source gains a new hookable communication/sync call, or a raw
`DataCopy`/`DataCopyPad`/`SetAtomic*` manual-review candidate appears,
installed verification becomes incomplete until the shim metadata, runner,
manifest coverage, or an explicit unsupported gate is updated. Source-aligned
oracle entries such as `ep_dispatch` do not pretend to have a production-header
shim; instead, their installed manifest names a runner marker and pins runtime
event coverage to source lines in `tilexr_ep_dispatch_kernel.cpp`. `ep_combine`
is currently supported by the runtime checker, but it does not yet have an
installed production-header trace manifest because there is no production
combine kernel header to wrap in this tree.

Both verifier modes accept `--verification-report <file>` for CI or agent
handoff. The JSON report records `mode`, `adapter_name`, `complete`,
`probe_compile`, `runtime_smoke`, `runtime_smoke_cases`, `runtime_events`,
`missing_artifacts`, `missing_manifest_sections`, `unchecked_items`, and the
human summary string. The file is written for both pass and fail results, so a
failed generated bundle can be diagnosed without parsing stdout.

## Reports

Each checker run writes four files to `--output-dir`:

- `summary.txt`: top-line status, case, selected algorithm, source file, top
  finding, triggering event id, source location, rank/server location, mismatch
  count, expected/actual values for output mismatches, mismatch context,
  `server scope` (`same_server`, `cross_server`, or `unknown`), and for timing
  findings a short timeline pointer with related-event count and the first
  useful relation.
- `findings.json`: structured findings for ordering hazards, stale flags,
  missing producers, mutual flag-wait deadlocks, pipe waits without producer
  events, peer user-buffer reads, output mismatches, and unsupported cases.
- `events.jsonl`: trace event stream, one JSON object per line.
- `checker_report.json`: machine-readable run summary, topology, top finding,
  algorithm selection explanation, source excerpt around the top finding when
  the source file is locally available, a compact `event_context` window around
  the triggering event, a related `top_finding.timeline`, artifact basenames,
  and full `artifact_paths` for downstream tools. In `--validate-event-trace`
  mode it also records `mode:"event_trace_validation"` and the original
  `event_trace_input` path.

Start with `summary.txt`. If it fails, use its `timeline events` and
`timeline first relation` lines to decide whether the first pass is a flag-slot,
pipe-event, source-nearby, or raw-nearby problem. Then use `findings.json` for
the first actionable source location,
`checker_report.json.top_finding.source_excerpt` for the exact source snippet
when the path exists locally, and
`checker_report.json.top_finding.event_context` for the nearby trace events. For
flag and pipe timing problems, inspect
`checker_report.json.top_finding.timeline` next: it keeps the trigger plus
related `same_flag_slot`, `same_pipe_event`, `near_source`, and `near_event`
entries with source line, rank/server, slot, pipe, event id, magic epoch, and
magic value. Only then fall back to `events.jsonl` for the full raw stream around
that rank, peer, server, peer server, event id, slot, offset, and source line.
`checker_report.json.top_finding.server_scope` mirrors the summary and should
be the first filter for multi-machine ordering failures.
Relative source paths in findings are resolved against `--repo-root` when
building `source_excerpt`, so reports stay useful even when the checker is run
from a build directory or another working directory.
`--validate-event-trace` accepts the same JSONL records emitted by
`events.jsonl`; required fields are `kind`, `rank`, `peer_rank`, `server`,
`peer_server`, `core`, `pipe`, `event_id`, `buffer_role`, `slot`, `magic`,
`offset`, `bytes`, `allow_future_producer`, `source_file`, `source_line`, and
`detail`.
For output mismatches, `summary.txt`, `findings.json`, and
`checker_report.json.top_finding` all include `expected`, `actual`, and
`context` so a bad operator can be triaged without first opening the raw event
stream. When the event log contains a `WRITE` or `COPY` covering the mismatched
user-output offset, the mismatch finding is also linked to that producer event,
so the summary source location and `event_context` point at the suspected write
path instead of `<unknown>`.

## Source Preservation

Production collective and EP sources must remain checker-free. Checker support
may wrap or compile those headers with checker include paths, but must not
require edits in production operator headers.

This invariant is enforced by source-preservation tests and production header
probe tests. If a checker change requires adding checker-only symbols inside a
production kernel header, it is a design regression.

## EP Coverage

Current EP support covers dispatch and combine at the source-aligned behavior
level:

- deterministic rank/token/hidden payload generation;
- deterministic expert-id routing with invalid-route filtering semantics;
- per-destination expanded payload, assist tuple, receive-count, and local
  expert-count oracles;
- combine-side inverse routing from dispatch payload/assist metadata back to
  each source rank's `[bs, top_k, h]` output layout;
- assist, receive-count, and local expert-count trace events;
- multi-server topology fields on route events;
- reports that identify the EP kernel source file and nearby event context;
- registered-window read/write ordering diagnostics for combine-side checks.

Future EP work should add stale-window reuse checks, route-bound failure
injection, and direct source-preserving trace adapters for any EP headers that
can be compiled under checker shims. If a production
`tilexr_ep_combine_kernel` body is added later, the checker should install a
source-preserving runner or generated trace manifest for that implementation;
the current combine support is a checker CPU oracle, not direct AscendC
instruction-level execution.
