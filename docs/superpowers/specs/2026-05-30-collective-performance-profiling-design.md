# TileXR Collective Performance Profiling Design

## Context

Communication operator development needs an operator-internal profiling path that can show where time is spent across stages, ranks, and AICore blocks. TileXR already has basic DFX fields in `CommArgs` and a legacy `dumpAddr`/`LcclDumpLogInfo` path in collective kernels, but that path is event-like, raw, and not sufficient for structured bottleneck analysis or multi-rank visualization.

This design targets the collectives implementation under `src/collectives` as the first integration point. The data format and helper APIs should remain generic enough for later use by MC2 or other AscendC kernels.

## Goals

- Measure operator-internal collective stages with `AscendC::GetSystemCycle()`.
- Convert cycles to microseconds with chip-aware rules: A5/Ascend950-class paths use `cycles / 1000`, other supported chips use `cycles / 50`.
- Visualize multiple ranks and cores in one report.
- Make bottleneck analysis the primary report entry point, with timeline drilldown available.
- Keep disabled profiling out of the production fast path.
- Support multiple cores executing the same operator without shared kernel-side update contention.
- Account for AscendC asynchronous instructions by placing barriers around measured regions when accurate timing requires them.
- Provide deterministic offline analysis and optional AI-assisted analysis.

## Non-Goals

- First release does not fully instrument every collective algorithm branch.
- First release does not provide a live web service or remote dashboard.
- First release does not require a model service to produce useful analysis.
- First release does not replace msprof; it provides operator-specific internal stage timing.

## Selected Approach

Add a reusable `TileXRPerfTrace` data plane. Host code creates an optional trace session, allocates a device trace buffer, passes a trace control pointer to the collective kernel launch, copies the buffer back after stream synchronization, and emits JSON/CSV/HTML/analysis artifacts.

Kernel code updates fixed per-rank/per-core/per-stage aggregate slots. It does not append unbounded events and does not merge across cores inside the kernel. Host-side parsing computes rank, global, and imbalance summaries.

The first complete instrumentation target is the collectives big-data AllGather branch implemented in `src/collectives/kernels/kernels/lcal_allgather_big_data.cce`.

## Build and Runtime Control

Profiling uses two gates:

1. Compile-time gate: default OFF. A new CMake option such as `TILEXR_COLLECTIVES_ENABLE_PROFILING=ON` adds a CCE macro and includes kernel profiling helpers. With the option OFF, helper calls compile away and the production kernel has no profiling branch.
2. Runtime gate: profile builds check whether a valid trace control block is present and enabled. This allows the same profile build to run with sampling disabled, enabled, or selectively configured.

This split provides a zero-overhead production build and a practical debug build that does not require source edits to turn capture on or off.

## Trace Data Format

Add a public, versioned header such as `src/include/tilexr_perf_trace.h`.

`TileXRPerfTraceHeader` records:

- magic and version
- total buffer size and schema sizes
- rank, rank size, block dimension, max core count
- launch id, operator type, data type, message size
- SOC/chip class and `cycleToUsDivisor`
- flags for enabled features and sampling mode

`TileXRPerfStageDesc` records:

- stage id
- short stage name
- category, for example copy, wait, sync, total
- barrier policy
- display order

`TileXRPerfCoreStageStats` records one aggregate slot:

- rank
- core/block id
- stage id
- count
- sum cycles
- min cycles
- max cycles
- first start cycle
- last end cycle
- auxiliary counters for loop iterations, poll counts, or bytes when useful

The primary stats layout is fixed:

```text
stats[rank][core][stage]
```

The kernel computes offsets directly. Each core writes only its own slots. Host tools aggregate after copying the trace buffer back.

Raw cycles are always preserved in JSON. Microsecond values are derived during parsing with `cycleToUsDivisor`.

## Kernel Helper Semantics

Kernel helpers live in a collectives-owned kernel header at first, but use only the generic data structures.

The core operations are:

- `StageBegin(trace, stageId, policy)`
- `StageEnd(trace, stageId, token, policy)`
- `AccumulateDuration(trace, stageId, startCycle, endCycle)`
- optional counter helpers for bytes, loop iterations, and polling attempts

`StageEnd` updates the current rank/core/stage slot:

- increments `count`
- adds duration to `sumCycles`
- updates `minCycles` and `maxCycles`
- records `firstStartCycle` if unset or earlier
- records `lastEndCycle` if later

Barrier policy is explicit:

- `Barriered`: `PipeBarrier<PIPE_ALL>()` before begin and before end timing. Use for stages that include asynchronous `DataCopy`, `CpGM2UB`, `CpUB2GM`, vector, or mixed-pipe work and must be accurately measured.
- `EndBarrierOnly`: use when the beginning is already synchronized but the end must wait for outstanding asynchronous work.
- `NoBarrier`: use around pure scalar polling, already-barriered regions, or metadata-only sections.

The first implementation should avoid adding redundant barriers where the surrounding collective code already has a `PipeBarrier<PIPE_ALL>()`.

## First Instrumented Branch

The first complete stage schema targets `TileXRAllGatherBigData`.

Initial stages:

- `kernel_total`
- `chunk_total`
- `post_sync`
- `local_input_to_ipc`
- `flag_poll_wait`
- `peer_ipc_to_output`
- `chunk_barrier`

The branch has two core roles:

- producer cores copy local input into the rank IPC buffer and publish progress flags
- consumer cores poll peer progress flags and copy ready peer IPC data into output

The trace must preserve that distinction in either stage categories or auxiliary counters so reports can separate local publishing cost from peer read/wait cost.

Other collectives should receive only the common launch plumbing in the first release. They may pass a null trace control pointer or record only a coarse unsupported/inactive status. Full stage instrumentation is deferred.

## Host API and Integration

Existing `TileXRAllGather` and `TileXRAllToAll` APIs remain unchanged.

Add optional profiling APIs or a small C++ wrapper for tools:

- `TileXRCollectivePerfConfig`
- `TileXRCollectivePerfSessionCreate`
- `TileXRCollectivePerfSessionDestroy`
- `TileXRCollectivePerfBeginLaunch`
- `TileXRCollectivePerfEndLaunch`
- `TileXRCollectivePerfWriteReport`

`TileXRCollectivePerfConfig` includes:

- enabled
- output directory
- sample interval, for example `sampleEveryN`
- whether to emit timeline data
- whether to emit AI prompt/export
- optional AI command hook

`AscendCCLKernelArgs` gains a trailing optional trace control pointer. This is an ABI-sensitive change inside TileXR's private host/kernel boundary, not a public C API change. Host launch fills the pointer only when a session is active and the profile build supports it.

The collectives perf tool should expose this through command-line flags such as:

- `--profile 0|1`
- `--profile-dir PATH`
- `--profile-ai-prompt 0|1`
- `--profile-sample-every N`

Environment variables may mirror these flags for script-driven runs.

## Report Artifacts

Default output directory:

```text
${TILEXR_PROF_HOME}/collectives/<timestamp>-<op>-rank<rank>/
```

If `TILEXR_PROF_HOME` is unset, use `run/prof/collectives/...` under the repository or current run directory.

Artifacts:

- `trace.json`: complete structured trace, schema, metadata, raw cycles, derived microseconds, and per-rank/core/stage stats.
- `summary.csv`: flat stage/rank/core summary for command-line analysis.
- `analysis.md`: deterministic offline bottleneck analysis.
- `report.html`: static bottleneck-first report with timeline drilldown.
- `ai_prompt.md`: optional compact prompt for AI-assisted analysis.
- `ai_analysis.md`: optional output from an AI command hook.

`report.html` starts with:

- top bottleneck stages
- slowest rank and core
- rank imbalance
- stage time percentage
- abnormal wait or sync patterns

Timeline drilldown shows rank/core rows and stage bars using `firstStartCycle` to `lastEndCycle`. Because the first release stores aggregates instead of all loop events, timeline is stage-level rather than per-iteration.

## Offline and AI Analysis

Offline analysis is mandatory and deterministic. It should flag:

- stages with high percentage of total time
- high rank imbalance
- high core imbalance within a rank
- wait-heavy profiles, especially `flag_poll_wait`
- copy-heavy profiles with low imbalance, indicating possible bandwidth limitation
- sync-heavy profiles, indicating progress skew or over-synchronization

AI analysis is optional. The tool generates a compact `ai_prompt.md` containing:

- launch metadata
- top bottlenecks
- rank/core outliers
- relevant stage schema
- offline analysis findings
- a short explanation of cycle conversion and profiling limitations

The prompt must not include the full trace JSON by default. If `TILEXR_PERF_AI_CMD` or an equivalent config is set, the report tool pipes the prompt to that command and saves the result as `ai_analysis.md`.

## Testing

Host-only tests:

- trace header and stats layout offsets
- cycle-to-us conversion for A5/Ascend950 and non-A5 chips
- summary aggregation from synthetic rank/core/stage data
- bottleneck rule analysis
- JSON/CSV/HTML artifact generation
- AI prompt generation and size limiting

Source/compile tests:

- default build does not require profiling symbols in kernels
- profile build includes stage schema and helper calls for `lcal_allgather_big_data.cce`
- public headers compile under C++14 host code

Hardware/manual tests:

- run `tilexr_collective_perf --op allgather --profile 1` on a message size that selects the big-data AllGather branch
- verify trace files exist for every rank
- verify `flag_poll_wait` and `peer_ipc_to_output` show nonzero values where expected
- compare non-profile build, profile build with runtime disabled, and profile build with runtime enabled to estimate overhead

## Open Implementation Notes

- The implementation should reuse existing `TILEXR_PROF_HOME` environment setup from `scripts/common_env.sh`.
- `.superpowers/` visual companion files are development artifacts and must not be committed.
- The legacy `dumpAddr` path should not be removed in this first release unless it directly conflicts with the new trace control block.
- Stage names should be stable strings in host-side schema rather than hard-coded only in the report tool.
- The trace format version should be bumped whenever struct layout changes.

## Implementation Verification

Implementation plan verification requires host-only collectives tests, profile-build compilation, and an optional hardware run of `tilexr_collective_perf --profile 1` on the big-data AllGather path. Hardware verification is reported as skipped when no usable NPU devices are available.
