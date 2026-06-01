# TileXR Collective Profile Timeline Aggregation Design

## Context

TileXR collectives already have an opt-in operator-internal profiling path. A profile run writes one report directory
per sampled measured launch:

```text
<profile-dir>/rank<N>/launch<M>/{trace.json,summary.csv,analysis.md,report.html,ai_prompt.md}
```

The current `report.html` is useful for bottleneck summaries, but its timeline is a table of aggregate stage
bounds. It is hard to compare multiple measured iterations, inspect fluctuation, or see rank/core lanes as a
timeline.

The perf runner already supports warmup through `--warmup-iters`. Warmup execution should remain unchanged. The
new work only records the warmup count in run-level metadata when available and focuses the visual timeline on
sampled measured iterations.

## Goals

- Add a more visual chronological timeline for sampled measured iterations.
- Support zoom in, zoom out, fit-to-view, and horizontal panning in the static HTML report.
- Show multiple sampled launches in the same timeline by default, ordered as `launch0`, `launch1`, `launch2`, and
  so on.
- Preserve existing per-launch report artifacts and single-launch tests.
- Support multi-rank visualization by aggregating `rank*/launch*/trace.json` after all rank processes finish.
- Keep kernel profiling overhead unchanged. Aggregation happens after the measured run.
- Keep the report self-contained and usable on a remote machine without CDN or frontend build tooling.

## Non-Goals

- Do not change warmup execution or profile warmup launches in the first iteration.
- Do not add per-event kernel trace rings in this step. The first implementation uses existing aggregate stage
  stats: `first_start_cycle`, `last_end_cycle`, `sum_cycles`, `count`, `min_cycles`, and `max_cycles`.
- Do not replace existing single-launch `report.html` files.
- Do not depend on synchronized raw cycle counters across NPUs.

## Selected Approach

Use a report-layer aggregation path.

Each `tilexr_collective_perf` rank process continues to write per-launch artifacts exactly as it does now. After
`run_collective_perf.sh` observes that all rank processes have exited successfully, it invokes a lightweight
profile aggregation helper for the profile root. The helper reads all available trace JSON files, groups compatible
launches, and writes a root-level run report.

This keeps the existing profiling data path stable and avoids a race where rank 0 tries to read files while other
ranks are still writing them.

## Artifacts

For a profile root such as `run/prof/collectives`, the aggregation helper writes:

```text
run/prof/collectives/report.html
run/prof/collectives/trace_index.json
run/prof/collectives/analysis.md
run/prof/collectives/ai_prompt.md    # only when AI prompt export is requested
```

The existing per-launch files remain under `rank<N>/launch<M>/`.

`trace_index.json` is the machine-readable run-level artifact. It records:

- schema name, such as `tilexr_perf_trace_run.v1`
- profile root
- warmup iteration count when known
- measured iteration count when known
- profile sample interval
- rank count inferred from traces
- groups keyed by operation, rank size, message bytes, stage count, and cycle conversion divisor
- launches inside each group
- source trace path for each rank/launch
- normalized timeline bars for each stage/rank/core
- missing or invalid trace diagnostics

The helper should tolerate extra fields in existing `trace.json` files. Host-side trace JSON may add optional
fields such as `rank`, `launch_id`, and `data_type` for clarity, but the first aggregation path can infer rank and
launch from directory names.

## Timeline Semantics

The default view is chronological. A group with sampled launches `launch0`, `launch1`, and `launch2` appears as
three adjacent sections on one horizontal timeline. Launch labels and vertical separators make iteration boundaries
visible.

Within each launch, bars are normalized to a local zero point. Because raw `GetSystemCycle` values from different
NPUs are not assumed to be globally synchronized, the report must not interpret cross-rank raw cycle offsets as
true host-side launch skew. Rank/core comparisons should use duration, wait time, and stage composition. The report
can still display raw cycles in tooltips for debugging.

For each launch and rank, `launch_rank_zero` is the smallest nonzero `first_start_cycle` among that rank's stats in
that launch. This keeps each rank lane internally ordered without claiming cross-device clock alignment.

The timeline bars use existing aggregate fields:

- visual start: `first_start_cycle - launch_rank_zero`
- visual end: `last_end_cycle - launch_rank_zero`
- duration: `last_end_cycle - first_start_cycle`
- recorded work time: `sum_cycles`
- repeat count: `count`
- max event time: `max_cycles`

For stages with many repeated events, such as `flag_poll_wait`, the bar spans the first-to-last observed window and
the tooltip shows that `sum_cycles` may be smaller than the span.

## Report UX

The root `report.html` starts with bottleneck-first summaries:

- operation and message bytes
- sampled launch count and expected measured iteration count when known
- warmup iteration count when known
- slowest launch by max `kernel_total`
- top bottleneck stage by summed microseconds
- rank/core with the largest stage max
- fluctuation summary across sampled launches

The visual timeline follows the summary. Required controls:

- zoom in
- zoom out
- fit all
- horizontal pan by dragging the timeline or using a scrollbar
- stage filters
- rank and core lane folding
- launch selection

The report should use stable stage colors by category:

- total stages
- copy stages
- wait stages
- sync/barrier stages

Hovering a bar shows:

- launch id
- rank
- core
- stage name
- start and end in normalized cycles or microseconds
- duration
- `sum_us`
- `count`
- `max_cycles`
- source `trace.json`

Clicking a bar or launch label opens the existing per-launch `report.html` for drilldown.

The report is static HTML with embedded JSON and vanilla JavaScript/CSS. It should not require npm, webpack, CDN
assets, or an internet connection.

## Runner Integration

`run_collective_perf.sh` is the safe place to trigger root aggregation because it already waits for all rank
processes. It should parse enough forwarded arguments to know:

- whether profiling is enabled
- the profile root directory
- warmup iteration count
- measured iteration count
- profile sample interval
- whether AI prompt export was requested

If profiling is enabled and a profile root is known, the script calls the aggregation helper after all rank
processes finish with status zero.

Direct invocation of `tilexr_collective_perf` remains supported. It continues to write per-launch reports. Users can
run the aggregation helper manually against an existing profile directory.

## Error Handling

The aggregation helper should prefer a partial report with clear diagnostics over all-or-nothing failure.

It should:

- list missing rank/launch traces in the report and `analysis.md`
- reject incompatible traces within one group instead of mixing them silently
- preserve sparse launch ids when `--profile-sample-every > 1`
- handle one rank or one launch as a valid degenerate case
- keep per-launch artifacts untouched if root report generation fails
- produce a static fallback table in `report.html` so useful data remains visible if JavaScript fails

Incompatibility checks include:

- unsupported schema
- inconsistent operation
- inconsistent rank size
- inconsistent message bytes
- inconsistent stage count
- inconsistent cycle-to-microsecond divisor inside one group

## AI Analysis Export

The run-level `ai_prompt.md` should summarize variance and outliers across sampled measured iterations. It should
include:

- run configuration and warmup count when known
- launch count and sampled launch ids
- top bottleneck stages
- slowest launch
- rank/core outliers
- missing trace diagnostics
- a short note that cross-NPU raw cycle offsets are not assumed to be synchronized

The prompt should stay compact enough to paste into an assistant without including full raw trace data.

## Tests

Host-only tests should cover the aggregation helper using generated fixture directories:

- multi-rank, multi-launch happy path
- sparse launch ids from `profile_sample_every > 1`
- missing rank trace
- invalid schema
- incompatible message bytes
- single rank and single launch fallback
- generated `trace_index.json` contains chronological launch sections
- generated `report.html` contains zoom controls, launch labels, stage filters, and drilldown links
- generated `analysis.md` reports bottleneck and fluctuation summaries

Existing C++ report tests should remain in place to protect single-launch reports. Source checks should confirm that
the shell runner invokes aggregation only after all ranks have completed successfully.

Hardware verification should run a profile build on at least:

- 2-rank AllGather with multiple measured iterations
- 4-rank AllGather with multiple measured iterations

Verification should confirm that the root report contains multiple launch sections, rank/core lanes, working
drilldown links, and no additional kernel-side profiling overhead.
