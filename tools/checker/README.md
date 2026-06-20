# TileXR Checker

The checker is an opt-in, no-NPU verification surface for TileXR collective semantics. It executes a small software model of selected collective schedules, emits deterministic diagnostics, and preserves the production collective sources as the source of truth.

## Purpose

- Verify selected collective invariants without requiring CANN runtime setup or NPU hardware.
- Provide a narrow CLI for reproducible pass/fail runs and report artifact generation.
- Catch source-preservation regressions so checker support does not leak into production collective headers.

## Non-goals

- The checker is not a replacement for hardware validation, CANN runtime validation, or collectives performance testing.
- The checker does not execute production AscendC kernels directly in the MVP.
- The checker does not require a CANN CPU-domain twin debugging flow; cpudebug integration is an optional future adapter, not part of the required workflow.

## Build

The checker is off by default. Enable it explicitly:

```bash
source scripts/common_env.sh
cmake -S . -B build-checker -DTILEXR_BUILD_CHECKER=ON -DBUILD_TESTING=ON
cmake --build build-checker -j"$(nproc)"
```

The CLI binary is produced at `build-checker/tools/checker/tilexr_checker`.

## CLI

Usage:

```text
tilexr_checker --op <allgather|allreduce> --rank-size <n> --count <n> \
  --datatype int32 [--reduce-op sum] [--scheduler <serial|round_robin>] \
  [--output-dir <dir>] [--inject-read-before-copy] [--inject-peer-user-read]
```

Examples:

```bash
build-checker/tools/checker/tilexr_checker \
  --op allgather \
  --rank-size 4 \
  --count 16 \
  --datatype int32 \
  --scheduler round_robin \
  --output-dir /tmp/tilexr-checker-smoke
```

```bash
build-checker/tools/checker/tilexr_checker \
  --op allreduce \
  --reduce-op sum \
  --rank-size 2 \
  --count 16 \
  --datatype int32 \
  --scheduler serial \
  --output-dir /tmp/tilexr-checker-allreduce
```

```bash
build-checker/tools/checker/tilexr_checker \
  --op allgather \
  --rank-size 2 \
  --count 16 \
  --datatype int32 \
  --scheduler serial \
  --inject-peer-user-read \
  --output-dir /tmp/tilexr-checker-fail
```

The failure injection flags are for checker diagnostics only:

- `--inject-read-before-copy` inserts a read-before-copy hazard.
- `--inject-peer-user-read` inserts a peer-visible user-buffer read hazard.

## MVP Matrix

The current MVP support is intentionally explicit and narrow:

| op | datatype | ranks | reduce-op | schedulers |
| --- | --- | --- | --- | --- |
| AllGather | int32 | 2, 4 | n/a | serial, round_robin |
| AllReduce | int32 | 2 | sum | serial, round_robin |

Any other combination is outside the documented MVP contract and should be treated as unsupported.

## Report Artifacts

Each checker CLI run writes four files to `--output-dir`:

- `summary.txt`: human-readable top-line status, case description, event count, mismatch count, and the leading finding.
- `findings.json`: structured findings list for hazards such as read-before-copy or peer user-buffer misuse.
- `events.jsonl`: event stream emitted by the checker executor, one JSON object per line.
- `checker_report.json`: machine-readable summary that records status plus the artifact file names for the run.

`summary.txt` is the fastest artifact to inspect first; `findings.json` and `events.jsonl` are the detailed follow-up surfaces when a run fails.

## Source Preservation Invariant

Production collective and EP sources must remain checker-free. The checker is allowed to model those sources, wrap them, or validate their ownership boundaries, but it must not require edits inside the production kernel headers to function.

This invariant is enforced by checker tests such as source-preservation and header-probe coverage. If a checker change requires production-source instrumentation to pass, that is a design regression.

## EP Extension Boundary

EP support is a future extension that should reuse the checker infrastructure rather than widening the MVP surface in place. The intended boundary is:

- add an `EpExecutor` on top of the same `RankWorld`, `EventLog`, `FindingSet`, and report writer primitives;
- keep EP-specific invariants focused on route bounds, expert-count conservation, dispatch/combine readiness, peer-visible window slots, and stale-window reuse;
- preserve the same source-preservation rule for production EP sources.

Until that executor exists, this README documents collective-checker MVP behavior only.

## CPU-Domain Twin Debugging

CANN CPU-domain twin debugging is not required for checker development or usage. A cpudebug-backed adapter may be added later as an auxiliary event source, but the checker runtime and MVP verification flow must remain usable without it.
