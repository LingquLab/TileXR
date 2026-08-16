# MoonEP Combine V2 Continuous Iteration Plan

## Goal

Remove all test-harness work between timed Combine V2 launches. Align ranks once,
submit the complete timed batch continuously, synchronize once, and report each
rank's batch time divided by the iteration count.

## Scope

1. Refactor the hardware benchmark to batch warmup and timed launches. Keep one
   pre-batch rank barrier and capture only the final profiling record.
2. Aggregate `COMBINE_V2_RANK_PERF avg_ms` directly while retaining support for
   older logs that contain per-iteration samples.
3. Mark final-iteration rank selection as kernel-profile timing rather than ACL
   Event timing and keep the three-file Chrome Trace export contract.
4. Update the maintained performance and profiling guides.

## Constraints

- No barrier, synchronization, D2H copy, profile parsing, logging, or event task
  may occur between timed kernel launches.
- Correctness remains independent from performance reporting.
- Profiling reports iteration `iterations - 1` only.
- Existing historical logs and trace fixtures remain readable.

## Verification

- Run the focused Python trace tests.
- Run source-level assertions for the continuous benchmark loop and rank-perf
  aggregation contract.
- Build the focused host target when the local environment supports it; hardware
  performance validation remains a separate remote test.
