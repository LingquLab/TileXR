# MoonEP Model Replay Cache Execution Plan

## Scope

Implement the design in
`docs/specs/2026-08-16-moonep-model-replay-cache-design.md` on the isolated task
worktree. Preserve existing cases, MoonEP APIs, C++14/CANN 9.1 compatibility, and
runtime safety. Do not import historical kernel changes or generated route fixtures.

## Ordered Work

1. Add focused Python tests for canonical cache identity, manifest validation,
   all-rank capture completeness, checksum failures, atomic publication, cache hit,
   miss, and force refresh. Add shell tests for required/prompted S/K/H/EP/R and
   argument forwarding.
2. Implement a small cache module responsible only for canonical identity, artifact
   validation, and atomic publication. Keep model launching outside this module.
3. Port the focused historical adapter capture hook and replay builder, generalized to
   shape metadata and strict manifest validation. Keep it opt-in through environment
   variables and preserve all public return contracts.
4. Port the user-level model-flow replay engine and 55-call order contract. Replace
   case-ID/static-fixture lookup with a supplied validated cache artifact. Reuse current
   benchmark timers and report byte formulas.
5. Extend the MindSpeed controller/node runner with explicit shape, capture directory,
   capture ID, and profiler controls. Run route capture once on miss, build and publish
   the cache, then run a capture-off model performance sample.
6. Integrate `--model-replay` into `run_moonep.sh`. Validate or prompt for the five
   shape values before environment/device setup, resolve hit/miss/refresh, launch the
   model capture when required, run single-node or multi-node replay, aggregate every
   rank, and print the comparison on rank 0.
7. Extend report aggregation and terminal rendering for ordered model-flow comparisons
   without changing current six-stage and dispatch-hot-loop output.
8. Run focused pytest suites, Bash syntax checks, Python compile checks, cache failure
   injection, final diff review, and an explicit unchanged-call-surface comparison for
   `tools/moonep/test_npu_e2e.py`. Record reusable lessons in the maintained MoonEP
   runner/debugging documentation.
9. Inspect `.195` and `.223` environment provenance and retained HCCL evidence. Do not
   rerun official HCCL Test when a matching one-time record exists. Deploy Windows to
   `.195` with Mutagen only, then Linux to `.223` with rsync without destructive delete
   flags.
10. Validate the four requested configurations, including miss, hit, and forced
    refresh. Preserve all-rank correctness, model logs, cache manifests, loaded-library
    hashes, and performance tables. Diagnose failures from the first failing boundary
    with one-variable checks. Do not run `mpirun test`.
11. Commit the first fully passing hardware state immediately. Audit DFX, trace, dump,
    stage barrier, framework profiler, and compile-time profiling switches. Retain the
    profiler option, rerun profiler-off model and replay, compare material stage gaps,
    document the evidence, and create the second scoped commit.
12. Fetch and rebase onto latest `origin/main`. If source changes, repeat the affected
    local and hardware checks. Generate one patch relative to latest main and report
    commands, artifacts, commits, comparison tables, and residual validation limits.

## Verification Gates

- Local behavior and failure-path tests pass before remote deployment.
- No generated capture or model output is tracked by Git.
- The six MoonEP API calls in `test_npu_e2e.py` are byte-for-byte unchanged.
- A valid cache hit provably avoids a model rerun.
- A failed forced refresh leaves the old valid cache readable.
- Hardware results include every rank and distinguish profiler-on from profiler-off.
- Final patch contains only the task implementation, tests, and maintained docs.
