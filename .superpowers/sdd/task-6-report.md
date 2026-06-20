Status: DONE

Summary:
- Added `CollectiveExecutor` with `RunResult` and MVP semantic execution for INT32 `AllGather` and SUM `AllReduce`.
- Kept execution deterministic and aligned event production with the existing Task 4/5 diagnostics model.
- Added executor unit coverage for pass cases, unsupported datatype handling, and test-only failure injection helpers.

Files changed:
- `tools/checker/include/tilexr/checker/executor.h`
- `tools/checker/src/executor.cpp`
- `tests/checker/unit/test_checker_executor.cpp`
- `tools/checker/CMakeLists.txt`
- `tests/checker/CMakeLists.txt`

TDD record:
1. Added `test_tilexr_checker_executor.cpp` and target wiring first.
2. Verified red failure via missing `src/executor.cpp` during CMake generation.
3. Implemented the executor and iterated until the requested checker test set passed.

Implementation notes:
- `Run()` now validates the case, clears prior events, fills deterministic rank-pattern inputs, executes the selected collective, compares outputs, merges ordering and output findings, and returns `Ok()` only when there are no error findings and no mismatches.
- Unsupported cases return `CheckerStatus::kUnsupported` and include an `UNSUPPORTED_API` finding.
- Publish/read semantics use `CommData(owner=rank, slot=rank)` as the physical location for each rank's staged payload.
- Read events for peer comm data are emitted directly with `peer_rank=owner` and `slot=owner` so diagnostics observe the real producer/consumer mapping.
- Scheduler behavior is deterministic phase ordering across all ranks. This satisfies the brief's conservative MVP guidance and keeps pass-path diagnostics clean for both scheduler modes covered by tests.

Verification:
- Command:
  `bash -lc 'source scripts/common_env.sh >/tmp/tilexr_checker_env.out && cmake -S . -B build-checker -DTILEXR_BUILD_CHECKER=ON -DBUILD_TESTING=ON && cmake --build build-checker --target test_tilexr_checker_executor test_tilexr_checker_diagnostics test_tilexr_checker_oracles -j"$(nproc)" && ctest --test-dir build-checker -R "test_tilexr_checker_executor|test_tilexr_checker_diagnostics|test_tilexr_checker_oracles" --output-on-failure'`
- Result: pass

Self-review:
- Confirmed write scope stays within the five Task 6 files from the brief.
- Confirmed failure injection helpers exist only in `tests/checker/unit/test_checker_executor.cpp`.
- Confirmed `.superpowers/sdd/task-6-report.md` is created for handoff and remains untracked.

Concerns:
- `SchedulerMode::kSerial` and `kRoundRobin` currently share the same phase-by-phase execution shape; both are deterministic and tested, but the MVP does not yet distinguish them with different event schedules beyond supported interface handling.

Fix for review findings:
- What you changed:
  - Made `SchedulerMode` observable in `tools/checker/src/executor.cpp` by splitting execution into deterministic `kSerial` rank-major order and deterministic `kRoundRobin` phase-major order.
  - Hardened `CollectiveExecutor::Run()` so non-ok semantic executor results return immediately with merged findings and `event_count`, without calling `CompareInt32Output()`.
  - Expanded `tests/checker/unit/test_checker_executor.cpp` to assert event kind/rank ordering for serial vs round-robin AllGather, prove the schedules differ, and verify unsupported execution returns zero mismatches.
  - Seeded comm-data storage from filled inputs before event emission so serial execution can remain deterministic without relying on phase-major producer timing.
- Verification command and result:
  - `bash -lc 'source scripts/common_env.sh >/tmp/tilexr_checker_env.out && cmake -S . -B build-checker -DTILEXR_BUILD_CHECKER=ON -DBUILD_TESTING=ON && cmake --build build-checker --target test_tilexr_checker_executor test_tilexr_checker_diagnostics test_tilexr_checker_oracles -j"$(nproc)" && ctest --test-dir build-checker -R "test_tilexr_checker_executor|test_tilexr_checker_diagnostics|test_tilexr_checker_oracles" --output-on-failure'`
  - Result: pass
- Files changed:
  - `tools/checker/src/executor.cpp`
  - `tests/checker/unit/test_checker_executor.cpp`

Serial ordering follow-up (2026-06-20):
- Failing test observed before the fix:
  - Command:
    `bash -lc 'source scripts/common_env.sh >/tmp/tilexr_checker_env.out && cmake -S . -B build-checker -DTILEXR_BUILD_CHECKER=ON -DBUILD_TESTING=ON && cmake --build build-checker --target test_tilexr_checker_executor -j"$(nproc)" && ctest --test-dir build-checker -R test_tilexr_checker_executor --output-on-failure'`
  - Expected failure snippet:
    `serial injected peer run fails: actual=0 expected=1`
    `serial injected stale run fails: actual=0 expected=1`
- Final fix summary:
  - `CollectiveExecutor::RunAllGatherInt32()` and `RunAllReduceSumInt32()` now always feed executor-owned traces through `CheckOrdering()`, including `SchedulerMode::kSerial`.
  - Removed the hidden comm-data pre-seed path. The executor now emits real publish copies before any consumer reads, so the trace stays causally valid.
  - `kSerial` and `kRoundRobin` remain deterministic but differ in read-phase order: serial drains reads/writes per rank after all publishes/stores/waits, while round-robin interleaves ranks by peer during reads.
  - Executor tests now inject ordering faults through `CollectiveExecutor::SetPostTraceHookForTest()` so `Run()` itself surfaces serial findings.
- Final verification:
  - Command:
    `bash -lc 'source scripts/common_env.sh >/tmp/tilexr_checker_env.out && cmake --build build-checker -j"$(nproc)" && ctest --test-dir build-checker -R "test_tilexr_checker_executor|test_tilexr_checker_diagnostics|test_tilexr_checker_oracles" --output-on-failure'`
  - Result:
    `100% tests passed, 0 tests failed out of 3`
- Commit hash:
  - pending
- Concerns:
  - The test seam is intentionally small, but it does live in the production executor header for now because there was no existing executor-owned injection hook.
