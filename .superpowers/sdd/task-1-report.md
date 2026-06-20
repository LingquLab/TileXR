# Task 1 Report: Add Opt-In Checker Build Surface

## Scope

Implemented the opt-in checker build surface only. The change adds `TILEXR_BUILD_CHECKER`, wires `tools/checker` and `tests/checker` under that option, and adds the first source-guard smoke test.

## Changes

- Added `option(TILEXR_BUILD_CHECKER "Build TileXR no-NPU checker" OFF)` to the root `CMakeLists.txt`.
- Added gated subdirectory wiring for `tools/checker` and `tests/checker`.
- Added `tools/checker/CMakeLists.txt` with the `tilexr-checker-core` interface target.
- Added `tools/checker/README.md` describing scope, no-NPU intent, and the later MVP operator scope.
- Added `tests/checker/CMakeLists.txt` with the `tilexr_checker_test` helper and `test_tilexr_checker_sources`.
- Added `tests/checker/unit/test_checker_sources.cpp` to assert the checker surface is present in the source tree.

## Verification

Ran the task verification command from the brief:

```bash
bash -lc 'source scripts/common_env.sh >/tmp/tilexr_checker_env.out && cmake -S . -B build-checker -DTILEXR_BUILD_CHECKER=ON -DBUILD_TESTING=ON && cmake --build build-checker -j"$(nproc)" && ctest --test-dir build-checker -R test_tilexr_checker_sources --output-on-failure'
```

Result: configure, build, and `test_tilexr_checker_sources` all passed.

## Notes

- I first confirmed the checker option was missing by configuring with `-DTILEXR_BUILD_CHECKER=ON`; CMake warned that the variable was unused, which matched the expected pre-change state.
- Temporary build directories created during verification were removed before finalizing the task.
