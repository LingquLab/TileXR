# Combine V2 Step0 Detailed Profiling Plan

## Goal

Expand MoonEP Combine V2 profiling so the Step0 send interval reports named,
accumulated costs for selection, self-copy, and remote WQE submission while
preserving the existing cumulative timestamps and failure diagnostics.

## Scope

- Version the internal profile record and add eight cycle counters.
- Instrument the profiling-only kernel path with non-overlapping counters for
  selection load/select, self route/copy, and remote route/descriptor/build/
  publish work.
- Copy and print the counters as named microsecond fields in the hardware probe.
- Update layout and source-contract tests for the new record contract.
- Build and run the existing focused tests, then build and validate the profile
  artifact on the approved 198/226 16P environment using Mutagen source sync.

## Non-Goals

- Do not optimize the kernel in this change.
- Do not change the public Combine V2 launch ABI or result calculations.
- Do not use detailed-profile timings as the final non-profile performance result.

## Implementation Order

1. Extend `combine_v2_profile.h` with profile v2 metric indices and capacity.
2. Add profiling-only metric accumulation to the kernel and serialize it.
3. Extend the hardware probe sample validation and named output.
4. Update focused unit expectations and add source-contract coverage.
5. Run local/static tests, compile remotely, sync the runtime from the primary
   server, and run the 16P BS=8192 profile case.

## Risks And Verification

Per-row cycle reads perturb profile builds, so counters are compiled out when
profiling is disabled and are used only for attribution. Profile record size is
an internal workspace ABI; host, kernel, and probe must be rebuilt and deployed
together. Verification must cover record size/layout, profile source wiring,
target compilation, record validation, correctness, and named metric output.
