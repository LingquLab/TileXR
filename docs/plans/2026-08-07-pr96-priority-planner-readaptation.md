# PR96-Priority Planner Re-adaptation

## Goal

Rework the PR96 Planner V2 integration so PR96 owns every shared Planner
V2 responsibility while retaining current-main MoonEP functionality through an
isolated V3 compatibility backend. PR106/PR107 remain authoritative outside the
Planner V2 ownership boundary.

## Scope

- Make `src/moonep/planner_v2` create and install `tilexr-moonep-planner`.
- Restore PR96 V2 legacy layout/Host files and point `tests/moonep_planner` at
  them.
- Rename current-main V3 internal Planner namespaces to prevent duplicate V2 symbols.
- Build current-main V3 as PIC objects appended to the V2-owned shared library.
- Preserve optimized V2, historical V2, current-main V3, and the authoritative
  PR106 MoonEP ABI.
- Keep pure-AICore embedding, Runtime V2 launch, C++14 Host, CANN 9.1, A5-only
  Planner support, IPC metadata, and `$ORIGIN` RPATH.

## Non-Goals

- Reverting PR107 Dispatch or PR106 PrefetchWeight, Combine, or ReduceGrad.
- Reverting the current-main MoonEP Plan V1 layout to the older PR96 branch state.
- Restoring the prohibited PR96 Host `kernel<<<...>>>` wrapper or a kernel
  shared-object runtime dependency.
- Adding UDMA to Planner V2 or changing transport behavior.

## Work

1. Restore PR96 V2 legacy Host/layout sources and V2-oriented Planner tests.
2. Isolate V3 internal names and verify current-main V3 behavior remains unchanged.
3. Invert CMake ownership so V2 creates the shared target and V3 appends
   compatibility objects.
4. Update source guards to enforce PR96 ownership and current CANN launch rules.
5. Run V2, V3, MoonEP C++, Python fake-runtime, compileall, and diff checks.
6. Sync the frozen worktree, rebuild/install with CANN 9.1, inspect exports and
   RPATH, then run bounded 1/2/8-rank Planner and MoonEP hardware correctness.

## Acceptance

- Target creation and `tests/moonep_planner` resolve to `planner_v2`.
- PR96 V2 behavior and tests are authoritative where V2/V3 previously collided.
- V3 remains available as the current-main compatibility backend; PR106/PR107
  public ABI and native Dispatch/ReduceGrad behavior remain authoritative.
- No duplicate Host symbols, wrapper launch syntax, runtime `devlib`, or active
  `reference/` dependency is present.
- Validation results are reported separately for mock, compile, and hardware.
