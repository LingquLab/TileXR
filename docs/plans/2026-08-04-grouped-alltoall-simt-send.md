# Grouped AllToAll SIMT Send Migration Plan

## Goal And Boundaries

Implement the approved design in
`docs/specs/2026-08-04-grouped-alltoall-simt-send-design.md`. Port only the
validated `b8f0ed4` SIMT send plane into the current grouped AllToAll
architecture. Preserve current SDMA/MTE copyout, credit, terminal SyncAll,
prewarm, multi-region shared-QP, route, and trace behavior.

Do not stage or alter the existing user changes in
`src/include/tilexr_sdma_a5.h` or
`tests/udma/demo/tilexr_udma_alltoall_group_trace_to_chrome.py` unless a
target build proves a narrowly scoped compatibility change is required.

## Task 1: Device SIMT Send Helper

Objective: add the proven SIMT task builder and posting pipeline in a dedicated
grouped demo header.

Scope:

- `tests/udma/demo/tilexr_udma_alltoall_group_simt.h`
- `tests/udma/CMakeLists.txt` only if needed for device compilation

Constraints:

- Adapt from `b8f0ed4`, not from earlier experimental SIMT commits.
- Use current registry, UDMA types, route helpers, and error/trace contracts.
- Preserve payload-before-signal ordering when regions differ.
- Do not create QPs or memory regions.

Acceptance:

- Device compilation succeeds with b131.
- The helper supports current group width, pass, route, multi-channel, and
  shared-QP indexing.

## Task 2: Current Kernel Integration

Objective: select the SIMT sender without changing receive/copyout semantics.

Scope:

- `tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp`
- `tests/udma/demo/tilexr_udma_alltoall_group_layout.h`

Constraints:

- `SIMT=1` uses one send core; `SIMT=0` uses 32 send workers.
- Every receive worker offset and trace-core mapping uses the selected runtime
  send-worker count.
- Preserve the four compile-time batch/credit specializations and terminal
  synchronization.
- Resolve the registry from `CommArgs::udmaRegistryPtr`.

Acceptance:

- Block dimensions are 33 for SIMT+MTE and 2 for SIMT+single-core SDMA.
- Existing non-SIMT dimensions remain 64 and 33 respectively.
- Source-contract tests prove current ready-driven copyout and terminal credit
  calls remain present.

## Task 3: Host ABI And Launch

Objective: expose the mode and provide the required SIMT local-memory layout.

Scope:

- `tests/udma/demo/tilexr_udma_demo.cpp`
- `tests/udma/demo/tilexr_udma_alltoall_group_launcher.cpp`

Constraints:

- Add `TILEXR_DEMO_ALLTOALL_GROUP_SIMT=0/1`, default 0.
- Add `simtMode` consistently to credit and non-credit kernel argument ABIs.
- Keep `rtKernelLaunchWithFlagV2` and all four embedded function registrations.
- For SIMT, reserve 64 KiB DCache and set dynamic UBuf to the remaining
  per-vector-core UBuf reported by the runtime.
- Reject invalid UBuf information before launch.

Acceptance:

- ABI sizes and offsets have static assertions and unit coverage.
- Runtime logs print SIMT mode, selected send workers, copyout workers, and
  block dimension.

## Task 4: Verification And Delivery

Objective: establish functional and compatibility evidence before committing.

Scope:

- `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`
- Existing build/test scripts and A10 deployment scripts under `tmp/`

Verification:

1. Run formatting/diff checks and the grouped layout/source-contract test.
2. Build the b131 grouped demo and runtime artifacts.
3. Deploy the same artifacts to the selected A10 2x8 hosts.
4. Run SIMT off/on for 1 KiB single, 256 MiB multi, and 1 GiB multi with
   single pass and identical warmup/repeat settings.
5. Require all 16 ranks to return zero, print the success marker, and pass data
   correctness checks. Use trace only after trace-off succeeds.
6. Review the final scoped diff and create one implementation commit including
   this plan and the approved design, excluding unrelated dirty files.

Residual scale risk: 2x8 proves device semantics and current feature
composition, but 128+ rank performance and shared-QP pressure still require a
later scale regression.
