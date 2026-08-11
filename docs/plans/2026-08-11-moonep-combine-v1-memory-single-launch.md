# MoonEP Combine V1 Memory Single-Launch Implementation Plan

## Goal

Implement the approved single-launch Combine V1 Memory baseline while keeping
the complete flow on Combine V2 by default.

## Scope And Constraints

- Follow `docs/specs/2026-08-11-moonep-combine-v1-memory-single-launch-design.md`.
- Preserve C++14, CANN 9.1, pure AICore binary registration, and peer-memory
  cross-node semantics.
- Preserve existing unrelated Dispatch V2, Combine V2, PrefetchWeight, and
  ReduceGrad changes in this worktree.
- Do not commit, push, or create a PR unless separately requested.
- Real hardware validation is limited to `141.61.49.223`.

## Ordered Work

### 1. Lock The Host/Layout Contract

Objective: replace split-phase layout behavior with the single-launch peer
window layout and V2 block scheduling.

Scope:

- `src/include/tilexr_moonep.h`
- `src/moonep/combine/host/combine_layout.*`
- `src/moonep/combine/host/combine_host.*`
- `src/moonep/combine/host/combine_launch.*`
- `src/moonep/combine/common/*`
- focused tests under `tests/moonep/unit/`

Acceptance:

- flags other than NONE fail;
- the replacement V1 descriptor requires and forwards `dstLocal`;
- source, receive, control, failure, and chunk regions fit 512 MiB;
- smallest and multi-chunk shapes produce nonzero aligned chunks;
- supported rank sizes produce the V2 active core count;
- one V1 call delegates to one registered-kernel launch.

### 2. Replace The V1 Kernel

Objective: implement one-launch duplicate pre-reduction, V2 peer/core scheduling,
memory push, completion, receive reduction, and drain.

Scope:

- `src/moonep/combine/kernels/tilexr_moonep_combine_kernel.cpp`
- shared V2 scheduling helpers only when they are transport-independent
- kernel source guards and schedule/layout tests

Acceptance:

- old publish-only/consume-only and registered-UDMA branches are absent;
- duplicate groups are partitioned without overlapping writes;
- hidden duplicate rows are skipped after pre-reduction while weight routes are preserved;
- all active cores and steps cover every peer exactly once;
- payload MTE3 precedes done publication;
- output token partitions have no gaps or overlap;
- hidden and optional weights complete in the same kernel launch;
- target `bisheng` compile succeeds.

### 3. Add The Runtime Switch

Objective: default to V2 and allow explicit V1 Memory selection.

Scope:

- `integrations/moonep_torch/tilexr_moonep/runtime.py`
- `integrations/moonep_torch/tilexr_moonep/torch_api.py`
- `tools/moonep/benchmark.py`
- Python FFI, mode, and performance-report tests

Acceptance:

- unset/2 calls V2;
- 1 calls V1 exactly once with hidden and optional weights;
- invalid values fail before communicator creation;
- metadata and trace labels match the selected backend;
- no V1-only workspace activation is added.

### 4. Local And Target Validation

Objective: establish Host, compile, integration, and hardware evidence.

Checks:

- focused Python tests, then `python -m pytest tests/moonep/python -q`;
- focused CTest targets for V1 Host/layout/source and V2 regressions;
- target CANN 9.1 configure/build/install on the existing isolated deployment;
- 8-NPU V1 benchmark/reference/correctness;
- 8-NPU default V2 benchmark/reference/correctness;
- `git diff --check`, final diff inspection, and original-workspace preservation.

The benchmark uses warmup 5 and 20 measured iterations. Performance results are
reported with the benchmark's own validity scope and are not generalized to
unvalidated multi-node transport.

## Dependencies And Coordination

The tasks are intentionally sequential. Host layout fields define the kernel
ABI; the kernel ABI defines the Python selection tests and target build. No task
is safely independent while these shared contracts are changing.
