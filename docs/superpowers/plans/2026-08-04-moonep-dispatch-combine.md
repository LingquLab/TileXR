# TileXR MoonEP Native Dispatch and Combine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: install and use
> `superpowers-neo` (`brainstorming`, `test-driven-development`, and
> `executing-plans` or `subagent-driven-development`) plus the
> `ascendc-development` operator skills before implementing this plan. Track every
> step with the checkboxes below and do not skip the design, precision, or profiling
> gates.

**Goal:** Replace the current local memcpy stubs behind
`TileXRMoonEpDispatchV1` and `TileXRMoonEpCombineV1` with A5-native Ascend C
operators, placed beside `src/moonep/planner`, so a saved MoonEP plan can execute
`planning -> dispatch -> expert work -> combine` on Ascend NPU.

**Architecture:** Keep the stable V1 C ABI in `libtilexr-moonep.so.1`. Add
independent `dispatch` and `combine` Host/kernel libraries under `src/moonep` and
delegate the public V1 entry points to them. The first backend is an A5 same-host
peer-window protocol using `CommArgs::peerMems[] + TileXR::IPC_DATA_OFFSET`, fresh
magic values from `TileXRCommNextMagic`, reusable magic-tagged synchronization, and
bounded device-side peer waits. Dispatch clears the destination window, decodes every
planner `dst` entry, scatters all routes, and drains the local window into the caller's
output. Combine publishes each rank's expert output to its local peer window, pulls the
rows named by the saved `dst`, and either reduces hidden rows or gathers route weights.
Negative `dst` entries are decoded and materialized rather than requiring the upstream
CUDA duplicate-builder/prologue/epilogue metadata that is absent from the V1 ABI.

**Tech Stack:** C++14 Host code, CMake 3.16, CANN 9.1.0, Bisheng Ascend C kernels for
`dav-c310-vec`, TileXR `tile-comm`, ACL/runtime, source-only C++ tests, Python/Torch-NPU
integration tests, A5 hardware correctness runs, and `msprof` profiling.

---

## Fixed Scope and Contracts

- Keep `TileXRMoonEpTensorV1`, `TileXRMoonEpPlanV1`,
  `TileXRMoonEpDispatchArgsV1`, and `TileXRMoonEpCombineArgsV1` binary layouts
  unchanged.
- `dispatchedCapacity` remains exactly `S * K`; planner output is the only route map.
- Decode each route with 64-bit arithmetic:
  `raw = dst >= 0 ? dst : -int64_t(dst) - 1`,
  `destRank = raw / dispatchedCapacity`, and
  `destOffset = raw % dispatchedCapacity`.
- Materialize negative/duplicate routes in the correctness backend. Do not add the
  upstream `dup_groups`, `dup_loffs`, `dup_counts`, or `zero_fill_ranges` tensors to
  V1.
- Hidden dispatch accepts contiguous BF16 `[S,H]` and produces BF16
  `[S*K,H]`. Hidden combine accepts BF16 `[S*K,H]`, accumulates each token's K rows
  in FP32, and produces BF16 `[S,H]`.
- Route-weight dispatch accepts contiguous FP32 `[S,K]` and produces FP32 `[S*K]`.
  Route-weight combine accepts FP32 `[S*K]` and gathers FP32 `[S,K]` without
  arithmetic.
- FP16 may be added only after BF16 correctness is complete and must have its own
  precision cases. INT32 tensors cease to be an execution dtype for native
  dispatch/combine and return `TILEXR_MOONEP_ERROR_NOT_SUPPORTED`.
- The first backend supports communicators whose required peers have valid same-host
  `peerMems[]` mappings. If `localRankSize < rankSize`, return
  `TILEXR_MOONEP_ERROR_NOT_SUPPORTED` before launch. Do not silently fall back to the
  old stub.
- Do not use `peerMems[]` as UDMA targets. Cross-node registered-memory UDMA is a
  separate versioned follow-up and is not part of this plan.
- A peer payload must fit entirely in `TileXR::IPC_BUFF_MAX_SIZE`; checked arithmetic
  rejects larger shapes before launch.
- Dispatch and combine remain asynchronous with respect to the Host. Public entry
  points enqueue kernels and never call `aclrtSynchronizeStream` internally.
- `3rdparty/moonep` and `reference/` are behavior references only. Active targets must
  not include, compile, or link their sources. The new MoonEP libraries must not link
  `src/ep` either.
- Runtime RPATH/RUNPATH is `$ORIGIN` only. CANN `devlib` may be a link-time search path
  but must never appear in runtime paths.
- Correctness is the first milestone. Start with one AIV block per rank. Parallel
  tiling is permitted only after the one-block path passes the complete correctness
  matrix and `msprof` identifies the limiting stages.

## Operator Data Flow

### Dispatch

1. Validate the V1 descriptors, communicator topology, dtype/shape contract, peer
   window size, and device `CommArgs` pointer.
2. Get a fresh magic value with `TileXRCommNextMagic`.
3. Clear this rank's peer payload window and publish the clear with a system-scope
   fence.
4. Synchronize all ranks at `dispatch-window-cleared`.
5. For every local `(token, topk)` route, decode `dst`, copy the hidden row or weight
   scalar to `peerMems[destRank] + IPC_DATA_OFFSET + destOffset * rowBytes`, including
   negative duplicate routes.
6. Synchronize all ranks at `dispatch-data-ready`.
7. Copy this rank's peer payload window to the caller-owned output tensor.
8. Synchronize all ranks at `dispatch-window-drained` before another stage can reuse
   the peer window.

### Combine

1. Validate the saved plan and input/output descriptor contract.
2. Get a fresh magic value.
3. Copy the complete local `[S*K,...]` expert output into this rank's peer payload
   window, then publish it with a system-scope fence.
4. Synchronize all ranks at `combine-data-ready`.
5. For each local route, decode `dst` and pull the corresponding row/scalar from
   `peerMems[destRank]`.
6. Hidden mode accumulates K BF16 rows in FP32 and casts once to BF16. Weight mode
   copies each FP32 scalar to its original `[token,topk]` slot.
7. Synchronize all ranks at `combine-window-drained` before the next peer-window use.

### Device Status

- Reuse `plan->status` as the asynchronous stage-status location after planning.
- Add disjoint status ranges for dispatch and combine success, invalid decoded route,
  and peer-ready timeout. The Host return value reports validation/launch errors;
  device execution failures are observed after stream synchronization by reading
  `plan->status`.
- Every poll loop is bounded. A timed-out block writes one deterministic status and
  all blocks/ranks stop consuming unpublished data.

---

## Planned File Structure

**Create:**

- `src/moonep/common/moonep_peer_window.h`
- `src/moonep/dispatch/CMakeLists.txt`
- `src/moonep/dispatch/design.md`
- `src/moonep/dispatch/common/dispatch_common.h`
- `src/moonep/dispatch/host/dispatch_layout.h`
- `src/moonep/dispatch/host/dispatch_layout.cpp`
- `src/moonep/dispatch/host/dispatch_host.h`
- `src/moonep/dispatch/host/dispatch_host.cpp`
- `src/moonep/dispatch/host/dispatch_launch.h`
- `src/moonep/dispatch/host/dispatch_launch.cpp`
- `src/moonep/dispatch/host/tilexr_moonep_dispatch.cpp`
- `src/moonep/dispatch/kernels/tilexr_moonep_dispatch_kernel.cpp`
- `src/moonep/combine/CMakeLists.txt`
- `src/moonep/combine/design.md`
- `src/moonep/combine/common/combine_common.h`
- `src/moonep/combine/host/combine_layout.h`
- `src/moonep/combine/host/combine_layout.cpp`
- `src/moonep/combine/host/combine_host.h`
- `src/moonep/combine/host/combine_host.cpp`
- `src/moonep/combine/host/combine_launch.h`
- `src/moonep/combine/host/combine_launch.cpp`
- `src/moonep/combine/host/tilexr_moonep_combine.cpp`
- `src/moonep/combine/kernels/tilexr_moonep_combine_kernel.cpp`
- `tests/moonep/reference/moonep_stage_reference.h`
- `tests/moonep/reference/moonep_stage_reference.cpp`
- `tests/moonep/unit/test_tilexr_moonep_stage_layout.cpp`
- `tests/moonep/unit/test_tilexr_moonep_stage_reference.cpp`
- `tests/moonep/unit/test_tilexr_moonep_dispatch_host.cpp`
- `tests/moonep/unit/test_tilexr_moonep_combine_host.cpp`
- `tests/moonep/unit/test_tilexr_moonep_kernel_sources.cpp`
- `tests/moonep/cases/moonep_dispatch_combine_cases.jsonl`
- `tests/moonep/python/test_dispatch_combine_precision.py`
- `tests/moonep/python/run_dispatch_combine_precision_report.py`
- `tests/moonep/python/benchmark_dispatch_combine_msprof.py`
- `docs/moonep/DISPATCH_COMBINE.md`

**Modify:**

- `src/include/tilexr_moonep.h`
- `src/moonep/CMakeLists.txt`
- `src/moonep/host/tilexr_moonep.cpp`
- `tests/moonep/CMakeLists.txt`
- `tests/moonep/unit/test_tilexr_moonep_host.cpp`
- `tests/moonep/unit/test_tilexr_moonep_sources.cpp`
- `tests/moonep/unit/test_tilexr_moonep_abi_layout.cpp`
- `tests/moonep/demo/tilexr_moonep_flow_demo.cpp`
- `tests/moonep/demo/run_a5.sh`
- `integrations/moonep_torch/tilexr_moonep/abi.py`
- `integrations/moonep_torch/tilexr_moonep/torch_api.py`
- `tests/moonep/python/fakes.py`
- `tests/moonep/python/test_ffi_unittest.py`
- `tests/moonep/python/test_unittest_smoke.py`
- `tools/moonep/benchmark.py`
- `tools/moonep/report.py`
- `tools/moonep/README.md`
- `docs/BUILD_VERIFICATION.md`

---

## Task 1: Freeze the Design, Layout, and CPU Oracle

**Files:**

- Create: `src/moonep/common/moonep_peer_window.h`
- Create: `src/moonep/dispatch/design.md`
- Create: `src/moonep/combine/design.md`
- Create: `src/moonep/dispatch/host/dispatch_layout.{h,cpp}`
- Create: `src/moonep/combine/host/combine_layout.{h,cpp}`
- Create: `tests/moonep/reference/moonep_stage_reference.{h,cpp}`
- Create: `tests/moonep/unit/test_tilexr_moonep_stage_layout.cpp`
- Create: `tests/moonep/unit/test_tilexr_moonep_stage_reference.cpp`
- Modify: `tests/moonep/CMakeLists.txt`

- [x] **Step 1: Write both operator design documents before implementation**

Document the exact signatures, shape/dtype tables, route decoding, same-host topology
gate, peer-window offsets, three dispatch synchronization steps, two combine steps,
bounded timeout/status behavior, block tiling, CopyIn/Compute/CopyOut pseudocode, and UB
allocation tables. Derive and record buffer coefficients for BF16 hidden reduction and
FP32 weight copy. The design must explicitly explain why negative routes are copied in
the first implementation rather than using upstream duplicate metadata.

- [x] **Step 2: Write failing layout and CPU-reference tests**

Cover checked `capacity * rowBytes`, 32-byte alignment, exactly-at-100-MiB acceptance,
one-byte-over rejection, world/rank mismatch, BF16/FP32 mode selection, INT32 rejection,
and decoding of `0`, positive remote routes, `-raw-1`, and `INT32_MIN`.

The CPU oracle must accept all ranks' `dst` and tensors and verify:

- balanced and skewed routing;
- world sizes 1, 2, and 8;
- `K=1`, `K=2`, and duplicate-heavy `K=32`;
- empty experts and unused capacity rows;
- hidden dispatch placement;
- FP32 route-weight placement;
- identity combine;
- deterministic expert transforms followed by combine;
- saved-plan forward/backward reuse.

- [x] **Step 3: Run the new tests and confirm RED**

```bash
cmake -S tests/moonep -B tests/moonep/build
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_stage_layout \
  test_tilexr_moonep_stage_reference -j"$(nproc)"
```

Expected: compilation fails because the layout and reference implementations do not
exist yet.

- [x] **Step 4: Implement the minimal Host layout and CPU oracle**

Use checked unsigned arithmetic and keep all shared structs POD/kernel-safe. The peer
window header defines only constants and status/step codes; payload starts exactly at
`IPC_DATA_OFFSET` and consumes no hidden unvalidated suffix.

- [x] **Step 5: Run GREEN and commit**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_stage_layout \
  test_tilexr_moonep_stage_reference -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure \
  -R 'tilexr_moonep_stage_(layout|reference)'
git diff --check
git add src/moonep/common src/moonep/dispatch/design.md \
  src/moonep/dispatch/host/dispatch_layout.* src/moonep/combine/design.md \
  src/moonep/combine/host/combine_layout.* tests/moonep/reference \
  tests/moonep/unit/test_tilexr_moonep_stage_* tests/moonep/CMakeLists.txt
git commit -m "test: define moonep stage contracts"
```

Expected: both tests pass and no whitespace errors are reported.

## Task 2: Add Dispatch/Combine Build Skeletons and Dependency Guards

**Files:**

- Create: `src/moonep/dispatch/CMakeLists.txt`
- Create: `src/moonep/combine/CMakeLists.txt`
- Create: both `common`, `host`, and `kernels` skeleton files listed above
- Modify: `src/moonep/CMakeLists.txt`
- Modify: `tests/moonep/unit/test_tilexr_moonep_sources.cpp`
- Create: `tests/moonep/unit/test_tilexr_moonep_kernel_sources.cpp`
- Modify: `tests/moonep/CMakeLists.txt`

- [x] **Step 1: Extend source guards first**

Require sibling `dispatch` and `combine` directories, separate Host libraries, separate
kernel SOs, A5 `dav-c310-vec`, C++14 Host targets, `$ORIGIN` install RPATH, and public
MoonEP linking to both Host libraries. Reject includes/links containing
`3rdparty/moonep`, `reference/`, `src/ep`, HCCL, SHMEM, or runtime RPATH entries that
contain `devlib`.

- [x] **Step 2: Run source tests and confirm RED**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_sources test_tilexr_moonep_kernel_sources -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure \
  -R 'tilexr_moonep_(sources|kernel_sources)'
```

Expected: failures identify the missing sibling targets, explicit registration, and
direct runtime launch paths.

- [x] **Step 3: Add minimal compilable target structure**

Use the shared pure-AICore compile/link/embed policy without copying planner operator
logic. Produce:

- `libtilexr-moonep-dispatch.so.1` with its dispatch AICore ELF embedded;
- `libtilexr-moonep-combine.so.1` with its combine AICore ELF embedded.

Keep CANN `devlib` only in `target_link_directories`. Add build dependencies for every
kernel/common header that affects generated AICore binaries. Register each binary and
its unique stable signature explicitly before `rtKernelLaunchWithFlagV2`; do not build
or call a `kernel<<<...>>>` Host wrapper.

- [x] **Step 4: Run source guards and an A5 compile smoke**

```bash
source scripts/common_env.sh
cmake -S . -B build-moonep \
  -DTILEXR_BUILD_MOONEP=ON -DTILEXR_BUILD_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build-moonep --target \
  tilexr-moonep-dispatch tilexr-moonep-combine -j"$(nproc)"
```

Expected: both Host libraries and embedded pure-AICore artifacts build with CANN 9.1.0. Functional launch is
not expected yet.

- [x] **Step 5: Commit the build skeleton**

```bash
git add src/moonep/dispatch src/moonep/combine src/moonep/CMakeLists.txt \
  tests/moonep/CMakeLists.txt tests/moonep/unit/test_tilexr_moonep_*sources.cpp
git commit -m "build: add moonep dispatch and combine targets"
```

## Task 3: Implement Dispatch Host Validation and Launch Boundary

**Files:**

- Create/modify: `src/moonep/dispatch/host/dispatch_host.{h,cpp}`
- Create/modify: `src/moonep/dispatch/host/dispatch_launch.{h,cpp}`
- Create/modify: `src/moonep/dispatch/host/tilexr_moonep_dispatch.cpp`
- Create: `tests/moonep/unit/test_tilexr_moonep_dispatch_host.cpp`
- Modify: `tests/moonep/CMakeLists.txt`

- [x] **Step 1: Write failing Host tests with fakes**

Fake `TileXRGetCommArgsHost`, `TileXRGetCommArgsDev`, `TileXRCommNextMagic`, and the
generated dispatch launcher. Test null/short structures, ABI mismatch, flags, stream,
plan invariants, rank/world mismatch, BF16 and FP32 shape modes, dtype mismatch, INT32,
peer-window overflow, cross-node topology, missing device args, magic failure, and exact
launcher arguments.

- [x] **Step 2: Confirm RED**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_dispatch_host -j"$(nproc)"
```

Expected: link or assertion failure because the native Host boundary is absent.

- [x] **Step 3: Implement the minimum Host path**

Expose one internal function used by the stable wrapper, for example
`TileXRMoonEpRunDispatchV1`. It must validate without dereferencing device data, obtain
Host/device `CommArgs`, reject unsupported topology/dtype, build the peer layout, get a
fresh magic, and invoke the kernel on the caller stream. Do not enqueue Host memset or
memcpy and do not synchronize.

- [x] **Step 4: Run GREEN and commit**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_dispatch_host -j"$(nproc)"
tests/moonep/build/test_tilexr_moonep_dispatch_host
git add src/moonep/dispatch/host tests/moonep/unit/test_tilexr_moonep_dispatch_host.cpp \
  tests/moonep/CMakeLists.txt
git commit -m "feat: add moonep dispatch host launch"
```

## Task 4: Implement the Correctness-First Dispatch Kernel

**Files:**

- Modify: `src/moonep/dispatch/common/dispatch_common.h`
- Modify: `src/moonep/dispatch/kernels/tilexr_moonep_dispatch_kernel.cpp`
- Modify: `tests/moonep/unit/test_tilexr_moonep_kernel_sources.cpp`
- Modify: `tests/moonep/demo/tilexr_moonep_flow_demo.cpp`

- [x] **Step 1: Add failing kernel-source assertions**

Require the public generated launch symbol, device `CommArgs`,
`IPC_DATA_OFFSET`, `SyncCollectives`, magic-tagged clear/ready/drained steps,
system-scope publication, 64-bit negative-route decode, BF16 row copy, FP32 scalar
copy, full local-window zeroing, bounded polling, and device status writes. Reject
`aclrtSynchronizeStream`, HCCL, UDMA calls, `src/ep` helpers, and direct inclusion of
the upstream implementation.

- [x] **Step 2: Confirm RED**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_kernel_sources -j"$(nproc)"
tests/moonep/build/test_tilexr_moonep_kernel_sources
```

- [x] **Step 3: Implement the one-AIV dispatch pipeline**

Use UB tiles sized from the design table. Clear the entire output byte count before
the clear barrier, route-copy all `S*K` entries, publish before ready, then drain the
complete local payload. Predicates and negative decode must be identical for BF16 and
FP32 modes. No atomics are needed because planner destinations are unique.

- [x] **Step 4: Build and run focused single-rank hardware RED/GREEN**

First run the updated demo before replacing its stub expectations and record the
failure. Then update only the dispatch portion of the demo oracle and run:

```bash
cmake --build build-moonep --target tilexr-moonep-dispatch \
  tilexr_moonep_flow_demo -j"$(nproc)"
TILEXR_PHYSICAL_DEVICE_COUNT=1 \
  bash tests/moonep/demo/run_a5.sh 1 8 2 4 128 1
```

Expected: planner and dispatch placement pass; combine remains marked stub until its
task is complete.

- [x] **Step 5: Commit**

```bash
git add src/moonep/dispatch tests/moonep/unit/test_tilexr_moonep_kernel_sources.cpp \
  tests/moonep/demo/tilexr_moonep_flow_demo.cpp
git commit -m "feat: implement moonep peer-window dispatch"
```

## Task 5: Implement Combine Host Validation and Launch Boundary

**Files:**

- Create/modify: `src/moonep/combine/host/combine_host.{h,cpp}`
- Create/modify: `src/moonep/combine/host/combine_launch.{h,cpp}`
- Create/modify: `src/moonep/combine/host/tilexr_moonep_combine.cpp`
- Create: `tests/moonep/unit/test_tilexr_moonep_combine_host.cpp`
- Modify: `tests/moonep/CMakeLists.txt`

- [x] **Step 1: Write failing combine Host tests**

Mirror dispatch validation while asserting the inverse shapes, hidden reduction mode,
weight gather mode, topology rejection, full input publication size, fresh magic, and
exact generated-launch arguments.

- [x] **Step 2: Confirm RED, implement, and run GREEN**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_combine_host -j"$(nproc)"
```

Implement `TileXRMoonEpRunCombineV1` with no Host synchronization and rerun the target
until it exits zero.

- [x] **Step 3: Commit**

```bash
git add src/moonep/combine/host tests/moonep/unit/test_tilexr_moonep_combine_host.cpp \
  tests/moonep/CMakeLists.txt
git commit -m "feat: add moonep combine host launch"
```

## Task 6: Implement the Correctness-First Combine Kernel

**Files:**

- Modify: `src/moonep/combine/common/combine_common.h`
- Modify: `src/moonep/combine/kernels/tilexr_moonep_combine_kernel.cpp`
- Modify: `tests/moonep/unit/test_tilexr_moonep_kernel_sources.cpp`
- Modify: `tests/moonep/demo/tilexr_moonep_flow_demo.cpp`

- [x] **Step 1: Add failing combine source assertions**

Require local input publication, ready/drained synchronization, remote peer pulls,
64-bit `dst` decode, BF16-to-FP32 accumulation across exactly K routes, one final BF16
cast, exact FP32 weight gather, and bounded error/status handling.

- [x] **Step 2: Implement hidden reduction and weight gather**

The kernel chooses mode from the validated dtype/rank contract. BF16 mode tiles H into
UB, zeros an FP32 accumulator for each token, pulls K rows, accumulates, converts, and
writes `[S,H]`. FP32 mode copies one scalar per route to `[S,K]` with no reduction.

- [x] **Step 3: Run source, Host, single-rank, and two-rank checks**

```bash
cmake --build tests/moonep/build -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure
cmake --build build-moonep --target tilexr-moonep-combine \
  tilexr_moonep_flow_demo -j"$(nproc)"
bash tests/moonep/demo/run_a5.sh 1 8 2 4 128 1
bash tests/moonep/demo/run_a5.sh 2 16 4 8 256 2
```

Expected: dispatch placement, identity combine, deterministic transformed combine,
duplicate routes, and route-weight round-trip match the CPU oracle on every rank.

- [x] **Step 4: Commit**

```bash
git add src/moonep/combine tests/moonep/unit/test_tilexr_moonep_kernel_sources.cpp \
  tests/moonep/demo/tilexr_moonep_flow_demo.cpp
git commit -m "feat: implement moonep peer-window combine"
```

## Task 7: Replace V1 Stubs and Update Capabilities Without ABI Drift

**Files:**

- Modify: `src/include/tilexr_moonep.h`
- Modify: `src/moonep/host/tilexr_moonep.cpp`
- Modify: `src/moonep/CMakeLists.txt`
- Modify: `tests/moonep/unit/test_tilexr_moonep_host.cpp`
- Modify: `tests/moonep/unit/test_tilexr_moonep_abi_layout.cpp`
- Modify: `tests/moonep/unit/test_tilexr_moonep_sources.cpp`

- [x] **Step 1: Change tests first**

Require planning, dispatch, and combine in `nativeStages`; only prefetch-weight and
reduce-grad remain in `stubStages`. Preserve every existing `sizeof` and `offsetof`
assertion. Replace memset/memcpy expectations for dispatch/combine with native delegate
calls while retaining stub tests for the other two stages.

- [x] **Step 2: Confirm RED**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_host test_tilexr_moonep_abi_layout \
  test_tilexr_moonep_sources -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure \
  -R 'tilexr_moonep_(host|abi_layout|sources)'
```

- [x] **Step 3: Delegate public entry points**

Keep validation shared only where it removes real duplication. Public dispatch and
combine call their native internal functions. Delete their `RunLocalStub` branches;
do not disturb planning, prefetch-weight, or reduce-grad behavior.

- [x] **Step 4: Verify symbols and dependencies**

```bash
cmake --build build-moonep --target tilexr-moonep -j"$(nproc)"
cmake --install build-moonep
nm -D install/lib*/libtilexr-moonep.so.1 | \
  rg 'TileXRMoonEp(Dispatch|Combine)V1'
readelf -d install/lib*/libtilexr-moonep.so.1 | rg 'NEEDED|RPATH|RUNPATH'
```

Expected: both V1 symbols remain exported, both native libraries are dependencies,
and no RUNPATH contains `devlib`.

- [x] **Step 5: Commit**

```bash
git add src/include/tilexr_moonep.h src/moonep tests/moonep/unit
git commit -m "feat: enable native moonep dispatch and combine"
```

## Task 8: Update Torch Integration, Benchmark, and Native Flow Demo

**Files:**

- Modify: `integrations/moonep_torch/tilexr_moonep/{abi.py,torch_api.py}`
- Modify: `tests/moonep/python/{fakes.py,test_ffi_unittest.py,test_unittest_smoke.py}`
- Modify: `tests/moonep/demo/tilexr_moonep_flow_demo.cpp`
- Modify: `tests/moonep/demo/run_a5.sh`
- Modify: `tools/moonep/{benchmark.py,report.py,README.md}`

- [x] **Step 1: Write failing Python capability and call-order tests**

Expect dispatch/combine native capability bits, BF16 hidden buffers, FP32 route-weight
buffers, saved-plan reuse in backward, caller-stream forwarding, and no internal stream
synchronize. Invalid device, dtype, shape, contiguity, destroyed plan, and unsupported
topology cases remain deterministic.

- [x] **Step 2: Update the facade without changing native ABI structs**

Remove stub-only warnings for dispatch/combine, keep warnings for prefetch/reduce, and
expose correctness-validity fields separately from performance-validity fields.

- [x] **Step 3: Replace byte-prefix demo checks with route-aware checks**

Use deterministic BF16 bit patterns and FP32 weights. Validate dispatch against the
CPU route oracle, apply a deterministic expert transform to valid rows, validate
combine, rerun dispatch/combine with the saved plan for backward, and retain explicit
host rendezvous before communicator destruction.

- [x] **Step 4: Run fake-runtime and source tests**

```bash
python -m pytest tests/moonep/python -q
cmake --build tests/moonep/build -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure
```

Expected: Python and Host/source suites pass without NPU hardware.

- [ ] **Step 5: Run 1/2/8-rank A5 correctness**

```bash
bash tests/moonep/demo/run_a5.sh 1 32 2 8 128 1
bash tests/moonep/demo/run_a5.sh 2 64 4 16 256 2
bash tests/moonep/demo/run_a5.sh 8 128 8 64 1024 8
```

Expected: every physical rank reports `planning=native dispatch=native combine=native`,
all route/weight checks pass, and `transport_performance_valid` remains false until the
profiling task is complete.

Execution note (2026-08-04): the one-, two-, and four-rank non-oversubscribed flows
passed, including a four-rank run on physical NPU 4-7. Eight logical ranks also passed
on seven physical devices with `blockDim=32`; the exact eight-device command remains
unchecked because NPU 0 was occupied by an unrelated long-running process.

- [x] **Step 6: Commit**

```bash
git add integrations/moonep_torch tests/moonep tools/moonep
git commit -m "test: validate native moonep flow"
```

## Task 9: Generate the Precision Matrix and Report

**Files:**

- Create: `tests/moonep/cases/moonep_dispatch_combine_cases.jsonl`
- Create: `tests/moonep/python/test_dispatch_combine_precision.py`
- Create: `tests/moonep/python/run_dispatch_combine_precision_report.py`
- Generate: `tests/moonep/reports/moonep_dispatch_combine_precision.json`
- Generate: `tests/moonep/reports/moonep_dispatch_combine_precision.md`

- [ ] **Step 1: Define at least 30 hardware cases before running them**

The case matrix must cross regular, generalized, and boundary shapes with routing
families: balanced, skewed, empty-expert, same-destination duplicates, cross-rank
duplicates, `K=1`, `K=32`, small/large H, one rank, and multi-rank. Include exact-copy
FP32 weight cases and BF16 hidden reduction cases. Every case records expected peer
window bytes and stays within the design limits.

- [ ] **Step 2: Run pytest on A5**

```bash
python -m pytest tests/moonep/python/test_dispatch_combine_precision.py -q
python tests/moonep/python/run_dispatch_combine_precision_report.py
```

Expected: all cases pass. Dispatch and weight gather are bit-exact. Combine reports
MaxAbsErr, MeanAbsErr, MaxRelErr, MeanRelErr, and CosineSim against FP32 CPU reference,
using thresholds stated in both design documents.

- [ ] **Step 3: Review and commit the reproducible report inputs**

Do not commit machine-specific raw profiler directories. Commit the JSONL cases,
scripts, and concise Markdown/JSON result summaries with machine/CANN/driver metadata.

```bash
git add tests/moonep/cases tests/moonep/python/test_dispatch_combine_precision.py \
  tests/moonep/python/run_dispatch_combine_precision_report.py \
  tests/moonep/reports/moonep_dispatch_combine_precision.*
git commit -m "test: add moonep precision evaluation"
```

## Task 10: Profile With msprof and Establish an Honest Baseline

**Files:**

- Create: `tests/moonep/python/benchmark_dispatch_combine_msprof.py`
- Generate: `tests/moonep/reports/moonep_dispatch_combine_perf.json`
- Generate: `tests/moonep/reports/moonep_dispatch_combine_perf.md`
- Modify only if justified by measurements: dispatch/combine layout, Host, and kernel
  files plus their tests/design tables

- [ ] **Step 1: Define at least eight representative profile cases**

Cover BF16 hidden and FP32 weights, multiple H/S/K values, 1/2/8 ranks, balanced and
duplicate-heavy routing. Run 20 measured launches after 10 warmups per case.

- [ ] **Step 2: Collect with msprof**

```bash
python tests/moonep/python/benchmark_dispatch_combine_msprof.py \
  --cases tests/moonep/cases/moonep_dispatch_combine_cases.jsonl \
  --warmup 10 --iterations 20 --output tests/moonep/reports
```

Filter `op_summary_*.csv` by exact OP Type, not display name. Report Task Duration,
effective payload bandwidth, vector utilization, MTE utilization, and synchronization
share when available.

- [ ] **Step 3: Keep comparisons honest**

There is no native Ascend MoonEP operator equivalent. Compare the final kernel only to
the retained one-AIV correctness baseline and report absolute latency/bandwidth. Do not
compare NPU results to CUDA MoonEP as if hardware were equivalent, and do not label
stub memcpy timing as an operator baseline.

- [ ] **Step 4: Optimize only measured bottlenecks**

If route copy dominates, add multi-AIV route tiling with an explicit device grid
barrier and prove clear/ready/drained ordering. If combine reduction dominates, tune H
tiling and UB double buffering. Each optimization starts with a failing regression or
performance assertion, updates the UB table, reruns all precision cases, and is a
separate commit.

- [ ] **Step 5: Commit profiling artifacts**

```bash
git add tests/moonep/python/benchmark_dispatch_combine_msprof.py \
  tests/moonep/reports/moonep_dispatch_combine_perf.* \
  src/moonep/dispatch/design.md src/moonep/combine/design.md
git commit -m "perf: profile moonep dispatch and combine"
```

## Task 11: Documentation and Final Validation

**Files:**

- Create: `docs/moonep/DISPATCH_COMBINE.md`
- Modify: `docs/BUILD_VERIFICATION.md`
- Modify: `tools/moonep/README.md`
- Verify: every file changed by Tasks 1-10

- [x] **Step 1: Write the user-facing operator documentation**

Document V1 signatures, supported shapes/dtypes, saved-plan lifetime, negative-route
semantics, peer-window size formula, asynchronous status checking, same-host A5
limitation, unsupported cross-node behavior, examples, and the remaining native stubs.

- [x] **Step 2: Run the complete Host/static suite**

```bash
cmake -S tests/moonep -B tests/moonep/build
cmake --build tests/moonep/build -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure
python -m pytest tests/moonep/python -q
```

- [x] **Step 3: Run the final CANN build/install checks**

```bash
source scripts/common_env.sh
cmake -S . -B build-moonep \
  -DTILEXR_BUILD_MOONEP=ON -DTILEXR_BUILD_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build-moonep -j"$(nproc)"
cmake --install build-moonep
ldd install/lib*/libtilexr-moonep.so.1
readelf -d install/lib*/libtilexr-moonep.so.1 | rg 'RPATH|RUNPATH|NEEDED'
```

Expected: all libraries resolve from the intended install/CANN/driver locations; no
runtime path contains `${ASCEND_HOME_PATH}/${ARCH}-linux/devlib`; no HCCL, SHMEM,
`src/ep`, upstream MoonEP, or reference-only dependency appears.

- [ ] **Step 4: Run final A5 data-plane correctness and profiling**

Run 1, 2, and 8 physical ranks, then the complete precision and msprof suites. Record
NPU model/count, CANN version, driver version, commit, commands, logs, and case totals.

- [x] **Step 5: Enforce the evidence boundary**

Report Host/static, A5 compile, A5 runtime correctness, Torch-NPU, and msprof evidence
separately. Explicitly state:

- host/source tests do not prove NPU data movement;
- one-rank execution does not prove peer synchronization;
- oversubscribed logical ranks do not prove equivalent physical-card throughput;
- same-host peer-memory does not prove cross-node UDMA;
- 910B fallback execution does not validate the A5-only kernel or UDMA data plane.

- [x] **Step 6: Inspect final diff and commit docs**

```bash
git status --short
git diff --check
git diff --stat
git grep -n -i '3rdparty/moonep\|reference/\|src/ep\|hccl\|shmem' -- \
  src/moonep src/include/tilexr_moonep.h
git add docs/moonep/DISPATCH_COMBINE.md docs/BUILD_VERIFICATION.md tools/moonep/README.md
git commit -m "docs: document native moonep operators"
```

Review every grep match. Only explanatory comments that explicitly reject a dependency
are acceptable in active files.

---

## Deferred Follow-Ups

- Add a versioned V2 API with registered UDMA workspace for cross-node dispatch and
  combine while retaining peer-memory as best-effort same-host fallback.
- Add upstream-compatible duplicate metadata only if profiling proves materialization
  is a bottleneck and the additional plan lifetime/ABI cost is justified.
- Add FP16 hidden support after BF16 precision closure.
- Implement real `PrefetchWeight` and `ReduceGrad`; they remain accurately advertised
  stubs after this plan.

## Completion Definition

This plan is complete only when dispatch and combine are reported native, the V1 ABI
layout is unchanged, all Host/source/Python tests pass, at least 30 A5 precision cases
pass, 1/2/8-rank same-host hardware flows pass, `msprof` reports are generated, runtime
dependency/RPATH checks pass, and every validation claim is scoped to the hardware and
transport actually exercised.
