# MoonEP Upstream-Compatible V1, PrefetchWeight, and ReduceGrad Plan

## Goal

Replace the unstable TileXR MoonEP V1 contract with the upstream-shaped tensor and
saved-plan model, migrate native Dispatch/Combine to it, add native PrefetchWeight and
ReduceGrad sibling operators, update the Torch facade, and prove the complete default
flow on Ascend950PR NPU 4-7.

## Approved Scope

The authoritative design is
`docs/specs/2026-08-04-moonep-upstream-v1-prefetch-reduce-design.md`.

Included:

- Plan/Planning V1 migration for `N/R/E/B/NvS/K`, padding, zero-fill, and dedup data.
- Stage-specific Dispatch, PrefetchWeight, Combine, and ReduceGrad argument structs.
- Configurable `B` and `tokenPadding`.
- Optional FP32 route weights and saved-plan redispatch.
- Three-projection in-place BF16 prefetch.
- Three-projection in-place FP32 gradient reduction and consumed-slot clearing.
- C/C++ ABI, Host, CPU-reference, source, Python, target-build, and real-device tests.

Excluded:

- `zero_copy=True` implementation; it is explicitly rejected.
- Cross-node UDMA, FP16, performance tuning, and group-GEMM integration.
- Linking or copying active implementation sources from `3rdparty/moonep` or
  `reference/`.
- Git push, pull request creation, merge, or history rewriting.

## Repository and Environment

- Repository: TileXR, branch `codex/moonep-dispatch-combine`.
- Language/toolchain: C++14, CANN 9.1.0, A5 `dav-c310-vec` kernels.
- Remote build/test host: `root@141.61.49.223:/home/c30061605/TileXR`.
- Remote Conda environment: `ai_moe_test`.
- Synchronization: Mutagen session `tilexr-moonep`.
- Hardware run: `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`.
- Preserve user-owned `.gitmodules`, `3rdparty/*`, `Testing/`, and
  `tests/moonep/build-mingw/` changes.

## Cross-Cutting Constraints

- Use `TileXRCommNextMagic`; do not reset shared flag memory for a new round.
- Same-host peer traffic uses `CommArgs::peerMems[] + IPC_DATA_OFFSET`.
- Never use a Host `kernel<<<...>>>` wrapper. Compile/embed pure AICore ELF, call
  `rtDevBinaryRegister` and `rtFunctionRegister`, then launch the registered signature
  with `rtKernelLaunchWithFlagV2`.
- Never put CANN `${ARCH}-linux/devlib` in runtime RPATH/RUNPATH.
- All shape, element-count, byte-count, offset, and chunk calculations use checked
  arithmetic and reject impossible capacities.
- Native calls remain asynchronous on the caller stream. Hardware tests synchronize
  before reading plan status or tensor outputs.

## Task 1: Establish the New ABI and CPU Contract

**Objective and role:** Define the public contract and executable CPU oracles before
changing native kernels. This is the dependency for every later task.

**Prerequisites:** Read the approved design and the authoritative upstream files listed
there. Preserve the ABI version value `1`; the structure layout may change because V1
is explicitly unstable.

**Modification scope:**

- Modify `src/include/tilexr_moonep.h`.
- Modify `tests/moonep/unit/test_tilexr_moonep_abi_layout.cpp`.
- Modify `tests/moonep/unit/test_tilexr_moonep_c_header.c`.
- Modify `tests/moonep/reference/moonep_stage_reference.{h,cpp}`.
- Modify `tests/moonep/unit/test_tilexr_moonep_stage_reference.cpp`.
- Modify `tests/moonep/CMakeLists.txt` only for required test wiring.

**Work:**

1. Add failing ABI assertions for the new Plan and all stage-specific structs.
2. Add CPU references for upstream `NvS`, zero-fill ranges, duplicate-group encoding,
   optional route-weight dispatch/combine, in-place prefetch, ascending-order FP32
   reduction, and live-slot clearing.
3. Cover empty slots, repeated expert requests, local and remote owners, duplicate top-k
   routes, `B < E/R`, `B == E/R`, padding, and overflow rejection.
4. Replace the public structs and flags only after the RED checks are observable.

**Acceptance and verification:**

```bash
cmake -S tests/moonep -B tests/moonep/build
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_abi_layout test_tilexr_moonep_c_header \
  test_tilexr_moonep_stage_reference -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure \
  -R 'tilexr_moonep_(abi_layout|c_header|stage_reference)'
```

Expected: all three targets pass and encode the new layout/semantics. Existing generic
`input`/`output` stage fields are absent.

**Artifacts and interfaces:** The updated public header and CPU reference functions are
the authoritative contracts for Tasks 2-7.

## Task 2: Migrate Planning and Plan Ownership

**Objective and role:** Produce an upstream-shaped saved plan and padded `NvS` that all
four communication/weight stages can consume.

**Dependencies:** Task 1 ABI and reference contract.

**Modification scope:**

- Modify `src/include/tilexr_moonep_planner.h` only as required by its internal bridge.
- Modify `src/moonep/planner/common/planner_common.h`.
- Modify `src/moonep/planner/host/planner_{host,layout,launch}.{h,cpp}`.
- Modify `src/moonep/planner/host/tilexr_moonep_planner.cpp`.
- Modify `src/moonep/planner/kernels/tilexr_moonep_planner_kernel.cpp`.
- Modify `src/moonep/host/tilexr_moonep.cpp` Planning paths.
- Modify planner and MoonEP unit tests under `tests/moonep_planner/` and
  `tests/moonep/unit/`.

**Constraints and non-goals:** Do not alter communicator initialization or import the
upstream CUDA planner. Keep `K <= 32` and bounded peer synchronization.

**Work:**

1. Add RED Host/layout/reference cases for explicit `B`, `tokenPadding`, `NvS`,
   `cuSeqlens[E+B]`, `zeroFillRanges[E+B,2]`, and plan tensor validation.
2. Change the workspace query to accept `B` and `tokenPadding` and return upstream
   logical `NvS` using checked arithmetic.
3. Move `cuSeqlens` out of Plan V1 and into PlanningArgs V1.
4. Extend workspace/launch ABI and the planner kernel to populate padded group ends,
   zero-fill ranges, expert-copy slots, and saved scratch for fresh Dispatch dedup.
5. Initialize `dupCounts` deterministically; Dispatch owns fresh dedup materialization.
6. Update planner status and exact Host/kernel argument tests.

**Acceptance and verification:**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_host test_tilexr_moonep_stage_layout -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure \
  -R 'tilexr_moonep_(host|stage_layout)|moonep_planner'
```

Expected: planner outputs match CPU references for padded, unpadded, saved-plan, and
invalid-capacity cases.

**Artifacts and interfaces:** New workspace query, PlanningArgs, Plan V1, and planner
kernel arguments consumed by Tasks 3-6.

## Task 3: Migrate Dispatch and Combine Together

**Objective and role:** Preserve the already-proven peer-data path while changing it to
upstream tensor arguments, padded `NvS`, optional route weights in one call, and saved
dedup metadata.

**Dependencies:** Tasks 1-2. Dispatch and Combine share plan encoding and peer-window
chunk layout, so they are one serial workstream.

**Modification scope:**

- Modify all active files under `src/moonep/dispatch/` and `src/moonep/combine/`.
- Modify `src/moonep/common/moonep_peer_window.h` only for genuinely shared chunk or
  status constants.
- Modify dispatch/combine Host, source, layout, and reference tests under
  `tests/moonep/`.

**Constraints and non-goals:** Do not restore wrapper launches or introduce UDMA.
Do not silently fall back to the old two-call hidden/weight API.

**Work:**

1. Add RED Host tests for paired optional weight tensors, new field names, padded NvS,
   fresh-dedup flag, reuse immutability, unsupported zero-copy, and exact launch args.
2. Update layout builders so a hidden chunk satisfies `NvS * chunkBytes <= 100 MiB`;
   iterate chunks instead of requiring the entire payload to fit.
3. On fresh Dispatch, construct `(primary, dup_start, dup_count)` groups and compact
   duplicate offsets. On saved-plan Dispatch, do not mutate those tensors.
4. Clear planned padding, scatter BF16 hidden and optional FP32 weights in one stage
   call, and materialize duplicate rows before draining `[NvS,*]` outputs.
5. Stage Combine inputs, pre-reduce duplicate rows, gather primary routes with FP32
   accumulation, and gather optional weights bit-exactly.
6. Retain bounded clear/ready/drained synchronization and per-stage status codes.

**Acceptance and verification:**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_dispatch_host test_tilexr_moonep_combine_host \
  test_tilexr_moonep_kernel_sources test_tilexr_moonep_stage_reference -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure \
  -R 'tilexr_moonep_(dispatch|combine|kernel_sources|stage_reference)'
```

Expected: route-aware BF16 and FP32 references pass, saved dedup fields remain bytewise
unchanged, and wrapper/source guards pass.

**Artifacts and interfaces:** Updated native Dispatch/Combine libraries and shared peer
chunk protocol used by the end-to-end demo.

## Task 4: Add Native PrefetchWeight

**Objective and role:** Add the missing sibling operator that updates all three full
weight projections in place according to `expertsToCopy`.

**Dependencies:** Tasks 1-2. It may reuse shared registration and peer protocol helpers
from Task 3 but must not depend on Dispatch implementation objects.

**Modification scope:**

- Create `src/moonep/prefetch_weight/CMakeLists.txt`.
- Create `src/moonep/prefetch_weight/common/prefetch_weight_common.h`.
- Create Host/layout/launch files under `src/moonep/prefetch_weight/host/`.
- Create `src/moonep/prefetch_weight/kernels/tilexr_moonep_prefetch_weight_kernel.cpp`.
- Create `tests/moonep/unit/test_tilexr_moonep_prefetch_weight_host.cpp`.
- Extend source/kernel guards and CMake test wiring.

**Constraints and non-goals:** BF16 rank-3 tensors only. A `-1` slot is not written.
Different trailing shapes for gate/up/down are valid. No Host synchronization.

**Work:**

1. Add RED Host/layout/source tests for all tensor, plan, topology, overflow, peer,
   stream, magic, and launch contracts.
2. Add the pure-AICore build/embed/registration target with a unique signature.
3. Implement deterministic projection/slot/chunk processing: the expert owner publishes
   source chunks, the destination drains to row `E+b`, and all ranks advance through
   bounded magic-tagged barriers.
4. Delegate `TileXRMoonEpPrefetchWeightV1` to the native Host path and remove its stub.

**Acceptance and verification:**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_prefetch_weight_host \
  test_tilexr_moonep_kernel_sources -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure \
  -R 'tilexr_moonep_(prefetch_weight_host|kernel_sources|stage_reference)'
```

Expected: local, remote, repeated, and unused slots match the BF16 oracle for three
different projection shapes.

**Artifacts and interfaces:** `tilexr-moonep-prefetch-weight` Host library with embedded
kernel and the public native entry-point delegation.

## Task 5: Add Native ReduceGrad

**Objective and role:** Add the missing sibling operator that reduces all ranks'
prefetch-slot gradients into owned expert rows and clears consumed source slots.

**Dependencies:** Tasks 1-2 and the peer chunk protocol established by Task 3 or 4.

**Modification scope:**

- Create `src/moonep/reduce_grad/CMakeLists.txt`.
- Create `src/moonep/reduce_grad/common/reduce_grad_common.h`.
- Create Host/layout/launch files under `src/moonep/reduce_grad/host/`.
- Create `src/moonep/reduce_grad/kernels/tilexr_moonep_reduce_grad_kernel.cpp`.
- Create `tests/moonep/unit/test_tilexr_moonep_reduce_grad_host.cpp`.
- Extend source/kernel guards and CMake test wiring.

**Constraints and non-goals:** FP32 only. Only locally owned expert rows may change.
Accumulate in ascending `(sourceRank, slot)` order. Preserve unused slots.

**Work:**

1. Add RED Host/layout/source tests for six tensors, pairwise trailing shapes,
   `[R,B,...]` reduce layout, rank ownership, overflows, magic, and exact launch args.
2. Add the pure-AICore build/embed/registration target with a distinct signature.
3. For each projection and row chunk, publish the local rank's B slot chunks, barrier,
   consume all source ranks in ascending rank/slot order, barrier, and clear local live
   slots. Use FP32 accumulation without dtype conversion.
4. Delegate `TileXRMoonEpReduceGradV1` to the native Host path and remove its stub.

**Acceptance and verification:**

```bash
cmake --build tests/moonep/build --target \
  test_tilexr_moonep_reduce_grad_host \
  test_tilexr_moonep_kernel_sources -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure \
  -R 'tilexr_moonep_(reduce_grad_host|kernel_sources|stage_reference)'
```

Expected: fan-in, repeated expert, empty, asymmetric, and random plans match the CPU
oracle; nonlocal expert rows and unused slots remain unchanged.

**Artifacts and interfaces:** `tilexr-moonep-reduce-grad` Host library with embedded
kernel and the public native entry-point delegation.

## Task 6: Integrate Libraries, Capabilities, and Installation

**Objective and role:** Make the five-stage native implementation one coherent installed
surface without dependency or runtime-path regressions.

**Dependencies:** Tasks 2-5.

**Modification scope:**

- Modify `src/moonep/CMakeLists.txt`.
- Modify `src/moonep/host/tilexr_moonep.cpp`.
- Modify `tests/moonep/unit/test_tilexr_moonep_host.cpp`.
- Modify `tests/moonep/unit/test_tilexr_moonep_sources.cpp`.
- Modify installation/source contract tests.

**Work:**

1. Add both new sibling libraries and link them into `libtilexr-moonep.so`.
2. Advertise all five stages as native and no stages as stubs.
3. Remove obsolete local-copy stub helpers and generic stage validation.
4. Require five unique embedded binaries/signatures and reject kernel SO installation,
   wrapper syntax, upstream dependencies, and devlib runtime paths.
5. Validate exported V1 symbols and installed dependencies.

**Acceptance and verification:**

```bash
cmake --build tests/moonep/build -j"$(nproc)"
ctest --test-dir tests/moonep/build --output-on-failure
```

On the remote CANN build:

```bash
cmake --build build-moonep-direct -j"$(nproc)"
cmake --install build-moonep-direct
nm -D install-moonep-direct/lib*/libtilexr-moonep.so.1 | \
  grep -E 'TileXRMoonEp(Planning|Dispatch|PrefetchWeight|Combine|ReduceGrad)V1'
readelf -d install-moonep-direct/lib*/libtilexr-moonep.so.1 | \
  grep -E 'NEEDED|RPATH|RUNPATH'
```

Expected: all V1 symbols exist, both new libraries resolve, RPATH is `$ORIGIN`, and no
dynamic path contains `devlib`.

**Artifacts and interfaces:** Complete installed native C ABI and libraries.

## Task 7: Align the Torch Facade and Test/Demo Callers

**Objective and role:** Expose upstream-facing Python parameters and exercise the exact
native ABI from realistic callers.

**Dependencies:** Tasks 1-6.

**Modification scope:**

- Modify `integrations/moonep_torch/tilexr_moonep/{abi,runtime,torch_api}.py`.
- Modify `tests/moonep/python/{fakes,test_ffi_unittest,test_unittest_smoke}.py`.
- Modify `tests/moonep/demo/tilexr_moonep_flow_demo.cpp` and `run_a5.sh`.
- Modify `tools/moonep/benchmark.py`, config/report files as required by the ABI.

**Constraints and non-goals:** Match upstream default and saved-plan signatures.
`zero_copy=True` must fail before native launch. Do not claim performance validity.

**Work:**

1. Update ctypes layouts and exact fake-library call-record tests.
2. Add context fields for `B` and `tokenPadding`; allocate Plan outputs at upstream
   shapes and retain workspace/scratch through saved-plan reuse.
3. Make Dispatch plan-optional: fresh calls run Planning and return `cuSeqlens`; reuse
   calls skip Planning and return `None` for it.
4. Pass hidden and optional route weights in one native Dispatch/Combine call.
5. Make PrefetchWeight mutate the caller's three full tensors and return `None` or an
   event. Pass six ReduceGrad tensors in one native call.
6. Extend the C++ flow demo with deterministic BF16 prefetch and FP32 reduction oracles,
   status markers, saved-plan redispatch, and local-slot clearing checks.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python -q
ctest --test-dir tests/moonep/build --output-on-failure
```

Expected: fake-runtime call order and structures match the C ABI; the demo builds and
prints all five stages as native.

**Artifacts and interfaces:** Upstream-shaped TileXR Torch facade and hardware demo.

## Task 8: Remote Build and Hardware Validation

**Objective and role:** Establish target-toolchain, installed-runtime, and real peer-data
evidence without conflating them.

**Dependencies:** Tasks 1-7 complete locally.

**Modification scope:** No source changes unless a reproduced build/runtime defect is
diagnosed and returned to its owning task. Store temporary logs outside tracked source
or under ignored test output paths.

**Work:**

1. Flush the existing Mutagen session and confirm synchronization is healthy.
2. Activate `ai_moe_test`, source CANN 9.1, configure/build/install in the existing
   `build-moonep-direct` and `install-moonep-direct` directories.
3. Run the complete configured CTest suite and Python smoke suite.
4. Inspect embedded artifacts, exported symbols, NEEDED libraries, RPATH/RUNPATH, and
   real driver HAL resolution.
5. Run a bounded single-rank case first.
6. Run four ranks on physical NPU 4-7 with nontrivial routes, optional weights, all
   three prefetch projections, all three gradient projections, repeated expert fan-in,
   unused slots, and saved-plan backward redispatch.
7. Aggregate every rank log independently. Require native markers and success status
   for Planning, Dispatch, PrefetchWeight, Combine, and ReduceGrad on all ranks.

**Acceptance and verification:**

Representative remote commands:

```bash
mutagen sync flush tilexr-moonep

ssh root@141.61.49.223
cd /home/c30061605/TileXR
source /home/pkg/b061/cann-9.1.T560/set_env.sh
eval "$(conda shell.bash hook)"
conda activate ai_moe_test
cmake -S . -B build-moonep-direct \
  -DTILEXR_BUILD_MOONEP=ON -DTILEXR_BUILD_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/install-moonep-direct"
cmake --build build-moonep-direct -j"$(nproc)"
cmake --install build-moonep-direct
ctest --test-dir build-moonep-direct --output-on-failure

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
  bash tests/moonep/demo/run_a5.sh 4 64 4 32 512 4
```

Expected: target build/install and all Host/static tests pass; four independent rank
logs prove all five kernels execute and all tensor/status oracles pass.

**Artifacts and interfaces:** Commands, concise aggregated evidence, and any retained
validation documentation update.

## Task 9: Documentation, Review, and Completion Evidence

**Objective and role:** Make the changed ABI maintainable and verify the final diff
against repository constraints.

**Dependencies:** Tasks 1-8.

**Modification scope:**

- Replace or update `docs/moonep/DISPATCH_COMBINE.md` with the complete five-stage API.
- Modify `docs/BUILD_VERIFICATION.md` and `tools/moonep/README.md`.
- Review every changed source/test/integration file.

**Work:**

1. Document the new V1 structs, tensor shapes, Plan lifetime, B/padding rules, in-place
   ownership, status semantics, supported topology, and zero-copy limitation.
2. Run `git diff --check`, inspect the scoped diff, and search active code for forbidden
   dependencies, wrapper launches, stale generic ABI fields, and old stub claims.
3. Use `superpowers-neo-requesting-code-review` because this is a broad public-interface
   and concurrent multi-rank change. Resolve evidence-backed findings before completion.
4. Use `superpowers-neo-verification-before-completion` and report each evidence layer
   separately.

**Acceptance and verification:**

```bash
git diff --check
git grep -n -E '<<<|RunLocalStub|dispatchedCapacity|args->input|args->output' -- \
  src/moonep src/include/tilexr_moonep.h integrations/moonep_torch
git grep -n -i -E '3rdparty/moonep|reference/|src/ep|hccl|shmem' -- \
  src/moonep src/include/tilexr_moonep.h
```

Every match must be an intentional rejection test or explanatory documentation. Final
completion requires green local tests, green remote CANN tests, a clean installation
inspection, and four-rank NPU 4-7 correctness evidence for all five native stages.

## Dependency Order

```text
Task 1 ABI/reference
  -> Task 2 Planning/Plan
      -> Task 3 Dispatch/Combine
      -> Task 4 PrefetchWeight
      -> Task 5 ReduceGrad
          -> Task 6 Integration/install
              -> Task 7 Torch/demo
                  -> Task 8 Remote/NPU
                      -> Task 9 Docs/review/completion
```

Tasks 4 and 5 are conceptually independent after Tasks 1-2, but execution in the shared
workspace remains serial because they both touch public integration and common test
guards. The main agent owns integration; no subagent or parallel writer is required.
