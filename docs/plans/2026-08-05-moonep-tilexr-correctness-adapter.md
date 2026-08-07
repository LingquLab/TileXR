# TileXR MoonEP Correctness Adapter Implementation Plan

## Status

- Approved and completed on 2026-08-05.
- Local evidence: MoonEP Python 79/79, MoonEP Host/source CTest 11/11,
  Planner CTest 5/5, UDMA JSON/source guards, `compileall`, and diff/static checks.
- Remote evidence: b131 CANN 9.1.0 build/install and configured CTest 22/22;
  four-rank `skewed-padding` differential passed all five stages on physical NPU 4-7
  with `candidate_backend=tilexr_native` and `performance_valid=false`.

## Approved Scope Amendment

- On 2026-08-05, one-rank differential validation exposed that the native Planner
  emitted endpoint pairs instead of upstream V1 `(pad_start, pad_row_count)` values in
  `zero_fill_ranges`.
- The user authorized the minimal Planner kernel, C++ stage reference, and regression
  test correction required to close this defect. No ABI, Host launch, registration,
  Dispatch payload, or other AICore behavior is added to scope.
- Four-rank validation also exposed heap corruption in the optional UDMA root-info
  regex parser. The user authorized this defect fix. The implementation replaces the
  failing regex operations with a bounded C++14 field scanner and preserves a standard-
  exception fallback boundary; `RaInit` unavailability now disables UDMA while IPC
  communicator initialization continues.
- Post-completion transport decision: all MoonEP stages remain same-node IPC-only and
  reject cross-node communicators. Core communicator initialization may probe UDMA,
  but every error disables that optional capability and continues on IPC; a successful
  probe does not change the MoonEP data path.

## Goal

Implement a built-in `TileXRMoonEPBackend` that adapts the real TileXR Torch facade to
`tools.moonep.contracts.MoonEPBackend`. This closes the current gap between the
five-stage reference/differential runner and the installed native TileXR implementation,
so `--mode correctness` compares Planning, Dispatch, PrefetchWeight, Combine, and
ReduceGrad outputs and mutations stage by stage on Ascend NPU.

## Scope

- Add the normalized TileXR candidate backend and factory.
- Preserve native NPU tensors and native Plan/workspace ownership across all stages.
- Make explicit Planning followed by Dispatch build dedup exactly once.
- Add deterministic backend cleanup after the final rank-wide correctness gate.
- Make the built-in TileXR backend the default candidate for `--mode correctness`, while
  preserving `--candidate-backend MODULE:FACTORY` as an override.
- Validate locally with fakes and on the CANN 9.1 host with one-rank and four-rank
  TileXR-vs-reference differential runs.

## Non-Goals

- No C ABI, Host launch, Tiling, AICore kernel, peer protocol, or status-code changes.
- No `kernel<<<...>>>` Host wrappers; existing pure-AICore registration and
  `rtKernelLaunchWithFlagV2` remain unchanged.
- No reference-value emulation inside the TileXR adapter.
- No compact-to-padded buffer translation that could hide a native NvS defect.
- No performance claim, profiling, cross-node MoonEP support, zero-copy support, or
  `inter_rank_sync=False` support.
- No standalone `--stage` CLI in this change; the backend methods remain independently
  callable, while the correctness runner executes the dependency-ordered five stages.

## Authoritative Context

- Repository constraints: `AGENTS.md`.
- Native ABI/facade plan:
  `docs/plans/2026-08-04-moonep-upstream-v1-prefetch-reduce.md`.
- Existing correctness design and deferred Task 6:
  `docs/plans/2026-08-04-moonep-python-correctness-reference.md`.
- Normalized protocol: `tools/moonep/contracts.py`.
- Stage runner/comparators: `tools/moonep/correctness.py`.
- Independent reference backend: `tools/moonep/torch_npu_backend.py`.
- Native facade: `integrations/moonep_torch/tilexr_moonep/torch_api.py`.
- Candidate loading and managed launcher: `tools/moonep/benchmark.py` and
  `tools/moonep/launcher.py`.

## Settled Design

### Backend Ownership

`TileXRMoonEPBackend` owns one `TileXRMoonEPBuffer`, which owns its context, native
runtime, communicator, pending asynchronous references, and native Plan workspaces.
The adapter exposes normalized result dataclasses but does not copy native outputs to
Host or substitute reference tensors.

The adapter keeps a private registry keyed by normalized Plan identity. Each entry owns
both the normalized Plan and its native Plan. Unknown or cloned Plans are rejected with
a precise contract error; the adapter never silently re-plans.

### Plan Translation

- Planning returns a normalized Plan whose core tensors directly reference native NPU
  tensors: `dst`, `cu_seqlens`, `experts_to_copy`, `zero_fill_ranges`, and
  `remote_stats`.
- The Planning result exposes `dedup=None`, even though native dedup storage is already
  allocated, because contents are not valid until fresh Dispatch completes.
- The first successful Dispatch attaches normalized `DedupPlan` fields that directly
  reference native `dup_groups`, `dup_loffs`, and `dup_counts`.
- Saved-plan Dispatch reuses those tensors and must preserve them bytewise.
- Native `status` and `workspace` remain private native Plan state. Adapter
  `synchronize()` delegates to `TileXRMoonEPBuffer.synchronize()`, which checks the
  correct success marker for the last enqueued stage.

### Fresh Dedup State

The correctness runner calls Planning and Dispatch as separate backend methods. The
facade therefore needs a buffer-owned state for Plans that have completed Planning but
have not completed fresh Dispatch.

- `planning()` records the native Plan as needing dedup.
- `dispatch()` uses `BUILD_DEDUP` when the Plan is in that state, regardless of whether
  Planning was inline or explicit.
- The state is consumed only after successful native Dispatch enqueue.
- Synchronizing between Planning and Dispatch does not consume it.
- A failed Dispatch retains it for a valid retry.
- Saved-plan Dispatch omits `BUILD_DEDUP` and does not mutate dedup tensors.
- `close()` clears retained Plan state.

This changes no public Dispatch signature and preserves the upstream fresh/saved-plan
return contract.

### Capability and Topology Gate

The built-in factory uses `TileXRMoonEPContext.from_env()` with normalized dimensions,
`args.install_prefix`, and `args.wait_iterations`. Before Planning it requires:

- normalized dimensions exactly match the created context;
- all five V1 stage bits are native and no required bit is stubbed;
- `transport_correctness_valid` is true;
- current NPU device and planner rank/world match launcher environment;
- the run is same-node, because the current MoonEP native validation scope is same-host.

Any failure closes a partially created context and raises `BackendUnavailableError`
with the missing capability or mismatched field.

### Lifecycle and Teardown

Add `close()` to the normalized backend protocol. The reference backend implements a
no-op close because benchmark main owns the HCCL process group. The TileXR backend closes
its buffer idempotently.

`run_correctness_case()` closes the candidate after the runner's final rank-wide stage
gate and before benchmark main destroys HCCL. Cleanup failures are recorded; they do not
replace an earlier stage failure. This preserves peer-window lifetime until every rank
has quiesced or reported a coordinated failure.

## Task 1: Add Lifecycle Contract and RED Cleanup Tests

**Objective and role:** Make backend resource ownership explicit before introducing a
communicator-owning candidate.

**Background and prerequisites:** The current protocol ends at `synchronize()`, and
`run_correctness_case()` never closes a candidate. Benchmark main separately owns the
global HCCL process group.

**Modification scope:**

- Modify `tools/moonep/contracts.py`.
- Modify `tools/moonep/torch_npu_backend.py`.
- Modify candidate lifecycle handling in `tools/moonep/benchmark.py`.
- Modify `tests/moonep/python/test_moonep_correctness_runner.py` and
  `tests/moonep/python/test_moonep_modes.py`.

**Constraints and non-goals:** Do not destroy the HCCL process group from a backend.
Do not hide the primary stage exception with a later close exception.

**Work:**

1. Add `close() -> None` to `MoonEPBackend`.
2. Add an idempotent no-op close to `TorchNpuMoonEPBackend` and forwarding close to test
   wrapper backends.
3. Add RED tests proving protocol rejection for a backend without close, close after
   successful correctness, close after stage failure, and primary-error preservation.
4. Move candidate factory creation inside the case result/error envelope so capability
   or construction failures still produce `result.json` without a false stage artifact.
5. Make `run_correctness_case()` close owned backends in deterministic candidate-then-
   reference order and record cleanup errors in `result.json`.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python/test_moonep_correctness_runner.py \
  tests/moonep/python/test_moonep_modes.py -q
```

Expected: lifecycle tests fail before implementation, then pass; HCCL ownership remains
with benchmark main.

**Artifacts and interfaces:** Updated `MoonEPBackend.close()` contract consumed by the
factory and runner.

## Task 2: Support Explicit Planning Followed by Fresh Dispatch

**Objective and role:** Make the native facade compatible with a staged correctness
runner without exposing a second Dispatch implementation.

**Background and prerequisites:** Current `dispatch(plan=...)` always treats the Plan as
saved and skips dedup construction, even when that Plan was just returned by explicit
`planning()`.

**Modification scope:**

- Modify `integrations/moonep_torch/tilexr_moonep/torch_api.py`.
- Modify `tests/moonep/python/fakes.py` only if a failure injection hook is required.
- Modify `tests/moonep/python/unittest_smoke.py`.

**Constraints and non-goals:** Do not change the public ABI, ctypes layout, public
Dispatch signature, kernel flags, or saved-plan output contract. Consume fresh state
only after successful native enqueue.

**Work:**

1. Add RED tests for explicit Planning followed by Dispatch with `build_dedup=True`.
2. Cover synchronization between those calls, retry after immediate Dispatch failure,
   inline Planning, saved-plan redispatch, dedup immutability, and close cleanup.
3. Add a buffer-owned strong Plan registry for native Plans needing first Dispatch.
4. Derive `build_dedup` from registry state and remove the entry only after successful
   runtime Dispatch.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python/test_unittest_smoke.py \
  tests/moonep/python/test_ffi_unittest.py -q
```

Expected: explicit and inline fresh paths build once; reuse never rebuilds or mutates
dedup; existing status and stream-lifetime tests remain green.

**Artifacts and interfaces:** One native facade path usable by both upstream inline
Dispatch and the staged adapter.

## Task 3: Implement `TileXRMoonEPBackend`

**Objective and role:** Bridge real native calls to every normalized stage result
without coupling correctness validators to native ctypes/facade types.

**Background and prerequisites:** Tasks 1-2. The normalized protocol, reference backend,
case factory, and stage comparators are already complete.

**Modification scope:**

- Create `tools/moonep/tilexr_backend.py`.
- Create `tests/moonep/python/test_tilexr_correctness_adapter.py` using
  `unittest.TestCase` so it also runs in the remote environment without pytest.
- Reuse test fakes from `tests/moonep/python/fakes.py`; do not add production emulation.

**Constraints and non-goals:** Return real native tensors. Do not import or execute
`3rdparty/moonep` or `reference/`. Reject Plans not owned by this backend. Keep optional
route weights paired and preserve all in-place ownership rules.

**Work:**

1. Implement `TileXRMoonEPBackend` with `name="tilexr_native"`, normalized dimensions,
   buffer ownership, native Plan registry, `synchronize()`, and idempotent `close()`.
2. Implement Planning translation with core native tensors and `dedup=None`.
3. Implement Dispatch translation, attach native dedup after successful fresh Dispatch,
   and return native hidden/optional weight outputs.
4. Implement in-place PrefetchWeight and ReduceGrad mapping between
   `ProjectionTensors` and facade arguments.
5. Implement Combine mapping with optional weights.
6. Implement `create_backend(torch_module, dimensions, case, args)` using
   `TileXRMoonEPContext.from_env()` and the capability/topology gate.
7. Add unit tests for exact calls and identities, tensor shapes, optional weights,
   first/saved Dispatch, unknown Plan rejection, status propagation, capability
   rejection, partial-construction cleanup, and idempotent close.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python/test_tilexr_correctness_adapter.py -q
cd tests/moonep/python
python -m unittest test_tilexr_correctness_adapter -v
```

Expected: the class satisfies `MoonEPBackend`, never substitutes reference data, and
normalizes all five native stages with correct Plan lifetime.

**Artifacts and interfaces:** `tools.moonep.tilexr_backend:TileXRMoonEPBackend` and
`tools.moonep.tilexr_backend:create_backend`.

## Task 4: Integrate the Built-In Candidate and Documentation

**Objective and role:** Make real TileXR differential validation a supported command,
while preserving generic candidate injection.

**Background and prerequisites:** Task 3.

**Modification scope:**

- Modify `tools/moonep/benchmark.py` and `tools/moonep/launcher.py` only where needed for
  the built-in default and lifecycle reporting.
- Modify `tests/moonep/python/test_moonep_modes.py`.
- Modify `tools/moonep/README.md`, `docs/BUILD_VERIFICATION.md`, and
  `docs/moonep/DISPATCH_COMBINE.md`.
- Mark deferred Task 6 complete in
  `docs/plans/2026-08-04-moonep-python-correctness-reference.md` only after verification.

**Constraints and non-goals:** Correctness work remains untimed and
`performance_valid=false`. External `MODULE:FACTORY` candidates remain supported.

**Work:**

1. Define the built-in candidate spec `tools.moonep.tilexr_backend:create_backend`.
2. Make `--mode correctness` use it when `--candidate-backend` is omitted; retain the
   explicit option as an override.
3. Replace the deferred-adapter error and documentation with installed-library,
   same-node, and capability prerequisites.
4. Ensure reports name `tilexr_native`, contain five per-stage artifacts, and preserve
   untimed/performance-invalid metadata.
5. Document one-rank triage and four-rank correctness commands separately from native
   benchmark timing.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python/test_moonep_modes.py \
  tests/moonep/python/test_moonep_correctness_runner.py -q
python -m tools.moonep.launcher --help
```

Expected: correctness mode resolves the built-in backend by default, explicit external
candidates still load, and reports remain untimed.

**Artifacts and interfaces:** Supported `--mode correctness` TileXR command and updated
validation documentation.

## Task 5: Local Regression and Static Validation

**Objective and role:** Prove Python/facade integration without claiming local NPU
correctness.

**Dependencies:** Tasks 1-4.

**Modification scope:** Tests and documentation only if a reproduced defect requires a
targeted fix in its owning task.

**Constraints and non-goals:** Windows fake/Host tests do not prove native device
transfer. Preserve user-owned submodules and generated build directories.

**Work and verification:**

```bash
python -m pytest tests/moonep/python -q
ctest --output-on-failure
git diff --check
rg -n '<<<|RunLocalStub' src/moonep integrations/moonep_torch tools/moonep
rg -n -i '3rdparty/moonep|reference/' \
  src/moonep integrations/moonep_torch tools/moonep/tilexr_backend.py
```

Run CTest with `tests/moonep/build-mingw` as the process working directory on Windows.
Every source match must be an intentional guard or documentation statement.

**Acceptance:** Full Python regression and 11 focused MoonEP Host/source tests pass;
diff/static checks are clean; no wrapper or active upstream dependency is introduced.

**Artifacts and interfaces:** Local evidence for contracts, mapping, lifecycle, and
source constraints.

## Task 6: Remote CANN and Physical NPU Differential Validation

**Objective and role:** Establish real installed-runtime and tensor-level differential
evidence for all five native stages.

**Dependencies:** Task 5.

**Environment:**

- Remote: `root@141.61.49.223:/home/c30061605/TileXR`.
- Sync: Mutagen session `tilexr-moonep`.
- CANN: `/home/pkg/b131/cann-9.1.0` (`V100R001C11SPC001B241`).
- Conda: `ai_moe_test`.
- Physical devices: `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`.

**Constraints and non-goals:** Serialize SSH sessions. Run one rank before four ranks.
Do not install pytest into the environment. Do not call the run a performance test.

**Work:**

1. Flush Mutagen and verify the session is connected/watching.
2. Rebuild/install with CANN 9.1 and run all configured CTests.
3. Run unittest-compatible FFI, smoke, and adapter tests in `ai_moe_test`.
4. Run `planning-small` correctness on physical NPU 4.
5. Run `skewed-padding` correctness on physical NPU 4-7 with four ranks.
6. Aggregate every rank's five stage artifacts. Require exact core Plan fields,
   Dispatch valid-region hidden/weights and semantic dedup, all three PrefetchWeight
   projections, Combine hidden/weights, ReduceGrad full gradients and consumed/unused
   buffers, and candidate name `tilexr_native`.
7. Confirm `performance_valid=false`, clean communicator teardown, no residual rank
   processes, and no `devlib` runtime path regression.

**Representative commands:**

```bash
mutagen sync flush tilexr-moonep

source /home/pkg/b131/cann-9.1.0/set_env.sh
eval "$(conda shell.bash hook)"
conda activate ai_moe_test
export TILEXR_BUILD_DIR="$PWD/build-moonep-b131"
export TILEXR_INSTALL_PREFIX="$PWD/install-moonep-b131"
export LD_LIBRARY_PATH="$TILEXR_INSTALL_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

cmake -S . -B "$TILEXR_BUILD_DIR" \
  -DTILEXR_BUILD_MOONEP=ON -DTILEXR_BUILD_TESTS=ON -DBUILD_TESTING=ON \
  -DCMAKE_INSTALL_PREFIX="$TILEXR_INSTALL_PREFIX"
cmake --build "$TILEXR_BUILD_DIR" -j"$(nproc)"
cmake --install "$TILEXR_BUILD_DIR"
ctest --test-dir "$TILEXR_BUILD_DIR" --output-on-failure --timeout 120

ASCEND_RT_VISIBLE_DEVICES=4 \
python -m tools.moonep.launcher \
  --mode correctness \
  --cases tools/moonep/cases/correctness.json \
  --case-ids planning-small \
  --output-dir /tmp/tilexr-moonep-correctness-1r \
  --install-prefix "$TILEXR_INSTALL_PREFIX" \
  --physical-device-count 1 --world-size 1

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
python -m tools.moonep.launcher \
  --mode correctness \
  --cases tools/moonep/cases/correctness.json \
  --case-ids skewed-padding \
  --output-dir /tmp/tilexr-moonep-correctness-4r \
  --install-prefix "$TILEXR_INSTALL_PREFIX" \
  --physical-device-count 4 --world-size 4
```

**Acceptance:** One-rank and four-rank summaries pass; every rank has five passing
stage artifacts comparing real TileXR outputs with the independent Torch/HCCL reference.

**Artifacts and interfaces:** Retained remote result/stage JSON and installed-runtime
evidence.

## Task 7: Review and Completion Evidence

**Objective and role:** Close the public-facade, lifecycle, and multi-rank correctness
risks before declaring the adapter supported.

**Dependencies:** Task 6.

**Modification scope:** Only evidence-backed fixes in files already owned by Tasks 1-4,
plus final documentation/status updates.

**Work:**

1. Review Plan/native identity mapping, tensor aliasing, first/saved dedup transition,
   asynchronous status propagation, failure cleanup, rank teardown, and capability
   rejection.
2. Use `superpowers-neo-requesting-code-review` because this crosses a public facade,
   dynamic candidate loading, and multi-rank resource lifetime.
3. Resolve evidence-backed findings and rerun invalidated checks.
4. Use `superpowers-neo-verification-before-completion` and report local, remote build,
   reference/HCCL, and real TileXR differential evidence separately.
5. Update the old deferred Task 6 status and durable docs only after all gates pass.

**Acceptance:** No unresolved P0/P1 findings; all current-state checks pass; remaining
limitations are explicitly scoped to same-node correctness and non-performance use.

**Artifacts and interfaces:** Reviewed implementation, durable documentation, and final
validation summary.

## Dependency Order

```text
Task 1 lifecycle contract
  -> Task 2 explicit Planning / fresh dedup
      -> Task 3 TileXR backend and factory
          -> Task 4 CLI and documentation
              -> Task 5 local validation
                  -> Task 6 remote NPU differential validation
                      -> Task 7 review and completion
```

The work should execute serially in the current shared workspace. Tasks share the
normalized protocol, facade state, benchmark lifecycle, and tests; parallel writers
would create unnecessary coordination risk.

## Final Acceptance Criteria

- `--mode correctness` works without an external candidate spec and selects
  `tilexr_native`.
- Every normalized backend stage uses real TileXR native tensors and operations.
- Explicit Planning followed by Dispatch builds dedup exactly once.
- Saved-plan Dispatch preserves dedup bytewise.
- Per-stage synchronization reports native status failures at the owning stage.
- Candidate cleanup is coordinated, idempotent, and does not destroy HCCL ownership.
- Local Python and Host/source regressions pass.
- One-rank and four-rank physical NPU differential runs pass all five stage artifacts.
- Reports remain untimed with `performance_valid=false`.
- No Host wrapper launch, active upstream/reference dependency, or `devlib` runtime path
  is introduced.
