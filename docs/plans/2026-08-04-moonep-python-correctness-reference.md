# MoonEP Python Reference and Stage Correctness Plan

## Execution Status

- Tasks 1-8: implemented and validated by 2026-08-05.
- Task 6 was completed by the follow-on TileXR correctness-adapter workstream with a
  built-in `tilexr_native` backend, native Plan/tensor ownership, and deterministic
  cleanup.
- Local evidence: 79 MoonEP Python tests, `compileall`, CLI parsing, Host/source CTest,
  UDMA parser/source guards, and `git diff --check` passed.
- Hardware evidence: the final `skewed-padding` case passed all five stages on four
  Ascend950PR devices through torch_npu 2.10.0.post2 and HCCL. Results are correctness
  evidence only and record `performance_valid=false`.

## Goal

Add an upstream-compatible MoonEP correctness framework that constructs deterministic
Planning inputs and expected outputs, implements all five stages with ordinary PyTorch
operations on Ascend NPU through `torch_npu` and HCCL, and can compare that reference
with any candidate backend after every stage.

The framework must identify the first failing stage and tensor. It must never treat a
successful final checksum, a native status code, or repeated deterministic output as a
substitute for stage correctness.

## Settled Scope

Included:

- A test-facing five-stage backend protocol for Planning, Dispatch, PrefetchWeight,
  Combine, and ReduceGrad.
- Deterministic route, hidden, route-weight, expert-weight, expert-output, full-gradient,
  and reduction-buffer construction.
- A pure-Python reference backend using `torch.*` tensor operations on NPU tensors.
- HCCL collectives through `torch.distributed` for rank-visible reference data.
- Input contract, input-value, output contract, output-value, and mutation checks for
  every stage.
- A candidate-backend injection point using the same test-facing protocol.
- An untimed benchmark correctness mode that runs both backends stage by stage.
- CPU tests for contracts/oracles and NPU/HCCL tests for the real reference data path.

Excluded:

- Native Ascend C implementation or optimization of any TileXR MoonEP stage.
- The normalized TileXR adapter wrapper. Another owner will implement it later against
  the backend protocol defined here.
- Performance tuning of the Python reference backend.
- Running the Python reference inside TileXR performance timing.
- Importing, linking, or copying executable code from `3rdparty/moonep` into active
  TileXR targets. The checkout is an authoritative comparison source only.
- Cross-node correctness claims in the first version.
- Git push, pull-request creation, merge, history rewriting, or cleanup of user files.

## Authoritative Semantics and Repository Boundaries

The compatibility target is the checked-in upstream comparison source:

- `3rdparty/moonep/moonep/api.py`
- `3rdparty/moonep/moonep/planning.py`
- `3rdparty/moonep/moonep/dispatch.py`
- `3rdparty/moonep/moonep/prefetch.py`
- `3rdparty/moonep/moonep/combine.py`
- `3rdparty/moonep/moonep/grad_reduce.py`
- `3rdparty/moonep/tests/planning_reference.py`
- `3rdparty/moonep/tests/test_{planning,dispatch,prefetch,combine,grad_reduce}.py`

Active Python code must not import `3rdparty.moonep`. Algorithms are implemented
independently with generic PyTorch operations and checked against small known fixtures.

This plan complements, but does not modify or replace:

- `docs/specs/2026-08-04-moonep-upstream-v1-prefetch-reduce-design.md`
- `docs/plans/2026-08-04-moonep-upstream-v1-prefetch-reduce.md`

Those documents own the native ABI and kernels. This plan owns the independent Python
oracle, differential runner, and correctness-mode benchmark integration.

Preserve C++14, CANN 9.1, driver 25.5.0+, magic-tag synchronization, registered-memory
rules, and the direct AICore launch restrictions in `AGENTS.md`. Preserve all existing
user-owned changes and untracked `3rdparty`, `Testing`, and build output directories.

## Dimensions and Test Data

```text
R             EP rank count
S             tokens per rank
K             routes per token
N             S * K
E             global expert count, divisible by R
epn           E / R
B             prefetch slots per rank
P             token-padding alignment
NvS           upstream padded physical route capacity per rank
H             hidden dimension
Hf            intermediate/projection dimension
```

The first correctness cases use small shapes and exactly representable values. Weight
layouts are gate/up `[E+B,H,Hf]` and down `[E+B,Hf,H]`; gradient and reduction-buffer
layouts match their corresponding projection. Larger benchmark shapes remain separate.

Cases must cover balanced and skewed routing, duplicate destination ranks within one
token's top-k, empty expert groups, padded groups, local and remote experts, repeated
prefetch requests, unused `-1` slots, multi-source gradient fan-in, and saved-plan
redispatch.

## Test-Facing Backend Contract

Create a backend-neutral contract under `tools/moonep` with these logical methods:

```text
planning(topk_experts, tokens_per_expert) -> PlanningResult
dispatch(plan, hidden_sh, route_weights_sk=None) -> DispatchResult
prefetch_weight(plan, projections) -> PrefetchResult
combine(plan, expert_output_nvsh, route_weights_nvs=None) -> CombineResult
reduce_grad(plan, full_grads, reduce_buffers) -> ReduceGradResult
```

The normalized plan includes `N/R/E/B/NvS/K`, `dst`, `cu_seqlens`,
`experts_to_copy`, `zero_fill_ranges`, `remote_stats`, and optional dedup metadata.
Planning does not validate dedup contents because upstream materializes them during a
fresh Dispatch. Dispatch returns the post-call plan state and makes dedup ready.

In-place stages return explicit post-call state even if the native facade returns
`None`:

- `PrefetchResult.projections`
- `ReduceGradResult.full_grads`
- `ReduceGradResult.reduce_buffers`

The wrapper may translate native object names and return conventions, but it must not
weaken, truncate, or reinterpret the normalized upstream contract.

## Mandatory Stage Gate

Every stage executes this sequence on every rank:

1. Validate canonical input shape, dtype, device, contiguity, ranges, dimensions, and
   upstream invariants.
2. Compare each backend's input values exactly with the canonical stage input.
3. Snapshot tensors and alias/storage information needed for mutation checks.
4. Give the reference and TileXR backends independent deep clones.
5. Invoke the reference and TileXR methods, synchronize their streams, and surface
   asynchronous status failures.
6. Validate each backend output contract and internal semantic invariants.
7. Validate forbidden mutations and all required in-place mutations.
8. Compare defined reference and TileXR outputs tensor by tensor.
9. Use an HCCL rank-wide pass/fail agreement before entering the next stage.
10. On failure, stop coherently and write a stage artifact with the tensor name, first
    mismatch index, expected and actual value, max absolute/relative/ULP error, masks,
    and mutation result.

Defined output regions are always checked. Upstream-defined undefined storage, notably
the tail after the final `cu_seqlens` end, is excluded with an explicit mask and recorded
in the artifact; it is not silently treated as a successful tensor-wide comparison.

## Stage-Specific Acceptance Rules

### Planning

- Inputs: `topk_experts [S,K] int32`, `tokens_per_expert [E] int32`.
- Input values must satisfy `bincount(topk_experts) == tokens_per_expert`, expert ids
  are in `[0,E)`, and inputs remain bytewise unchanged.
- Compare `dst`, `cu_seqlens`, `experts_to_copy`, `zero_fill_ranges`, and
  `remote_stats` exactly.
- Check token conservation, destination bounds, duplicate encoding, padded segment
  monotonicity, capacity, and expert-copy ownership.
- Dedup arrays are not a Planning value check; only their allocation contract may be
  checked before Dispatch.

### Dispatch

- Inputs: validated plan, `hidden_sh [S,H] bf16`, optional
  `route_weights_sk [S,K] fp32`.
- Hidden and route-weight inputs remain unchanged. Core Planning fields remain
  unchanged. Fresh Dispatch may materialize dedup metadata only.
- Compare defined `hidden_nvsh [NvS,H]` and optional `route_weights_nvs [NvS]`
  exactly. Planned padding rows must be zero.
- Compare dedup groups as semantic `(primary, duplicate-offset-set)` records because
  atomic construction order is not an upstream guarantee.

### PrefetchWeight

- Inputs: validated plan and three contiguous BF16 full-weight tensors.
- Rows `[0,E)` must remain unchanged.
- For each live local slot `b`, row `E+b` must equal the source row selected by
  `experts_to_copy[rank,b]` exactly.
- A `-1` slot must retain its pre-call sentinel exactly.
- Validate all three projections independently, including different trailing shapes.

### Combine

- Inputs: validated post-Dispatch plan, `expert_output_nvsh [NvS,H] bf16`, and optional
  `route_weights_nvs [NvS] fp32`.
- Inputs remain unchanged in normalized non-zero-copy mode.
- Compare `hidden_sh [S,H]` and optional `route_weights_sk [S,K]`.
- Route weights are bit-exact. Initial hidden fixtures use BF16-friendly values so the
  required result is exact; a later precision suite may explicitly permit one ULP.
- Duplicate routes contribute once through their primary after semantic pre-reduction.

### ReduceGrad

- Inputs: validated plan, three FP32 full-gradient tensors, and three FP32
  `[R,B,...]` reduction-buffer tensors.
- Only the current rank's owned expert rows may change. Non-owned expert rows and
  prefetch rows in the full tensors remain unchanged.
- Accumulate live contributions in ascending `(source_rank, slot)` order.
- Every live reduction slot is zeroed after consumption; every `-1` slot is unchanged.
- Initial fixtures use exactly representable FP32 values and require exact comparison.

## Task 1: Add Contracts and Contract Tests

**Objective and role:** Define the normalized five-stage API, tensor containers, clone
rules, and validation errors that every later task consumes.

**Background and prerequisites:** Use the settled contract and upstream references
above. Existing `integrations/moonep_torch/tilexr_moonep/torch_api.py` is compact V1
and is not the normalized correctness contract.

**Modification scope:**

- Create `tools/moonep/contracts.py`.
- Modify `tools/moonep/__init__.py` only for stable exports.
- Create `tests/moonep/python/test_correctness_contracts.py`.

**Constraints and non-goals:** Do not import `torch_npu` at module import time. Keep
containers usable with CPU Torch and test doubles. Do not alter the public native ABI.

**Work:**

1. Add dimension/config, plan, three-projection, and stage-result dataclasses.
2. Define a structural backend protocol and a precise unavailable/capability error.
3. Implement recursive clone/snapshot helpers that preserve dtype/device but do not
   share mutable tensor storage between backends.
4. Add contract validation for shapes, dtypes, contiguity, storage offset, device/rank,
   option pairing, plan lifecycle, and checked dimensions.
5. Add failing-then-passing unit cases for malformed inputs and accidental aliases.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python/test_correctness_contracts.py -q
```

Expected: valid CPU/fake-NPU contracts pass; every malformed shape, dtype, range,
missing optional pair, wrong plan state, and shared mutable input fails with a named
stage/tensor error.

**Artifacts and interfaces:** `MoonEPBackend` and result/container types consumed by
Tasks 2-7.

## Task 2: Add Deterministic Case Construction

**Objective and role:** Construct complete, reproducible stage inputs whose expected
effects are distinguishable by rank, token, expert, projection, and slot.

**Dependencies:** Task 1.

**Modification scope:**

- Create `tools/moonep/case_factory.py`.
- Create `tools/moonep/cases/correctness.json`.
- Create `tests/moonep/python/test_moonep_case_factory.py`.
- Modify `tools/moonep/config.py` for `B`, `token_padding`, `intermediate_size`, and
  routing-pattern fields without changing existing case defaults.

**Constraints and non-goals:** Seeds must produce the same logical values on CPU and
NPU. Do not rely on backend-specific random-number sequences for golden data.

**Work:**

1. Generate routing patterns on CPU and derive `tokens_per_expert` with a structured
   operation, then transfer cloned tensors to the selected device.
2. Encode deterministic hidden/route-weight values and sentinel-filled weight slots.
3. Encode three projections and matching full-grad/reduce-buffer tensors with values
   that identify their rank/expert/slot/indices and remain exactly representable.
4. Provide isolated stage fixtures plus a pipeline fixture whose downstream canonical
   inputs are derived only from already-validated upstream outputs.
5. Cover balanced, skewed, duplicate, padding, unused-slot, and fan-in cases.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python/test_moonep_case_factory.py -q
python -m pytest tests/moonep/python/test_correctness_contracts.py -q
```

Expected: repeated construction is exact; counts match routes; sentinels and rank/value
encodings distinguish all required semantic paths.

**Artifacts and interfaces:** Canonical case objects consumed by the reference backend
and differential runner.

## Task 3: Implement Upstream Planning in the Torch-NPU Backend

**Objective and role:** Produce the upstream Planning oracle independently of the
TileXR planner.

**Dependencies:** Tasks 1-2.

**Modification scope:**

- Create `tools/moonep/torch_npu_backend.py`.
- Extend or replace `tools/moonep/planner_reference.py` only through compatibility
  wrappers that preserve existing compact benchmark tests.
- Create `tests/moonep/python/test_torch_npu_planning_reference.py`.

**Constraints and non-goals:** Use generic `torch` operations such as `bincount`,
`cumsum`, `sort`, `searchsorted`, `where`, indexing, and HCCL collectives. Do not use a
TileXR planner output as the expected value. Do not claim an existing `torch_npu` MoE
operator has MoonEP Planning semantics.

**Work:**

1. Add a distributed context that validates the initialized HCCL group and rank
   metadata without taking ownership of unrelated process groups.
2. Gather per-rank expert counts in source-rank order.
3. Implement upstream allocation/balancing, prefetch-slot selection, padded VM-group
   layout, zero-fill ranges, remote stats, destination encoding, and duplicate marking.
4. Initialize plan lifecycle metadata so Dispatch owns dedup materialization.
5. Add hand-computed single-rank and multi-rank fixtures plus comparison with the
   checked-in upstream reference as a test-only development check, never a runtime
   import.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python/test_torch_npu_planning_reference.py -q
```

On an Ascend host with an initialized HCCL environment:

```bash
python tools/moonep/run_benchmark.py \
  --mode reference \
  --cases tools/moonep/cases/correctness.json \
  --case-ids planning-small \
  --world-size 4 --physical-device-count 4 \
  --output-dir Testing/moonep-reference-planning
```

Expected: all ranks produce exact plan fields and pass every Planning invariant.

**Artifacts and interfaces:** A usable `TorchNpuMoonEPBackend.planning` implementation
and HCCL context for Task 4.

## Task 4: Implement Dispatch, PrefetchWeight, Combine, and ReduceGrad References

**Objective and role:** Complete the independent pure-Python backend for all downstream
stage semantics after a plan is fixed.

**Dependencies:** Task 3.

**Modification scope:**

- Modify `tools/moonep/torch_npu_backend.py`.
- Create `tests/moonep/python/test_torch_npu_stage_reference.py`.
- Create `tests/moonep/python/npu_reference_worker.py` if a dedicated multi-rank worker
  is clearer than the benchmark worker.

**Constraints and non-goals:** Use `torch.*` on NPU and HCCL data exchange. Correctness
comes before memory or communication efficiency. Use rank-ordered all-gather data for
operations whose exact accumulation order matters. Do not time these methods as TileXR
performance.

**Work:**

1. Dispatch: exchange source hidden/weights, scatter by decoded `dst`, clear planned
   padding, and materialize normalized semantic dedup metadata.
2. PrefetchWeight: make source expert rows rank-visible and copy each selected expert
   into local row `E+b` for all three projections, preserving `-1` slots.
3. Combine: make per-rank expert outputs visible, pre-reduce duplicates, gather routes,
   and perform the upstream FP32 accumulation/final BF16 conversion.
4. ReduceGrad: gather reduction buffers, update only locally owned expert rows in
   ascending source/slot order, and clear only consumed live slots.
5. Test every stage independently before adding an end-to-end reference pipeline.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python/test_torch_npu_stage_reference.py -q
```

On Ascend NPU/HCCL:

```bash
python tools/moonep/run_benchmark.py \
  --mode reference \
  --cases tools/moonep/cases/correctness.json \
  --world-size 4 --physical-device-count 4 \
  --output-dir Testing/moonep-reference-all-stages
```

Expected: all five stages pass exact output and mutation oracles on every rank. Logs
must distinguish CPU-only tests from actual NPU/HCCL execution.

**Artifacts and interfaces:** Complete `TorchNpuMoonEPBackend` consumed by differential
correctness mode.

## Task 5: Add Stage Validators and the Differential Runner

**Objective and role:** Enforce the mandatory stage gate and produce actionable
mismatch artifacts rather than a final checksum-only result.

**Dependencies:** Tasks 1-4.

**Modification scope:**

- Create `tools/moonep/correctness.py`.
- Create `tests/moonep/python/test_moonep_correctness_runner.py`.
- Extend `tests/moonep/python/fakes.py` with fault-injecting normalized backends.

**Constraints and non-goals:** Never continue to a later collective after one rank has
failed a stage. Do not use a global allclose tolerance. Undefined regions require named
masks. Input mutation is a first-class failure.

**Work:**

1. Implement reusable exact, masked exact, ULP, semantic-set, invariant, and mutation
   comparisons.
2. Implement a rank-coherent stage runner with pre-call agreement, post-call stream
   synchronization, and post-validation agreement.
3. Write one JSON artifact per rank/stage and a compact case summary.
4. Add fault injection for every stage's input, output, padding, dedup, sentinel,
   mutation, tolerance, and remote-rank failure path.
5. Prove the runner stops at the first failed stage and reports the first mismatch.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python/test_moonep_correctness_runner.py -q
```

Expected: each injected defect fails at its owning stage with a stable diagnostic; no
later backend method is called; a remote-rank failure produces a coherent case failure.

**Artifacts and interfaces:** `run_reference_case` and `run_differential_case`, plus a
versioned stage-artifact schema.

## Task 6: Add the Normalized TileXR Backend Wrapper

**Status:** Completed and validated on 2026-08-05. The final four-rank
`skewed-padding` run passed Planning, Dispatch, PrefetchWeight, Combine, and ReduceGrad
against the independent Torch/HCCL reference on NPU 4-7. The report identifies
`tilexr_native` and records `performance_valid=false`.

**Objective and role:** Adapt the upstream-shaped TileXR Torch facade to the same
correctness protocol without coupling validators to ctypes/native object details.

**Dependencies:** Tasks 1 and 5, plus the native facade contract owned by
`docs/plans/2026-08-04-moonep-upstream-v1-prefetch-reduce.md`.

**Modification scope:**

- Create `tools/moonep/tilexr_backend.py`.
- Modify `integrations/moonep_torch/tilexr_moonep/torch_api.py` only for missing
  upstream-complete public data needed by the wrapper; coordinate with its owning
  native migration.
- Create `tests/moonep/python/test_tilexr_correctness_adapter.py`.

**Constraints and non-goals:** Do not emulate missing native outputs with reference
values. Do not translate compact `S*K` buffers into padded `NvS` while claiming native
correctness. If the installed adapter lacks a required capability, fail or mark the
case unavailable before stage execution with the exact missing field/capability.

**Work:**

1. Normalize plan/result names, plan lifecycle, synchronous completion, and in-place
   return state.
2. Validate that `B`, `P`, `NvS`, zero-fill, dedup, rank-3 weights, and rank-4 reduce
   buffers are genuinely accepted by the installed adapter.
3. Surface native asynchronous status after synchronization.
4. Add fake-adapter tests for all normalized calls, optional weights, saved-plan reuse,
   and unavailable capability reporting.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python/test_tilexr_correctness_adapter.py -q
python -m pytest tests/moonep/python/test_ffi_unittest.py -q
```

Expected: the wrapper passes exact normalized arguments to a compatible adapter and
rejects the current compact/stub surface without producing a false correctness pass.

**Artifacts and interfaces:** `TileXRMoonEPBackend`, interchangeable with the reference
backend in Task 5.

## Task 7: Separate Benchmark, Reference, and Correctness Modes

**Objective and role:** Integrate the reference and differential paths into the existing
launcher while preserving the meaning and timing of the native benchmark.

**Dependencies:** Tasks 2-6. Real TileXR differential execution uses the completed
Task 6 built-in candidate adapter.

**Modification scope:**

- Modify `tools/moonep/{benchmark,launcher,config,report}.py`.
- Modify `tools/moonep/run_benchmark.py` only if entry-point behavior changes.
- Modify `tools/moonep/README.md`.
- Create or modify benchmark CLI tests under `tests/moonep/python/`.

**Constraints and non-goals:** Keep existing behavior as `--mode benchmark`. Reference
and differential work is untimed. Do not include reference/HCCL latency in native
performance samples or derive performance claims from correctness cases.

**Work:**

1. Add `--mode {benchmark,reference,correctness}` and forward it through the managed
   launcher. Add a generic candidate-backend factory option for correctness mode;
   this is an injection boundary, not a TileXR adapter implementation.
2. `benchmark`: retain adapter-only warmup/timing and existing report compatibility.
3. `reference`: run only the Python backend and all five stage validators.
4. `correctness`: instantiate the reference and injected candidate backends, run the
   mandatory stage gate, and emit per-stage artifacts. Without a candidate factory,
   fail before stage execution with a precise unavailable message.
5. Extend aggregation with explicit backend names, comparison policy, stage status,
   failed rank/stage/tensor, and `performance_valid=false` for untimed modes.
6. Replace the old checksum/stub correctness label with an explicit legacy/native-status
   field where backward report compatibility is needed.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python -q
python tools/moonep/run_benchmark.py --help
```

On compatible Ascend hardware:

```bash
python tools/moonep/run_benchmark.py \
  --mode correctness \
  --candidate-backend tilexr_moonep_correctness:create_backend \
  --cases tools/moonep/cases/correctness.json \
  --world-size 4 --physical-device-count 4 \
  --output-dir Testing/moonep-differential
```

Validated result: each case reports five passed stages on every rank. Reference mode
remains independently runnable, correctness mode uses the built-in TileXR candidate by
default, and benchmark samples contain only TileXR timings.

**Artifacts and interfaces:** Stable CLI modes and report schema for CI/manual NPU
validation.

## Task 8: Regression, Hardware Evidence, Documentation, and Review

**Objective and role:** Establish credible evidence at every affected boundary and make
the correctness framework maintainable.

**Dependencies:** Tasks 1-7.

**Modification scope:**

- Modify `tools/moonep/README.md` and `docs/BUILD_VERIFICATION.md`.
- Add focused documentation under `docs/moonep/` only if the README cannot clearly hold
  the stage contracts and artifact schema.
- Review all files changed by this plan; do not alter unrelated native work.

**Work:**

1. Run CPU contract, case-factory, oracle, runner, adapter, CLI, and existing MoonEP
   Python regression tests.
2. On CANN 9.1/torch_npu, run one rank first, then four ranks on known A5/Ascend950
   devices with HCCL initialized.
3. Capture Planning exactness, all four downstream exactness/mutation checks, rank-wide
   agreement, and a deliberate mismatch diagnostic.
4. Run native benchmark mode separately and prove no reference/HCCL call is inside its
   timed region.
5. Inspect the diff for runtime imports from `3rdparty/moonep`, accidental changes to
   native ABI/kernel files, stale stub-correctness claims, and broad tolerances.
6. Use `superpowers-neo-requesting-code-review` because the change is cross-module and
   defines a shared test protocol. Use `superpowers-neo-verification-before-completion`
   before describing the work as complete.

**Acceptance and verification:**

```bash
python -m pytest tests/moonep/python -q
git diff --check
git grep -n -E 'from +3rdparty|import +3rdparty' -- tools/moonep tests/moonep/python
git diff --name-only -- src/moonep src/include
```

The final NPU evidence must state the exact CANN, torch, torch_npu, driver, device,
rank count, command, cases, and per-stage results. CPU tests alone do not prove NPU or
HCCL behavior; single-rank tests do not prove peer communication; A5/Ascend950 local
tests do not prove cross-node UDMA.

**Artifacts and interfaces:** Updated usage/verification documentation, retained
correctness artifacts, review findings, and a scoped completion report.

## Dependency Order and Coordination

```text
Task 1 contracts
  -> Task 2 deterministic cases
      -> Task 3 Planning reference
          -> Task 4 downstream references
              -> Task 5 validators/runner
                  -> Task 6 TileXR wrapper
                      -> Task 7 benchmark modes/reports
                          -> Task 8 hardware evidence/review
```

Task 6 was completed by the follow-on correctness-adapter workstream before final Task 7
and Task 8 validation. The generic candidate-backend injection contract remains
available for explicit overrides. Tasks 3 and 4 share the reference backend and execute
serially. No parallel writer was required in the dirty workspace.

## Completion Criteria

The plan is complete only when:

- Planning inputs and all defined Planning outputs are validated exactly.
- Every downstream stage validates canonical inputs, backend-local outputs, expected
  mutations, forbidden mutations, and reference-vs-TileXR values.
- All ranks agree before advancing to the next stage and stop coherently on failure.
- The Python backend executes with `torch_npu` tensors and real HCCL in a multi-rank
  test.
- The generic candidate boundary runs a conforming fake backend and reports a precise
  unavailable error when no real adapter is installed; it never produces a partial
  false pass.
- Native benchmark timing excludes the Python reference and correctness comparisons.
- Local and hardware evidence is reported with its actual validation boundary.
