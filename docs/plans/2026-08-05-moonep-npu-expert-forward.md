# MoonEP Torch-NPU Expert Forward Plan

## Goal

Replace the correctness adapter's preconstructed Expert Output with one mandatory
BF16 Expert Forward implementation:

`torch_npu.npu_grouped_matmul -> torch_npu.npu_swiglu -> torch_npu.npu_grouped_matmul`

The computation must consume each backend's own Dispatch output, normalized Plan, and
PrefetchWeight output before passing the weighted Expert Output to Combine.

## Scope

- Remove `CanonicalMoonEPCase.expert_output` and its deterministic construction.
- Add an Expert Forward result and correctness-runner stage between PrefetchWeight and
  Combine.
- Pack gate/up projection weights for GMM1 and use down projection weights for GMM2.
- Use the complete `plan.cu_seqlens` as the cumulative `group_list` for the `E+B`
  VM groups; the normalized Plan stores group ends and has no leading zero entry.
  Convert its ABI `int32` storage to a contiguous `int64` tensor at the Torch-NPU call
  boundary because `npu_grouped_matmul` requires an `int64` group list.
- Run both GMMs over the defined Dispatch prefix ending at `cu_seqlens[-1]`, then
  zero-fill the remaining physical `NvS` capacity before Combine.
- Clone the defined Dispatch prefix and all three prefetched projections at the
  Torch-NPU boundary so GMM owns independent input storage and cannot alter
  backend-owned Dispatch or PrefetchWeight results.
- Multiply GMM2 output by dispatched route weights before Combine, matching the
  existing Torch facade forward contract.
- Add CPU fake-API contract tests and run real Torch-NPU validation on the configured
  Ascend host.

## Non-Goals

- No precomputed or pure-Python Expert Forward fallback.
- No native TileXR or Ascend C grouped-matmul kernel.
- No INT4/INT8 activation or weight quantization.
- No changes to the five native MoonEP stage ABIs or their IPC-only transport.
- No Git staging, commit, push, or worktree cleanup.

## Implementation Tasks

### 1. Define and test the Expert Forward contract

**Objective:** Establish the mandatory grouped-GEMM data flow before production
implementation.

**Scope:** `tools/moonep/contracts.py`, `tests/moonep/python/`.

**Acceptance:** Tests require two `torch_npu.npu_grouped_matmul` calls, one
`torch_npu.npu_swiglu` call, packed `[E+B,H,2I]` GMM1 weights, `[E+B,I,H]` GMM2
weights, cumulative group boundaries, BF16 output shape `[NvS,H]`, and route-weight
application.

### 2. Implement and integrate the stage

**Objective:** Make the correctness runner execute real Expert Forward for every
backend after PrefetchWeight and before Combine.

**Scope:** `tools/moonep/expert_forward.py`, `tools/moonep/correctness.py`,
`tools/moonep/case_factory.py`, `tools/moonep/benchmark.py`, and narrowly related
report/CLI code if required by the existing stage-artifact contract. The timed forward
benchmark must use the same helper instead of its scalar placeholder.

**Constraints:** Import `torch_npu` lazily with a clear unavailable-backend error.
Keep each backend's tensors isolated. Do not emulate or replace backend results.

**Acceptance:** The precomputed field and path are absent; Combine receives the
Expert Forward output; the new stage participates in differential validation and
artifact reporting without changing native backend interfaces. Aggregation requires
the exact ordered six-stage list and one matching artifact file per stage and rank.

### 3. Update regressions and documentation

**Objective:** Align fixtures, expected stage lists, README flow, and launcher tests
with the six-checkpoint correctness pipeline.

**Scope:** `tests/moonep/python/`, `tools/moonep/README.md`, and relevant durable
MoonEP correctness documentation only.

**Acceptance:** Focused contract tests and the complete local Python suite pass; source
searches find no `expert_output` case field or precomputed Expert Forward branch.

### 4. Validate on Ascend hardware

**Objective:** Prove the actual Torch-NPU operators execute in the TileXR end-to-end
flow.

**Prerequisites:** Mutagen session `tilexr-moonep`; CANN
`/home/pkg/b131/cann-9.1.0`; conda `ai_moe_test`; install prefix
`/home/c30061605/TileXR/install-moonep-b131-20260805`; NPU 4-7 when available.

**Acceptance:** A bounded one-rank run first, then a four-rank differential run when
resources permit. Results must contain a passing ExpertForward artifact in addition to
the five native MoonEP stages, and logs must show no fallback or precomputed path.

## Verification Strategy

1. Focused fake Torch-NPU unit tests prove operator signatures, shapes, ordering, and
   error handling without requiring local CANN.
2. Full MoonEP Python regression protects normalized contracts, backend lifecycle,
   stage comparison, and CLI propagation.
3. `compileall`, `git diff --check`, and source searches catch syntax, whitespace, and
   stale-path defects.
4. Real NPU runs are required for runtime/operator compatibility claims; local fake
   tests do not prove `npu_grouped_matmul` execution on AICore.

## Key Risks

- `npu_grouped_matmul` signature and accepted group-list layout are Torch-NPU-version
  specific; use the installed `ai_moe_test` behavior as authoritative runtime evidence.
- BF16 GMM output can differ across backend data paths; Expert Forward and downstream
  floating-point comparisons require explicit tolerances rather than exact equality.
- Empty VM groups must remain represented by repeated cumulative boundaries so weight
  rows stay aligned with `E+B` group indices.
- Dispatch allocates physical `NvS` capacity, while its defined prefix can be shorter;
  passing the undefined tail to GMM would violate the cumulative group-list contract.

## Completion Evidence

- Local MoonEP Python suite: 88/88 passed, plus `compileall` and `git diff --check`.
- Remote environment: CANN `/home/pkg/b131/cann-9.1.0`, Torch-NPU 2.10.0.post2,
  driver 25.1.rc1.b188, Ascend950PR.
- One-rank `planning-small` differential passed all six checkpoints at
  `/tmp/tilexr-moonep-expert-forward-gmm-1r-v4-20260805`.
- Four-rank `skewed-padding` differential passed on physical NPU 4-7 with 24 stage
  artifacts at `/tmp/tilexr-moonep-expert-forward-gmm-4r-20260805`.
- One-rank benchmark mode passed the complete native forward/backward flow and emitted
  an `expert_forward` device-event timing at
  `/tmp/tilexr-moonep-expert-forward-benchmark-1r-20260805`.
- Fresh four-rank runs for all three modes passed on physical NPU 4-7:
  `/tmp/tilexr-moonep-allmodes-benchmark-4r-20260805`,
  `/tmp/tilexr-moonep-allmodes-reference-4r-20260805`, and
  `/tmp/tilexr-moonep-allmodes-correctness-4r-20260805`. Reference and correctness
  each contain 24 ordered six-stage artifacts; benchmark records a positive
  `expert_forward` device-event timing on every rank.
- The passing correctness runs report `candidate_backend=tilexr_native`, include a passing
  `expert_forward` artifact, and contain no precomputed or fallback marker.
