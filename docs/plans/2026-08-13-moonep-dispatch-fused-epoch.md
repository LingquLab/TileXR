# MoonEP Dispatch Fused Epoch Implementation Plan

## Scope

Implement the approved design in
`docs/specs/2026-08-13-moonep-dispatch-fused-epoch-design.md`. Preserve the
public MoonEP and Python APIs. Do not modify comparison-only `reference/` code or
add Weight-only Dispatch.

## Tasks

### 1. Layout And Host Contract

Update the URMA layout to append Hidden and Weight active regions rather than
overlay them. Extend launch parameters and the registered direct-Kernel ABI to
carry both pointer pairs and active offsets. Refactor `RunDispatchUrma` so a
paired call validates both descriptors but invokes the launcher once. Add layout
and Host/launch tests that prove one magic and one launch.

### 2. Fused WQE Accounting

Extend the common batch helpers and UB WQE builder so one selected route emits
one Hidden WQE and an optional immediately following Weight WQE on the same
logical QP. Preserve completion WQE ordering, SQ capacity reserve, ring wrap,
CQE final-BB accounting, and grouped staged-doorbell behavior. Add Host-testable
helpers for WQE counts and selected indices.

### 3. One-Epoch Kernel

Refactor the current single-payload Kernel body into one fused execution path.
Stage both sources before the first local barrier, scan routes once, service
both local and remote payloads, wait once, copy both outputs, and quiet once.
Keep communication convergence on device-detected errors and preserve sticky
status. Ensure UB resources are reset only after their producers/consumers have
completed.

### 4. Diagnostics And Tools

Version or feature-mark fused diagnostics without breaking existing record-size
consumers. Preserve separate per-payload records and final paired Kernel status.
Update `dispatch_hot_loop.py`, reporting tests, source guards, and API tests.

### 5. Validation And Documentation

Run focused C++ and Python tests, target CANN 9.1 Host/Kernel build, and inspect
the final diff. On `141.61.49.195`, establish NPU/CANN/source/binary provenance,
reuse the retained one-time matching HCCL Test result (create it only if this
environment has no prior record), deploy to a task-specific directory, and
execute the approved correctness/stability/performance ladder. Record reusable
fused-epoch workspace and completion-order lessons in the maintained Dispatch
design.
