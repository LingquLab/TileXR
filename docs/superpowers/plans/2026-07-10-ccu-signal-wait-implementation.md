# CCU Signal/Wait Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build internal two-rank Direct CCU signal/wait synchronization in `TileXRCcuBackend`, then compose a two-rank barrier from the same mechanism.

**Architecture:** Add internal request/plan types to the CCU backend boundary and route preparation through the existing Direct CCU install pipeline. Generate role-specific CCU synchronization programs, reuse lower-layer resource proof and repository/mission install, and validate with unit tests plus the two-rank smoke probe.

**Tech Stack:** C++14, CANN ACL/runtime, TileXR Direct CCU modules under `src/comm/ccu`, Python `unittest`, shell smoke runner.

## Global Constraints

- Do not expose public C API in `src/include/tilexr_api.h`.
- Do not link or include hcomm/HCCL private CCU producer APIs.
- Reuse existing Direct CCU runtime lifecycle, resource-window registration, peer allgather, lower-layer install, repository install, mission install, and `rtCCULaunch`.
- Implement single-direction signal/wait first, then two-direction barrier.
- Hardware validation must prove device-side synchronization with delayed signal timing, not host marker success.

---

### Task 1: Internal Backend Types and Submission Surface

**Files:**
- Modify: `src/comm/ccu/tilexr_ccu_backend.h`
- Modify: `src/comm/ccu/tilexr_ccu_backend.cpp`
- Test: `tests/ccu/test_tilexr_ccu_backend_boundary.py`

**Interfaces:**
- Produces: `enum class TileXRCcuSignalWaitRole { Signal, Wait, SignalAndWait };`
- Produces: `struct TileXRCcuSignalWaitRequest`
- Produces: `struct TileXRCcuSignalWaitPlan`
- Produces: `int TileXRCcuBackend::PrepareSignalWait(const TileXRCcuSignalWaitRequest&, TileXRCcuSignalWaitPlan*)`
- Produces: `int TileXRCcuBackend::SubmitSignalWait(const TileXRCcuSignalWaitPlan&, aclrtStream, TileXRCcuDirectSubmitReport*)`

- [ ] Add failing boundary tests that assert internal signal/wait types exist in `tilexr_ccu_backend.h` and do not appear in `src/include/tilexr_api.h`.
- [ ] Add backend declarations and simple forwarding methods.
- [ ] Run `python3 -m unittest tests.ccu.test_tilexr_ccu_backend_boundary`.

### Task 2: Signal/Wait Program Builder

**Files:**
- Create: `src/comm/ccu/tilexr_ccu_signal_wait_program.h`
- Create: `src/comm/ccu/tilexr_ccu_signal_wait_program.cpp`
- Modify: `src/comm/CMakeLists.txt`
- Test: `tests/ccu/test_tilexr_ccu_signal_wait_program.py`

**Interfaces:**
- Consumes: `TileXRCcuSignalWaitRole`
- Produces: `struct TileXRCcuSignalWaitProgramSpec`
- Produces: `int TileXRCcuBuildSignalWaitProgram(const TileXRCcuSignalWaitProgramSpec&, std::vector<TileXRCcuInstr>*, TileXRCcuBarrierProgramReport*)`

- [ ] Add tests for signal-only, wait-only, signal-and-wait, and invalid resource cases.
- [ ] Implement the builder using existing microcode encoders, not duplicate opcode packing.
- [ ] Add the new source file to the `tile-comm` target.
- [ ] Run `python3 -m unittest tests.ccu.test_tilexr_ccu_signal_wait_program`.

### Task 3: Planner Prepare Path

**Files:**
- Modify: `src/comm/ccu/tilexr_ccu_collective_planner.h`
- Modify: `src/comm/ccu/tilexr_ccu_collective_planner.cpp`
- Modify: `src/comm/ccu/tilexr_ccu_direct_orchestrator.h`
- Modify: `src/comm/ccu/tilexr_ccu_direct_orchestrator.cpp`
- Test: `tests/ccu/test_tilexr_ccu_direct_orchestrator.py`

**Interfaces:**
- Consumes: `TileXRCcuSignalWaitRequest`
- Produces: `int TileXRCcuCollectivePlanner::PrepareSignalWait(...)`
- Produces: `int TileXRCcuRunDirectSignalWaitInstallAttempt(...)`

- [ ] Add orchestrator tests that build signal/wait launch packages and verify submit tasks are produced when install evidence is satisfied.
- [ ] Implement direct signal/wait install attempt by reusing `RunDirectInstallAttemptImpl` structure and selecting the signal/wait program builder.
- [ ] Add planner method that validates rank size 2, peer rank, role, runtime availability, and basic info.
- [ ] Run `python3 -m unittest tests.ccu.test_tilexr_ccu_direct_orchestrator`.

### Task 4: Backend Wiring

**Files:**
- Modify: `src/comm/ccu/tilexr_ccu_backend.cpp`
- Test: `tests/ccu/test_tilexr_ccu_backend_boundary.py`

**Interfaces:**
- Consumes: `TileXRCcuCollectivePlanner::PrepareSignalWait`
- Produces: working `TileXRCcuBackend::PrepareSignalWait` and `SubmitSignalWait`

- [ ] Add tests for null plan, unavailable runtime, not-ready submit, and null stream.
- [ ] Wire backend preparation through planner and submission through `TileXRCcuSubmitPreparedTasks`.
- [ ] Run `python3 -m unittest tests.ccu.test_tilexr_ccu_backend_boundary`.

### Task 5: Smoke Probe and Runner

**Files:**
- Modify: `tests/ccu/ccu_tilexr_direct_smoke_probe.cpp`
- Modify: `tests/ccu/run_tilexr_ccu_direct_smoke.sh`
- Modify: `tests/ccu/test_tilexr_ccu_direct_smoke_probe.py`
- Modify: `tests/ccu/test_tilexr_ccu_direct_smoke_runner.py`

**Interfaces:**
- Consumes: `TileXRCcuBackend::PrepareSignalWait`
- Produces env-gated smoke modes:
  - `TILEXR_CCU_DIRECT_SMOKE_SIGNAL_WAIT=1`
  - `TILEXR_CCU_DIRECT_SMOKE_SIGNAL_RANK=<rank>`
  - `TILEXR_CCU_DIRECT_SMOKE_BARRIER=1`

- [ ] Add source tests for signal/wait mode selection and inactive/active timing behavior.
- [ ] Add runner env propagation and log assertions for signal/wait and barrier.
- [ ] Implement smoke probe path using internal backend methods.
- [ ] Run `python3 -m unittest tests.ccu.test_tilexr_ccu_direct_smoke_probe tests.ccu.test_tilexr_ccu_direct_smoke_runner`.

### Task 6: Local and Remote Validation

**Files:**
- No new source files; run validation commands.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified Direct CCU signal/wait and barrier behavior.

- [ ] Run focused CCU unit tests:
  `python3 -m unittest tests.ccu.test_tilexr_ccu_signal_wait_program tests.ccu.test_tilexr_ccu_direct_orchestrator tests.ccu.test_tilexr_ccu_backend_boundary tests.ccu.test_tilexr_ccu_direct_smoke_probe tests.ccu.test_tilexr_ccu_direct_smoke_runner`
- [ ] Build `tile-comm` on the NPU server.
- [ ] Run no-hcomm dependency guard.
- [ ] Run two-card `rank0 -> rank1` signal/wait smoke with delayed signal rank.
- [ ] Run two-card `rank1 -> rank0` signal/wait smoke with delayed signal rank.
- [ ] Run two-card barrier smoke.

