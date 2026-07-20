# CCU AlltoAll Loop Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate ten consecutive two-rank CCU AlltoAll submissions using one prepared mission and loop-specific device synchronization markers.

**Architecture:** Extend PreSync with a marker loaded from SQE argument zero and synchronized through a dedicated XN with mask `0x1`; address and token retain masks `0x2` and `0x4`, and the receiver waits for `0x7`. The smoke probe prepares once, updates buffers and task arg zero per loop, submits repeatedly with phase-specific host gates, then validates both the peer marker and full destination data.

**Tech Stack:** C++14, CCU microcode encoders, Python `unittest`, Bash hardware runner, ACL runtime.

---

### Task 1: Encode The Loop Marker In PreSync

**Files:**
- Modify: `src/comm/ccu/tilexr_ccu_alltoall_program.h`
- Modify: `src/comm/ccu/tilexr_ccu_alltoall_program.cpp`
- Modify: `src/comm/ccu/tilexr_ccu_direct_orchestrator.cpp`
- Modify: `src/comm/ccu/tilexr_ccu_collective_planner.cpp`
- Test: `tests/ccu/test_tilexr_ccu_alltoall_program.py`
- Test: `tests/ccu/test_tilexr_ccu_direct_orchestrator.py`

- [ ] **Step 1: Write failing generator tests**

Assert that marker-enabled PreSync emits `LoadSqeArgsToX(markerLocalXn, 0)`, then `SyncXn(markerRemoteXn, markerLocalXn, preChannel, notifyCke, 0x1)`, followed by the existing address/token notifications and a `SetCke` wait mask of `0x7`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m unittest tests.ccu.test_tilexr_ccu_alltoall_program tests.ccu.test_tilexr_ccu_direct_orchestrator
```

Expected: failure because the program spec has no marker fields and the instruction count remains `5 + 64 * 7`.

- [ ] **Step 3: Implement marker-enabled PreSync**

Add `preSyncLocalMarkerXn`, `preSyncRemoteMarkerXn`, `preSyncMarkerArgIndex`, and `preSyncMarkerEnabled` to the program spec. Encode the marker load with `TileXRCcuEncodeLoadSqeArgsToX`, notify mask `0x1`, and wait mask `0x7`. Increase the AlltoAll instruction capacity to `7 + 64 * 7` and map the copy resource XNs to the marker pair.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

### Task 2: Repeat One Prepared Mission With Per-Loop Data

**Files:**
- Modify: `tests/ccu/ccu_tilexr_direct_smoke_probe.cpp`
- Modify: `tests/ccu/run_tilexr_ccu_direct_smoke.sh`
- Test: `tests/ccu/test_tilexr_ccu_direct_smoke_probe.py`
- Test: `tests/ccu/test_tilexr_ccu_direct_smoke_runner.py`

- [ ] **Step 1: Write failing loop-control tests**

Require `TILEXR_CCU_ALLTOALL_LOOP_COUNT`, a default of one, preparation outside the loop, task argument zero updated per loop, `phase=loopIndex` ready/done gates, source/destination refresh, peer marker readback, and loop-indexed result output.

- [ ] **Step 2: Run focused smoke tests and verify RED**

Run:

```bash
python3 -m unittest tests.ccu.test_tilexr_ccu_direct_smoke_probe tests.ccu.test_tilexr_ccu_direct_smoke_runner
```

Expected: failure because no loop-count environment variable or repeated submission exists.

- [ ] **Step 3: Implement repeated submission**

Add a validated loop count, rank-and-loop pattern generation, destination reset, marker encoding, prepared task arg mutation, phase-specific gates, marker XN readback, and per-loop result reporting. Keep allocation, registration, plan preparation, and installation outside the loop. Forward the environment variable from the runner.

- [ ] **Step 4: Run focused smoke tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

### Task 3: Regression And Hardware Verification

**Files:**
- Verify only; no planned production edits.

- [ ] **Step 1: Run affected CCU tests**

```bash
python3 -m unittest \
  tests.ccu.test_tilexr_ccu_alltoall_program \
  tests.ccu.test_tilexr_ccu_direct_orchestrator \
  tests.ccu.test_tilexr_ccu_direct_smoke_probe \
  tests.ccu.test_tilexr_ccu_direct_smoke_runner \
  tests.ccu.test_tilexr_ccu_lower_layer_plan_builder
```

Expected: zero failures.

- [ ] **Step 2: Build `tile-comm` and the smoke probe**

```bash
source scripts/common_env.sh
cmake --build build_ccu_direct --target tile-comm ccu_tilexr_direct_smoke_probe -j2
```

Expected: both targets build successfully.

- [ ] **Step 3: Wait for NPU 6 and 7 to become idle**

Poll `npu-smi info` through `tests/ccu/ccu_npu_smi_busy_guard.py --devices 6,7` every 30 seconds. Do not terminate unknown jobs.

- [ ] **Step 4: Run loop-count ten hardware validation**

Set `TILEXR_CCU_ALLTOALL_LOOP_COUNT=10` with the established two-rank long-mission configuration. Expected: each rank prints ten successful loop results, every peer marker matches its loop, and every loop reports `mismatches=0`.

- [ ] **Step 5: Inspect the final diff**

Run `git diff --check` and confirm unrelated untracked files remain untouched.
