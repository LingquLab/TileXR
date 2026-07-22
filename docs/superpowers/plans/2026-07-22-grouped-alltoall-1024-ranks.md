# Grouped AllToAll 1024-Rank Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing grouped full-mesh AllToAll execution path and its TileXR Comm/UDMA dependencies to accept up to 1024 ranks.

**Architecture:** Raise the existing global fixed-array capacity instead of introducing a second communication ABI. Preserve full-mesh per-peer QPs and the 16-peer grouped schedule, while increasing grouped trace storage to 128 MiB.

**Tech Stack:** C++14, Ascend C, TileXR UDMA transport, CMake, physical Ascend950/CANN b101 regression.

## Global Constraints

- Maximum rank size is exactly 1024.
- Grouped AllToAll continues to require `rankSize % 8 == 0`.
- Group width remains 16 peers, split into eight positive and eight negative distances.
- Existing full-mesh per-peer QP and route behavior remains unchanged.
- Registered memory remains limited to 1 GiB per rank.
- Grouped trace capacity is exactly 128 MiB.
- Physical commands use a 60-second runtime timeout and `/home/pkg/b101/cann`.

---

### Task 1: Expand Comm And UDMA Registry Rank Capacity

**Files:**
- Modify: `tests/udma/unit/test_tilexr_udma_registry.cpp`
- Modify: `src/include/comm_args.h`

**Interfaces:**
- Consumes: `TileXR::TILEXR_MAX_RANK_SIZE` in Comm, transport, registry, and kernels.
- Produces: a global fixed-array capacity of 1024 ranks.

- [ ] **Step 1: Write the failing registry capacity test**

Change `TestRankScaleLimit()` to assert `TILEXR_MAX_RANK_SIZE == 1024`, construct a valid registry with `rankSize = 1024`, populate `regions[1023]`, and verify rank 1023 is addressable while rank 1024 is rejected.

- [ ] **Step 2: Run the registry test and verify RED**

Run:

```bash
cmake -S tests/udma -B tests/udma/build-local
cmake --build tests/udma/build-local --target test_tilexr_udma_registry -j8
./tests/udma/build-local/test_tilexr_udma_registry
```

Expected: FAIL because the current maximum is 256 and a 1024-rank registry is invalid.

- [ ] **Step 3: Raise the global capacity**

Change:

```cpp
constexpr int TILEXR_MAX_RANK_SIZE = 1024;
```

Keep the existing `CommArgs`, peer-memory, magic, count-matrix, and registry array layouts otherwise unchanged.

- [ ] **Step 4: Run the registry test and verify GREEN**

Run the commands from Step 2. Expected: `TileXR UDMA registry checks passed`.

- [ ] **Step 5: Commit**

```bash
git add src/include/comm_args.h tests/udma/unit/test_tilexr_udma_registry.cpp
git commit -m "feat(udma): expand communication capacity to 1024 ranks"
```

### Task 2: Expand Group Scheduling And Trace Capacity

**Files:**
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`
- Modify: `tests/udma/demo/tilexr_udma_alltoall_group_layout.h`
- Modify: `tests/udma/demo/tilexr_udma_alltoall_group_trace.h`

**Interfaces:**
- Consumes: `AllToAllGroupValidRankSize(int)`, `AllToAllGroupCount(int)`, and `AllToAllGroupTraceLayoutFits(uint32_t, uint32_t, uint32_t)`.
- Produces: valid grouped schedules through 1024 ranks and a 128 MiB trace capacity.

- [ ] **Step 1: Write failing 1024-rank scheduling tests**

Extend schedule coverage to `{8, 16, 24, 32, 40, 64, 128, 256, 512, 1024}`. Assert:

```cpp
CHECK_EQ(TileXR::Demo::AllToAllGroupCount(1024), 64U);
CHECK_EQ(TileXR::Demo::AllToAllGroupCount(1032), 0U);
```

Add a 128 MiB-per-rank plan using `rankSize = 1024` and `elementsPerPeer = 32768`; verify two payload planes fit below 1 GiB.

- [ ] **Step 2: Write failing trace capacity tests**

Include `tilexr_udma_alltoall_group_trace.h` and assert:

```cpp
CHECK_EQ(TileXR::Demo::kAllToAllGroupTraceBytes,
    128ULL * 1024ULL * 1024ULL);
CHECK_EQ(TileXR::Demo::AllToAllGroupTraceLayoutFits(50U, 64U, 4U), true);
CHECK_EQ(TileXR::Demo::AllToAllGroupTraceLayoutFits(50U, 64U, 5U), false);
```

- [ ] **Step 3: Run the grouped layout test and verify RED**

Run:

```bash
cmake --build tests/udma/build-local --target test_tilexr_udma_alltoall_group_layout -j8
./tests/udma/build-local/test_tilexr_udma_alltoall_group_layout
```

Expected: FAIL because rank 1024 is rejected and trace capacity is 8 MiB.

- [ ] **Step 4: Implement the new grouped limits**

Set:

```cpp
constexpr int32_t kAllToAllGroupMaxRankSize = 1024;
constexpr size_t kAllToAllGroupTraceBytes = 128ULL * 1024ULL * 1024ULL;
```

Do not change peer ordering, group-width arithmetic, pass limits, registered-memory limits, or trace record layout.

- [ ] **Step 5: Run grouped and registry tests and verify GREEN**

Run:

```bash
cmake --build tests/udma/build-local --target test_tilexr_udma_registry test_tilexr_udma_alltoall_group_layout -j8
./tests/udma/build-local/test_tilexr_udma_registry
./tests/udma/build-local/test_tilexr_udma_alltoall_group_layout
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/udma/demo/tilexr_udma_alltoall_group_layout.h tests/udma/demo/tilexr_udma_alltoall_group_trace.h tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp
git commit -m "feat(udma): support 1024-rank grouped alltoall"
```

### Task 3: Build And Physical 2x8 Regression

**Files:**
- Verify: `src/comm/tilexr_comm.cpp`
- Verify: `src/comm/udma/tilexr_udma_transport.cpp`
- Verify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Modify local test utility: `tmp/run_grouped_alltoall_b101_2x8.py`

**Interfaces:**
- Consumes: the 1024-rank Comm ABI and grouped constants from Tasks 1-2.
- Produces: build evidence and a correct physical 2x8 run.

- [ ] **Step 1: Run local source and unit verification**

```bash
cmake --build tests/udma/build-local -j8
ctest --test-dir tests/udma/build-local --output-on-failure
git diff --check
```

Expected: build succeeds, all configured tests pass, and `git diff --check` reports no errors.

- [ ] **Step 2: Commit the complete HEAD to a bundle and deploy both hosts**

Update the ignored local runner's trace-size check from `8388608` to
`134217728`, matching the new trace ABI. This utility-only edit is not committed.

```bash
git bundle create tmp/grouped_alltoall_1024_head.bundle HEAD
python tmp/deploy_grouped_bundle.py tmp/grouped_alltoall_1024_head.bundle /home/h30059441/tilexr_grouped_alltoall_b101 141.61.49.223 141.61.50.31
python tmp/build_grouped_peer_striping.py 141.61.49.223 141.61.50.31
```

Expected: both hosts check out the same HEAD; Bisheng build and grouped layout, AllToAll layout, transport layout, and trace converter tests pass.

- [ ] **Step 3: Check NPU availability**

```bash
python tmp/check_npu.py 141.61.49.223 141.61.50.31
```

Expected: devices 0-7 on both hosts show no running process. If occupied, run only as a correctness check and label performance untrusted.

- [ ] **Step 4: Run physical 2x8 regression**

```bash
python tmp/run_grouped_alltoall_b101_2x8.py target 48 0 /home/h30059441/tilexr_grouped_alltoall_b101 8 0 141.61.49.223 141.61.50.31
```

Expected: all 16 ranks report `ok=True`, all 16 trace files have the configured trace size, and the command finishes within its per-rank 60-second timeout.

- [ ] **Step 5: Record final evidence**

Report the local test results, remote commit, physical result directory, all-rank correctness, host mean, and any environment contention. Do not claim real 1024-rank runtime validation.
