# UDMA Full-Mesh GM Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record raw system-cycle spans for every physical full-mesh core task and convert the 8 MiB per-rank trace to Chrome Trace Event JSON.

**Architecture:** A shared header defines a fixed kernel-span area and runtime-indexed task spans. Host code owns allocation and file output; device workers only record nullable spans. A standalone Python converter reads one or more rank binaries and emits rank/core lanes.

**Tech Stack:** C++14 Host code, Ascend C device code, Python 3 unittest, Chrome Trace Event JSON.

## Global Constraints

- Trace allocation is exactly 8 MiB per rank.
- Record at most 50 iterations and cores 0 through 34.
- Preserve full-mesh communication and the 12:4 payload split.
- Do not average timestamps on device or Host.
- Commit before every remote bundle deployment.
- Use `TILEXR_IPC_PID_MODE=pid` for physical validation.

---

### Task 1: Define And Test The Binary Layout

**Files:**
- Create: `tests/udma/demo/tilexr_udma_fullmesh_trace.h`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_layout.cpp`
- Modify: `tests/udma/CMakeLists.txt`

**Interfaces:**
- Produces: `FullmeshTraceHeader`, `FullmeshTraceSpan`, `FullmeshTraceLayoutBytes`, `FullmeshTraceTaskSpanOffset`, and phase constants.

- [ ] **Step 1: Add failing layout assertions**

Require magic `0x464d5452`, version 1, 8 MiB, 4 KiB header, 50 iterations,
35 cores, two kernel regions, and 14 task phases. Assert that the physical
2x8 repeat50/pass1 layout fits and repeat50/pass4 does not.

- [ ] **Step 2: Commit and remotely verify RED**

Commit the test, create a full Git bundle, rebuild
`test_tilexr_udma_alltoall_layout` remotely, and confirm failure because the
new header and symbols do not exist.

- [ ] **Step 3: Implement the shared layout**

Define:

```cpp
constexpr uint32_t kFullmeshTraceMagic = 0x464d5452U;
constexpr uint32_t kFullmeshTraceVersion = 1U;
constexpr size_t kFullmeshTraceBytes = 8ULL * 1024ULL * 1024ULL;
constexpr size_t kFullmeshTraceHeaderBytes = 4096ULL;
constexpr uint32_t kFullmeshTraceMaxIterations = 50U;
constexpr uint32_t kFullmeshTraceMaxCores = 35U;
constexpr uint32_t kFullmeshTraceKernelRegions = 2U;
constexpr uint32_t kFullmeshTracePhaseCount = 14U;
constexpr uint64_t kFullmeshTraceCyclesPerUs = 1000ULL;
```

Use runtime `passCount` and `rankSize` in task indexing, but reserve kernel
spans for all 50 iterations and 35 cores. Return a value greater than 8 MiB
when multiplication would overflow or dimensions exceed their limits.

- [ ] **Step 4: Verify GREEN remotely and commit**

Run the layout test on both hosts, then commit the header, test, and CMake
dependency.

### Task 2: Build The Chrome Trace Converter

**Files:**
- Create: `tests/udma/demo/tilexr_udma_fullmesh_trace_to_chrome.py`
- Create: `tests/udma/unit/test_tilexr_udma_fullmesh_trace_to_chrome.py`

**Interfaces:**
- Produces: `read_rank_trace(path)` and `build_chrome_trace(rank_traces)`.

- [ ] **Step 1: Write failing converter tests**

Create a synthetic 8 MiB rank trace with one kernel span and task spans for
core16/peer8 data-put and core34/peer8 ACK. Require:

```python
self.assertEqual(data_put["tid"], 16)
self.assertEqual(data_put["args"]["peer"], 8)
self.assertEqual(data_put["dur"], 0.1)
self.assertEqual(ack["name"], "ACK")
self.assertNotIn("displayTimeUnit", trace)
```

Also test unknown versions, truncated files, half-written spans, capacity
overflow, and independent rank clock normalization.

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m unittest tests.udma.unit.test_tilexr_udma_fullmesh_trace_to_chrome -v
```

Expected: import failure because the converter does not exist.

- [ ] **Step 3: Implement the converter**

Parse header format `<8I4Q`, validate all dimensions and offsets, and index
task spans with:

```python
index = ((((iteration * MAX_CORES + core) * pass_count + pass_index) *
          rank_size + peer) * PHASE_COUNT + phase)
```

Emit metadata events for `rank N` and `coreN <role>`. Emit complete events
with raw cycle arguments plus iteration, pass, peer, phase, and role.

- [ ] **Step 4: Verify GREEN and commit**

Run the unittest command above and commit the converter and tests.

### Task 3: Add Host Allocation And Binary Output

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_layout.cpp`

**Interfaces:**
- Consumes: `kFullmeshTraceBytes` and `FullmeshTraceLayoutBytes`.
- Produces: launch arguments `GM_ADDR fullmeshTrace` and `uint32_t fullmeshTraceIteration`.

- [ ] **Step 1: Add failing Host source checks**

Require environment names `TILEXR_UDMA_FULLMESH_TRACE` and
`TILEXR_UDMA_FULLMESH_TRACE_DIR`, capacity validation, `aclrtMalloc`, H2D/D2H
copies, per-iteration launch indices, and output filename
`tilexr_fullmesh_trace_rank_<rank>.bin`.

- [ ] **Step 2: Verify RED remotely**

Commit and bundle the tests, rebuild the layout test remotely, and confirm the
new Host checks fail.

- [ ] **Step 3: Implement Host plumbing**

Enable tracing only for test type 7, multi-node full-mesh,
`REMOTE_PUT_ONLY=0`, and the environment flag. Initialize this header:

```cpp
FullmeshTraceHeader header {};
header.magic = kFullmeshTraceMagic;
header.version = kFullmeshTraceVersion;
header.rank = rank;
header.iterationCount = allToAllRepeat;
header.passCount = bigDataPlan.passCount;
header.coreCount = kFullmeshTraceMaxCores;
header.rankSize = rankSize;
header.phaseCount = kFullmeshTracePhaseCount;
header.cyclesPerUs = kFullmeshTraceCyclesPerUs;
header.traceBytes = kFullmeshTraceBytes;
header.kernelSpanOffset = kFullmeshTraceHeaderBytes;
header.taskSpanOffset = FullmeshTraceTaskSpanBaseOffset();
```

Pass null/zero when disabled. Copy and write all 8 MiB after stream sync and
before freeing the trace allocation. Treat every allocation/copy/write error
as demo failure.

- [ ] **Step 4: Verify Host tests and commit**

Rebuild Host and layout tests remotely, run the layout test, and commit.

### Task 4: Instrument Full-Mesh Core Tasks

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_layout.cpp`

**Interfaces:**
- Consumes: nullable trace pointer, iteration, pass count, rank size, peer, and phase.
- Produces: deterministic kernel and task spans for cores 0 through 34.

- [ ] **Step 1: Add failing kernel source checks**

Require `GetSystemCycle`, kernel/work span recording, and each phase in the
copy, remote-send, local-send, and receive worker slices. Require trace helpers
to guard `trace == nullptr` before reading cycles or writing GM.

- [ ] **Step 2: Verify RED remotely**

Commit and bundle the tests, rebuild the layout test, and confirm the phase
checks fail.

- [ ] **Step 3: Implement trace helpers**

Add guarded helpers:

```cpp
BigDataFullmeshTraceRecordKernelSpan(trace, iteration, core, region, begin, end);
BigDataFullmeshTraceRecordTaskSpan(
    trace, iteration, core, pass, peer, phase,
    passCount, rankSize, begin, end);
```

Each helper validates all dimensions before using `DataCopyPad` to write the
16-byte span.

- [ ] **Step 4: Instrument worker boundaries**

Record the 14 phases at the actual operation boundaries defined in the design.
Use `rank` as the peer index for pass and self-copy spans. Do not combine
separate peer operations into one span. Record kernel/work end spans on every
active full-mesh core after the pass loop.

- [ ] **Step 5: Build, run tests, and commit**

Build the Ascend C kernel and Host demo on both remote hosts. Run layout,
transport-layout, and converter tests. Commit only after all pass.

### Task 5: Run And Export Physical 2x8 Trace

**Files:**
- Create: `tmp/test0701_fullmesh_trace_2x8/` artifacts only.

**Interfaces:**
- Consumes: committed trace-enabled demo.
- Produces: 16 binary traces and one Chrome Trace JSON.

- [ ] **Step 1: Bundle and deploy**

Create and verify a complete branch bundle, upload it to both hosts, detach at
the committed HEAD, rebuild, install, and rerun unit tests.

- [ ] **Step 2: Run physical 2x8 repeat50**

Use:

```bash
export TILEXR_DEMO_BIGDATA_REMOTE_PUT_ONLY=0
export TILEXR_DEMO_BIGDATA_PROFILE_STAGE=8
export TILEXR_DEMO_ALLTOALL_REPEAT=50
export TILEXR_UDMA_FULLMESH_TRACE=1
export TILEXR_UDMA_FULLMESH_TRACE_DIR=<result_dir>
export TILEXR_IPC_PID_MODE=pid
```

Expected: 16/16 ranks pass full output validation and each writes one 8 MiB
binary trace.

- [ ] **Step 3: Download and convert**

Download all rank binaries into `tmp/test0701_fullmesh_trace_2x8/` and run:

```powershell
python tests/udma/demo/tilexr_udma_fullmesh_trace_to_chrome.py `
  tmp/test0701_fullmesh_trace_2x8/*.bin `
  --output tmp/test0701_fullmesh_trace_2x8/fullmesh_2x8_trace.json
```

Verify JSON parsing, event count, all 16 process IDs, all active core lanes,
and absence of `displayTimeUnit`.
