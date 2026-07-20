# Grouped AllToAll Route-Stage Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in three-stage grouped AllToAll diagnostic that measures local, six-port, and two-port traffic without overlap, then validate physical 2x8 bandwidth at 128 MiB per rank.

**Architecture:** Introduce a shared route-stage policy with a default combined mode and three diagnostic modes. In diagnostic mode the Host launches one filtered kernel per stage, synchronizes and executes the existing TCP all-rank barrier between stages, and writes one 8 MiB trace per stage; default execution remains one kernel with no added barrier.

**Tech Stack:** C++14 Host runtime, AscendC AIV kernel, TileXR UDMA same-QP put-signal/quiet, ACL runtime events, Python trace converter, physical CANN `/home/pkg/b101/cann`.

## Global Constraints

- Supported rank sizes remain `N * 8`, from 8 through 128.
- The default path retains one kernel launch and no Host barrier.
- Diagnostic stage order is exactly `local`, `primary`, `secondary`.
- Payload and ready signal use the same QP followed by `UDMAQuietStatusOnQp`.
- Do not add ACK or `SyncAll` to the grouped kernel.
- Keep invocation tokens and two-plane ping-pong unchanged across stages.
- Each stage trace remains exactly 8 MiB.
- Physical runs use `TILEXR_IPC_PID_MODE=pid`, `/home/pkg/b101/cann`, warmup 5,
  repeat 50, 16 copyout workers, and `timeout 60s`.

---

### Task 1: Route-Stage Policy

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_alltoall_group_route.h`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`

**Interfaces:**
- Produces: `enum class AllToAllGroupRouteStage : uint32_t` with `kCombined`,
  `kLocal`, `kPrimary`, and `kSecondary`.
- Produces: `AllToAllGroupValidRouteStage(uint32_t) -> bool`.
- Produces: `AllToAllGroupPeerInRouteStage(int, int, AllToAllGroupRouteStage) -> bool`.

- [ ] **Step 1: Write failing route-stage coverage tests**

Add `TestRouteStages()` and call it from `main()`:

```cpp
void TestRouteStages()
{
    using TileXR::Demo::AllToAllGroupRouteStage;
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidRouteStage(0U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidRouteStage(3U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidRouteStage(4U), false);
    for (int rankSize : {8, 16, 128}) {
        for (int rank = 0; rank < rankSize; ++rank) {
            int local = 0;
            int primary = 0;
            int secondary = 0;
            for (int peer = 0; peer < rankSize; ++peer) {
                if (peer == rank) continue;
                const bool l = TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kLocal);
                const bool p = TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kPrimary);
                const bool s = TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kSecondary);
                CHECK_EQ(static_cast<int>(l) + static_cast<int>(p) +
                    static_cast<int>(s), 1);
                local += l; primary += p; secondary += s;
                CHECK_EQ(TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kCombined), true);
            }
            CHECK_EQ(local, 7);
            CHECK_EQ(primary + secondary, rankSize - 8);
            CHECK_EQ(secondary, rankSize == 8 ? 0 : 2 * (rankSize / 8 - 1));
        }
    }
}
```

- [ ] **Step 2: Run the test and verify RED**

```bash
cmake --build tests/udma/build_b101 --target test_tilexr_udma_alltoall_group_layout -j
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_group_layout
```

Expected: compile failure because the stage enum and helpers do not exist.

- [ ] **Step 3: Implement the minimal shared policy**

Add after the existing route constants:

```cpp
enum class AllToAllGroupRouteStage : uint32_t {
    kCombined = 0U, kLocal = 1U, kPrimary = 2U, kSecondary = 3U,
};

inline bool AllToAllGroupValidRouteStage(uint32_t value)
{
    return value <= static_cast<uint32_t>(AllToAllGroupRouteStage::kSecondary);
}

inline bool AllToAllGroupPeerInRouteStage(
    int rank, int peer, AllToAllGroupRouteStage stage)
{
    if (rank < 0 || peer < 0 || rank == peer) return false;
    const bool crossNode = AllToAllGroupIsCrossNode(rank, peer);
    switch (stage) {
        case AllToAllGroupRouteStage::kCombined: return true;
        case AllToAllGroupRouteStage::kLocal: return !crossNode;
        case AllToAllGroupRouteStage::kPrimary:
            return crossNode && !AllToAllGroupUseSecondaryRoute(rank, peer);
        case AllToAllGroupRouteStage::kSecondary:
            return crossNode && AllToAllGroupUseSecondaryRoute(rank, peer);
    }
    return false;
}
```

- [ ] **Step 4: Run the test and verify GREEN**

Run Step 2 again. Expected: grouped layout checks pass.

- [ ] **Step 5: Commit**

```bash
git add tests/udma/demo/tilexr_udma_alltoall_group_route.h \
  tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp
git commit -m "feat(udma): classify grouped route stages"
```

---

### Task 2: Symmetric Kernel Stage Filtering

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`

**Interfaces:**
- Consumes: numeric stage values from Task 1.
- Extends: `launch_tilexr_udma_all_to_all_group(..., uint32_t copyoutWorkers,
  uint32_t routeStage)`.
- Produces: `AllToAllGroupPeerInRouteStageDevice(rank, peer, routeStage)`.

- [ ] **Step 1: Add failing kernel source guards**

```cpp
CHECK_CONTAINS(kernel, "uint32_t copyoutWorkers, uint32_t routeStage");
CHECK_CONTAINS(kernel, "AllToAllGroupPeerInRouteStageDevice");
CHECK_CONTAINS(kernel, "if (!AllToAllGroupPeerInRouteStageDevice(rank, peer, routeStage))");
CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL");
CHECK_NOT_CONTAINS(kernel, "SyncAll");
```

- [ ] **Step 2: Run the layout test and verify RED**

Run Task 1 Step 2. Expected: source guards fail.

- [ ] **Step 3: Implement device validation and filtering**

Add constants `0U..3U` matching Task 1. Reject values above 3 in the kernel
configuration check. Implement the predicate as:

```cpp
if (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_COMBINED) return true;
if (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL) return !crossNode;
if (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_PRIMARY)
    return crossNode && !AllToAllGroupUseSecondaryRouteDevice(rank, peer);
return crossNode && AllToAllGroupUseSecondaryRouteDevice(rank, peer);
```

Perform self-copy only for combined/local stages. Filter receive peers before
token calculation and waiting. Filter send peers before QP selection and UDMA.
Append `routeStage` to the kernel and wrapper signatures.

- [ ] **Step 4: Build and verify GREEN**

```bash
cmake --build tests/udma/build_b101 --target tilexr_udma_demo \
  test_tilexr_udma_alltoall_group_layout -j
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_group_layout
```

Expected: build and test pass.

- [ ] **Step 5: Commit**

```bash
git add tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp \
  tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp
git commit -m "feat(udma): filter grouped kernels by route stage"
```

---

### Task 3: Opt-In Host Scheduler And Independent Traces

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp`

**Interfaces:**
- Consumes: extended launch wrapper from Task 2 and existing
  `DemoBarrierAll(rank, rankSize, step)`.
- Produces: `TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES=0|1`.
- Produces: `tilexr_group_trace_{local|primary|secondary}_rank_<rank>.bin`.

- [ ] **Step 1: Add failing Host structure tests**

Require the option, exact stage names, ACL event timing, conditional barrier,
and default combined mode:

```cpp
CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES");
CHECK_CONTAINS(demo, "{\"local\", \"primary\", \"secondary\"}");
CHECK_CONTAINS(demo, "aclrtEventElapsedTime");
CHECK_CONTAINS(grouped, "DemoBarrierAll(rank, rankSize, barrierStep)");
CHECK_CONTAINS(grouped, "AllToAllGroupRouteStage::kCombined");
CHECK_CONTAINS(demo, "tilexr_group_trace_" + stageName + "_rank_");
```

Replace the old unconditional grouped-function `DemoBarrierAll` prohibition
with checks that the `routeStages == false` branch launches `kCombined`
directly and does not call the staged launch helper.

- [ ] **Step 2: Run the layout test and verify RED**

Run Task 1 Step 2. Expected: Host structure checks fail.

- [ ] **Step 3: Parse and validate the option**

```cpp
const int routeStagesValue = GetEnvInt(
    "TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES", 0);
if (routeStagesValue != 0 && routeStagesValue != 1) {
    std::cerr << "[rank " << rank
              << "] ERROR: TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES must be 0 or 1, got "
              << routeStagesValue << std::endl;
    return false;
}
const bool routeStages = routeStagesValue == 1;
```

- [ ] **Step 4: Generalize trace ownership and filenames**

Use one trace slot in default mode and three in staged mode. Each slot owns one
device pointer and one host vector initialized with the existing version-1
header. Change `WriteGroupTraceBinary` to accept `stageName`; omit the suffix
for combined mode and include it for diagnostic modes. The release lambda must
free every allocated trace pointer on partial failure.

- [ ] **Step 5: Implement stage launch and ACL timing**

Create/reuse ACL start and end events around one kernel launch. Synchronize the
end event, call `aclrtEventElapsedTime`, accumulate milliseconds by stage, then
call `DemoBarrierAll` only in diagnostic mode. Use one `invocationId` for all
three stages of a logical iteration and increment it after secondary completes:

```cpp
const AllToAllGroupRouteStage stages[] = {
    AllToAllGroupRouteStage::kLocal,
    AllToAllGroupRouteStage::kPrimary,
    AllToAllGroupRouteStage::kSecondary,
};
const char* stageNames[] = {"local", "primary", "secondary"};
```

Warmup follows the same stage sequence and barriers with null traces. Default
mode remains one `kCombined` launch per invocation.

- [ ] **Step 6: Report and validate**

Print mean device microseconds for each stage and their sum. Copy/write all
stage traces after repeat50, then run the existing debug check and full output
validation exactly once.

- [ ] **Step 7: Build and verify GREEN**

```bash
cmake --build tests/udma/build_b101 --target tilexr_udma_demo \
  test_tilexr_udma_alltoall_group_layout -j
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_group_layout
```

Expected: build and test pass.

- [ ] **Step 8: Commit**

```bash
git add tests/udma/demo/tilexr_udma_demo.cpp \
  tests/udma/unit/test_tilexr_udma_alltoall_group_layout.cpp
git commit -m "feat(udma): stage grouped route diagnostics"
```

---

### Task 4: Trace Compatibility And Regressions

**Files:**
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_group_trace_to_chrome.py`
- Verify unchanged: `tests/udma/demo/tilexr_udma_alltoall_group_trace_to_chrome.py`

**Interfaces:**
- Consumes: one standard version-1 binary per stage.
- Produces: proof that suffixed traces convert independently.

- [ ] **Step 1: Add a suffixed-file compatibility test**

```python
def test_reads_suffixed_stage_trace(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tilexr_group_trace_primary_rank_0.bin"
        self.write_valid_trace(path)
        trace = MODULE.read_rank_trace(path)
        chrome = MODULE.build_chrome_trace([trace])
        self.assertEqual(trace["path"], str(path))
        self.assertIn(str(path), chrome["otherData"]["sources"])
```

- [ ] **Step 2: Run the Python test**

```bash
python -m unittest tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
```

Expected: pass without converter changes. If the fixture hard-codes the old
filename, first observe that failure and then make only the fixture accept a
`path` argument.

- [ ] **Step 3: Run all local regressions**

```bash
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_group_layout
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_layout
./tests/udma/install_b101/bin/test_tilexr_udma_transport_layout
python -m unittest tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
git diff --check
```

Expected: all tests pass and `git diff --check` is silent.

- [ ] **Step 4: Commit**

```bash
git add tests/udma/unit/test_tilexr_udma_alltoall_group_trace_to_chrome.py
git commit -m "test(udma): cover grouped stage trace filenames"
```

---

### Task 5: Bundle, Deploy, And Measure Physical 2x8

**Files:**
- Create ignored: `tmp/udma-grouped-alltoall-route-stages.bundle`
- Create ignored artifacts under: `tmp/grouped_alltoall_route_stages_b101_2x8/`

**Interfaces:**
- Consumes: committed implementation HEAD and existing remote runner workflow.
- Produces: correctness logs, three loop49 traces, and isolated QP0/QP4 bandwidth.

- [ ] **Step 1: Create and verify a complete bundle**

```bash
git bundle create tmp/udma-grouped-alltoall-route-stages.bundle --all
git bundle verify tmp/udma-grouped-alltoall-route-stages.bundle
git bundle list-heads tmp/udma-grouped-alltoall-route-stages.bundle
```

Expected: verification succeeds and includes the current branch HEAD.

- [ ] **Step 2: Upload and build on `.223`**

Upload to both hosts, fetch into the dedicated validation checkout, reset that
validation branch to bundle HEAD, and build with:

```bash
export TILEXR_CANN_HOME=/home/pkg/b101/cann
source scripts/common_env.sh
cmake --build tests/udma/build_b101 --target tilexr_udma_demo \
  test_tilexr_udma_alltoall_group_layout test_tilexr_udma_alltoall_layout \
  test_tilexr_udma_transport_layout -j
```

Expected: build exits 0. Verify demo and grouped-kernel SHA-256 match on both
hosts.

- [ ] **Step 3: Run remote regressions**

Run all three C++ tests and Python trace tests on `.223`.

Expected: all pass before the physical workload starts.

- [ ] **Step 4: Run staged physical 2x8**

Use ranks 0-7 on `141.61.50.31` and ranks 8-15 on `141.61.49.223`:

```bash
export TILEXR_CANN_HOME=/home/pkg/b101/cann
export TILEXR_IPC_PID_MODE=pid
export TILEXR_DEMO_ALLTOALL_WARMUP=5
export TILEXR_DEMO_ALLTOALL_REPEAT=50
export TILEXR_DEMO_ALLTOALL_GROUP_COPYOUT_WORKERS=16
export TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES=1
export TILEXR_UDMA_GROUP_TRACE=1
export TILEXR_UDMA_GROUP_TRACE_DIR=/home/h30059441/grouped_route_stages_b101_2x8
timeout 60s ./tests/udma/install_b101/bin/tilexr_udma_demo \
  16 "$rank" 8 2097152 8 0
```

Expected: all ranks exit 0, validate output, and write three 8 MiB traces.

- [ ] **Step 5: Extract and validate loop49**

Convert each stage's 16 files independently to:

```text
tmp/grouped_alltoall_route_stages_b101_2x8/local/loop49.json
tmp/grouped_alltoall_route_stages_b101_2x8/primary/loop49.json
tmp/grouped_alltoall_route_stages_b101_2x8/secondary/loop49.json
```

Require:

```text
local same-node sends = 112, self-copy = 16
primary cross-node QP0 sends = 96
secondary cross-node QP4 sends = 32
duplicate peers across stages = 0
missing peers including self = 0
```

- [ ] **Step 6: Calculate isolated bandwidth**

```text
primary GiB/s = 384 MiB * 976.5625 / primary envelope us
secondary GiB/s = 128 MiB * 976.5625 / secondary envelope us
per-card GiB/s = node GiB/s / 8
```

Report both directions and averages beside concurrent loop49 baselines:

```text
QP0 concurrent = 731.72 / 859.54 GiB/s
QP4 concurrent = 294.95 / 286.71 GiB/s
```

- [ ] **Step 7: Final verification**

```bash
git status --short --branch
git diff --check
git bundle verify tmp/udma-grouped-alltoall-route-stages.bundle
```

Expected: no unexpected tracked changes, no whitespace errors, and a verified
bundle containing implementation HEAD.
