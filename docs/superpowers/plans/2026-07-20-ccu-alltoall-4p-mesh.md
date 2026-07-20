# CCU AlltoAll 4P Mesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and validate one true four-rank direct-CCU Mesh1D AlltoAll mission that moves four 2 MiB chunks per rank, including the self chunk, and supports ten submissions with unchanged communicator, mission, and transport resources.

**Architecture:** Keep the validated two-rank API unchanged and add a four-rank Mesh program, orchestrator entry point, and planner entry point. Each rank owns three peer descriptors sorted by peer rank and nine sync routes (`copy`, `pre`, `token` per peer); the mission publishes all three peer handshakes before any wait, then performs three remote CCU copies plus one CCU local copy and waits for all completions. Prepare/import/install happens once, while each loop rewrites data and the SQE marker and uses phase-specific four-rank ready/done gates.

**Tech Stack:** C++14, TileXR direct CCU runtime and microcode encoders, Python `unittest` source/compile probes, Bash hardware runner, CMake, ACL runtime, Mutagen, SSH.

---

## File Map

- `src/comm/ccu/tilexr_ccu_microcode.{h,cpp}`: encode the hardware local-memory-to-local-memory transfer instruction.
- `src/comm/ccu/tilexr_ccu_alltoall_program.{h,cpp}`: describe and build one rank's four-rank Mesh mission.
- `src/comm/ccu/tilexr_ccu_direct_orchestrator.{h,cpp}`: allocate nine resources, build the Mesh program, check capacity, and package one prepared mission.
- `src/comm/ccu/tilexr_ccu_collective_planner.{h,cpp}`: AllGather four endpoints, import three destinations, map nine routes, and invoke the Mesh orchestrator.
- `tests/ccu/ccu_tilexr_direct_smoke_probe.cpp`: prepare once, submit repeatedly, verify three peer markers and all 8 MiB of receive data.
- `tests/ccu/run_tilexr_ccu_direct_smoke.sh`: launch and evaluate a rank-size-driven process set.
- `tests/ccu/test_tilexr_ccu_*.py`: unit, source-contract, compile-probe, and runner tests for every layer.

### Task 1: Encode CCU Local-To-Local Transfer

**Files:**
- Modify: `src/comm/ccu/tilexr_ccu_microcode.h`
- Modify: `src/comm/ccu/tilexr_ccu_microcode.cpp`
- Test: `tests/ccu/test_tilexr_ccu_microcode.py`

- [ ] **Step 1: Write the failing encoder tests**

Add a compile-and-run probe that calls the new API with distinct field values and decodes all four words:

```cpp
TileXR::TileXRCcuMemTransferSpec spec;
spec.localGsa = 0x101;
spec.localXn = 0x102;
spec.remoteGsa = 0x201; // destination GSA for the local-copy opcode
spec.remoteXn = 0x202;  // destination XN for the local-copy opcode
spec.lengthXn = 0x103;
spec.channelId = 0x104;
spec.setCkeId = 0x105;
spec.setCkeMask = 0x7;
spec.waitCkeId = 0x106;
spec.waitCkeMask = 0x8;
TileXR::TileXRCcuInstr instr;
assert(TileXR::TileXRCcuEncodeTransLocMemToLocMem(spec, &instr) == TileXR::TILEXR_SUCCESS);
assert(slot(instr.words[0], 0) == 0x100a);
assert(slot(instr.words[0], 1) == spec.remoteGsa);
assert(slot(instr.words[0], 2) == spec.remoteXn);
assert(slot(instr.words[0], 3) == spec.localGsa);
assert(slot(instr.words[1], 0) == spec.localXn);
assert(slot(instr.words[1], 1) == spec.lengthXn);
assert(slot(instr.words[1], 2) == spec.channelId);
assert(slot(instr.words[3], 0) == spec.setCkeId);
assert(slot(instr.words[3], 1) == spec.setCkeMask);
assert(slot(instr.words[3], 2) == spec.waitCkeId);
assert(slot(instr.words[3], 3) == spec.waitCkeMask);
```

Also assert null output, zero GSA/XN/length/channel, half-specified CKE pairs, and invalid reduce fields return `TILEXR_ERROR_PARA_CHECK_FAIL`.

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_microcode -v`

Expected: FAIL because `TileXRCcuEncodeTransLocMemToLocMem` is undeclared/undefined.

- [ ] **Step 3: Add the public declaration and minimal encoder**

Add to the header:

```cpp
int TileXRCcuEncodeTransLocMemToLocMem(
    const TileXRCcuMemTransferSpec& spec,
    TileXRCcuInstr* instr);
```

Add opcode `0x100a` beside the two existing transfer opcodes and encode using the same validation and flag packing:

```cpp
int TileXRCcuEncodeTransLocMemToLocMem(const TileXRCcuMemTransferSpec& spec, TileXRCcuInstr* instr)
{
    if (ValidateInstrOutput(instr) != TILEXR_SUCCESS || ValidateTransferSpec(spec) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    instr->words[0] = PackSlots(TILEXR_CCU_TRANS_LOC_MEM_TO_LOC_MEM_HEADER,
        spec.remoteGsa, spec.remoteXn, spec.localGsa);
    instr->words[1] = PackSlots(spec.localXn, spec.lengthXn, spec.channelId, TransferControlSlot(spec));
    instr->words[2] = PackSlots(0, 0, 0, TransferFlagSlot(spec));
    instr->words[3] = PackSlots(spec.setCkeId, spec.setCkeMask, spec.waitCkeId, spec.waitCkeMask);
    return TILEXR_SUCCESS;
}
```

- [ ] **Step 4: Run the focused and full microcode tests**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_microcode -v`

Expected: PASS, including the existing remote/local transfer encoders.

- [ ] **Step 5: Commit**

```bash
git add src/comm/ccu/tilexr_ccu_microcode.h src/comm/ccu/tilexr_ccu_microcode.cpp tests/ccu/test_tilexr_ccu_microcode.py
git commit -m "feat(ccu): encode local memory transfers"
```

### Task 2: Build The Four-Rank Mesh Mission

**Files:**
- Modify: `src/comm/ccu/tilexr_ccu_alltoall_program.h`
- Modify: `src/comm/ccu/tilexr_ccu_alltoall_program.cpp`
- Test: `tests/ccu/test_tilexr_ccu_alltoall_program.py`

- [ ] **Step 1: Add failing rank-parameterized Mesh tests**

Define a fixture for each `localRank` in `0..3` with peers in intentionally unsorted input order. Decode the generated instructions and assert:

```python
self.assertEqual([p for p in range(4) if p != local_rank], report.peerRanks)
self.assertEqual(3, report.peerCount)
self.assertEqual(9, report.syncResourceCount)
self.assertEqual(64, report.remoteBlockCountPerPeer)
self.assertEqual(64, report.selfBlockCount)
self.assertLess(last_peer_post_index, first_peer_wait_index)
self.assertEqual(local_rank * 2 * 1024 * 1024, decoded_self_source_offset)
self.assertEqual(local_rank * 2 * 1024 * 1024, decoded_self_destination_offset)
```

For every peer `p`, assert remote source offset is `localRank * chunkBytes`, remote destination offset is `localRank * chunkBytes`, the peer route IDs are `3*ordinal+{0,1,2}`, and the PreSync wait mask is `0x7`. Add failures for rank size other than four, duplicate/missing/self peer, zero token/address, non-2-MiB-aligned size, duplicate resource IDs, and insufficient output pointer/report.

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_alltoall_program -v`

Expected: FAIL because the Mesh types and builder do not exist.

- [ ] **Step 3: Add explicit Mesh types**

Add these interfaces without changing `TileXRCcuAllToAll2RankProgramSpec`:

```cpp
struct TileXRCcuAllToAllMeshPeerSpec {
    uint32_t peerRank = 0;
    uint64_t remoteRecvAddr = 0;
    uint64_t remoteRecvToken = 0;
    uint16_t copyResourceIndex = 0;
    uint16_t preSyncResourceIndex = 0;
    uint16_t tokenResourceIndex = 0;
};

struct TileXRCcuAllToAllMeshProgramSpec {
    uint32_t rankSize = 4;
    uint32_t localRank = 0;
    uint64_t localSendAddr = 0;
    uint64_t localSendToken = 0;
    uint64_t localRecvAddr = 0;
    uint64_t localRecvToken = 0;
    uint64_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    uint16_t markerArgIndex = 0;
    uint16_t localGsa = 0;
    uint16_t selfDestinationGsa = 0;
    uint16_t localXn = 0;
    uint16_t selfDestinationXn = 0;
    uint16_t lengthXn = 0;
    uint16_t selfChannelId = 0;
    uint16_t selfCompletionCke = 0;
    std::vector<TileXRCcuAllToAllMeshPeerSpec> peers;
};

int TileXRCcuBuildAllToAllMeshProgram(
    const TileXRCcuAllToAllMeshProgramSpec& spec,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report);
```

Extend the report with peer ranks, peer/resource counts, self/remote block counts, and local/remote completion counts.

- [ ] **Step 4: Implement deterministic validation and emission**

Implement this exact high-level ordering using the existing `Append*` helpers and the Task 1 local-copy encoder:

```cpp
ValidateFourRankShapeAndUniqueResources(spec);
auto peers = spec.peers;
std::sort(peers.begin(), peers.end(), ByPeerRank);
LoadMarkerFromSqeArg(spec.markerArgIndex);
for (const auto& peer : peers) {
    PostMarker(peer);
    PostRemoteDestinationAddress(peer);
    PostRemoteDestinationToken(peer);
}
for (const auto& peer : peers) {
    WaitForPeerPreSync(peer, 0x7);
}
for (const auto& peer : peers) {
    EmitRemoteBlocks(peer, spec.localRank * spec.chunkBytes, spec.chunkBytes);
}
EmitLocalBlocks(spec.localRank * spec.chunkBytes, spec.chunkBytes);
for (const auto& peer : peers) {
    WaitForRemoteCopyCompletion(peer);
}
WaitForSelfCopyCompletion(spec.selfCompletionCke);
EmitFinish();
```

Do not reuse one peer's channel/XN/CKE IDs, do not emit a wait inside the publish loop, and derive `report.totalInstructionCount` from `program->size()`.

- [ ] **Step 5: Run Mesh and 2P regression tests**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_alltoall_program -v`

Expected: PASS for ranks 0..3 and all existing 2P cases.

- [ ] **Step 6: Commit**

```bash
git add src/comm/ccu/tilexr_ccu_alltoall_program.h src/comm/ccu/tilexr_ccu_alltoall_program.cpp tests/ccu/test_tilexr_ccu_alltoall_program.py
git commit -m "feat(ccu): build four-rank all-to-all mesh mission"
```

### Task 3: Allocate And Package Mesh Resources

**Files:**
- Modify: `src/comm/ccu/tilexr_ccu_direct_orchestrator.h`
- Modify: `src/comm/ccu/tilexr_ccu_direct_orchestrator.cpp`
- Test: `tests/ccu/test_tilexr_ccu_direct_orchestrator.py`

- [ ] **Step 1: Write failing resource/package tests**

Add offline tests that provide a four-peer endpoint spec and assert one mission, one task, nine sync resources, peer-ordinal route mapping `0/1/2`, `3/4/5`, `6/7/8`, and exact program-sized repository installation. Mutate each basic-info capacity (`mission`, `instruction`, `channel`, `xn`, `cke`, `gsa`) below the requested count and require the report message to contain both `requested=` and `available=`.

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_direct_orchestrator -v`

Expected: FAIL because `TileXRCcuRunDirectAllToAllMeshInstallAttempt` is missing.

- [ ] **Step 3: Add the orchestrator API**

```cpp
struct TileXRCcuDirectAllToAllMeshPeerSpec {
    uint32_t peerRank = 0;
    uint64_t remoteRecvAddr = 0;
    uint64_t remoteRecvToken = 0;
};

struct TileXRCcuDirectAllToAllMeshSpec {
    uint32_t rankSize = 4;
    uint32_t localRank = 0;
    uint64_t localSendAddr = 0;
    uint64_t localSendToken = 0;
    uint64_t localRecvAddr = 0;
    uint64_t localRecvToken = 0;
    uint64_t chunkBytes = 2ULL * 1024ULL * 1024ULL;
    std::vector<TileXRCcuDirectAllToAllMeshPeerSpec> peers;
};

int TileXRCcuRunDirectAllToAllMeshInstallAttempt(
    const TileXRCcuDirectInstallOptions& options,
    const TileXRCcuDirectAllToAllMeshSpec& alltoall,
    TileXRCcuDirectInstallAttempt* attempt,
    TileXRCcuDirectInstallReport* report);
```

- [ ] **Step 4: Build once to size, validate capacity, then install once**

Create a resource request with `syncResourceCount=9`, allocate distinct copy/pre/token resources per peer, reserve self-copy GSA/XN/channel/CKE, build the Mesh program into a temporary vector, and use its exact size before repository installation:

```cpp
const uint32_t requestedInstructions = static_cast<uint32_t>(program.size());
const uint32_t availableInstructions = RangeAvailableCount(
    attempt->specInfo.instructionNum, options.instructionStartId);
if (requestedInstructions > availableInstructions) {
    return CapacityError("instruction", requestedInstructions,
        availableInstructions, report);
}
attempt->package.missions.resize(1);
attempt->submitTasks.resize(1);
attempt->submitTasks[0].argSize = std::max<uint16_t>(1, options.sqeArgCount);
```

Preserve the existing 2P path byte-for-byte except for shared pure helpers extracted to avoid duplication.

- [ ] **Step 5: Run orchestrator and AlltoAll tests**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_direct_orchestrator tests.ccu.test_tilexr_ccu_alltoall_program -v`

Expected: PASS; invalid capacities identify the exhausted resource and exact counts.

- [ ] **Step 6: Commit**

```bash
git add src/comm/ccu/tilexr_ccu_direct_orchestrator.h src/comm/ccu/tilexr_ccu_direct_orchestrator.cpp tests/ccu/test_tilexr_ccu_direct_orchestrator.py
git commit -m "feat(ccu): package four-rank all-to-all resources"
```

### Task 4: Gather Four Endpoints And Import Three Destinations

**Files:**
- Modify: `src/comm/ccu/tilexr_ccu_collective_planner.h`
- Modify: `src/comm/ccu/tilexr_ccu_collective_planner.cpp`
- Test: `tests/ccu/test_tilexr_ccu_collective_planner.py`
- Test: `tests/ccu/test_tilexr_ccu_direct_backend.py`

- [ ] **Step 1: Write failing planner tests**

Use the existing fake session/backend to return four gathered `DirectCcuMemoryCopyEndpoint` values. Assert exactly one AllGather, exactly three imports in peer-rank order, nine lower-layer routes, three distinct imported `targetSegVa` values repeated only across that peer's three routes, and no import for the local endpoint. Add invalid endpoint tests for wrong rank size, duplicate endpoint rank, `valid=0`, wrong byte count, missing token, and failed import.

- [ ] **Step 2: Run planner tests to verify RED**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_collective_planner tests.ccu.test_tilexr_ccu_direct_backend -v`

Expected: FAIL because the Mesh planner API and multi-override storage are absent.

- [ ] **Step 3: Add the Mesh planner entry point**

```cpp
int PrepareDirectCcuAllToAllMeshInstallAttempt(
    TileXRCcuRuntimeSession& session,
    const TileXRCcuDirectInstallOptions& options,
    uint64_t localSourceAddr,
    uint64_t localDestinationAddr,
    uint64_t chunkBytes,
    TileXRCcuDirectInstallAttempt* attempt,
    TileXRCcuDirectInstallReport* report);
```

Require `session.RankSize() == 4` and register `4 * chunkBytes` for both local buffers while retaining `chunkBytes` as the per-peer transfer size.

- [ ] **Step 4: Replace the single testing override with route-indexed overrides**

```cpp
struct DirectCcuRemoteRouteMemoryOverride {
    uint32_t syncRouteIndex = 0;
    TileXRCcuRemoteCcuBufferInfo buffer;
};
std::vector<DirectCcuRemoteRouteMemoryOverride> directCcuRemoteRouteMemoryOverrides_;
```

`SetDirectCcuRemoteRouteMemoryOverride` must keep its old all-routes behavior for 2P tests. `SetDirectCcuRemoteRouteMemoryOverrideForSyncRoute` updates/inserts one indexed entry, `Apply...` overlays every matching route, and `Clear...` clears the vector.

- [ ] **Step 5: Implement one gather and three imports**

```cpp
std::vector<DirectCcuMemoryCopyEndpoint> endpoints(4);
TILEXR_RETURN_IF_ERROR(session.AllGather(&localEndpoint, sizeof(localEndpoint), endpoints.data()));
for (uint32_t peer = 0, ordinal = 0; peer < 4; ++peer) {
    if (peer == static_cast<uint32_t>(session.Rank())) continue;
    ValidateEndpoint(endpoints[peer], peer, 4 * chunkBytes);
    auto imported = ImportDestination(session, endpoints[peer]);
    mesh.peers.push_back({peer, endpoints[peer].destinationRemoteImport.addr,
        TileXRCcuPackMemoryToken(endpoints[peer].destinationRemoteImport.tokenId,
            endpoints[peer].destinationRemoteImport.tokenValue, true)});
    for (uint32_t route = 0; route < 3; ++route) {
        SetDirectCcuRemoteRouteMemoryOverrideForSyncRoute(3 * ordinal + route, peer,
            imported.targetSegVa, endpoints[peer].destinationRemoteImport.tokenId,
            endpoints[peer].destinationRemoteImport.rawTokenId,
            endpoints[peer].destinationRemoteImport.tokenValue);
    }
    ++ordinal;
}
```

Set `syncResourceCount=9`, derive `syncInstructionCount` from the built Mesh program rather than a stale constant, invoke the Mesh orchestrator, and clear overrides on every return path with a small RAII guard.

- [ ] **Step 6: Run planner, backend, and 2P regression tests**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_collective_planner tests.ccu.test_tilexr_ccu_direct_backend tests.ccu.test_tilexr_ccu_direct_orchestrator -v`

Expected: PASS; the existing 2P single-route behavior remains unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/comm/ccu/tilexr_ccu_collective_planner.h src/comm/ccu/tilexr_ccu_collective_planner.cpp tests/ccu/test_tilexr_ccu_collective_planner.py tests/ccu/test_tilexr_ccu_direct_backend.py
git commit -m "feat(ccu): plan four-rank all-to-all endpoints"
```

### Task 5: Reuse One Prepared Mission For Ten Mesh Submissions

**Files:**
- Modify: `tests/ccu/ccu_tilexr_direct_smoke_probe.cpp`
- Modify: `tests/ccu/test_tilexr_ccu_direct_smoke_probe.py`

- [ ] **Step 1: Write failing source-contract and helper tests**

Add tests requiring Mesh mode to allocate `rankSize * chunkBytes`, call `PrepareDirectCcuAllToAllMeshInstallAttempt` before the loop, create the stream before the loop, and call neither prepare nor stream creation inside it. Require a phase path containing `ready.phase<loopIndex>` and `done.phase<loopIndex>`, three marker validations per loop, full-buffer validation, and failure output fields `rank`, `loopIndex`, `peerRank`, `route`, `channel`, `xn`, `cke`, `currentInstruction`, `selfCopyCompletion`, `sourceRank`, `chunkOffset`, and `globalOffset`.

- [ ] **Step 2: Run smoke-probe tests to verify RED**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_direct_smoke_probe -v`

Expected: FAIL because the probe remains two-rank-specific.

- [ ] **Step 3: Add generation-specific data and marker helpers**

```cpp
uint8_t ExpectedAllToAllByte(uint32_t source, uint32_t target, uint32_t loop, uint64_t offset)
{
    return static_cast<uint8_t>((source * 67U + target * 29U + loop * 17U +
        static_cast<uint32_t>(offset * 13U)) & 0xffU);
}

uint64_t MakeLoopMarker(uint32_t rank, uint32_t loop)
{
    return 0x5458524100000000ULL | (static_cast<uint64_t>(rank & 0xffU) << 8U) |
        static_cast<uint64_t>(loop & 0xffU);
}
```

Fill `send[target][offset]` from these fields, reset all 8 MiB of receive memory each loop, and validate `recv[source][offset]` with source/local-rank/loop.

- [ ] **Step 4: Prepare once and execute phase-isolated loops**

```cpp
PrepareMeshOnce(&attempt, &report);
CreateStreamOnce(&stream);
const auto stable = CaptureMissionAndResourceIdentity(attempt);
for (uint32_t loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    FillFourSendChunks(rank, loopIndex);
    ResetCompleteReceiveBuffer();
    attempt.submitTasks[0].args[0] = MakeLoopMarker(rank, loopIndex);
    FourRankGate(workDir / ("ready.phase" + std::to_string(loopIndex)), rank, 0);
    SubmitAndSynchronizeSameTask(attempt.submitTasks[0], stream);
    ValidateThreePeerMarkers(loopIndex);
    const auto localResult = ValidateCompleteReceiveBuffer(rank, loopIndex);
    FourRankGate(workDir / ("done.phase" + std::to_string(loopIndex)), rank, localResult);
    StopAllRanksIfAnyDoneResultFailed(loopIndex);
    AssertMissionAndResourceIdentity(stable, attempt, loopIndex);
}
```

The ready/done payload must include the loop marker, not a reused boolean. Any failure prints diagnostics and exits before beginning the next phase.

- [ ] **Step 5: Run probe tests**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_direct_smoke_probe -v`

Expected: PASS for Mesh contracts and existing 2P loop-reuse contracts.

- [ ] **Step 6: Commit**

```bash
git add tests/ccu/ccu_tilexr_direct_smoke_probe.cpp tests/ccu/test_tilexr_ccu_direct_smoke_probe.py
git commit -m "test(ccu): exercise repeated four-rank all-to-all"
```

### Task 6: Generalize The Smoke Runner To Four Processes

**Files:**
- Modify: `tests/ccu/run_tilexr_ccu_direct_smoke.sh`
- Modify: `tests/ccu/test_tilexr_ccu_direct_smoke_runner.py`

- [ ] **Step 1: Add failing four-rank runner tests**

Create fake probe logs for ranks `0..3`. Verify device list `4,5,6,7`, per-rank endpoint/EID/resource-window forwarding, four PIDs/statuses/logs, and exact result rules:

```python
self.assertIn("expectedResults=40", result.stdout)
self.assertIn("expectedMarkerMatches=120", result.stdout)
self.assertIn("rank=3 device=7", result.stdout)
self.assertNotIn("resultCount>=1", runner_source)
```

Add negative cases for one missing rank log, one nonzero status, 39/40 results, 119/120 markers, invalid device count, and duplicate devices.

- [ ] **Step 2: Run runner tests to verify RED**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_direct_smoke_runner -v`

Expected: FAIL because launch/status/log handling is hard-coded to two ranks.

- [ ] **Step 3: Replace rank0/rank1 variables with indexed arrays**

```bash
rank_size="${TILEXR_CCU_RANK_SIZE:-2}"
IFS=',' read -r -a devices <<< "${TILEXR_CCU_DEVICE_LIST:-0,1}"
[ "${#devices[@]}" -eq "${rank_size}" ] || fail "device count does not match rank size"
declare -a pids statuses logs
for ((rank=0; rank<rank_size; ++rank)); do
    export_rank_environment "${rank}"
    launch_rank "${rank}" "${devices[$rank]}" >"${work_dir}/ccu_rank${rank}.log" 2>&1 &
    pids[$rank]=$!
    logs[$rank]="${work_dir}/ccu_rank${rank}.log"
done
```

Wait every PID even after one fails, record every status, and never kill unrelated processes. `export_rank_environment` must forward rank-specific EID, endpoint, token, XN, CKE, and resource-window fields for ranks `0..3`; default Mesh EID index is `3`.

- [ ] **Step 4: Enforce exact aggregate counts**

```bash
expected_results=$((rank_size * loop_count))
expected_markers=$((rank_size * (rank_size - 1) * loop_count))
[ "${result_count}" -eq "${expected_results}" ] || fail "result count mismatch"
[ "${marker_count}" -eq "${expected_markers}" ] || fail "marker count mismatch"
```

Keep rank size two as the default so current callers remain compatible.

- [ ] **Step 5: Run runner and probe tests**

Run: `python -m unittest tests.ccu.test_tilexr_ccu_direct_smoke_runner tests.ccu.test_tilexr_ccu_direct_smoke_probe -v`

Expected: PASS for 2P and 4P dry-run/fake-process cases.

- [ ] **Step 6: Commit**

```bash
git add tests/ccu/run_tilexr_ccu_direct_smoke.sh tests/ccu/test_tilexr_ccu_direct_smoke_runner.py
git commit -m "test(ccu): launch four-rank direct smoke"
```

### Task 7: Local Regression And Remote Hardware Validation

**Files:**
- Modify only if a test exposes a defect in files from Tasks 1-6.
- Verify: all affected CCU suites and `build_ccu_direct`.

- [ ] **Step 1: Run all affected local tests**

```bash
python -m unittest \
  tests.ccu.test_tilexr_ccu_microcode \
  tests.ccu.test_tilexr_ccu_alltoall_program \
  tests.ccu.test_tilexr_ccu_direct_orchestrator \
  tests.ccu.test_tilexr_ccu_collective_planner \
  tests.ccu.test_tilexr_ccu_direct_backend \
  tests.ccu.test_tilexr_ccu_direct_smoke_probe \
  tests.ccu.test_tilexr_ccu_direct_smoke_runner -v
```

Expected: PASS with zero failures/errors.

- [ ] **Step 2: Build the library and smoke probe on Linux**

```bash
source scripts/common_env.sh
cmake --build build_ccu_direct --target tile-comm -j2
cmake --build build_ccu_direct --target ccu_tilexr_direct_smoke_probe -j2
```

Expected: both targets complete successfully and link against the real runtime libraries, not the `devlib` HAL stub.

- [ ] **Step 3: Confirm passwordless access and the existing remote path**

Run from Windows:

```powershell
ssh -o BatchMode=yes root@141.61.50.31 'pwd; test -d /root/TileXR && echo TILEXR_REMOTE_OK'
```

Expected: passwordless login succeeds and prints the confirmed repository path. Do not modify `authorized_keys`.

- [ ] **Step 4: Reuse or create the confirmed Mutagen sync**

Inspect first:

```powershell
& 'C:\Users\l00654177\AppData\Local\Programs\Mutagen\mutagen.exe' sync list
```

If no session exactly matches local `C:\Users\l00654177\Desktop\TileXR` and confirmed `root@141.61.50.31:/root/TileXR`, create one with ignores for `.git`, build/cache directories, credentials, and the unrelated untracked home-directory artifacts:

```powershell
& 'C:\Users\l00654177\AppData\Local\Programs\Mutagen\mutagen.exe' sync create `
  --name tilexr-141-61-50-31 `
  --ignore-vcs `
  --ignore 'build*' --ignore '.anaconda' --ignore '.conda' --ignore '.mutagen*' --ignore '.ssh' `
  'C:\Users\l00654177\Desktop\TileXR' 'root@141.61.50.31:/root/TileXR'
```

Then run `mutagen sync monitor tilexr-141-61-50-31` until status is `Watching for changes`. If Mutagen cannot use the existing SSH configuration, record the error and transfer only Task 1-6 changed files with `scp`.

- [ ] **Step 5: Poll devices 4,5,6,7 until idle**

On the server, run the repository busy guard or `npu-smi info`; if any selected device is occupied, query again every 30 seconds. Do not stop, signal, or reconfigure another workload.

- [ ] **Step 6: Run 4P loop=1 in a fresh directory**

```bash
TILEXR_RUN_CCU_DIRECT_SMOKE_PROBE=1 \
TILEXR_CCU_DIRECT_SMOKE_ALLTOALL=1 \
TILEXR_CCU_DIRECT_SMOKE_ALLTOALL_MESH=1 \
TILEXR_CCU_RANK_SIZE=4 \
TILEXR_CCU_DEVICE_LIST=4,5,6,7 \
TILEXR_CCU_ALLTOALL_BYTES=2097152 \
TILEXR_CCU_ALLTOALL_LOOP_COUNT=1 \
TILEXR_CCU_SMOKE_WORK_DIR=/tmp/tilexr-ccu-4p-loop1-$(date +%s) \
bash tests/ccu/run_tilexr_ccu_direct_smoke.sh
```

Expected: exit 0, exactly 4 successful loop results, exactly 12 peer marker matches, and zero mismatches over four 8 MiB receive buffers.

- [ ] **Step 7: Run 4P loop=10 in another fresh directory**

Use the Step 6 command with `TILEXR_CCU_ALLTOALL_LOOP_COUNT=10` and a `loop10` work directory.

Expected: exit 0, exactly 40 successful loop results, exactly 120 peer marker matches, zero mismatches, phase files `0..9`, and stable mission ID/key/instruction range/task count/QP/jetty/CKE/XN/channel IDs across all ten loops.

- [ ] **Step 8: Re-run the 2P loop=10 regression**

After polling the selected validated pair idle, run the existing 2P command with `TILEXR_CCU_RANK_SIZE=2`, its two-device list, and `TILEXR_CCU_ALLTOALL_LOOP_COUNT=10`.

Expected: exit 0, exactly 20 successful loop results, exactly 20 peer marker matches, and unchanged 2P mission/resource identity.

- [ ] **Step 9: Inspect evidence and repository hygiene**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended Task 1-6 files are modified. Preserve unrelated untracked files.

- [ ] **Step 10: Commit the verified implementation**

Stage only the explicit Task 1-6 paths and commit:

```bash
git commit -m "feat(ccu): validate four-rank all-to-all mesh"
```

Record in the final verification report the three hardware commands, work directories, exact result/marker counts, mission/resource identity evidence, and any temporary Mutagen session name.
