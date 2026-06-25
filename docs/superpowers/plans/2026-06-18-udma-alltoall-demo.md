# UDMA All-to-All Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `test_type=2` all-to-all path to the TileXR UDMA demo under `tests/udma/demo`.

**Architecture:** Reuse the existing `tilexr_udma_demo` host executable, launch script, local TCP barrier, and UDMA registered-memory setup. Add one AICore all-to-all kernel that writes each local destination slice into the matching remote output slice using `TileXR::UDMAPutNbi`, with host-side validation after all ranks complete.

**Tech Stack:** C++14 host code, Ascend C AICore kernel code, TileXR public C API, `tilexr_udma.h` device wrapper, CMake, bash, remote Ascend950/A5 validation.

## Global Constraints

- CANN version: 9.1.0.
- Target OS: Ubuntu 20.04 LTS; root user required for device access.
- UDMA data-plane validation targets A5 / Ascend950 / 950 hardware.
- Do not add shmem includes or shmem API calls.
- Existing `test_type=0` all-gather and `test_type=1` put-signal behavior must remain unchanged.
- All-to-all uses `int32_t` only.
- Remote validation must create a new directory under `/home/aiv-perf/` on `root@141.61.95.18`.

---

## File Structure

- Modify `tests/udma/demo/tilexr_udma_demo_kernel.cpp`: add `tilexr_udma_all_to_all_kernel` and `launch_tilexr_udma_all_to_all`.
- Modify `tests/udma/demo/tilexr_udma_demo.cpp`: declare the launch wrapper, allocate the all-to-all input/output layout for `test_type=2`, initialize inputs, launch the new kernel, and validate outputs.
- Modify `tests/udma/demo/run_tilexr_udma_demo.sh`: update printed help text to include `test_type=2`.
- Modify `tests/udma/demo/README.md`: document the all-to-all path and example command.
- Optionally modify `tests/udma/demo/ASCEND_VERIFICATION.md` only if it contains an exhaustive test-type list that would become stale.

---

### Task 1: Add Device Kernel And Launch Wrapper

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`

**Interfaces:**
- Consumes: `TileXR::UDMAPutNbi<int32_t>(args, targetRank, localSrc, byteOffset, byteCount)` and `TileXR::UDMAQuiet(args, targetRank)` from `src/include/tilexr_udma.h`.
- Produces:
  ```cpp
  void launch_tilexr_udma_all_to_all(
      uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
      GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset);
  ```

- [ ] **Step 1: Add the kernel before the registered smoke kernel**

Add this code to `tests/udma/demo/tilexr_udma_demo_kernel.cpp` after `tilexr_udma_put_signal_kernel`:

```cpp
extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t outputByteOffset)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = static_cast<int32_t>(outputByteOffset);
    }
    if (!enabled) {
        return;
    }

    uint32_t bytes = static_cast<uint32_t>(elementsPerPeer * sizeof(int32_t));
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        auto localSrc = input + peer * elementsPerPeer;
        uint64_t remoteOffset = outputByteOffset +
            static_cast<uint64_t>(rank) * elementsPerPeer * sizeof(int32_t);
        if (peer == rank) {
            auto localDst = output + rank * elementsPerPeer;
            for (int32_t i = 0; i < elementsPerPeer; ++i) {
                localDst[i] = localSrc[i];
            }
            continue;
        }
        TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc, remoteOffset, bytes);
        TileXR::UDMAQuiet(args, peer);
    }
}
```

- [ ] **Step 2: Add the host launch wrapper**

Add this code near the other `launch_tilexr_udma_*` wrappers:

```cpp
void launch_tilexr_udma_all_to_all(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset)
{
    tilexr_udma_all_to_all_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, outputByteOffset);
}
```

- [ ] **Step 3: Run a local source check**

Run:

```powershell
rg -n "tilexr_udma_all_to_all|launch_tilexr_udma_all_to_all" tests/udma/demo/tilexr_udma_demo_kernel.cpp
```

Expected: one kernel definition and one launch wrapper definition are shown.

---

### Task 2: Add Host-Side All-To-All Layout, Launch, And Validation

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo.cpp`

**Interfaces:**
- Consumes:
  ```cpp
  void launch_tilexr_udma_all_to_all(
      uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
      GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset);
  ```
- Produces:
  ```cpp
  bool ValidateAllToAllData(
      int rank, int rankSize, const std::vector<int32_t>& output, int32_t elementsPerPeer);
  ```

- [ ] **Step 1: Declare the new launch wrapper**

Add this declaration after `launch_tilexr_udma_put_signal`:

```cpp
extern void launch_tilexr_udma_all_to_all(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset);
```

- [ ] **Step 2: Add all-to-all constants**

Add this constant beside the existing demo constants:

```cpp
constexpr int32_t kAllToAllBaseValue = 100000;
```

- [ ] **Step 3: Add a validator**

Add this function after `ValidateData`:

```cpp
bool ValidateAllToAllData(
    int rank, int rankSize, const std::vector<int32_t>& output, int32_t elementsPerPeer)
{
    bool ok = true;
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        int32_t expected = kAllToAllBaseValue + srcRank * 1000 + rank;
        for (int32_t i = 0; i < elementsPerPeer; ++i) {
            size_t offset = static_cast<size_t>(srcRank) * elementsPerPeer + i;
            if (output[offset] != expected) {
                std::cerr << "[rank " << rank << "] ALLTOALL MISMATCH at src=" << srcRank
                          << " elem=" << i << " offset=" << offset
                          << " got=" << output[offset] << " expected=" << expected << std::endl;
                ok = false;
                break;
            }
        }
    }

    std::cout << "[rank " << rank << "] alltoall output sample:";
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        size_t offset = static_cast<size_t>(srcRank) * elementsPerPeer;
        std::cout << " from" << srcRank << "=" << output[offset];
    }
    std::cout << std::endl;
    return ok;
}
```

- [ ] **Step 4: Split payload layout by test type**

Replace the current data/signal payload sizing block with:

```cpp
bool isAllToAll = testType == 2;
size_t dataCount = static_cast<size_t>(rankSize) * elementsPerRank;
size_t dataBytes = dataCount * sizeof(int32_t);
size_t inputOffset = 0;
size_t outputOffset = isAllToAll ? dataBytes : 0;
size_t signalOffset = isAllToAll ? dataBytes * 2 : dataBytes;
size_t signalBytes = static_cast<size_t>(rankSize) * sizeof(uint64_t);
size_t payloadBytes = signalOffset + signalBytes;
```

Keep the existing `registeredBytes` calculation immediately after this block.

- [ ] **Step 5: Define typed buffer pointers**

Replace the current `data` and `signals` pointer setup with:

```cpp
auto data = static_cast<int32_t*>(registeredMemory);
auto input = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(registeredMemory) + inputOffset);
auto output = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(registeredMemory) + outputOffset);
auto signals = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(registeredMemory) + signalOffset);
```

- [ ] **Step 6: Initialize host buffers for both modes**

Replace the current `hostData` initialization with:

```cpp
std::vector<int32_t> hostData(dataCount, -1);
std::vector<int32_t> hostOutput(dataCount, -1);
if (isAllToAll) {
    for (int dstRank = 0; dstRank < rankSize; ++dstRank) {
        int32_t value = kAllToAllBaseValue + rank * 1000 + dstRank;
        std::fill(hostData.begin() + static_cast<size_t>(dstRank) * elementsPerRank,
                  hostData.begin() + static_cast<size_t>(dstRank + 1) * elementsPerRank,
                  value);
    }
} else {
    std::fill(hostData.begin() + static_cast<size_t>(rank) * elementsPerRank,
              hostData.begin() + static_cast<size_t>(rank + 1) * elementsPerRank,
              1000 + rank);
}
```

- [ ] **Step 7: Copy input and output buffers to device**

Replace the data H2D copy part with:

```cpp
bool initOk = CopyHostToDevice(rank, input, dataCount * sizeof(int32_t),
        hostData.data(), dataCount * sizeof(int32_t), isAllToAll ? "alltoall input" : "data");
if (isAllToAll) {
    initOk = CopyHostToDevice(rank, output, dataCount * sizeof(int32_t),
        hostOutput.data(), dataCount * sizeof(int32_t), "alltoall output") && initOk;
}
if (!initOk ||
    !CopyHostToDevice(rank, signals, hostSignals.size() * sizeof(uint64_t),
        hostSignals.data(), hostSignals.size() * sizeof(uint64_t), "signals") ||
    !CopyHostToDevice(rank, debug, hostDebug.size() * sizeof(int32_t),
        hostDebug.data(), hostDebug.size() * sizeof(int32_t), "debug")) {
```

Keep the existing cleanup block inside this `if`.

- [ ] **Step 8: Launch the all-to-all kernel for `test_type=2`**

Replace the launch selection with:

```cpp
if (testType == 2) {
    PrintStatus(rank, "launch all-to-all kernel");
    launch_tilexr_udma_all_to_all(
        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
        reinterpret_cast<GM_ADDR>(debug), elementsPerRank, static_cast<uint64_t>(outputOffset));
} else if (testType == 1) {
    PrintStatus(rank, "launch put-signal kernel");
    launch_tilexr_udma_put_signal(
        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(data), reinterpret_cast<GM_ADDR>(signals),
        reinterpret_cast<GM_ADDR>(debug), elementsPerRank, kSignalValue);
} else {
    PrintStatus(rank, "launch all-gather kernel");
    launch_tilexr_udma_all_gather(
        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(data), reinterpret_cast<GM_ADDR>(debug),
        elementsPerRank);
}
```

- [ ] **Step 9: Copy output back for all-to-all**

Replace the data D2H copy with:

```cpp
bool copyBackOk = CopyDeviceToHost(rank, hostData.data(), dataCount * sizeof(int32_t),
        data, dataCount * sizeof(int32_t), "data");
if (isAllToAll) {
    copyBackOk = CopyDeviceToHost(rank, hostOutput.data(), dataCount * sizeof(int32_t),
        output, dataCount * sizeof(int32_t), "alltoall output") && copyBackOk;
}
if (!copyBackOk ||
    !CopyDeviceToHost(rank, hostSignals.data(), hostSignals.size() * sizeof(uint64_t),
        signals, hostSignals.size() * sizeof(uint64_t), "signals") ||
    !CopyDeviceToHost(rank, hostDebug.data(), hostDebug.size() * sizeof(int32_t),
        debug, hostDebug.size() * sizeof(int32_t), "debug")) {
```

Keep the existing cleanup block inside this `if`.

- [ ] **Step 10: Validate all-to-all output**

Replace the final validation selection with:

```cpp
bool ok = isAllToAll ? ValidateAllToAllData(rank, rankSize, hostOutput, elementsPerRank)
                     : ValidateData(rank, rankSize, hostData, elementsPerRank);
if (testType == 1) {
    ok = ValidateSignals(rank, rankSize, hostSignals) && ok;
}
```

- [ ] **Step 11: Run local source checks**

Run:

```powershell
rg -n "all-to-all|alltoall|isAllToAll|ValidateAllToAllData|launch_tilexr_udma_all_to_all" tests/udma/demo/tilexr_udma_demo.cpp
```

Expected: declaration, validation function, launch path, and copy/initialization branches are shown.

---

### Task 3: Update Demo Documentation And Script Text

**Files:**
- Modify: `tests/udma/demo/run_tilexr_udma_demo.sh`
- Modify: `tests/udma/demo/README.md`
- Modify if stale: `tests/udma/demo/ASCEND_VERIFICATION.md`

**Interfaces:**
- Consumes: `test_type=2` behavior from Tasks 1 and 2.
- Produces: documented command `bash demo/run_tilexr_udma_demo.sh 2 2 16 2 0`.

- [ ] **Step 1: Update script test-type text**

In `tests/udma/demo/run_tilexr_udma_demo.sh`, change:

```bash
echo "Test type:         ${test_type} (0=all-gather put, 1=put-signal)"
```

to:

```bash
echo "Test type:         ${test_type} (0=all-gather put, 1=put-signal, 2=all-to-all)"
```

- [ ] **Step 2: Update README run examples**

In `tests/udma/demo/README.md`, add this command to the Run section:

```bash
bash demo/run_tilexr_udma_demo.sh 2 2 16 2 0
```

Update the argument list so it includes:

```text
- `test_type=2`: all-to-all UDMA put. Rank `src` sends input slice `dst` to rank `dst`; each output is ordered by source rank.
```

- [ ] **Step 3: Check `ASCEND_VERIFICATION.md` for stale test-type lists**

Run:

```powershell
rg -n "test_type|all-gather|put-signal|all-to-all" tests/udma/demo/ASCEND_VERIFICATION.md
```

If it lists only test types 0 and 1, update the list to include:

```text
test_type=2 validates all-to-all registered-memory UDMA puts with output ordered by source rank.
```

- [ ] **Step 4: Run documentation check**

Run:

```powershell
rg -n "test_type=2|all-to-all|run_tilexr_udma_demo.sh 2" tests/udma/demo/README.md tests/udma/demo/run_tilexr_udma_demo.sh tests/udma/demo/ASCEND_VERIFICATION.md
```

Expected: README and script both mention `test_type=2`.

---

### Task 4: Local Build-Oriented Verification

**Files:**
- Verify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Verify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`
- Verify: `tests/udma/CMakeLists.txt`

**Interfaces:**
- Consumes: code from Tasks 1-3.
- Produces: local confidence before remote hardware validation.

- [ ] **Step 1: Review the diff**

Run:

```powershell
git diff -- tests/udma/demo/tilexr_udma_demo.cpp tests/udma/demo/tilexr_udma_demo_kernel.cpp tests/udma/demo/run_tilexr_udma_demo.sh tests/udma/demo/README.md tests/udma/demo/ASCEND_VERIFICATION.md
```

Expected: only all-to-all related changes are present.

- [ ] **Step 2: Check that CMake tracks the modified kernel source**

Run:

```powershell
rg -n "tilexr_udma_demo_kernel.cpp|tilexr_udma_demo.cpp|tilexr_udma_demo" tests/udma/CMakeLists.txt
```

Expected: existing custom command depends on `demo/tilexr_udma_demo_kernel.cpp`; no CMake changes are required.

- [ ] **Step 3: Check formatting-sensitive syntax around modified host blocks**

Run:

```powershell
rg -n "payloadBytes|registeredBytes|initOk|copyBackOk|testType == 2|ValidateAllToAllData" tests/udma/demo/tilexr_udma_demo.cpp
```

Expected: each symbol appears in the expected host flow.

- [ ] **Step 4: Check working tree before remote copy**

Run:

```powershell
git status --short
```

Expected: only the planned demo files and plan file are modified or added.

---

### Task 5: Remote Build And Runtime Validation

**Files:**
- Verify remotely under a new `/home/aiv-perf/` directory on `root@141.61.95.18`.

**Interfaces:**
- Consumes: local repository changes from Tasks 1-4.
- Produces: remote build and runtime evidence for all-to-all.

- [ ] **Step 1: Create a unique remote validation directory**

Run from the local workspace:

```powershell
ssh root@141.61.95.18 "set -e; d=/home/aiv-perf/tilexr-udma-alltoall-$(date +%Y%m%d-%H%M%S); mkdir -p \"$d\"; echo \"$d\""
```

Expected: command prints the new remote directory path.

- [ ] **Step 2: Copy the repository to the remote directory**

Use `rsync` if available:

```powershell
rsync -a --delete --exclude .git --exclude build --exclude install --exclude tests/udma/build --exclude tests/udma/install --exclude tests/udma/logs ./ root@141.61.95.18:<REMOTE_DIR>/
```

If `rsync` is unavailable on Windows, use `scp` with a compressed archive created outside the repo build artifacts:

```powershell
tar --exclude .git --exclude build --exclude install --exclude tests/udma/build --exclude tests/udma/install --exclude tests/udma/logs -czf $env:TEMP\tilexr-udma-alltoall.tgz .
scp $env:TEMP\tilexr-udma-alltoall.tgz root@141.61.95.18:<REMOTE_DIR>/
ssh root@141.61.95.18 "set -e; cd <REMOTE_DIR>; tar -xzf tilexr-udma-alltoall.tgz; rm tilexr-udma-alltoall.tgz"
```

Expected: remote directory contains `scripts/common_env.sh` and `tests/udma/demo`.

- [ ] **Step 3: Build TileXR core on remote**

Run:

```powershell
ssh root@141.61.95.18 "set -e; cd <REMOTE_DIR>; source scripts/common_env.sh; mkdir -p build; cd build; cmake -DCMAKE_INSTALL_PREFIX=../install ..; make -j$(nproc); make install"
```

Expected: `install/lib/libtile-comm.so` exists under the remote directory.

- [ ] **Step 4: Build UDMA demo on remote**

Run:

```powershell
ssh root@141.61.95.18 "set -e; cd <REMOTE_DIR>/tests/udma; bash build.sh"
```

Expected: `tests/udma/install/bin/tilexr_udma_demo` exists, and the output does not say the demo was skipped because `bisheng` is missing.

- [ ] **Step 5: Run all-to-all with 2 ranks**

Run:

```powershell
ssh root@141.61.95.18 "set -e; cd <REMOTE_DIR>/tests/udma; bash demo/run_tilexr_udma_demo.sh 2 2 16 2 0"
```

Expected: every rank prints `TileXR UDMA demo success`; log tails include `alltoall output sample` with `from0=100000` on rank 0, `from0=100001` on rank 1, `from1=101000` on rank 0, and `from1=101001` on rank 1.

- [ ] **Step 6: Run optional wider validation if at least 4 NPUs are available**

Run:

```powershell
ssh root@141.61.95.18 "set -e; cd <REMOTE_DIR>/tests/udma; bash demo/run_tilexr_udma_demo.sh 2 4 16 4 0"
```

Expected: every rank prints `TileXR UDMA demo success`.

- [ ] **Step 7: Capture failure logs if validation fails**

Run:

```powershell
ssh root@141.61.95.18 "cd <REMOTE_DIR>/tests/udma; latest=$(ls -td logs/tilexr_udma_demo_* 2>/dev/null | head -1); if [ -n \"$latest\" ]; then for f in \"$latest\"/rank_*.log; do echo \"===== $f =====\"; tail -n 120 \"$f\"; done; fi"
```

Expected: logs show whether failure is UDMA enablement, build/runtime environment, kernel debug words, or output mismatch.

---

### Task 6: Final Review And Commit

**Files:**
- Review all modified files.

**Interfaces:**
- Consumes: completed implementation and verification evidence.
- Produces: final commit for all-to-all demo implementation.

- [ ] **Step 1: Inspect final diff**

Run:

```powershell
git diff --stat
git diff -- tests/udma/demo/tilexr_udma_demo.cpp tests/udma/demo/tilexr_udma_demo_kernel.cpp tests/udma/demo/run_tilexr_udma_demo.sh tests/udma/demo/README.md tests/udma/demo/ASCEND_VERIFICATION.md
```

Expected: changes match the approved spec and no unrelated files are modified.

- [ ] **Step 2: Check git status**

Run:

```powershell
git status --short
```

Expected: modified implementation/doc files plus this plan file.

- [ ] **Step 3: Commit implementation after successful validation**

Run:

```powershell
git add tests/udma/demo/tilexr_udma_demo.cpp tests/udma/demo/tilexr_udma_demo_kernel.cpp tests/udma/demo/run_tilexr_udma_demo.sh tests/udma/demo/README.md tests/udma/demo/ASCEND_VERIFICATION.md docs/superpowers/plans/2026-06-18-udma-alltoall-demo.md
git commit -m "feat: add udma alltoall demo"
```

Expected: one implementation commit is created.

---

## Self-Review

- Spec coverage: Tasks 1 and 2 implement the `test_type=2` all-to-all kernel, layout, launch, synchronization, and validation. Task 3 documents the command. Task 5 covers the required remote validation under `/home/aiv-perf/`.
- Placeholder scan: No task uses TBD/TODO/fill-in wording. `<REMOTE_DIR>` is an execution-time value produced by Task 5 Step 1 and intentionally reused by later commands.
- Type consistency: The launch wrapper signature is identical in Task 1 and Task 2. Host uses `elementsPerRank` as the user-facing argument and passes it as `elementsPerPeer` to the kernel.
