# TileXR Auto Route Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish TileXR auto transport routing so a single public interface can choose memory or direct URMA, the real data-plane entry points use it, and local plus 62/70 verification can reproduce the result.

**Architecture:** Keep route policy in `src/include/tilexr_transport.h` as header-only AscendC-safe code. Add thin auto Put/Get wrappers above existing peer-memory and UDMA primitives, then migrate focused EP data-movement call sites to those wrappers without changing the 4 MiB threshold policy.

**Tech Stack:** C++14, AscendC device headers, TileXR `CommArgs`, TileXR UDMA API, CMake, Bash, mutagen, SSH to `root@141.62.24.62` and `root@141.62.24.70`.

---

## File Structure

- Modify `src/include/tilexr_transport.h`: owns route enum, availability check, auto selector, and auto Put/Get wrapper interfaces.
- Modify `src/ep/kernels/tilexr_ep_dispatch_kernel.cpp`: replace focused UDMA/manual branch sites with auto wrapper calls when the existing source and destination are expressed as peer rank plus offset.
- Modify `src/ep/kernels/tilexr_ep_kernel_common.h`: mirror the same focused migration for common kernel helpers if duplicated there.
- Modify `tests/udma/unit/test_tilexr_transport_auto_route.cpp`: extend host-only tests for wrapper-visible boundary behavior and keep selector matrix intact.
- Modify `tests/udma/unit/test_tilexr_udma_source_guard.cpp` or add a small source guard under `tests/udma/unit/`: verify auto wrapper names are present and raw UDMA calls are not reintroduced at migrated call sites.
- Modify `tests/udma/CMakeLists.txt`, `tests/udma/build.sh`, `tests/udma/run_tests.sh`: keep auto route in the build/run path.

---

### Task 1: Lock The Public Auto Wrapper API

**Files:**
- Modify: `tests/udma/unit/test_tilexr_transport_auto_route.cpp`
- Modify: `src/include/tilexr_transport.h`

- [ ] **Step 1: Write the failing host-only compile test**

Extend `tests/udma/unit/test_tilexr_transport_auto_route.cpp` with compile-time checks for wrapper symbols. Use host mode only; the test proves the public header exposes the final API names.

```cpp
void TestAutoWrapperSymbolsAreDeclared()
{
    auto args = MakeArgs(false);
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(&args, 1), TileXR::TileXRTransportKind::MEMORY);
}
```

The wrapper implementations themselves are device-only. Host compilation should still include the header cleanly and should not require AscendC symbols.

- [ ] **Step 2: Run test to verify current state**

Run:

```powershell
$exe = Join-Path $env:TEMP 'tilexr_transport_auto_route_local.exe'
g++ -std=c++14 -ID:\l00929943\TileXR\src\include D:\l00929943\TileXR\tests\udma\unit\test_tilexr_transport_auto_route.cpp -o $exe
& $exe
Remove-Item -LiteralPath $exe
```

Expected before production changes: selector tests pass, wrapper-specific source checks are absent.

- [ ] **Step 3: Add auto wrapper declarations and device implementations**

In `src/include/tilexr_transport.h`, include `tilexr_udma.h` under AscendC device compilation and add:

```cpp
template <typename T>
TILEXR_TRANSPORT_INLINE TileXRTransportKind TileXRPutAutoNbi(
    const TILEXR_TRANSPORT_GM CommArgs* args,
    int targetRank,
    const TILEXR_TRANSPORT_GM T* localSrc,
    uint64_t remoteOffset,
    uint32_t bytes)
{
    TileXRTransportKind route = TileXRSelectAutoTransport(args, bytes);
#if TILEXR_ASCENDC_AICORE_COMPILE
    if (route == TileXRTransportKind::DIRECT_URMA) {
        UDMAPutNbi<T>(args, targetRank, localSrc, remoteOffset, bytes);
    }
#else
    (void)targetRank;
    (void)localSrc;
    (void)remoteOffset;
#endif
    return route;
}
```

Add the analogous `TileXRGetAutoNbi`. Memory path remains a returned route at this layer; call sites continue to execute the existing DataCopy path when the wrapper returns `MEMORY`.

- [ ] **Step 4: Run the host-only test again**

Run the same local command. Expected: `TileXR transport auto route checks passed`.

---

### Task 2: Migrate Focused EP UDMA Branches To Auto Route

**Files:**
- Modify: `src/ep/kernels/tilexr_ep_dispatch_kernel.cpp`
- Modify: `src/ep/kernels/tilexr_ep_kernel_common.h`
- Modify/Test: `tests/udma/unit/test_tilexr_udma_source_guard.cpp` or new source guard

- [ ] **Step 1: Write the failing source guard**

Add checks that migrated files include `tilexr_transport.h` and contain `TileXRPutAutoNbi<uint8_t>` / `TileXRGetAutoNbi<uint8_t>` at the data send/get sites.

```cpp
CheckContains(dispatchKernelPath, dispatchKernel, "#include \"tilexr_transport.h\"");
CheckContains(dispatchKernelPath, dispatchKernel, "TileXR::TileXRPutAutoNbi<uint8_t>");
CheckContains(dispatchKernelPath, dispatchKernel, "TileXR::TileXRGetAutoNbi<uint8_t>");
```

- [ ] **Step 2: Run the guard to verify it fails**

Build/run the guard target locally or on 62 with:

```bash
cd /home/l00929943/TileXR_codex_auto_route_20260720_151042/tests/udma
BUILD_TILEXR_UDMA_DEMO=OFF bash build.sh
./install/bin/test_tilexr_udma_source_guard
```

Expected before migration: failure because EP files do not include/call auto wrapper yet.

- [ ] **Step 3: Migrate the focused UDMA branches**

For each existing branch shaped like:

```cpp
if (TileXR::UDMARegistryEnabled(args)) {
    TileXR::UDMAPutNbi<uint8_t>(args, peer, src, offset, bytes);
} else {
    CopyPeerMemory(...);
}
```

replace the UDMA condition with:

```cpp
if (TileXR::TileXRPutAutoNbi<uint8_t>(args, peer, src, offset, bytes) ==
    TileXR::TileXRTransportKind::MEMORY) {
    CopyPeerMemory(...);
}
```

For get:

```cpp
if (TileXR::TileXRGetAutoNbi<uint8_t>(args, peer, dst, offset, bytes) ==
    TileXR::TileXRTransportKind::MEMORY) {
    CopyPeerMemory(...);
}
```

Keep the existing memory fallback block intact.

- [ ] **Step 4: Run source guard**

Expected: guard passes.

---

### Task 3: Keep Build And Run Scripts Reproducible

**Files:**
- Modify: `tests/udma/build.sh`
- Modify: `tests/udma/run_tests.sh`
- Modify: `tests/udma/CMakeLists.txt`

- [ ] **Step 1: Verify auto route is installed and runnable**

Run on both remote hosts:

```bash
cd /home/l00929943/TileXR_codex_auto_route_20260720_151042
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr_core_build_${HOSTNAME}_auto -DCMAKE_INSTALL_PREFIX=$PWD/install -DBUILD_TESTING=OFF -DTILEXR_BUILD_TESTS=OFF
cmake --build /tmp/tilexr_core_build_${HOSTNAME}_auto -j$(nproc) --target install
cd tests/udma
BUILD_TILEXR_UDMA_DEMO=OFF bash build.sh
bash run_tests.sh
```

Expected summary includes:

```text
Test 1 (Auto Route):      PASS
Test 5 (TileXR Single):   PASS
```

- [ ] **Step 2: If runtime loader fails, keep lib64 and lib paths**

`tests/udma/run_tests.sh` must export:

```bash
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib64:${INSTALL_DIR}/lib:${TILEXR_ROOT}/install/lib64:${TILEXR_ROOT}/install/lib:/usr/local/lib:${LD_LIBRARY_PATH}"
```

---

### Task 4: Final Remote Verification

**Files:**
- No code edits unless verification exposes a bug.

- [ ] **Step 1: Flush mutagen sessions**

Run:

```powershell
mutagen sync flush tilexr-auto-route-62-20260720-151042 tilexr-auto-route-70-20260720-151042
```

- [ ] **Step 2: Run local checks**

Run:

```powershell
git diff --check
$exe = Join-Path $env:TEMP 'tilexr_transport_auto_route_local.exe'
g++ -std=c++14 -ID:\l00929943\TileXR\src\include D:\l00929943\TileXR\tests\udma\unit\test_tilexr_transport_auto_route.cpp -o $exe
& $exe
Remove-Item -LiteralPath $exe
```

- [ ] **Step 3: Run 62 and 70 checks**

Run `bash tests/udma/run_tests.sh` on both hosts and confirm the summary PASS lines.

- [ ] **Step 4: Report remaining limits**

State whether full multi-rank MPI ran or was skipped. If skipped due to no `mpirun` or insufficient usable NPUs, report the exact skip reason from the script.

---

## Self-Review

- Spec coverage: B public wrapper is Task 1; A EP data-plane migration is Task 2; C script and remote verification are Tasks 3-4.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: wrapper names match `TileXRPutAutoNbi`, `TileXRGetAutoNbi`, `TileXRTransportKind`, and `TileXRSelectAutoTransport`.
