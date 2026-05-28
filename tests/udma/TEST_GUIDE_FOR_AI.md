# TileXR UDMA Test Guide For AI Agents

This guide is for an AI agent running tests on an Ascend machine.

Important: TileXR UDMA means UnifiedBus DMA. The runtime UDMA path is only for A5 / Ascend950 / 950 hardware. Do not report 910B, 310P, or other devices as successful UDMA validation targets.

## Current Acceptance Scope

Validate the TileXR registered-memory UDMA path:

- `TileXRUDMARegister` registers ordinary `aclrtMalloc` memory;
- `tilexr_udma.h` device wrappers issue UDMA put / put-signal operations;
- demo result data and signal values are correct;
- host demo code does not include `shmem.h`.
- no standalone shmem API test is part of the current acceptance flow.

## Files To Know

```text
tests/udma/unit/test_tilexr_udma_registry.cpp
tests/udma/integration/test_tilexr_udma.cpp
tests/udma/demo/tilexr_udma_demo.cpp
tests/udma/demo/tilexr_udma_demo_kernel.cpp
tests/udma/demo/run_tilexr_udma_demo.sh
tests/udma/demo/ASCEND_VERIFICATION.md
```

## Execution Steps

### Step 1: Inspect Environment

```bash
cd /path/to/TileXR
source scripts/common_env.sh
npu-smi info
```

Record:

- NPU model;
- NPU count;
- CANN version;
- driver version;
- TileXR commit.

Stop and report unsupported hardware if the machine is not A5 / Ascend950 / 950.

### Step 2: Build TileXR And Tests

```bash
cd /path/to/TileXR
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build-udma -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build-udma --target tile-comm -j"$(nproc)"
cmake --install /tmp/tilexr-build-udma

cd tests/udma
bash build.sh
```

Check:

```bash
test -x install/bin/test_tilexr_udma_registry
test -x install/bin/test_tilexr_udma
test -x install/bin/tilexr_udma_demo
test -f install/lib/libtilexr_udma_demo_kernel.so
```

### Step 3: Run Host Checks

```bash
cd /path/to/TileXR/tests/udma
./install/bin/test_tilexr_udma_registry

source ../../scripts/common_env.sh
export LD_LIBRARY_PATH="$PWD/install/lib:../../install/lib:${LD_LIBRARY_PATH:-}"
RANK=0 RANK_SIZE=1 ./install/bin/test_tilexr_udma
```

Expected:

- registry test prints `TileXR UDMA registry checks passed`;
- communicator smoke test exits 0 and prints `Failed: 0`.

### Step 4: Run UDMA Demo

All-gather style UDMA put:

```bash
cd /path/to/TileXR/tests/udma
bash demo/run_tilexr_udma_demo.sh 0 2 16 2 0
```

UDMA put with signal:

```bash
bash demo/run_tilexr_udma_demo.sh 1 2 16 2 0
```

Expected:

- both commands exit 0;
- every rank log contains `TileXR UDMA demo success`;
- every rank log contains `TileXRUDMARegister success`;
- all-gather logs show result samples such as `seg0=1000 seg1=1001`;
- put-signal logs show non-local signal entries equal to `1000`.

### Step 5: Check Logs

```bash
latest=$(ls -td logs/tilexr_udma_demo_* | head -n1)
grep -R "TileXR UDMA demo success" "$latest"
grep -R "TileXRUDMARegister success" "$latest"
grep -R "DATA MISMATCH\\|expected non-local signals\\|TileXR UDMA demo failed\\|ERROR" "$latest" || true
```

The last command should print nothing for a successful run.

## Report Format

```text
Environment:
- Machine:
- NPU model:
- NPU count:
- CANN version:
- Driver version:
- TileXR commit:

Results:
- Build:
- Registry unit test:
- TileXR communicator smoke:
- UDMA all-gather demo:
- UDMA put-signal demo:
- Log directory:

Failures or warnings:
- ERROR lines:
- DATA MISMATCH lines:
- Signal mismatch lines:
- Other observations:
```

## Common Issues

| Symptom | Meaning | Action |
| --- | --- | --- |
| Demo prints UDMA disabled | Runtime transport was unavailable | Confirm A5 / Ascend950 hardware and driver/CANN setup |
| `tilexr_udma_demo` missing | AICore compiler was unavailable or demo build skipped | Check `bisheng` and rebuild `tests/udma` |
| `DATA MISMATCH` | Data-plane transfer failed | Save all rank logs and report the command used |
| Signal mismatch | Put-signal path failed | Save all rank logs and report signal values |
| Test runs on 910B/310P | Unsupported validation target | Mark as compile/smoke only, not UDMA runtime validation |
