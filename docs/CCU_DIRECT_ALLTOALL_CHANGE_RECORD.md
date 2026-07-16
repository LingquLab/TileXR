# CCU Direct AllToAll Change Record

## Scope

This change records the TileXR-owned Direct CCU alltoall work on the
`direct-ccu-rebased` branch. The implementation keeps HCCL and hcomm as
reference-only inputs: TileXR does not include HCCL/hcomm headers, call their
private CCU interfaces, or link new HCCL/hcomm dependencies.

The validated target is the 2-rank P2P alltoall smoke path with a 2 MB payload.

## Code Changes

### AllToAll program builder

- Added a TileXR-owned 2-rank Direct CCU alltoall program flow.
- Uses fixed 4 KB memory slices, `memSlicePerLoop <= 8`, 32 KB blocks, and
  64 blocks for the 2 MB smoke scale.
- Encodes the HCCL-style phase structure in TileXR microcode:
  - PreSync: remote notification and local wait before data movement.
  - Copy: CCU memory copy blocks over the existing P2P copy route.
  - PostSync: completion notification and optional peer wait.
- Split synchronization masks for PreSync and PostSync so the phases no longer
  reuse one ambiguous CKE bit.
- Added instruction-count reporting for pre-sync, copy, post-sync, and finish
  sections.

### Planner and orchestrator

- Added planner/orchestrator entry points for preparing a 2-rank alltoall
  launch package.
- Added direct resource configuration for alltoall sync resources, channels,
  XNs, CKEs, GSA addresses, and remote endpoint routes.
- Kept the production backend guarded: Direct CCU prepare/submit helpers remain
  test-only under `TILEXR_CCU_TESTING`.

### Smoke probe and runner

- Added alltoall smoke switches:
  - `TILEXR_CCU_DIRECT_SMOKE_ALLTOALL`
  - `TILEXR_CCU_DIRECT_SMOKE_ALLTOALL_LONG_MISSION`
  - `TILEXR_CCU_ALLTOALL_BYTES`
  - `TILEXR_CCU_ALLTOALL_MEM_SLICE_PER_LOOP`
- Added `TILEXR_CCU_DIRECT_SMOKE_SYNC_XN_PING` for a minimal SyncXn route
  sanity check.
- Added bounded stream synchronization through
  `aclrtSynchronizeStreamWithTimeout` for signal/wait, SyncXn ping, and
  alltoall submit paths.
- Fixed multi-phase host submit coordination by making the submit-ready and
  submit-done files phase-scoped:
  - `rank0.phase0.ready`
  - `rank0.phase0.done`
  - `rank1.phase1.ready`
  - `rank1.phase1.done`
- Extended the P2P copy direction parser to accept `local_to_remote`,
  `LocalToRemote`, and `1`.

### Tests

- Added and updated unit tests for:
  - alltoall program instruction layout and resource usage,
  - planner/orchestrator alltoall prepare paths,
  - smoke runner environment forwarding,
  - phase-scoped collective submit gates,
  - bounded synchronization in smoke modes,
  - lower-layer resource planning expectations.

## Validated Commands

Local structural tests:

```bash
python -m unittest \
  tests.ccu.test_tilexr_ccu_alltoall_program \
  tests.ccu.test_tilexr_ccu_direct_orchestrator \
  tests.ccu.test_tilexr_ccu_direct_smoke_probe \
  tests.ccu.test_tilexr_ccu_lower_layer_plan_builder
```

Result:

```text
Ran 105 tests in 3.728s
OK (skipped=52)
```

Remote build and dependency guard on `141.62.24.62`:

```bash
cd /home/tileXR
source scripts/common_env.sh
timeout 180s cmake -S . -B build_ccu_direct \
  -DTILEXR_CCU_TESTING=1 \
  -DCMAKE_INSTALL_PREFIX=/home/tileXR/install
timeout 240s cmake --build build_ccu_direct --target tile-comm -j2
bash tests/ccu/check_tile_comm_no_hcomm_deps.sh \
  build_ccu_direct/src/comm/libtile-comm.so
```

Result: build succeeded and the dependency guard reported no hcomm/HCCL private
CCU dependency or symbol reference.

Remote P2P positive-control smoke on devices `6,7`:

```bash
env ASCEND_HOME_PATH=/home/Hccl_QQTest/Ascend/cann-9.1.0 \
  TILEXR_RUN_CCU_DIRECT_SMOKE_PROBE=1 \
  TILEXR_TILE_COMM_LIB=/home/tileXR/build_ccu_direct/src/comm/libtile-comm.so \
  TILEXR_CCU_DIRECT_TRACE=1 \
  TILEXR_CCU_DIRECT_TRACE_ENDPOINT_ROUTE=1 \
  TILEXR_CCU_DIRECT_RESOURCE_WINDOW_EID_INDEX_RANK0=3 \
  TILEXR_CCU_DIRECT_RESOURCE_WINDOW_EID_INDEX_RANK1=3 \
  TILEXR_CCU_SMOKE_DEVICES=6,7 \
  TILEXR_CCU_DIRECT_SMOKE_P2P_CCU_COPY=1 \
  TILEXR_CCU_DIRECT_SMOKE_P2P_CCU_COPY_DIRECTION=remote_to_local \
  TILEXR_CCU_DIRECT_SMOKE_SUBMIT=1 \
  TILEXR_CCU_SMOKE_WORK_DIR=build_ccu_direct/diag62_p2p_r2l_67_current \
  TILEXR_CCU_SMOKE_TIMEOUT=120 \
  TILEXR_CCU_DIRECT_SMOKE_READY_TIMEOUT_MS=30000 \
  TILEXR_CCU_DIRECT_SUBMIT_TIMEOUT=12000 \
  TILEXR_CCU_DIRECT_READBACK_INSTRUCTIONS=1 \
  timeout 150s bash tests/ccu/run_tilexr_ccu_direct_smoke.sh
```

Result: rank0 and rank1 exited with status 0.

Remote 2-rank 2 MB alltoall smoke on devices `6,7`:

```bash
env ASCEND_HOME_PATH=/home/Hccl_QQTest/Ascend/cann-9.1.0 \
  TILEXR_RUN_CCU_DIRECT_SMOKE_PROBE=1 \
  TILEXR_TILE_COMM_LIB=/home/tileXR/build_ccu_direct/src/comm/libtile-comm.so \
  TILEXR_CCU_DIRECT_TRACE=1 \
  TILEXR_CCU_DIRECT_TRACE_ENDPOINT_ROUTE=1 \
  TILEXR_CCU_DIRECT_RESOURCE_WINDOW_EID_INDEX_RANK0=3 \
  TILEXR_CCU_DIRECT_RESOURCE_WINDOW_EID_INDEX_RANK1=3 \
  TILEXR_CCU_SMOKE_DEVICES=6,7 \
  TILEXR_CCU_DIRECT_SMOKE_ALLTOALL=1 \
  TILEXR_CCU_DIRECT_SMOKE_SUBMIT=1 \
  TILEXR_CCU_ALLTOALL_BYTES=2097152 \
  TILEXR_CCU_ALLTOALL_MEM_SLICE_PER_LOOP=8 \
  TILEXR_CCU_SMOKE_WORK_DIR=build_ccu_direct/diag62_alltoall_host_phased_67_current \
  TILEXR_CCU_SMOKE_TIMEOUT=180 \
  TILEXR_CCU_DIRECT_SMOKE_READY_TIMEOUT_MS=30000 \
  TILEXR_CCU_DIRECT_SUBMIT_TIMEOUT=12000 \
  TILEXR_CCU_DIRECT_READBACK_INSTRUCTIONS=1 \
  timeout 220s bash tests/ccu/run_tilexr_ccu_direct_smoke.sh
```

Key log evidence:

```text
tilexr_ccu_alltoall config rank=0 peer=1 bytes=2097152 memSlicePerLoop=8 blockCount=64 hostPhases=2
tilexr_ccu_direct_smoke collectiveSubmitReady rank=0 phase=0 localReady=1 allRanksReady=1
tilexr_ccu_direct_smoke collectiveSubmitDone rank=0 phase=0 localResult=0 allRanksDone=1 allRanksSucceeded=1
tilexr_ccu_direct_smoke collectiveSubmitReady rank=0 phase=1 localReady=1 allRanksReady=1
tilexr_ccu_direct_smoke collectiveSubmitDone rank=0 phase=1 localResult=0 allRanksDone=1 allRanksSucceeded=1
tilexr_ccu_alltoall result passed=1 rank=0 ret=0 readRet=0 mismatches=0
tilexr_ccu_alltoall config rank=1 peer=0 bytes=2097152 memSlicePerLoop=8 blockCount=64 hostPhases=2
tilexr_ccu_direct_smoke collectiveSubmitReady rank=1 phase=0 localReady=1 allRanksReady=1
tilexr_ccu_direct_smoke collectiveSubmitDone rank=1 phase=0 localResult=0 allRanksDone=1 allRanksSucceeded=1
tilexr_ccu_direct_smoke collectiveSubmitReady rank=1 phase=1 localReady=1 allRanksReady=1
tilexr_ccu_direct_smoke collectiveSubmitDone rank=1 phase=1 localResult=0 allRanksDone=1 allRanksSucceeded=1
tilexr_ccu_alltoall result passed=1 rank=1 ret=0 readRet=0 mismatches=0
```

## Environment Notes

- File transfer to the server used mutagen, as required by the project
  instructions.
- On `141.62.24.62`, the default system CANN `libra.so` does not export
  `RaCustomChannel`. Hardware CCU smoke tests must set:

```bash
ASCEND_HOME_PATH=/home/Hccl_QQTest/Ascend/cann-9.1.0
```

- Hardware test commands use an outer `timeout` and also pass
  `TILEXR_CCU_DIRECT_SUBMIT_TIMEOUT` for stream-level timeout control.

## Current Limitations

- The validated alltoall path is 2-rank P2P with host-phased submission.
- The formal LoopGroup version is not claimed as validated by this record.
- The validated route uses devices `6,7` on `141.62.24.62`.
