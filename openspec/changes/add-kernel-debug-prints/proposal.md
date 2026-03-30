## Why

After implementing dual-transport (SDMA P2P + RDMA ibverbs) on the host side for single-server MC2 scenarios, we need a kernel-side verification mechanism to confirm that both transport paths are correctly established at runtime. Currently there is no way to observe from the kernel whether P2P windows are mapped or ibverbs data is filled.

## What Changes

- Add `[cwh]`-prefixed `AscendC::printf` debug prints in `AllGatherMatmulFullMesh::Init()` to dump key fields from `HcclA2CombineOpParam`:
  - `rankId`, `rankNum`, `winSize` — basic comm group info
  - `windowsIn[0..rankNum-1]` and `windowsOut[0..rankNum-1]` — P2P window addresses per rank
  - `data` pointer and `dataSize` — ibverbs transport data pointer

## Capabilities

### New Capabilities
- `kernel-debug-prints`: Kernel-side debug printing of `HcclA2CombineOpParam` fields to verify P2P window mapping and ibverbs data fill at launch time.

### Modified Capabilities
<!-- None - no spec-level behavior changes, this is debug instrumentation only -->

## Impact

- **File**: `src/mc2/all_gather_matmul/op_kernel/all_gather_matmul_full_mesh.h`
- **Scope**: Debug/instrumentation only, guarded by `GetBlockIdx() == 0` — no functional behavior change
- **Output**: Visible via `plog_grep.sh cwh` after running `ops_only_run.sh`
- **Dependencies**: None beyond existing `HcclA2CombineOpParam` definition in `src/mc2/common/inc/kernel/moe_distribute_base.h`
