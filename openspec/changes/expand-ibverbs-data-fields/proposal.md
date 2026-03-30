## Why

上一轮 kernel debug prints 已确认 P2P windows 地址非零，但 `data=0x0`（ibverbs 未填入）。为了在 ibverbs 路径建立后验证其内容，需要将 `data` 数组按 `IbVerbsData` 结构体展开，逐 rank 打印 `remoteInput`、`remoteOutput`、`localInput`、`localOutput` 四个 `MemDetails` 字段（`addr`、`size`、`key`）。

## What Changes

- 在 `AllGatherMatmulFullMesh::Init()` 的 `GetBlockIdx() == 0` 块中，当 `data` 非零时，将其作为 `IbVerbsData` 数组展开，逐 rank 打印每个 `MemDetails` 的 `addr`、`size`、`key` 字段

## Capabilities

### New Capabilities
- `ibverbs-data-expand`: 将 `HcclA2CombineOpParam.data` 按 `IbVerbsData[rankNum]` 展开，打印每个 rank 的四个 MemDetails 字段

### Modified Capabilities
- `kernel-debug-prints`: 在已有 `data` 指针打印后追加结构体展开内容

## Impact

- **File**: `src/mc2/all_gather_matmul/op_kernel/all_gather_matmul_full_mesh.h`
- **Scope**: 仅在 `data != 0` 时执行，不影响 `data=0` 时的现有行为
- **依赖结构体**: `AscendC::IbVerbsData`（定义于 CANN SDK `hccl_inner_def.h`，包含 `remoteInput`/`remoteOutput`/`localInput`/`localOutput` 四个 `MemDetails`）
