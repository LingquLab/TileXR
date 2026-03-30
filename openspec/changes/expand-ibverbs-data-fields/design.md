## Context

`HcclA2CombineOpParam.data` 是 `__gm__ AscendC::IbVerbsData*`，指向一个长度为 `rankNum` 的数组（每个元素对应一个 rank 的 ibverbs 内存信息）。`IbVerbsData` 结构体包含四个 `MemDetails`（每个有 `addr`/`size`/`key`）：`remoteInput`、`remoteOutput`、`localInput`、`localOutput`。

当前 `data=0x0` 表示 ibverbs 路径尚未在这台测试机上填入；一旦填入后，展开打印可验证各字段是否正确。

## Goals / Non-Goals

**Goals:**
- 当 `data != 0` 时，遍历 `data[0..rankNum-1]`，对每个 rank 打印：
  - `remoteInput.addr` / `.size` / `.key`
  - `remoteOutput.addr` / `.size` / `.key`
  - `localInput.addr` / `.size` / `.key`
  - `localOutput.addr` / `.size` / `.key`
- 所有打印加 `[cwh]` 前缀

**Non-Goals:**
- `data=0` 时不执行任何额外操作（已有打印 `data[0x0]` 即可）
- 不打印 `qpInfo` 或其他字段（`IbVerbsData` 中无此字段，在 `TransportDeviceNormalData` 中才有）

## Decisions

### 以 `uint64_t` 偏移访问数组元素
`data` 是 `__gm__ IbVerbsData*`，直接用 `data[i].remoteInput.addr` 即可访问，无需手动计算偏移。kernel 侧支持 `__gm__` 指针下标访问。

### 用 `uint64_t` 打印地址/大小，`uint32_t` 打印 key
`MemDetails.addr` 和 `size` 均为 `uint64_t`（用 `%lu`），`key` 为 `uint32_t`（用 `%u`）。

### guard 条件：`data != 0`
在已有 `data` 打印行之后紧接一个 `if ((uint64_t)winContext_->data != 0)` 块，保持与现有 `tileXrContext` null check 风格一致。

## Risks / Trade-offs

- [Risk] `data` 指向的设备内存未 ready 时访问会触发硬件异常 → Mitigation: 仅在 `data != 0` 时展开，与 `SetDevIbverbsData` 填入时机一致，data 非零说明内存已分配并写入
- [Risk] 每个 rank 打印 4 行，2 rank 场景共 8 行，日志量可接受
