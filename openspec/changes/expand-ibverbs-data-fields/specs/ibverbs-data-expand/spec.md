## ADDED Requirements

### Requirement: 展开打印 IbVerbsData 数组
当 `HcclA2CombineOpParam.data` 非零时，kernel SHALL 将其作为 `IbVerbsData[rankNum]` 数组展开，对每个 rank 打印 `remoteInput`、`remoteOutput`、`localInput`、`localOutput` 四个 `MemDetails` 的 `addr`、`size`、`key` 字段。

#### Scenario: data 非零时展开打印
- **WHEN** `(uint64_t)winContext_->data != 0` 且 `GetBlockIdx() == 0`
- **THEN** 对 `i` 从 0 到 `rankNum-1`，分别打印：
  `[cwh] data[i].remoteInput: addr=0x... size=... key=...`
  `[cwh] data[i].remoteOutput: addr=0x... size=... key=...`
  `[cwh] data[i].localInput: addr=0x... size=... key=...`
  `[cwh] data[i].localOutput: addr=0x... size=... key=...`

#### Scenario: data 为零时不展开
- **WHEN** `(uint64_t)winContext_->data == 0`
- **THEN** 仅打印已有的 `[cwh] data[0x0] dataSize[0]`，不执行展开循环

### Requirement: 展开打印使用 [cwh] 前缀
展开打印的所有行 SHALL 以 `[cwh]` 为前缀，确保可通过 `plog_grep.sh cwh` 过滤。

#### Scenario: plog 过滤
- **WHEN** 用户运行 `bash plog_grep.sh cwh`
- **THEN** 展开的 MemDetails 行与其他 `[cwh]` 行一同出现在输出中
