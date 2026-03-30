## 1. Kernel 展开打印

- [x] 1.1 在 `all_gather_matmul_full_mesh.h` 的 `Init()` 中，已有 `data` 打印行之后，添加 `if ((uint64_t)winContext_->data != 0)` 块
- [x] 1.2 块内循环 `i = 0..rankNum-1`，对每个 `data[i]` 打印 `remoteInput`/`remoteOutput`/`localInput`/`localOutput` 的 `addr`、`size`、`key`

## 2. 验证

- [x] 2.1 同步到测试服务器，执行 `bash ops_build_run.sh` 确认编译通过
- [x] 2.2 运行 `bash ops_only_run.sh`，`bash plog_grep.sh cwh` 确认：当 `data=0x0` 时无展开行；当 `data` 非零时出现 `data[i].remoteInput` 等行
