## 1. Kernel Debug Prints

- [x] 1.1 在 `all_gather_matmul_full_mesh.h` 的 `Init()` 函数 `GetBlockIdx() == 0` 块中，添加打印 `rankId`、`rankNum`、`winSize`
- [x] 1.2 添加循环打印 `windowsIn[i]` 和 `windowsOut[i]`（i 从 0 到 `rankNum-1`）
- [x] 1.3 添加打印 `data` 指针和 `dataSize`

## 2. 验证

- [x] 2.1 将修改同步到测试服务器，执行 `bash ops_build_run.sh` 重新编译
- [x] 2.2 运行 `bash ops_only_run.sh`
- [x] 2.3 执行 `bash plog_grep.sh cwh` 确认以下输出存在：
  - `[cwh] rankId[X] rankNum[Y] winSize[Z]` 行
  - 每个 rank 的 `windowsIn/windowsOut` 地址行（非零）
  - `[cwh] data[0xNNNN] dataSize[Z]`（data 非 null 说明 ibverbs 已填入）
