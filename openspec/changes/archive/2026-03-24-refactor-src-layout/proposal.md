## Why

源码目录（`comm/`、`include/`、`mc2/`）与构建脚本、环境配置、第三方库等混放在仓库根目录，项目结构不清晰，且 `mc2/` 内部存在深达 6 层的相对 `#include` 路径，脆弱且难以维护。统一迁入 `src/`，并通过 `TILEXR_INCS` 变量集中管理编译时头文件搜索路径。

## What Changes

- 新建 `src/` 目录，将 `comm/`、`include/`、`mc2/` 整体移入 `src/`
- 根 `CMakeLists.txt` 调整 `include_directories` 和 `add_subdirectory` 指向 `src/`
- `src/comm/` 内的相对 `#include "../include/..."` 路径保持不变（兄弟目录关系不变）
- `src/mc2/` 内的深层相对 `#include "../../../../../../include/..."` 改为 `#include "comm_args.h"` 等直接引用（依赖 `-I` 传入）
- `mc2/build.sh` 中 `TILEXR_INCS` 追加 `-I<src/include 绝对路径>`，覆盖所有 example 编译命令
- `ops_build_run.sh` 中 `cp` 路径从 `${TILEXR_HOME}/mc2/` 更新为 `${TILEXR_HOME}/src/mc2/`，`include/` 同理

## Capabilities

### New Capabilities

- `src-layout`: 源码集中在 `src/` 下的目录组织规范

### Modified Capabilities

- `autoconf-pkg-install`: 无需求变更，仅目录结构影响

## Impact

- 涉及文件：`CMakeLists.txt`、`src/comm/` 内所有头文件、`src/mc2/` 深层 example/op_host 头文件、`mc2/build.sh`、`ops_build_run.sh`
- 外部行为不变：构建产物 `libtile-comm.so` 路径、安装目标不变
- **BREAKING**：任何直接引用 `${TILEXR_HOME}/comm/`、`${TILEXR_HOME}/include/`、`${TILEXR_HOME}/mc2/` 的外部脚本需同步更新
