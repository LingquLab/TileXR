## Context

当前仓库根目录混放了源码（`comm/`、`include/`、`mc2/`）和构建脚本、环境配置、第三方依赖。`mc2/` 内的 example 和 op_host 代码通过最多 6 层 `../` 引用公共头文件，路径与目录深度强耦合，一旦文件移动即失效。`mc2/build.sh` 中 `TILEXR_INCS` 已有占位（当前仅有 `-I${ASCEND_HOME_PATH}/pkg_inc`），适合扩展注入 TileXR 自身的头文件路径。

## Goals / Non-Goals

**Goals:**
- `comm/`、`include/`、`mc2/` 整体移入 `src/`
- `mc2/build.sh` 的 `TILEXR_INCS` 追加 `-I` 路径，覆盖所有 example 编译命令
- `src/mc2/` 内深层相对 `#include` 改为直接文件名引用
- `CMakeLists.txt` 和 `ops_build_run.sh` 路径同步更新

**Non-Goals:**
- 不改变构建产物路径（`libtile-comm.so` 安装位置不变）
- 不修改 `3rdparty/`、测试脚本、`op-simulator/` 等无关模块
- 不调整 `comm/` 内部 `../include/` 相对路径（移动后兄弟关系不变，无需改动）

## Decisions

### 决策 1：src/ 层级扁平放置，不再细分
`src/comm/`、`src/include/`、`src/mc2/` 直接位于 `src/` 下，不引入额外子分层（如 `src/lib/`、`src/ops/`）。

**理由**：三者均为一级源码目录，扁平结构保持 `comm/` 与 `include/` 的兄弟关系，内部 `../include/` 路径无需修改，改动最小。

**备选**：将 `include/` 提升为 `src/include/` 并设为全局头文件目录，通过 CMake `target_include_directories PUBLIC` 传播。当前 `comm/CMakeLists.txt` 未使用 PUBLIC include，改动量较大，暂不采用。

### 决策 2：TILEXR_INCS 用绝对路径
`TILEXR_INCS` 中追加的路径使用脚本运行时计算的绝对路径（`$(realpath ...)/src/include`），而非相对路径。

**理由**：`mc2/build.sh` 的工作目录在执行中会切换（`cd ${BUILD_PATH}`），相对路径不可靠。

### 决策 3：深层 #include 改为直接文件名
`#include "../../../../../../include/comm_args.h"` → `#include "comm_args.h"`，依赖 `TILEXR_INCS` 的 `-I` 注入。

**理由**：路径层数与目录深度解耦，后续目录结构调整不会再影响 `#include`。

**备选**：改为 `#include "tilexr/comm_args.h"` 加命名空间目录。增加迁移成本，当前规模不必要。

## Risks / Trade-offs

- **[风险] ops_build_run.sh 漏改路径** → 逐行 grep `TILEXR_HOME}/mc2` 和 `TILEXR_HOME}/include` 确保全量替换
- **[风险] mc2/build.sh 中 TILEXR_INCS 赋值时 src/include 路径尚未确定** → 在赋值行前用 `TILEXR_SRC_INCLUDE=$(realpath "${CURRENT_DIR}/../src/include")` 计算，`CURRENT_DIR` 在脚本顶部已定义为 `mc2/build.sh` 所在目录
- **[风险] comm/ 内 sock_exchange.h 用 `"tilexr_types.h"` 裸引用（无路径前缀）**，依赖 CMake `include_directories` 已含 `src/include/` → CMakeLists.txt 更新后自动解决，无需改头文件

## Migration Plan

1. `git mv comm include mc2` → `src/`（保留 git 历史）
2. 更新根 `CMakeLists.txt`（2 处路径）
3. 更新 `mc2/build.sh` 的 `TILEXR_INCS`
4. 修改 `src/mc2/` 内深层相对 `#include`（2 个文件）
5. 更新 `ops_build_run.sh`（4 处路径）
6. 本地执行 `cmake . && make` 验证编译通过

回滚：`git revert` 即可，无数据库或外部系统变更。

## Open Questions

- `src/mc2/build.sh` 移动后，`ops_build_run.sh` 中 `cp -f ${TILEXR_HOME}/mc2/build.sh ${TILEXR_OPS_HOME}/build.sh` 是否需同步改为 `src/mc2/build.sh`？（是，需要改）
