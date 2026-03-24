## Context

`prepare.sh` 负责安装 TileXR 所依赖的本地工具（time、pigz、patch、ccache、cmake、ripgrep、sshpass、mpich）。其中 time、patch、sshpass、mpich 均为标准 autoconf 项目，安装流程完全相同，却各自写了一份内联代码。

昨天的重构确立了两条原则：公共函数统一到共享文件、环境依赖函数放 `common_env.sh`。本次变更延续这一原则。

## Goals / Non-Goals

**Goals:**
- 提取 autoconf 安装模式为 `_install_autoconf_pkg()` 函数，放入 `common_env.sh`
- `prepare.sh` 中四个重复块替换为函数调用
- 修复 `cd $PWD` bug
- 去掉无意义的 `prepare_3rd()` 函数包装
- 补齐 `colorful_time` 的遗漏调用

**Non-Goals:**
- 不改变任何工具的安装路径或安装参数
- 不处理 cmake 的版本检查逻辑（保持内联，它是特例）
- 不改动 pigz、ccache、ripgrep（它们不用 autoconf，保持内联）

## Decisions

### 决策 1：`_install_autoconf_pkg` 放 `common_env.sh` 而非 `common_util.sh`

函数内部使用 `${TILEXR_3RD_OPEN_HOME}`、`${TILEXR_UTIL_HOME}`、`${TILEXR_TEMP_HOME}`、`${TILEXR_HOME}` 等环境变量。放入 `common_util.sh` 会造成工具层反向依赖环境层（昨天刚修复的问题）。放入 `common_env.sh` 末尾，所有变量已定义，依赖方向正确。

### 决策 2：函数签名 `_install_autoconf_pkg <pkg_name> <tarball> [configure_args...]`

- `pkg_name`：决定 `${TILEXR_UTIL_HOME}/<pkg_name>/` 安装路径和 temp 目录
- `tarball`：文件名（在 `${TILEXR_3RD_OPEN_HOME}/` 下）
- 可变参数透传给 `./configure`（如 mpich 的 `--disable-fortran`）
- mpich 的 `${MPI_HOME}` 与 `${TILEXR_UTIL_HOME}/mpich` 等价，直接复用函数

### 决策 3：去掉 `prepare_3rd()` 包装，改为顶层执行

其他所有脚本均为顶层执行。包装后立即调用没有复用价值，且 `return` 语义在顶层和函数内不同，去掉更清晰。

## Risks / Trade-offs

- **mpich 路径等价性**：`MPI_HOME=${TILEXR_UTIL_HOME}/mpich`，函数用 `${TILEXR_UTIL_HOME}/${pkg_name}` 安装，两者完全等价。若未来 `MPI_HOME` 改到其他路径，需同步修改此处 → 低风险，两处都在 `common_env.sh` 内，可见性好
- **`cd ${TILEXR_HOME}` 在函数末尾**：函数通过 bash 子 shell 调用时无副作用；`source` 调用时会改变调用方 `$PWD`，但 `prepare.sh` 始终通过 `bash` 调用，无问题
