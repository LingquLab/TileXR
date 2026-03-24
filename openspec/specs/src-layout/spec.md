## ADDED Requirements

### Requirement: Source code lives under src/
所有 C++ 源码目录（`comm/`、`include/`、`mc2/`）SHALL 位于仓库根目录的 `src/` 子目录下，不得与构建脚本、环境配置、第三方库混放在根目录。

#### Scenario: 目录结构符合规范
- **WHEN** 查看仓库根目录
- **THEN** 根目录中不存在 `comm/`、`include/`、`mc2/`，它们均在 `src/` 下

### Requirement: CMake 通过 src/ 找到源码
根 `CMakeLists.txt` 的 `include_directories` 和 `add_subdirectory` SHALL 指向 `src/` 内的路径。

#### Scenario: CMake 配置正确解析 include 路径
- **WHEN** 在项目根目录执行 `cmake .`
- **THEN** `include_directories` 中包含 `src/include/`，`add_subdirectory` 指向 `src/comm`

### Requirement: TILEXR_INCS 包含 src/include 路径
`mc2/build.sh` 中的 `TILEXR_INCS` 变量 SHALL 包含 `-I<src/include 绝对路径>`，使所有 example 编译命令无需深层相对路径即可找到公共头文件。

#### Scenario: TILEXR_INCS 注入后 example 编译成功
- **WHEN** 执行 `bash mc2/build.sh --run_example all_gather_matmul eager cust ...`
- **THEN** 编译命令中包含 `-I.../src/include`，编译成功，不报找不到头文件的错误

### Requirement: 消除深层相对 include 路径
`src/mc2/` 内所有使用 `../../../../../../include/` 形式的相对 `#include` SHALL 改为直接引用文件名（如 `#include "comm_args.h"`），依赖 `TILEXR_INCS` 提供的 `-I` 路径解析。

#### Scenario: 相对路径不超过两层
- **WHEN** 扫描 `src/mc2/` 下所有 `.h` 和 `.cpp` 文件的 `#include` 指令
- **THEN** 不存在超过两个 `../` 的相对路径

### Requirement: ops_build_run.sh 路径跟随 src/ 更新
`ops_build_run.sh` 中所有引用 `${TILEXR_HOME}/mc2/` 和 `${TILEXR_HOME}/include/` 的路径 SHALL 更新为 `${TILEXR_HOME}/src/mc2/` 和 `${TILEXR_HOME}/src/include/`。

#### Scenario: ops_build_run.sh 复制文件路径正确
- **WHEN** 执行 `bash ops_build_run.sh`
- **THEN** mc2 源码和 include 头文件从 `src/` 下正确复制到 ops-transformer 目标目录
